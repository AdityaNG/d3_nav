# NCCL_DEBUG=INFO NCCL_P2P_LEVEL=NVL CUBLAS_WORKSPACE_CONFIG=:16:8 nohup python3 -m d3nav.scripts.train &  # noqa
import math

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from nuscenes.nuscenes import NuScenes

# from nuscenes.utils.data_classes import Box
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from d3nav.datasets.nusc import NuScenesDataset
from d3nav.metric_stp3 import PlanningMetric
from d3nav.model.d3nav import D3Nav

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.use_deterministic_algorithms(True, warn_only=True)

effective_batch_size = 96
num_devices = 2
batch_size_per_device = 12

accumulate_grad_batches = int(
    effective_batch_size / (batch_size_per_device * num_devices)
)

assert (
    accumulate_grad_batches * batch_size_per_device * num_devices
    == effective_batch_size
)

learning_rate = 5e-5 * math.sqrt(effective_batch_size / 24)

num_layers = 3
traj_requires_grad = True
temporal_context = 1


class D3NavTrainingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = D3Nav(temporal_context=temporal_context)
        self.metric = PlanningMetric()

        self.model.dropout_rate = 0.0  # frame level dropout
        self.model.freeze_traj_dec(
            requires_grad=traj_requires_grad,
        )
        self.model.unfreeze_last_n_layers(num_layers=num_layers)
        self.clip_grad_norm = 1.0  # Standard value, adjust if needed

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_trajectory = self(x)
        loss = torch.nn.functional.l1_loss(pred_trajectory, y)
        lrs = self.scheduler.get_last_lr()

        if batch_idx % 10 == 0:
            self.log(
                "train_loss",
                loss,
            )
            self.log(
                "learning_rate",
                sum(lrs) / float(len(lrs)),
            )
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        batch_size = x.shape[0]

        pred_trajectory = self(x)
        loss = torch.nn.functional.l1_loss(pred_trajectory, y)
        self.log(
            "val_loss",
            loss,
        )

        l2_1s_l = []
        l2_2s_l = []
        l2_3s_l = []

        # Calculate metrics
        for batch_index in range(batch_size):
            l2_1s = self.metric.compute_L2(
                pred_trajectory[batch_index, :2, :2], y[batch_index, :2, :2]
            )
            l2_2s = self.metric.compute_L2(
                pred_trajectory[batch_index, :4, :2], y[batch_index, :4, :2]
            )
            l2_3s = self.metric.compute_L2(
                pred_trajectory[batch_index, :, :2], y[batch_index, :, :2]
            )

            l2_1s_l += [l2_1s]
            l2_2s_l += [l2_2s]
            l2_3s_l += [l2_3s]

        l2_1s = np.array(l2_1s_l).mean()
        l2_2s = np.array(l2_2s_l).mean()
        l2_3s = np.array(l2_3s_l).mean()

        self.log_dict(
            {
                "val_l2_1s": l2_1s,
                "val_l2_2s": l2_2s,
                "val_l2_3s": l2_3s,
            },
        )

        # TODO: fix this
        # if bboxes is not None:

        #     segmentation, pedestrian = self.planning_metric.get_label(
        #         bboxes, bboxes)
        #     occupancy = torch.logical_or(segmentation, pedestrian)

        #     obj_coll_sum, obj_box_coll_sum = self.metric.evaluate_coll(pred_trajectory[:, :, :2], y[:, :, :2], bboxes)  # noqa
        #     col_1s = obj_box_coll_sum[:2].sum() / (2 * len(batch))
        #     col_2s = obj_box_coll_sum[:4].sum() / (4 * len(batch))
        #     col_3s = obj_box_coll_sum.sum() / (6 * len(batch))

        #     self.log_dict({
        #         'val_col_1s': col_1s,
        #         'val_col_2s': col_2s,
        #         'val_col_3s': col_3s,
        #     })

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=learning_rate)

        # Warmup parameters
        num_training_steps = 4000
        num_warmup_steps = round(
            0.05 * num_training_steps
        )  # Typically 5-10% of total steps

        def lr_lambda(current_step: int):
            # Linear warmup
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            # Cosine decay after warmup
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "step",  # Update at each step rather than epoch
            "frequency": 1,
            "name": "d3nav_lr_scheduler",
        }

        self.scheduler = scheduler
        return [optimizer], [lr_scheduler]


def custom_collate(batch):
    x = []
    y = []
    bboxes = []
    for sample in batch:
        x.append(sample[0])
        y.append(sample[1])
        if len(sample) > 2:
            bboxes.append(sample[2])

    x = default_collate(x)
    y = default_collate(y)

    if bboxes:
        return x, y, bboxes
    return x, y


def main():
    # Initialize NuScenes
    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot="/media/NG/datasets/nuscenes/",
        verbose=True,
    )

    # Create datasets and dataloaders
    train_dataset = NuScenesDataset(
        nusc, is_train=True, temporal_context=temporal_context
    )
    val_dataset = NuScenesDataset(
        nusc, is_train=False, temporal_context=temporal_context
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ckpt = None
    ckpt = (
        "checkpoints/traj_quantizer/d3nav-traj-epoch-132-val_loss-0.2792.ckpt"
    )
    # ckpt = "checkpoints/d3nav/d3nav-epoch-06-val_loss-0.6668.ckpt"
    # ckpt = "checkpoints/d3nav/d3nav-epoch-03-val_loss-0.7735.ckpt"

    if ckpt is None:
        # Initialize training module
        training_module = D3NavTrainingModule()
    else:
        training_module = D3NavTrainingModule.load_from_checkpoint(
            ckpt, strict=False
        )

    # Initialize logger
    logger = WandbLogger(project="D3Nav-NuScenes")

    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=-1,  # Save all checkpoints
        filename="d3nav-{epoch:02d}-{val_loss:.4f}",
        every_n_epochs=1,  # Save every epoch
        dirpath="wandb/latest-run/checkpoints",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_devices,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=0.5,
        logger=logger,
        callbacks=[
            checkpoint_callback,
        ],
        precision="bf16-mixed",
        num_sanity_val_steps=1,
        strategy="ddp",  # distributed strategy
        sync_batchnorm=True,  # Synchronize batch normalization between processes  # noqa
    )

    # Train the model
    trainer.fit(
        training_module,
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    main()

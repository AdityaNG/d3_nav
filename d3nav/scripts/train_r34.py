# NCCL_DEBUG=INFO NCCL_P2P_LEVEL=NVL CUBLAS_WORKSPACE_CONFIG=:16:8 nohup python3 -m d3nav.scripts.train_r34 &  # noqa

import math

import lightning.pytorch as pl
import numpy as np
import torch
import torchvision.models as models
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


class ResNet34Trajectory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet34
        resnet = models.resnet34(pretrained=True)

        # Remove the last layer
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])

        # Add new layers for trajectory prediction
        # Output shape should be (batch_size, 6, 2) for 3 seconds, 2D coordinates  # noqa
        self.trajectory_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 12),  # 6 points * 2 coordinates
        )

    def forward(self, x):
        # Expected input: (batch_size, time, channels, height, width)

        x = x[:, 0, :, :, :]
        # x: (batch_size, channels, height, width)
        features = self.features(x)
        features = torch.flatten(features, 1)
        trajectory = self.trajectory_head(features)
        # Reshape to (batch_size, 6, 2) for trajectory coordinates
        return trajectory.view(-1, 6, 2)


class ResNet34TrainingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ResNet34Trajectory()
        self.metric = PlanningMetric()
        self.clip_grad_norm = 1.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_trajectory = self(x)
        loss = torch.nn.functional.l1_loss(
            pred_trajectory, y[..., :2]
        )  # Only use x,y coordinates
        lrs = self.scheduler.get_last_lr()

        if batch_idx % 10 == 0:
            self.log("train_loss", loss)
            self.log("learning_rate", sum(lrs) / float(len(lrs)))
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        batch_size = x.shape[0]
        pred_trajectory = self(x)
        loss = torch.nn.functional.l1_loss(
            pred_trajectory, y[..., :2]
        )  # Only use x,y coordinates
        self.log("val_loss", loss)

        l2_1s_l = []
        l2_2s_l = []
        l2_3s_l = []

        # Calculate metrics
        for batch_index in range(batch_size):
            l2_1s = self.metric.compute_L2(
                pred_trajectory[batch_index, :2], y[batch_index, :2, :2]
            )
            l2_2s = self.metric.compute_L2(
                pred_trajectory[batch_index, :4], y[batch_index, :4, :2]
            )
            l2_3s = self.metric.compute_L2(
                pred_trajectory[batch_index], y[batch_index, :, :2]
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

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=learning_rate)

        # Warmup parameters
        num_training_steps = 800 * 10
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
    # ... existing code ...
    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot="/media/NG/datasets/nuscenes/",
        verbose=True,
    )

    train_dataset = NuScenesDataset(nusc, is_train=True)
    val_dataset = NuScenesDataset(nusc, is_train=False)

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

    # Initialize training module
    training_module = ResNet34TrainingModule()

    # Initialize logger
    logger = WandbLogger(project="ResNet34-NuScenes-Baseline")

    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
        filename="resnet34-{epoch:02d}-{val_loss:.4f}",
        every_n_epochs=1,
        dirpath="wandb/latest-run/checkpoints",
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_devices,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=0.125,
        logger=logger,
        callbacks=[
            checkpoint_callback,
        ],
        precision="bf16-mixed",
        num_sanity_val_steps=1,
        strategy="ddp",
        sync_batchnorm=True,
    )

    trainer.fit(
        training_module,
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    main()

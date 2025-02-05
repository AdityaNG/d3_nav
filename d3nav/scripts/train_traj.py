import numpy as np
import pytorch_lightning as pl
import torch
from nuscenes.nuscenes import NuScenes

# from nuscenes.utils.data_classes import Box
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from d3nav.datasets.nusc import NuScenesDataset
from d3nav.metric_stp3 import PlanningMetric
from d3nav.model.d3nav import DEFAULT_DATATYPE, D3Nav

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.use_deterministic_algorithms(True, warn_only=True)


class D3NavTrainingModule(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = D3Nav().to(dtype=DEFAULT_DATATYPE)
        self.metric = PlanningMetric()

        self.model.freeze_traj_enc_dec(requires_grad=True)

    def forward(self, y):
        return self.model.traj_quantize(y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_trajectory = self(y)
        loss = torch.nn.functional.l1_loss(pred_trajectory, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        batch_size = x.shape[0]

        pred_trajectory = self(y)
        loss = torch.nn.functional.l1_loss(pred_trajectory, y)
        self.log("val_loss", loss)

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
            }
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


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
    train_dataset = NuScenesDataset(nusc, is_train=True)
    val_dataset = NuScenesDataset(nusc, is_train=False)

    batch_size = 512

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ckpt = None
    ckpt = (
        "checkpoints/traj_quantizer/d3nav-traj-epoch-42-val_loss-1.3925.ckpt"
    )

    if ckpt is None:
        # Initialize training module
        training_module = D3NavTrainingModule()
    else:
        training_module = D3NavTrainingModule.load_from_checkpoint(ckpt)

    # Initialize logger
    logger = WandbLogger(project="D3Nav-NuScenes-Traj")

    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=-1,  # Save all checkpoints
        filename="d3nav-traj-{epoch:02d}-{val_loss:.4f}",
        every_n_epochs=1,  # Save every epoch
        dirpath="wandb/latest-run/checkpoints",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision="bf16-mixed",
    )

    # Train the model
    trainer.fit(
        training_module,
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    main()

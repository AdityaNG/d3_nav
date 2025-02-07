import pytorch_lightning as pl
import torch
from nuscenes.nuscenes import NuScenes

# from nuscenes.utils.data_classes import Box
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from d3nav.datasets.nusc import NuScenesDataset
from d3nav.model.trainer import D3NavTrajTrainingModule

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.use_deterministic_algorithms(True, warn_only=True)


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
        training_module = D3NavTrajTrainingModule()
    else:
        training_module = D3NavTrajTrainingModule.load_from_checkpoint(ckpt)

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

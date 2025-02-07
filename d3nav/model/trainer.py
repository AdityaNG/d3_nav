import numpy as np
import pytorch_lightning as pl
import torch

from d3nav.metric_stp3 import PlanningMetric
from d3nav.model.d3nav import DEFAULT_DATATYPE, D3Nav


class D3NavTrajTrainingModule(pl.LightningModule):
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

# %%
from pathlib import Path
from typing import Any, Optional, Tuple
import pytorch_lightning as pl
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.metrics.functional import get_stats, iou_score
from segmentation_models_pytorch.losses import JaccardLoss
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    STEP_OUTPUT,
    EVAL_DATALOADERS,
    EPOCH_OUTPUT,
)
import time
from alceo.dataset.pits import PitsSiteDataset
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torchvision.transforms.functional as tvf
from torch.nn.functional import one_hot
from sorcery import dict_of, unpack_keys
from torchmetrics import MeanMetric, MetricCollection


class PitsChangeDetectionNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.siamese_network = Unet(
            encoder_name="resnet18",
            in_channels=3,
            encoder_depth=3,
        )
        self.change_network = Unet(
            encoder_name="resnet18",
            encoder_depth=3,
            in_channels=16,
            classes=2,
        )

    def forward(
        self, im1, im2
    ) -> Tuple[torch.TensorType, torch.TensorType, torch.TensorType]:
        # Siamese feature extraction
        features1 = self.siamese_network.decoder(*self.siamese_network.encoder(im1))
        features2 = self.siamese_network.decoder(*self.siamese_network.encoder(im2))
        features = torch.cat((features1, features2), dim=1)
        change_pred = self.change_network(features)

        return change_pred


class PitsLightningModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.batch_ids_to_monitor = None
        self.network = PitsChangeDetectionNetwork()
        self.loss_fn = JaccardLoss(mode="multilabel")
        self.mIoU = MetricCollection(
            {
                "stage_train": MeanMetric(dist_sync_on_step=True),
                "stage_validation": MeanMetric(dist_sync_on_step=True),
            }
        )
        self.mIoU1 = MetricCollection(
            {
                "stage_train": MeanMetric(dist_sync_on_step=True),
                "stage_validation": MeanMetric(dist_sync_on_step=True),
            }
        )
        self.mIoU2 = MetricCollection(
            {
                "stage_train": MeanMetric(dist_sync_on_step=True),
                "stage_validation": MeanMetric(dist_sync_on_step=True),
            }
        )

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = PitsSiteDataset(
            Path("/HDD1/gsech/source/alceo/dataset/pits/DURA_EUROPOS")
        )
        self.train_dataset, self.val_dataset, self.train_dataset = random_split(
            dataset, [0.6, 0.2, 0.2]
        )
        return super().setup(stage)

    def _dataloader_for_dataset(self, dataset: Dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=5,
            num_workers=5,
        )

    def configure_optimizers(self) -> Any:
        return torch.optim.SGD(params=self.parameters())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader_for_dataset(self.train_dataset)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader_for_dataset(self.validation_dataset)

    def _shared_step(self, batch, stage: str) -> STEP_OUTPUT:

        pits_appeared = batch["pits.appeared"]
        pits_disappeared = batch["pits.disappeared"]
        im1 = batch["im1"]
        im2 = batch["im2"]
        change_truth = torch.stack([pits_appeared, pits_disappeared], dim=1)

        change_pred = self.network(im1, im2)

        # computing loss


        # computing full loss!
        loss = self.loss_fn(change_pred, change_truth)
        

        return loss

    def _shared_step_end(self, stage, step_output):
        (
            loss,
            batch_loss,
            sem_loss1,
            sem_loss2,
            change_loss,
            regularization,
            iou,
            iou1,
            iou2,
            image_id,
        ) = unpack_keys(step_output)

        mean_iou = iou.mean(dim=1)
        mean_iou1 = iou1.mean(dim=1)
        mean_iou2 = iou2.mean(dim=1)

        mIoU = self.mIoU[f"stage_{stage}"]
        mIoU1 = self.mIoU1[f"stage_{stage}"]
        mIoU2 = self.mIoU2[f"stage_{stage}"]
        mIoU.update(mean_iou.to(device=mIoU.device))
        mIoU1.update(mean_iou1.to(device=mIoU1.device))
        mIoU2.update(mean_iou2.to(device=mIoU2.device))
        cli_tag = f"[rank {self.trainer.global_rank}][epoch {self.trainer.current_epoch}][step {self.trainer.global_step}]"
        assert isinstance(image_id, list), "Schifo"
        if self.batch_ids_to_monitor is not None:
            # print(f"{cli_tag} ids to monitor: {self.batch_ids_to_monitor}", end = "\n\n")
            # print(f"{cli_tag} ids to check: {image_id}", end = "\n\n")

            if self.global_rank == 0:
                for i, _id in enumerate(image_id):
                    print(
                        f"{cli_tag} checking id: {_id} (of {image_id}) (in {self.batch_ids_to_monitor})",
                        end="\n\n",
                        flush=True,
                    )
                    if _id in self.batch_ids_to_monitor:
                        print(f"{cli_tag} found monitored: {_id}", flush=True)
                        tag = f"{stage}_data/{_id}"
                        image_metrics = {
                            f"{tag}/loss": batch_loss[i],
                            f"{tag}/IoU": mean_iou[i],
                            f"{tag}/IoU1": mean_iou1[i],
                            f"{tag}/IoU2": mean_iou2[i],
                            f"{tag}/change_loss": change_loss[i],
                            f"{tag}/sem_loss1": sem_loss1[i],
                            f"{tag}/sem_loss2": sem_loss2[i],
                        }
                        self.log_dict(
                            image_metrics,
                            batch_size=1,
                            on_step=False,
                            on_epoch=True,
                        )

        batch_size = batch_loss.shape[0]
        mean_sem_loss1 = sem_loss1.mean()
        mean_sem_loss2 = sem_loss2.mean()
        mean_change_loss = change_loss.mean()

        to_log = {
            f"{stage}/{key}": val
            for key, val in dict_of(
                loss,
                mean_sem_loss1,
                mean_sem_loss2,
                mean_change_loss,
                regularization,
                mIoU,
                mIoU1,
                mIoU2,
            ).items()
        }

        self.log_dict(
            to_log,
            batch_size=batch_size,
            on_epoch=True,
            sync_dist=True,
        )

    # def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
    #     return self._shared_step_end("train", step_output)

    # def validation_step_end(self, step_output: STEP_OUTPUT) -> Optional[STEP_OUTPUT]:
    #     return self._shared_step_end("validation", step_output)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch, batch_idx, dataloader_idx=0, *args: Any, **kwargs: Any
    ) -> Optional[STEP_OUTPUT]:
        return self._shared_step(batch, "validation")

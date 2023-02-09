# %%
from typing import Any, Optional, Tuple
import pytorch_lightning as pl
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.metrics.functional import get_stats, iou_score
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    STEP_OUTPUT,
    EVAL_DATALOADERS,
    EPOCH_OUTPUT,
)
import time
from dataset.second import SECONDataset
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision.transforms.functional as tvf
from torch.nn.functional import one_hot
from sorcery import dict_of, unpack_keys
from torchmetrics import MeanMetric, MetricCollection


class ChangeDetectionNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.siamese_network = Unet(
            encoder_name="resnet18",
            in_channels=3,
        )
        self.semantic_network = Unet(
            encoder_name="resnet18",
            in_channels=16,
            classes=len(SECONDataset.label_names) - 1,
        )
        self.change_network = Unet(
            encoder_name="resnet18",
            in_channels=32,
            classes=2,
        )

    def forward(
        self, im1, im2
    ) -> Tuple[torch.TensorType, torch.TensorType, torch.TensorType]:
        # Siamese feature extraction
        features1 = self.siamese_network.decoder(*self.siamese_network.encoder(im1))
        features2 = self.siamese_network.decoder(*self.siamese_network.encoder(im2))
        features = torch.cat((features1, features2), dim=1)

        seg_pred1 = self.semantic_network(features1)
        seg_pred2 = self.semantic_network(features2)
        change_pred = self.change_network(features)

        return seg_pred1, seg_pred2, change_pred


class AlceoLightningModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.batch_ids_to_monitor = None
        self.network = ChangeDetectionNetwork()
        self.semantic_loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=6, reduction="none"
        )
        self.change_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
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
        self.train_dataset = SECONDataset(
            dataset_root="/home/gsech/Source/alceo/data/second_dataset",
            is_train=True,
        )
        self.validation_dataset = SECONDataset(
            dataset_root="/home/gsech/Source/alceo/data/second_dataset",
            is_train=False,
        )
        return super().setup(stage)

    def _dataloader_for_dataset(self, dataset: Dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=5,
            num_workers=5,
        )

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(params=self.parameters())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader_for_dataset(self.train_dataset)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader_for_dataset(self.validation_dataset)
    
# [rank 2][step 0] train image_id: 00045
# [rank 2][step 0] train image_id: 00068
# [rank 2][step 0] train image_id: 00031
# [rank 2][step 0] train image_id: 00046
# [rank 2][step 0] train image_id: 00056
# [rank 2][step 0] train image_id: 00028
# [rank 2][step 0] train image_id: 00043
# [rank 2][step 0] train image_id: 00089
# [rank 2][step 0] train image_id: 00050
# [rank 2][step 0] train image_id: 00083
# [rank 1][step 0] train image_id: 00011
# [rank 1][step 0] train image_id: 00079
# [rank 1][step 0] train image_id: 00088
# [rank 1][step 0] train image_id: 00058
# [rank 1][step 0] train image_id: 00013
# [rank 1][step 0] train image_id: 00003
# [rank 1][step 0] train image_id: 00086
# [rank 1][step 0] train image_id: 00041
# [rank 1][step 0] train image_id: 00057
# [rank 1][step 0] train image_id: 00078
# [rank 0][step 0] train image_id: 00038
# [rank 0][step 0] train image_id: 00029
# [rank 0][step 0] train image_id: 00052
# [rank 0][step 0] train image_id: 00017
# [rank 0][step 0] train image_id: 00065
# [rank 0][step 0] train image_id: 00025
# [rank 0][step 0] train image_id: 00015
# [rank 0][step 0] train image_id: 00016
# [rank 0][step 0] train image_id: 00034
# [rank 0][step 0] train image_id: 00042
# [rank 3][step 0] train image_id: 00067
# [rank 3][step 0] train image_id: 00071
# [rank 3][step 0] train image_id: 00040
# [rank 3][step 0] train image_id: 00076
# [rank 3][step 0] train image_id: 00020
# [rank 3][step 0] train image_id: 00018
# [rank 3][step 0] train image_id: 00080
# [rank 3][step 0] train image_id: 00021
# [rank 3][step 0] train image_id: 00060
# [rank 3][step 0] train image_id: 00026

    def _shared_step(self, batch, stage: str) -> STEP_OUTPUT:

        label1 = batch["label1"]
        label2 = batch["label2"]
        im1 = batch["im1"]
        im2 = batch["im2"]
        image_id = batch["image_id"]
        if stage == "train" and self.batch_ids_to_monitor is None:
            self.batch_ids_to_monitor = set(image_id)
            cli_tag = f"[rank {self.trainer.global_rank}][step {self.trainer.global_step}]"
            for i in list(self.trainer.train_dataloader.sampler):
                _image_id = self.trainer.train_dataloader.dataset.datasets.image_ids[i]
                print(f"{cli_tag} train image_id: {_image_id}", flush=True)
                
            print(f"{cli_tag} ids to monitor: {self.batch_ids_to_monitor}", flush=True)

        seg_pred1, seg_pred2, no_change_pred = self.network(im1, im2)

        # computing loss

        # producing change ground truth
        no_change_mask = (label1 == 6).to(dtype=torch.long)

        # computing l1 of parameters for regularization
        params = torch.cat([param.view(-1) for param in self.parameters()])
        regularization = params.norm(p=1)

        # computing semantic segmentation losses
        sem_loss1 = 1.0e-1 * self.semantic_loss_fn(seg_pred1, label1).sum(dim=(1, 2))
        sem_loss2 = 1.0e-1 * self.semantic_loss_fn(seg_pred2, label2).sum(dim=(1, 2))

        # computing loss for change class
        change_loss = self.change_loss_fn(no_change_pred, no_change_mask).sum(
            dim=(1, 2)
        )

        # computing full loss!
        batch_loss = sem_loss1 + sem_loss2 + change_loss + 1.0e-2 * regularization

        loss = batch_loss.mean()

        # computing metrics!

        # binary mask for no-change class
        no_change_pred_mask = no_change_pred.max(dim=1)[1] == 1

        seg1 = seg_pred1.max(dim=1)[1]
        seg1[no_change_pred_mask] = 6

        seg2 = seg_pred2.max(dim=1)[1]
        seg2[no_change_pred_mask] = 6

        stats1 = get_stats(seg1, label1, mode="multiclass", num_classes=7)
        stats2 = get_stats(seg2, label2, mode="multiclass", num_classes=7)

        iou1 = iou_score(*stats1, reduction="none")
        iou2 = iou_score(*stats2, reduction="none")

        iou = (iou1 + iou2) / 2

        assert not torch.logical_or(
            torch.isnan(iou1).any(), torch.isnan(iou1).any()
        ), "Found NaN or Inf"
        assert not torch.logical_or(
            torch.isnan(iou2).any(), torch.isnan(iou2).any()
        ), "Found NaN or Inf"
        assert not torch.logical_or(
            torch.isnan(iou).any(), torch.isnan(iou).any()
        ), "Found NaN or Inf"

        return dict_of(
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
        )

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
                    print(f"{cli_tag} checking id: {_id} (of {image_id}) (in {self.batch_ids_to_monitor})", end="\n\n", flush=True)
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

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        return self._shared_step_end("train", step_output)

    def validation_step_end(self, step_output: STEP_OUTPUT) -> Optional[STEP_OUTPUT]:
        return self._shared_step_end("validation", step_output)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch, batch_idx, dataloader_idx=0, *args: Any, **kwargs: Any
    ) -> Optional[STEP_OUTPUT]:
        return self._shared_step(batch, "validation")

        """
        
[rank 2][step 0] found monitored: 05682
[rank 2][step 0] found monitored: 08743
[rank 2][step 0] found monitored: 02255
[rank 0][step 0] found monitored: 01537
[rank 2][step 0] found monitored: 07400
[rank 2][step 0] found monitored: 05220
[rank 0][step 0] found monitored: 03423
[rank 0][step 0] found monitored: 09334
[rank 0][step 0] found monitored: 01662
[rank 3][step 0] found monitored: 04020
[rank 1][step 0] found monitored: 01725
[rank 0][step 0] found monitored: 02301
[rank 3][step 0] found monitored: 01883
[rank 1][step 0] found monitored: 06130
[rank 1][step 0] found monitored: 05548
[rank 1][step 0] found monitored: 09344
[rank 3][step 0] found monitored: 09794
[rank 1][step 0] found monitored: 04568
[rank 3][step 0] found monitored: 09548
[rank 3][step 0] found monitored: 10971




[rank 0][step 0] found monitored: 01537
[rank 0][step 0] found monitored: 03423
[rank 0][step 0] found monitored: 09334
[rank 0][step 0] found monitored: 01662
[rank 0][step 0] found monitored: 02301









05682
08743
02255
07400
05220
04539
03398
04264
01601
11370
08710
04410
24649
08070
02973
07305
03137
11863


        """
# %%
from typing import Any, List, Optional, Tuple
import pytorch_lightning as pl
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.metrics.functional import get_stats, iou_score
from segmentation_models_pytorch.losses import JaccardLoss
from pytorch_lightning.utilities.types import (
    STEP_OUTPUT,
)
import torch
import torchvision.transforms.functional as tvf
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class PitsChangeDetectionNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.siamese_network = Unet(
            encoder_name="resnet18",
            in_channels=4,
            encoder_depth=3,
            decoder_channels=(256, 128, 64),
        )
        self.change_network = Unet(
            encoder_name="resnet18",
            encoder_depth=3,
            decoder_channels=(256, 128, 64),
            in_channels=128,
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
    def __init__(
        self,
        network: PitsChangeDetectionNetwork,
        training_labels: List[str],
        validation_labels: List[str],
        test_labels: List[str],
    ) -> None:
        super().__init__()
        self.batch_ids_to_monitor = None
        self.network = network
        self.loss_fn = JaccardLoss(mode="multilabel")

        self.training_labels = training_labels
        self.validation_labels = validation_labels
        self.test_labels = test_labels
        self.setup_metrics()

    def update_for_tag(self, tag: str, predicted: torch.Tensor, correct: torch.Tensor):
        stats = get_stats(
            predicted,
            correct,
            mode="binary",
        )

        ious = iou_score(*stats)
        ne_ious = iou_score(*stats, zero_division=-1)
        ne_weights = torch.ones_like(ne_ious)
        ne_weights[ne_ious < 0.0] = 0.0

        self.torchmetrics[f"{tag}/mIoU"](ious)
        self.torchmetrics[f"{tag}/ne_mIoU"](ne_ious, ne_weights)

        self.torchmetrics[f"{tag}/IoU"](predicted, correct)
        self.torchmetrics[f"{tag}/F1"](predicted, correct)
        self.torchmetrics[f"{tag}/precision"](predicted, correct)
        self.torchmetrics[f"{tag}/recall"](predicted, correct)

    def log_for_tag(self, tag: str):
        for change_kind in ["appeared", "disappeared"]:
            log_tag = f"{tag}/{change_kind}"
            self.log(
                f"{log_tag}/mIoU",
                self.torchmetrics[f"{log_tag}/mIoU"],
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
            self.log(
                f"{log_tag}/ne_mIoU",
                self.torchmetrics[f"{log_tag}/ne_mIoU"],
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
            self.log(
                f"{log_tag}/IoU",
                self.torchmetrics[f"{log_tag}/IoU"],
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
            self.log(
                f"{log_tag}/F1",
                self.torchmetrics[f"{log_tag}/F1"],
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
            self.log(
                f"{log_tag}/precision",
                self.torchmetrics[f"{log_tag}/precision"],
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
            self.log(
                f"{log_tag}/recall",
                self.torchmetrics[f"{log_tag}/recall"],
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )

    def setup_metrics(self):
        def metrics_for_tag(tag: str):
            return {
                f"{tag}/mIoU": MeanMetric(),
                f"{tag}/ne_mIoU": MeanMetric(),
                f"{tag}/IoU": BinaryJaccardIndex(),
                f"{tag}/F1": BinaryF1Score(),
                f"{tag}/precision": BinaryPrecision(),
                f"{tag}/recall": BinaryRecall(),
            }

        # training metrics
        _metrics = {}

        for change_kind in ["appeared", "disappeared"]:
            _metrics.update(metrics_for_tag(f"training/{change_kind}"))
            for label in self.training_labels:
                _metrics.update(metrics_for_tag(f"training/{label}/{change_kind}"))

            _metrics.update(metrics_for_tag(f"validation/{change_kind}"))
            for label in self.validation_labels:
                _metrics.update(metrics_for_tag(f"validation/{label}/{change_kind}"))

            _metrics.update(metrics_for_tag(f"testing/{change_kind}"))
            for label in self.test_labels:
                _metrics.update(metrics_for_tag(f"testing/{label}/{change_kind}"))

        self.torchmetrics = MetricCollection(_metrics)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=1e-5,
        )

    def forward(self, batch) -> torch.Tensor:
        im1 = batch["im1"]
        im2 = batch["im2"]
        return self.network(im1, im2)

    def _shared_step(
        self,
        batch,
        stage_tag,
        dataloader_tag=None,
    ) -> STEP_OUTPUT:

        change_activation = self.forward(batch)
        change_target = torch.cat(
            [batch["pits.appeared"], batch["pits.disappeared"]], dim=1
        )

        # computing loss
        loss = self.loss_fn(change_activation, change_target)
        self.log(
            f"{stage_tag}/loss",
            loss,
            sync_dist=True,
            on_epoch=True,
        )
        if dataloader_tag is not None:
            self.log(
                f"{dataloader_tag}/loss",
                loss,
                sync_dist=True,
                on_epoch=True,
            )

        appeared_target = change_target[:, 0]
        disappeared_target = change_target[:, 1]
        appeared_pred = change_activation[:, 0] > 0.5
        disappeared_pred = change_activation[:, 1] > 0.5

        self.update_for_tag(f"{stage_tag}/appeared", appeared_pred, appeared_target)
        if dataloader_tag is not None:
            self.update_for_tag(
                f"{dataloader_tag}/appeared", appeared_pred, appeared_target
            )

        self.update_for_tag(
            f"{stage_tag}/disappeared", disappeared_pred, disappeared_target
        )
        if dataloader_tag is not None:
            self.update_for_tag(
                f"{dataloader_tag}/disappeared", disappeared_pred, disappeared_target
            )

        return loss

    def training_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=-1,
        *args,
        **kwargs,
    ) -> STEP_OUTPUT:
        if isinstance(batch, list):
            return None

        dataloader_tag = None
        if dataloader_idx > -1:
            dataloader_tag = f"training/{self.training_labels[dataloader_idx]}"
        return self._shared_step(batch, "training", dataloader_tag=dataloader_tag)

    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=-1,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[STEP_OUTPUT]:

        if isinstance(batch, list):
            return None

        dataloader_tag = None
        if dataloader_idx > -1:
            dataloader_tag = f"validation/{self.validation_labels[dataloader_idx]}"
        return self._shared_step(
            batch,
            "validation",
            dataloader_tag=dataloader_tag,
        )

    def test_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=-1,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[STEP_OUTPUT]:
        if isinstance(batch, list):
            return None

        dataloader_tag = None
        if dataloader_idx > -1:
            dataloader_tag = f"testing/{self.test_labels[dataloader_idx]}"
        return self._shared_step(batch, "testing", dataloader_tag=dataloader_tag)

    def on_train_epoch_end(self) -> None:
        stage_tag = "training"
        self.log_for_tag(stage_tag)
        for label in self.training_labels:
            dataloader_tag = f"{stage_tag}/{label}"
            self.log_for_tag(dataloader_tag)
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        stage_tag = "validation"
        self.log_for_tag(stage_tag)
        for label in self.validation_labels:
            dataloader_tag = f"{stage_tag}/{label}"
            self.log_for_tag(dataloader_tag)
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        stage_tag = "testing"
        self.log_for_tag(stage_tag)
        for label in self.test_labels:
            dataloader_tag = f"{stage_tag}/{label}"
            self.log_for_tag(dataloader_tag)
        return super().on_test_epoch_end()

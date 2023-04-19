from typing import Any, List, Optional
import pytorch_lightning as pl
from segmentation_models_pytorch.metrics.functional import get_stats, iou_score
from pytorch_lightning.utilities.types import (
    STEP_OUTPUT,
)
import torch
import torch.nn as nn
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

_TRAIN_IDX = 0
_VALIDATION_IDX = 1
_TEST_IDX = 2


class AlceoMetricModule(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss_fn: nn.Module,
        train_labels: List[str],
        validation_labels: List[str],
        test_labels: List[str],
    ) -> None:
        """A LightningModule for training, validating and testing change detection models on the ALCEO dataset.

        Args:
            network (nn.Module): A PyTorch Module that takes as input two images and returns multilabel activations for pits that appeared (channel 0) and pits that disappeared (channel 1)
            loss_fn (nn.Module): The loss function used to optimise the network.
            train_labels (List[str]): The tags to use for the training datasets
            validation_labels (List[str]): The tags to use for the validation datasets
            test_labels (List[str]): The tags to use for the test datasets
        """
        super().__init__()
        self.save_hyperparameters(ignore=["network", "loss_fn"])
        self.network = network
        self.loss_fn = loss_fn
        self.train_labels = self.hparams.train_labels
        self.validation_labels = self.hparams.validation_labels
        self.test_labels = self.hparams.test_labels
        self._phase_labels = [
            self.train_labels,
            self.validation_labels,
            self.test_labels,
        ]
        self._phases = ["training", "validation", "testing"]
        self.change_kinds = ["appeared", "disappeared"]
        self._setup_metrics()

    def forward(self, batch) -> torch.Tensor:
        im1 = batch["im1"]
        im2 = batch["im2"]
        return self.network(im1, im2)

    def shared_step(
        self,
        batch,
        phase_tag,
        dataloader_tag=None,
    ) -> STEP_OUTPUT:
        """The network logic used for processing a single batch in training/validation/testing.

        Args:
            batch (dict): A dictionary containing the two timesteps images (im1 and im2) as well as the ground truth (pits.appeared, pits.disappeared).
            stage_tag (str): one of "training", "validation", "testing"
            dataloader_tag (str, optional): The tag associated with the current dataloader. Defaults to None.

        Returns:
            torch.FloatTensor: the loss computed using loss_fn
        """

        change_activation = self.forward(batch)

        # computing loss
        change_target = torch.cat(
            [batch["pits.appeared"], batch["pits.disappeared"]], dim=1
        )
        loss = self.loss_fn(change_activation, change_target)
        self.log(
            f"{phase_tag}/loss",
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

        # computing rest of metrics
        appeared_target = change_target[:, 0]
        disappeared_target = change_target[:, 1]
        appeared_pred = change_activation[:, 0] > 0.5
        disappeared_pred = change_activation[:, 1] > 0.5

        self.update_for_tag(
            f"{phase_tag}/{self.change_kinds[0]}", appeared_pred, appeared_target
        )
        if dataloader_tag is not None:
            self.update_for_tag(
                f"{dataloader_tag}/{self.change_kinds[0]}",
                appeared_pred,
                appeared_target,
            )

        self.update_for_tag(
            f"{phase_tag}/{self.change_kinds[1]}", disappeared_pred, disappeared_target
        )
        if dataloader_tag is not None:
            self.update_for_tag(
                f"{dataloader_tag}/{self.change_kinds[1]}",
                disappeared_pred,
                disappeared_target,
            )

        return loss

    def update_for_tag(self, tag: str, predicted: torch.Tensor, correct: torch.Tensor):
        """Updates all the metrics for a given tag using the predicted and correct tensors

        Args:
            tag (str): base tag of the metrics to update (e.g. "validation/appeared", "validation/EB/appeared")
            predicted (torch.Tensor): The binary mask predicted by the network
            correct (torch.Tensor): The binary mask in the dataset ground truth
        """
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
        """Logs all the metrics of a given tag

        Args:
            tag (str): base tag of the metrics to update (e.g. "validation/appeared", "validation/EB/appeared")
        """
        for change_kind in self.change_kinds:
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

    def _setup_metrics(self):
        """Initializes all the metrics in the MetricCollection called torchmetrics."""

        def metrics_for_tag(tag: str):
            return {
                f"{tag}/mIoU": MeanMetric(),
                f"{tag}/ne_mIoU": MeanMetric(),
                f"{tag}/IoU": BinaryJaccardIndex(),
                f"{tag}/F1": BinaryF1Score(),
                f"{tag}/precision": BinaryPrecision(),
                f"{tag}/recall": BinaryRecall(),
            }

        _metrics = {}

        for i in range(len(self._phases)):
            _phase = self._phases[i]
            _labels = self._phase_labels[i]
            for change_kind in self.change_kinds:
                # create phase-level metrics.
                _metrics.update(metrics_for_tag(f"{_phase}/{change_kind}"))
                for label in _labels:
                    # create dataloader-level metrics.
                    _metrics.update(metrics_for_tag(f"{_phase}/{label}/{change_kind}"))

        self.torchmetrics = MetricCollection(_metrics)

    # def configure_optimizers(self) -> Any:
    #     return torch.optim.Adam(
    #         params=self.parameters(),
    #         lr=1e-5,
    #     )

    def _step(
        self,
        phase_idx,
        batch,
        batch_idx,
        dataloader_idx=-1,
    ):
        """Shared step logic for interfacing with Lightning's step subdivision.

        Args:
            phase_idx (int): id of the current phase.
            batch (dict): batch to process
            batch_idx (int): batch index provided by Lightning
            dataloader_idx (int, optional): id of the phase's dataloader. Defaults to -1.

        Returns:
            Optional[torch.FloatTensor]: Might return the loss value
        """
        if isinstance(batch, list):
            return None
        _phase = self._phases[phase_idx]
        _dataloader_labels = self._phase_labels[phase_idx]
        dataloader_tag = None
        if dataloader_idx > -1:
            dataloader_tag = f"{_phase}/{_dataloader_labels[dataloader_idx]}"
        return self.shared_step(batch, _phase, dataloader_tag=dataloader_tag)

    def _on_epoch_end(self, phase_idx) -> None:
        """Logs all metrics for phase and related dataloaders.

        Args:
            phase_idx (int): ID of the current phase 
        """
        _phase = self._phases[phase_idx]
        self.log_for_tag(_phase)
        for label in self._phase_labels[phase_idx]:
            dataloader_tag = f"{_phase}/{label}"
            self.log_for_tag(dataloader_tag)

    def training_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=-1,
        *args,
        **kwargs,
    ) -> STEP_OUTPUT:
        return self._step(_TRAIN_IDX, batch, batch_idx, dataloader_idx=dataloader_idx)

    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=-1,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[STEP_OUTPUT]:
        return self._step(
            _VALIDATION_IDX, batch, batch_idx, dataloader_idx=dataloader_idx
        )

    def test_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=-1,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[STEP_OUTPUT]:
        return self._step(_TEST_IDX, batch, batch_idx, dataloader_idx=dataloader_idx)

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end(_TRAIN_IDX)
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(_VALIDATION_IDX)
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(_TEST_IDX)
        return super().on_test_epoch_end()

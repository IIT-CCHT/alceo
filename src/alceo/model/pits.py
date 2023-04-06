# %%
from pathlib import Path
from typing import Any, List, Optional, Tuple
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
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import torch
import torchvision.transforms.functional as tvf
from torch.nn.functional import one_hot
from sorcery import dict_of, unpack_keys
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, BinaryRecall


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
    
    @property
    def stage_tag(self):
        if self.trainer.training:
            return "training"
        if self.trainer.validating:
            return "validation"
        if self.trainer.testing:
            return "testing"
        return "NE"
    
    @property
    def stage_labels(self):
        if self.trainer.training:
            return self.training_labels
        if self.trainer.validating:
            return self.validation_labels
        if self.trainer.testing:
            return self.test_labels
        return []
            
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
        self.log(f"{tag}/mIoU", self.torchmetrics[f"{tag}/mIoU"])
        self.log(f"{tag}/ne_mIoU", self.torchmetrics[f"{tag}/ne_mIoU"])
        self.log(f"{tag}/IoU", self.torchmetrics[f"{tag}/IoU"])
        self.log(f"{tag}/F1", self.torchmetrics[f"{tag}/F1"])
        self.log(f"{tag}/precision", self.torchmetrics[f"{tag}/precision"])
        self.log(f"{tag}/recall", self.torchmetrics[f"{tag}/recall"])
    
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
        _metrics = metrics_for_tag("training")
        for label in self.training_labels:
            _metrics.update(metrics_for_tag(f"training/{label}"))
            
        _metrics.update(metrics_for_tag("validation"))
        for label in self.validation_labels:
            _metrics.update(metrics_for_tag(f"validation/{label}"))
        
        _metrics.update(metrics_for_tag("testing"))
        for label in self.test_labels:
            _metrics.update(metrics_for_tag(f"testing/{label}"))
        
        self.torchmetrics = MetricCollection(_metrics)

    def __init__(self, network: PitsChangeDetectionNetwork, training_labels: List[str], validation_labels, test_labels,) -> None:
        super().__init__()
        self.batch_ids_to_monitor = None
        self.network = network
        self.loss_fn = JaccardLoss(mode="multilabel")
        
        self.training_labels = training_labels
        self.validation_labels = validation_labels
        self.test_labels = test_labels
        self.setup_metrics()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=1e-5,
        )

    def forward(self, batch):
        im1 = batch["im1"]
        im2 = batch["im2"]
        return self.network(im1, im2)

    def _shared_step(self, batch, stage: str, dataloader_idx=-1) -> STEP_OUTPUT:
        log_stage = stage
        if dataloader_idx >= 0:
            log_stage = f"{stage}/{self.datasets[dataloader_idx][0]}"

        pits_appeared = batch["pits.appeared"]
        pits_disappeared = batch["pits.disappeared"]

        change_truth = torch.cat([pits_appeared, pits_disappeared], dim=1)
        change_pred = self.forward(batch)

        # computing loss
        loss = self.loss_fn(change_pred, change_truth)
        self.log(f"{log_stage}/loss", loss, sync_dist=True, on_epoch=True)
        
        change_pred_mask = change_pred[:, 0] > 0
        change_target = change_truth[:, 0]
            
        self.update_for_tag(self.stage_tag, change_pred_mask, change_target)
        if dataloader_idx > 0:
            tag = f"{self.stage_tag}/{self.stage_labels[dataloader_idx]}"
            self.update_for_tag(tag, change_pred_mask, change_target)

        return loss

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        return self._shared_step(batch)

    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=-1,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[STEP_OUTPUT]:
        if isinstance(batch, list):
            for i in range(len(batch)):
                self._shared_step(batch[i], dataloader_idx=dataloader_idx)
        else:
            return self._shared_step(batch, dataloader_idx=dataloader_idx)
    
    def _shared_epoch_end(self):
        self.log_for_tag(self.stage_tag)
        for label in self.stage_labels:
            self.log_for_tag(f"{self.stage_tag}/{label}")



# %%
if __name__ == "__main__":
    # %%
    model = PitsLightningModule()
    model = model.load_from_checkpoint(
        "/HDD1/gsech/source/alceo/DvcLiveLogger/dvclive_run/checkpoints/epoch=99-step=4200.ckpt"
    )
    # %%
    from pytorch_lightning import seed_everything

    seed_everything(1234, workers=True)

    model.setup("fit")

    # %%
    import matplotlib.pyplot as plt

    item = model.test_datasets[0][18]
    item = model.on_after_batch_transfer(item, 0)
    pits_appeared = item["pits.appeared"]
    pits_disappeared = item["pits.disappeared"]
    im1 = item["im1"].unsqueeze(dim=0)
    im2 = item["im2"].unsqueeze(dim=0)

    change_pred = model.network(im1, im2).squeeze().detach().numpy()

    # %%
    fig, [[a_im1, a_im2], [a_app, a_dis]] = plt.subplots(nrows=2, ncols=2)
    a_im1.imshow(tvf.to_pil_image(im1[0, [0, 1, 2]]))
    a_im1.set_title(f"Image 1: {item['change_start']}")
    a_im1.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
    )
    a_im2.imshow(tvf.to_pil_image(im2[0, [0, 1, 2]]))
    a_im2.set_title(f"Image 2: {item['change_end']}")
    a_im2.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
    )
    a_app.imshow((pits_appeared[0]))
    a_app.set_title("Pits appeared")
    a_app.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
    )
    a_dis.axis("off")
    # a_dis.imshow((pits_disappeared[0]))
    # a_dis.set_title("Pits disappeared")
    # a_dis.tick_params(
    #     axis="both",
    #     which="both",
    #     labelbottom=False,
    #     labelleft=False,
    #     bottom=False,
    #     left=False,
    # )

    # %%
    import pandas as pd

    app_df = pd.read_csv(
        "/HDD1/gsech/source/alceo/dvclive/plots/metrics/validation/iou_appeared.tsv",
        sep="\t",
    )
    app_df
    disapp_df = pd.read_csv(
        "/HDD1/gsech/source/alceo/dvclive/plots/metrics/validation/iou_disappeared.tsv",
        sep="\t",
    )
    disapp_df
    # %%
    import matplotlib.pyplot as plt

    # fig, [app_ax, disapp_ax] = plt.subplots(nrows=2, ncols=1)
    app_ax = plt.axes()
    app_ax.plot(app_df["step"], app_df["iou_appeared"])
    app_ax.set_title("mIoU pits appeared")
    app_ax.set_ylabel("Validation mIoU")
    app_ax.set_xlabel("Training step")
    app_ax.set_ylim([0, 1])
    # disapp_ax.plot(disapp_df["step"], disapp_df["iou_disappeared"])
    # disapp_ax.set_title("mIoU pits disappeared")
    # disapp_ax.set_ylabel("Validation mIoU")
    # disapp_ax.set_xlabel("Training step")
    # disapp_ax.set_ylim([0, 1])
    plt.tight_layout()
# %%

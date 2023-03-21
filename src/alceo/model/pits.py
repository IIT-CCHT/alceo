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
    def __init__(self) -> None:
        super().__init__()
        self.batch_ids_to_monitor = None
        self.network = PitsChangeDetectionNetwork()
        self.loss_fn = JaccardLoss(mode="multilabel")
        self.mIoU_appeared = MetricCollection(
            {
                "stage_train": MeanMetric(dist_sync_on_step=True),
                "stage_validation": MeanMetric(dist_sync_on_step=True),
            }
        )
        self.mIoU_disappeared = MetricCollection(
            {
                "stage_train": MeanMetric(dist_sync_on_step=True),
                "stage_validation": MeanMetric(dist_sync_on_step=True),
            }
        )

    def setup(self, stage: Optional[str] = None) -> None:
        pits_dataset_path = Path("/HDD1/gsech/source/alceo/dataset/pits/")
        dataset = PitsSiteDataset(pits_dataset_path / "DURA_EUROPOS")
        self.datasets = [
            ("DURA_EUROPOS", PitsSiteDataset(pits_dataset_path / "DURA_EUROPOS")),
            ("ASWAN", PitsSiteDataset(pits_dataset_path / "ASWAN")),
        ]
        self.train_datasets, self.validation_datasets, self.test_datasets = [], [], []
        
        for label, dataset in self.datasets:
            train_dataset, validation_dataset, test_dataset = random_split(
                dataset, [0.8, 0.1, 0.1]
            )
            self.train_datasets.append(train_dataset)
            self.validation_datasets.append(validation_dataset)
            self.test_datasets.append(test_dataset)
            
        return super().setup(stage)

    def _dataloader_for_dataset(self, dataset: Dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=16,
            num_workers=5,
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch["im1"] = batch["im1"].float()
        batch["im2"] = batch["im2"].float()

        return super().on_after_batch_transfer(batch, dataloader_idx)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=1e-5,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader_for_dataset(self.train_dataset)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader_for_dataset(self.validation_dataset)

    def _shared_step(self, batch, stage: str) -> STEP_OUTPUT:

        pits_appeared = batch["pits.appeared"]
        pits_disappeared = batch["pits.disappeared"]
        im1 = batch["im1"]
        im2 = batch["im2"]
        change_truth = torch.cat([pits_appeared, pits_disappeared], dim=1)

        change_pred = self.network(im1, im2)

        # computing loss

        # computing full loss!
        loss = self.loss_fn(change_pred, change_truth)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        stats = get_stats(
            change_pred, change_truth, mode="multilabel", threshold=0.5, num_classes=2
        )

        ious = iou_score(*stats)

        iou_app = self.mIoU_appeared[f"stage_{stage}"]
        iou_diss = self.mIoU_disappeared[f"stage_{stage}"]
        iou_app(ious[:, 0])
        iou_diss(ious[:, 1])
        self.log(f"{stage}/iou_appeared", iou_app, on_epoch=True)
        self.log(f"{stage}/iou_disappeared", iou_diss, on_epoch=True)

        return loss

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

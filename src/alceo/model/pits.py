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
        dataset = PitsSiteDataset(
            Path("/HDD1/gsech/source/alceo/dataset/pits/DURA_EUROPOS")
        )
        self.train_dataset, self.validation_dataset, self.test_dataset = random_split(
            dataset, [0.8, 0.1, 0.1]
        )
        return super().setup(stage)

    def _dataloader_for_dataset(self, dataset: Dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=8,
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
        
        stats = get_stats(change_pred, change_truth, mode="multilabel", threshold=0.5, num_classes=2)
        
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
    model = model.load_from_checkpoint("/HDD1/gsech/source/alceo/epoch=60-step=5124.ckpt")
    # %%
    from pytorch_lightning import seed_everything
    seed_everything(1234, workers=True)
    # %%
    model.setup("fit")
    # %%
    for i in range(len(model.test_dataset)):
        item = model.test_dataset[i]
        item = model.on_after_batch_transfer(item, 0)
        print(f"index: {i}, changes: {item['pits.appeared'].sum()}")
        if i > 20:
            break
    # %%
    import matplotlib.pyplot as plt
    item = model.test_dataset[18]
    item = model.on_after_batch_transfer(item, 0)
    pits_appeared = item["pits.appeared"]
    pits_disappeared = item["pits.disappeared"]
    im1 = item["im1"].unsqueeze(dim=0)
    im2 = item["im2"].unsqueeze(dim=0)

    change_pred = model.network(im1, im2).squeeze().detach().numpy()
    # %%
    print("Pred appeared")
    plt.imshow(change_pred[0] > 0.5)
    # %%
    print("Pred disappeared")
    plt.imshow(change_pred[1] > 0.5)
    # %%
    print("Pits appeared")
    plt.imshow(pits_appeared[0])
    # %%
    print("Pits disappeared")
    plt.imshow(pits_disappeared[0])

    # %%
    item["tile_name"]
    # %%
    torch.all(pits_appeared == pits_disappeared)
    # %%

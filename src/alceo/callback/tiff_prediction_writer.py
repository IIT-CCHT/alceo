from pathlib import Path
from typing import Any, Literal, Optional, Sequence
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio as rio
import pytorch_lightning as pl
import torch
import os

class TiffPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        output_dir: Optional[str] = None,
    ) -> None:
        """A BasePredictionWriter for AlceoChangeDetectionDataset(s). It will call the GeoTIFF rasters with the same filename as the "im1" raster filename into "activation" and "mask" folders created inside the provided "output_dir".

        Args:
            write_interval (Literal['batch', 'epoch', 'batch_and_epoch'], optional): Defines when to write, this is kept exposed from the BasePredictionWriter class. Defaults to "batch".
            output_dir (Optional[str], optional): The path to the folder that will contain the resulting GeoTIFFs, if None the current working directory is used. Defaults to None.
        """
        super().__init__(write_interval)
        if output_dir is None:
            output_dir = Path(os.getcwd())
        else:
            output_dir = Path(output_dir)
        self.output_dir = output_dir
        self.activation_dir = output_dir / "activation"
        os.makedirs(self.activation_dir, exist_ok=True)
        self.mask_dir = output_dir / "mask"
        os.makedirs(self.mask_dir, exist_ok=True)

    def _handle_predicted(
        self,
        prediction: torch.Tensor,
        im1_path: str,
        tile_name: str,
    ):
        # opening im1 for getting the same GeoTIFF metadata (bounds + CRS + res)
        with rio.open(im1_path) as src:
            act_profile = src.profile
            act_profile.update(
                dtype=rio.float32,
                count=2) # network activation will be saved in float32.
            with rio.open(self.activation_dir / tile_name, mode="w", **act_profile) as dst:
                activation = prediction.cpu().numpy().astype(rio.float32) # convert to float 32
                dst.write(activation)
            mask_profile = src.profile
            mask_profile.update(
                dtype=rio.uint8,
                count=2,
            ) # network predicted binary mask of appeared (band 1) and disappeared (band 2) pits.
            with rio.open(self.mask_dir / tile_name, mode="w", **mask_profile) as dst:
                mask = (prediction > 0.5).cpu().numpy().astype(rio.uint8)
                dst.write(mask)
        

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        tile_names = batch["tile_name"]
        im1_paths = batch["im1_path"]
        for i in range(len(tile_names)):
            self._handle_predicted(prediction[i], im1_paths[i], tile_names[i])

from pathlib import Path
from typing import Literal, Optional
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio as rio

class TiffPredictionWriter(BasePredictionWriter):
    
    def __init__(self, write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch", output_dir: Optional[Path] = None) -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        
    def write_on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", prediction: Any, batch_indices: Optional[Sequence[int]], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)
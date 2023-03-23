# %%
from alceo.model.pits import PitsLightningModule
from pytorch_lightning import seed_everything
import rasterio as rio


def _load_raster(raster_path: Path):
        with rio.open(raster_path) as ref:
            raster = ref.read()
            return raster
checkpoint_path = "/HDD1/gsech/source/alceo/DvcLiveLogger/pits_change_detection/checkpoints/DURAEUROPOS-epoch=1758-validation/DURAEUROPOS/appeared/iou-no-empty=0.61686.ckpt"

seed_everything(1234)
model = PitsLightningModule.load_from_checkpoint(checkpoint_path)
model.setup("fit")
val_dataset = model.validation_datasets[0]
val_dataset[0]
# %%
model
# %%
from pathlib import Path

output_dir = Path("/HDD1/gsech/source/alceo/data/sites/DURA_EUROPOS/inference/DE_26_5_2013-DE_19_09_2014")

import os
os.makedirs(output_dir, exist_ok=True)

from glob import glob


for i in range(len(val_dataset)):
    import torch 
    import numpy as np
    item = val_dataset[i]

    model = model.eval()
    change_pred = model.network(item["im1"].float().unsqueeze(dim=0), item["im2"].float().unsqueeze(dim=0)).detach().numpy()

    change_mask = change_pred[0, 0] > 0.5

    with rio.open(val_dataset.dataset.im2_folder / item["tile_name"]) as src:
        profile = src.profile
        profile.update(
            dtype=rio.uint8,
            count=1)
        with rio.open(output_dir / item["tile_name"], "w", **profile) as dst:
            dst.write(change_mask.astype(rio.uint8), 1)
            print(f"Written: {item['tile_name']} ({change_mask.sum()} trues)")

# %%

# %%

# %%

# %%

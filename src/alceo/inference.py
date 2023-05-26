# %%
from alceo.model.pits import PitsLightningModule
from pytorch_lightning import seed_everything
import rasterio as rio
from pathlib import Path
import os
import torchvision.transforms.functional as tvf

def _load_raster(raster_path: Path):
        with rio.open(raster_path) as ref:
            raster = ref.read()
            return raster
checkpoint_path = "/HDD1/gsech/source/alceo/DvcLiveLogger/pits_change_detection/checkpoints/DURAEUROPOS-epoch=1758-validation/DURAEUROPOS/appeared/iou-no-empty=0.61686.ckpt"

seed_everything(1234)
model = PitsLightningModule.load_from_checkpoint(checkpoint_path)
model.setup("fit")
val_dataset = model.validation_datasets[0]
item = val_dataset[0]
item.keys()
# %%
model
# %%
output_dir = Path("/HDD1/gsech/source/alceo/data/sites/DURA_EUROPOS/inference/DE_26_5_2013-DE_19_09_2014")


os.makedirs(output_dir, exist_ok=True)

# %%
for i in range(len(val_dataset)):
    
    import torch 
    import numpy as np
    item = val_dataset[i]
    mask = item["pits.appeared"].squeeze(0) > 0
    model = model.eval()
    change_pred = model.network(item["im1"].float().unsqueeze(dim=0), item["im2"].float().unsqueeze(dim=0)).detach()

    change_mask = change_pred[0, 0] > 0.5 
    error = torch.bitwise_xor(change_mask, mask)
    with rio.open(val_dataset.dataset.im2_folder / item["tile_name"]) as src:
        profile = src.profile
        profile.update(
            dtype=rio.uint8,
            count=2)
        with rio.open(output_dir / item["tile_name"], "w", **profile) as dst:
            dst.write(change_mask.numpy().astype(rio.uint8), 1)
            dst.write(error.numpy().astype(rio.uint8), 2)
            print(f"Written: {item['tile_name']} ({change_mask.sum()} trues)")

# %%
import matplotlib.pyplot as plt
plt.imshow(mask.numpy().squeeze(0))

# %%
plt.imshow(change_mask)
# %%

# %%

# %%
from pathlib import Path
from typing import Any, Dict
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import pandas as pd
import rasterio as rio
import torch
import numpy as np


@dataclass
class PitsSiteDataset(Dataset[Dict[str, Any]]):
    dataset_root: Path
    tiles_df: pd.DataFrame = field(init=False)
    im1_folder: Path = field(init=False)
    im2_folder: Path = field(init=False)
    pits_appeared_folder: Path = field(init=False)
    pits_disappeared_folder: Path = field(init=False)

    def __post_init__(self):
        assert (
            self.dataset_root.exists()
        ), f"Provided dataset root does not exists {self.dataset_root}"

        self.im1_folder = self.dataset_root / "im1"
        self.im2_folder = self.dataset_root / "im2"
        self.pits_appeared_folder = self.dataset_root / "pits.appeared"
        self.pits_disappeared_folder = self.dataset_root / "pits.disappeared"

        assert (
            self.im1_folder
        ), f"No im1 folder found in dataset root {self.dataset_root}"
        assert (
            self.im2_folder
        ), f"No im2 folder found in dataset root {self.dataset_root}"
        assert (
            self.pits_appeared_folder
        ), f"No pits.appeared folder found in dataset root {self.dataset_root}"
        assert (
            self.pits_disappeared_folder
        ), f"No pits.disappeared folder found in dataset root {self.dataset_root}"

        assert (
            self.dataset_root / "tiles_meta.csv"
        ), f"No tiles_meta.csv file found in dataset root {self.dataset_root}"
        self.tiles_df = pd.read_csv(self.dataset_root / "tiles_meta.csv")

    def __len__(self):
        return len(self.tiles_df)

    def _load_raster(self, raster_path: Path):
        with rio.open(raster_path) as ref:
            raster = ref.read()
            return raster

    def __getitem__(self, index) -> Dict[str, Any]:
        item = self.tiles_df.loc[index].to_dict()
        im1_loaded = self._load_raster(self.im1_folder / item["tile_name"])
        
        item["im1"] = torch.from_numpy(im1_loaded.astype(np.int32))
        c, w, h = item["im1"].shape
        if c == 8:
            item["im1"] = item["im1"][[4, 2, 1, 7], :, :]

        im2_loaded = self._load_raster(self.im2_folder / item["tile_name"])
        item["im2"] = torch.from_numpy(im2_loaded.astype(np.int32))
        c, w, h = item["im2"].shape
        if c == 8:
            item["im2"] = item["im2"][[4, 2, 1, 7], :, :]

        pits_appeared_loaded = self._load_raster(
            self.pits_appeared_folder / item["tile_name"]
        )
        item["pits.appeared"] = torch.from_numpy(
            pits_appeared_loaded.astype(np.int32)
        )

        pits_disappeared_loaded = self._load_raster(
            self.pits_disappeared_folder / item["tile_name"]
        )
        item["pits.disappeared"] = torch.from_numpy(
            pits_disappeared_loaded.astype(np.int32)
        )

        return item


# %%
if __name__ == "__main__":
    # %%
    dset = PitsSiteDataset(Path("/HDD1/gsech/source/alceo/dataset/pits/DURA_EUROPOS"))
    tiles_meta = dset[0]
    print(tiles_meta)
    # %%

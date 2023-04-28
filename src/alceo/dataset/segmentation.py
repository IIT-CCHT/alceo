# %%
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import geopandas as gpd
import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.features import rasterize
import torch
import numpy as np
from shapely import union_all


@dataclass
class AlceoPitsImageSegmentationDataset(Dataset[Dict[str, Any]]):
    image_path: Path
    annotations_path: Optional[Path]
    date_string: str
    tiles_path: Path
    area_of_interest_path: Path
    bands: List[int]
    _annotation_df: Optional[gpd.GeoDataFrame] = field(init=False)
    _tiles_df: gpd.GeoDataFrame = field(init=False)
    
    def __post_init__(self):
        self.image_path = Path(self.image_path)
        self.tiles_path = Path(self.tiles_path)
        self.area_of_interest_path = Path(self.area_of_interest_path)
        if self.annotations_path is not None:
            self.annotations_path = Path(self.annotations_path)
        # Loading the image.
        assert self.image_path.exists(), f"Provided image path does not exists. {self.image_path}"
        crs = None
        with rio.open(self.image_path) as src:
            crs = src.crs
        
        # Loading annotations.
        if self.annotations_path is not None:
            assert self.annotations_path.exists(), f"Provided annotations image file does not exists. {self.annotations_path}"
            self._annotation_df: gpd.GeoDataFrame = gpd.read_file(self.annotations_path)
            assert "Day_Month_Year" in self._annotation_df.columns, f"Missing Day_Month_Year column in GeoDataFrame. {self.annotations_path}"
            self._annotation_df = self._annotation_df[self._annotation_df.Day_Month_Year == self.date_string]
            assert len(self._annotation_df) > 0, f"No annotation found for given date ({self.date_string}). {self.annotations_path}"
            
            if crs is not None:
                self._annotation_df = self._annotation_df.to_crs(crs)
        
        # Loading area of interest
        assert self.area_of_interest_path.exists(), f"Provided area of interest GeoJSON file does not exists. {self.area_of_interest_path}"
        _aoi_df: gpd.GeoDataFrame = gpd.read_file(self.area_of_interest_path)
        assert len(_aoi_df) > 0, f"At least one area of interest should be provided! {self.area_of_interest_path}"
        if crs is not None:
            _aoi_df = _aoi_df.to_crs(crs)
        _aoi_shape = union_all(_aoi_df.geometry)
        
        # Loading tiles.
        assert self.tiles_path.exists(), f"Provided tiles GeoJSON file does not exists. {self.tiles_path}"
        self._tiles_df: gpd.GeoDataFrame = gpd.read_file(self.tiles_path)
        
        if crs is not None:
            self._tiles_df = self._tiles_df.to_crs(crs)
        
        self._tiles_df = self._tiles_df[self._tiles_df.geometry.covered_by(_aoi_shape)]
        
        assert len(self._tiles_df) > 0, f"Not tiles found in given file for the area of interest ({self.area_of_interest_path}). {self.tiles_path}"
        
    def __len__(self):
        return len(self._tiles_df)
    
    def _load_tile(self, tile, src: rio.DatasetReader):
        tile_window = from_bounds(
            *tile.geometry.bounds,
            transform=src.transform,
        )
        _raster = src.read(
            indexes=self.bands,
            window=tile_window,
            out_dtype=np.int32,
            out_shape=(src.count, tile.height, tile.width),
            resampling=Resampling.bilinear,
        )
        return torch.from_numpy(_raster).float()

    def _load_mask(self, tile, src: rio.DatasetReader):
        if self._annotation_df is not None:
            pits = self._annotation_df[self._annotation_df.covered_by(tile.geometry)]
            _raster = rasterize(pits.geometry, out_shape=(tile.height, tile.width), transform=src.transform, fill=1, dtype=np.int32)
            return torch.from_numpy(_raster)
        return None

    def __getitem__(self, index) -> Dict[str, Any]:
        tile = self._tiles_df.loc[index]
        item = tile.to_dict()
        with rio.open(self.image_path) as src:
            item["raster"] = self._load_tile(tile, src)
            item["mask"] = self._load_mask(tile, src)
        return item
        
        
# %%
dataset = AlceoPitsImageSegmentationDataset(
    image_path="/home/gsech/alceo/data/sites/DURA_EUROPOS/images/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif",
    annotations_path="/home/gsech/alceo/data/sites/DURA_EUROPOS/annotations/pits.geojson",
    date_string="19/09/2014",
    tiles_path="/home/gsech/alceo/data/sites/DURA_EUROPOS/tiles.geojson",
    area_of_interest_path="/home/gsech/alceo/data/sites/DURA_EUROPOS/train_area.geojson",
    bands=[1, 2, 3, 4],
)
# %%

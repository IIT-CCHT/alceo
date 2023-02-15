# %%
import rasterio
from rasterio import windows
import rasterio.mask
from itertools import product
import shapely.geometry
from shapely import intersection_all, union_all
from pathlib import Path
import geopandas as gpd
from alceo.utils import in_notebook
from argparse import ArgumentParser
from rasterio.enums import Resampling
from tqdm import tqdm
import os
import numpy as np
# %%
def rasterize_tiles(
    tiles_geojson_path: Path,
    input_geotiff_path: Path,
    output_directory_path: Path,
    areas_of_interest_geojson: Path = None,
    keep_partial: bool = False,
):
    """Given a GeoJSON containing vectorial features of tiles and a input GeoTIFF raster, split the latter in tiles and save them into output_directory_path. keep_partials saves also tiles that do not completely fill the tile raster. Some areas of interest can be provided via a GeoJSON.

    Args:
        tiles_geojson_path (Path): Path to the tiles features GeoJSON.
        input_geotiff_path (Path): Path to the input raster.
        output_directory_path (Path): Directory where to save raster outputs.
        areas_of_interest_geojson (Path, optional): Features of the various areas of interest to keep. Defaults to None.
        keep_partial (bool, optional): Save tiles that are partially filled. Defaults to False.
    """
    output_filename = "{}.tif"
    # %% Create output directory if it doesn't exists

    if not output_directory_path.exists():
        os.makedirs(output_directory_path, exist_ok=True)

    tiles_gdf = gpd.read_file(tiles_geojson_path)
    
    if areas_of_interest_geojson is not None and areas_of_interest_geojson.exists():
        aoi_gdf = gpd.read_file(areas_of_interest_geojson, driver="GeoJSON").to_crs(tiles_gdf.crs)
        aoi_shape = union_all(aoi_gdf.geometry)
        tiles_gdf = tiles_gdf[tiles_gdf.geometry.covered_by(aoi_shape)]
    # %%
    for id, row in tqdm(tiles_gdf.iterrows()):
        # %%
        src = rasterio.open(input_geotiff_path)
        tile_window = windows.from_bounds(
            *row.geometry.bounds,
            transform=src.transform,
        )
        res = src.read(
            window=tile_window,
            out_shape=(src.count, row.height, row.width),
            resampling=Resampling.bilinear,
        )
        
        # if np.any(np.all(res == 0.0, axis=0)): # discard tiles that have pixels with all bands at zero. 
        #     continue
        # %%

        tile_path = output_directory_path / output_filename.format(row.tile_id
        )

        meta = src.meta.copy()
        meta["width"] = row.width
        meta["height"] = row.height
        meta["transform"] = windows.transform(tile_window, src.transform)
        with rasterio.open(tile_path, "w", **meta) as out:
            out.write(res)


if __name__ == "__main__":
    import logging

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    console_handler.setFormatter(formatter)
    logger = logging.getLogger("rasterio")
    logger.addHandler(console_handler)
    logger.setLevel(logging.ERROR)
    
    if not in_notebook():
        parser = ArgumentParser(
            "rasterize_tiles.py",
            description="""
    Given a GeoJSON containing geometrical features describing tiles locations and a GeoTIFF raster sample it and obtain a GeoTIFF for each tile.
    """,
        )
        parser.add_argument(
            "-t",
            "--tiles_geojson_path",
            type=Path,
            help="Input GeoJSON containing the geometries of the tiles as well as the expected pixel size in width, height properties.",
            required=True,
        )
        parser.add_argument(
            "-i",
            "--input_geotiff_path",
            type=Path,
            help="Input GeoTIFF containing the raster data to sample in tiles.",
            required=True,
        )
        parser.add_argument(
            "-o",
            "--output_directory_path",
            type=Path,
            help="Directory where the tiles GeoTIFFs will be stored.",
        )

        args = parser.parse_args()
        tiles_geojson_path = args.tiles_geojson_path
        input_geotiff_path = args.input_geotiff_path
        output_directory_path = args.output_directory_path
        areas_of_interest_geojson = args.areas_of_interest_geojson
    else:
        # %%
        tiles_geojson_path = Path(
            "/HDD1/gsech/source/alceo/data/sites/DURA_EUROPOS/tiles.json"
        )
        input_geotiff_path = Path(
            "/HDD1/gsech/source/alceo/data/sites/DURA_EUROPOS/images/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif"
        )
        output_directory_path = Path(
            "/HDD1/gsech/source/alceo/data/sites/DURA_EUROPOS/tiles/DE_19_09_2014/DE_19_09_2014_NN_diffuse"
        )
        areas_of_interest_geojson = Path(
            "/HDD1/gsech/source/alceo/data/sites/DURA_EUROPOS/area_of_interest.geojson"
        )
        
    # %%
    rasterize_tiles(
        tiles_geojson_path,
        input_geotiff_path,
        output_directory_path,
        areas_of_interest_geojson
    )

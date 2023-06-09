# %%
import rasterio
from rasterio import windows
import rasterio.mask
from rasterio.warp import reproject
from shapely import union_all
from shapely.geometry import box, mapping
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
):
    """Given a GeoJSON containing vectorial features of tiles and a input GeoTIFF raster, split the latter in tiles, resample them if needed, and save them into output_directory_path. For additional filtering, areas of interest can be provided via a GeoJSON.

    Args:
        tiles_geojson_path (Path): Path to the tiles features GeoJSON.
        input_geotiff_path (Path): Path to the input raster.
        output_directory_path (Path): Directory where to save raster outputs.
        areas_of_interest_geojson (Path, optional): Features of the various areas of interest to keep. Defaults to None.
    """
    # %% Create output directory if it doesn't exists
    output_filename = "{}.tif"

    if not output_directory_path.exists():
        os.makedirs(output_directory_path, exist_ok=True)

    tiles_gdf = gpd.read_file(tiles_geojson_path)

    if areas_of_interest_geojson is not None and areas_of_interest_geojson.exists():
        aoi_gdf = gpd.read_file(areas_of_interest_geojson, driver="GeoJSON").to_crs(
            tiles_gdf.crs
        )
        aoi_shape = union_all(aoi_gdf.geometry)
        tiles_gdf = tiles_gdf[tiles_gdf.geometry.covered_by(aoi_shape)]
    # %%
    with rasterio.open(input_geotiff_path) as src:
        # %%
        # src = rasterio.open(input_geotiff_path)
        image_box = box(*src.bounds)
        # %%
        for id, row in tqdm(tiles_gdf.iterrows()):
            if not image_box.covers(row.geometry):
                continue
            # %%
            tile_clipped, tile_transform = rasterio.mask.mask(
                src,
                [row.geometry],
                crop=True,
            )
            # %%
            reproj_tile, reproj_transform = reproject(
                source=tile_clipped,
                destination=np.zeros((src.count, row.height, row.width)),
                src_transform=tile_transform,
                src_crs=src.crs,
                dst_crs=src.crs,
                dst_nodata=src.nodata,
                resampling=Resampling.bilinear,
            )
            # %%

            tile_path = output_directory_path / output_filename.format(row.tile_id)
            profile = src.profile
            profile.update(
                transform=reproj_transform,
                driver="GTiff",
                width=row.width,
                height=row.height,
            )
            with rasterio.open(tile_path, "w", **profile) as out:
                out.write(reproj_tile)


# %%
if __name__ == "__main__":
    import logging

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    console_handler.setFormatter(formatter)
    logger = logging.getLogger("rasterio")
    logger.addHandler(console_handler)
    logger.setLevel(logging.ERROR)

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
    parser.add_argument(
        "-a",
        "--areas_of_interest_geojson",
        type=Path,
        help="GeoJSON containing areas of interest that have to completely cover tiles to be rasterized.",
    )

    args = parser.parse_args()
    tiles_geojson_path = args.tiles_geojson_path
    input_geotiff_path = args.input_geotiff_path
    output_directory_path = args.output_directory_path
    areas_of_interest_geojson = args.areas_of_interest_geojson

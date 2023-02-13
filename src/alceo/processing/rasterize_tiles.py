# %%
import rasterio
from rasterio import windows
from itertools import product
import shapely.geometry
from shapely import intersection_all
from pathlib import Path
import geopandas as gpd
from alceo.utils import in_notebook
from argparse import ArgumentParser
from rasterio.enums import Resampling
from tqdm import tqdm
import os

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
        nargs="+",
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
else:
    tiles_geojson_path = Path(
        "/home/gsech/Source/alceo/data/images/DURA_EUROPOS/tiles.json"
    )
    input_geotiff_path = Path(
        "/home/gsech/Source/alceo/data/images/DURA_EUROPOS/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif"
    )
    output_directory_path = Path(
        "/home/gsech/Source/alceo/data/images/DURA_EUROPOS/DE_19_09_2014/tiles"
    )

output_filename = input_geotiff_path.stem + "_t-{}-{}.tif"

# %% Create output directory if it doesn't exists

if not output_directory_path.exists():
    os.makedirs(output_directory_path, exist_ok=True)

tiles_gdf = gpd.read_file(tiles_geojson_path)

for id, row in tqdm(tiles_gdf.iterrows()):
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

    tile_path = output_directory_path / output_filename.format(
        int(tile_window.col_off), int(tile_window.row_off)
    )

    meta = src.meta.copy()
    meta["width"] = row.width
    meta["height"] = row.height
    meta["transform"] = windows.transform(tile_window, src.transform)

    with rasterio.open(tile_path, "w", **meta) as out:
        out.write(res)
# %%

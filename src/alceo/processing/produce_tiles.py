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


def produce_tiles(
    input_paths,
    output_geojson_path,
    tile_prefix,
):
    #%% Loading all images bounds and find intersection
    all_bounds = []
    _transform = None
    _crs = None
    for image_path in input_paths:
        with rasterio.open(image_path) as dataset:
            all_bounds.append(dataset.bounds)
        if _transform is None:  # save transform of first image (for simplicity)
            _transform = dataset.transform
        if _crs is None:
            _crs = dataset.crs

    all_geoms = [shapely.geometry.box(*bounds) for bounds in all_bounds]
    bounds_intersection = intersection_all(all_geoms)

    # Compute tiles for common area
    tile_width = 512
    tile_height = 512

    big_window = windows.from_bounds(
        *bounds_intersection.bounds,
        transform=_transform,
    )

    nrows, ncols = int(big_window.height), int(big_window.width)
    offsets = product(
        range(0, ncols, tile_width),
        range(0, nrows, tile_height),
    )

    _tiles_geoms = []
    _tiles_data = []
    for col_off, row_off in offsets:
        _window = windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=512,
            height=tile_height,
        ).intersection(big_window)
        _window_bounds = windows.bounds(_window, _transform)
        _tiles_geoms.append(shapely.geometry.box(*_window_bounds))
        _tiles_data.append({"width": tile_width, "height": tile_height, "tile_id": f"{tile_prefix}({_window_bounds[0]}, {_window_bounds[1]})"})

    # %% Create and save GeoDataFrame of tiles.
    tiles_gdf = gpd.GeoDataFrame(_tiles_data, geometry=_tiles_geoms, crs=_crs)
    tiles_gdf.to_file(output_geojson_path, index=False, driver="GeoJSON")

if __name__ == '__main__':
    if not in_notebook():
        parser = ArgumentParser(
            "change_from_annotations.py",
            description="""
    Given a GeoJSON containing geometrical features describing pits location for various dates (in a field called Day_Month_Year with dd/mm/YYYY format).
    Compute the pits that appeared, disappeared and persisted between two dates.
    """,
        )
        parser.add_argument(
            "-i",
            "--input_paths",
            nargs="+",
            type=Path,
            help="Input GeoJSON containing the geometries of the annotations done on the various images.",
            required=True,
        )
        parser.add_argument(
            "-o",
            "--output_geojson_path",
            type=Path,
            help="The output directory where the `.appeared`, `.disappeared` and `.persisted` features will be saved. Defaults to current directory.",
            default="./tiles.json",
        )
        parser.add_argument(
            "-p",
            "--tile_prefix",
            type=Path,
            help="Prefix used for generating tile id. Defaults to 'tile_'.",
            default="tile_",
        )

        args = parser.parse_args()
        input_paths = args.input_paths
        output_geojson_path = args.output_geojson_path
        tile_prefix = args.tile_prefix
    else:
        input_paths = [
            "/home/gsech/Source/alceo/data/images/DURA_EUROPOS/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif",
            "/home/gsech/Source/alceo/data/images/DURA_EUROPOS/DE_26_5_2013/DE_26_5_2013_NN_diffuse.tif",
        ]
        output_geojson_path = "/home/gsech/Source/alceo/data/images/DURA_EUROPOS/tiles.json"
        tile_prefix = "tile_"
    produce_tiles(input_paths, output_geojson_path, tile_prefix)
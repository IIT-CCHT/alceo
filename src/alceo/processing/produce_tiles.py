# %%
import rasterio
from rasterio import windows
from itertools import product
import shapely.geometry
from shapely import intersection_all, union_all, covered_by
from pathlib import Path
import geopandas as gpd
from alceo.utils import in_notebook
from argparse import ArgumentParser


def produce_tiles(
    input_paths,
    output_geojson_path,
    tile_prefix,
    areas_of_interest_geojson: Path,
    tile_width = 512,
    tile_height = 512,
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

    assert (
        areas_of_interest_geojson.exists()
    ), f"Provided areas of intereset GeoJSON ({areas_of_interest_geojson}) does not exists!"
    aoi_gdf = gpd.read_file(areas_of_interest_geojson, driver="GeoJSON").to_crs(_crs)
    aoi_geom = union_all(aoi_gdf.geometry)

    all_geoms = [shapely.geometry.box(*bounds) for bounds in all_bounds]
    bounds_intersection = intersection_all(all_geoms)

    # Compute tiles for common area
    

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
            width=tile_width,
            height=tile_height,
        ).intersection(big_window)
        _window_bounds = windows.bounds(_window, _transform)
        tile_box = shapely.geometry.box(*_window_bounds)
        if not covered_by(tile_box, aoi_geom):
            continue
        _tiles_geoms.append(tile_box)
        _tiles_data.append(
            {
                "width": tile_width,
                "height": tile_height,
                "tile_id": f"{tile_prefix}({_window_bounds[0]}, {_window_bounds[1]})",
            }
        )

    # %% Create and save GeoDataFrame of tiles.
    tiles_gdf = gpd.GeoDataFrame(_tiles_data, geometry=_tiles_geoms, crs=_crs)
    tiles_gdf.to_file(output_geojson_path, index=False, driver="GeoJSON")


if __name__ == "__main__":
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
        parser.add_argument(
            "-a",
            "--areas_of_interest_geojson",
            type=Path,
            help="GeoJSON containing areas of interest that have to completely cover tiles to be rasterized.",
            required=True,
        )
        parser.add_argument(
            "-tw",
            "--tile_width",
            type=int,
            default=512,
            help="Width of the tile in pixels.",
        )
        parser.add_argument(
            "-th",
            "--tile_height",
            type=int,
            default=512,
            help="Height of the tile in pixels.",
        )

        args = parser.parse_args()
        input_paths = args.input_paths
        output_geojson_path = args.output_geojson_path
        tile_prefix = args.tile_prefix
        areas_of_interest_geojson = args.areas_of_interest_geojson
        tile_width = args.tile_width
        tile_height = args.tile_height
        
    else:
        input_paths = [
            "/home/gsech/Source/alceo/data/images/DURA_EUROPOS/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif",
            "/home/gsech/Source/alceo/data/images/DURA_EUROPOS/DE_26_5_2013/DE_26_5_2013_NN_diffuse.tif",
        ]
        output_geojson_path = (
            "/home/gsech/Source/alceo/data/images/DURA_EUROPOS/tiles.json"
        )
        tile_prefix = "tile_"
        tile_width, tile_height = 512, 512
    produce_tiles(
        input_paths,
        output_geojson_path,
        tile_prefix,
        areas_of_interest_geojson,
        tile_width=tile_width,
        tile_height=tile_height,
    )

# %%
from typing import List
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
    input_paths: List[Path],
    output_geojson_path: Path,
    tile_prefix: str,
    areas_of_interest_geojson: Path,
    tile_width: int = 512,
    tile_height: int = 512,
):
    """Produces the vectorial representation of tiles for an area of interest of a site and saves such representation to a GeoJSON file.

    Args:
        input_paths (List[Path]): List of GeoTIFF images of the site, at least one should be provided.
        output_geojson_path (Path): Path to the output GeoJSON containing the vectorial representation of tiles.
        tile_prefix (str): Prefix used for generating the tile_id metadata feature.
        areas_of_interest_geojson (Path): Path to a GeoJSON with polygons describing the area of interest in which tiles should be computed.
        tile_width (int, optional): Width of the resulting tile in number of pixels, GSD is computed using the site images. Defaults to 512.
        tile_height (int, optional): Height of the resulting tile in number of pixels, GSD is computed using the site images. Defaults to 512.
    """
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
    parser = ArgumentParser(
        "produce_tiles.py",
        description="""
Produces the vectorial representation of tiles for an area of interest of a site and saves such representation to a GeoJSON file.
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

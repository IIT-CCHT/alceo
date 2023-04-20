# %%
import shutil
import os
from typing import List
import geopandas as gpd
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from glob import glob
from shapely.ops import unary_union

from alceo.utils import in_notebook

# %%
def pits_site_dataset(
    site_data_path: Path,
    output_path: Path,
    area_selection_path: Path,
    changes_selection: List[str],
):
    # %%
    tiles_geojson_path = site_data_path / "tiles.geojson"
    tiles_folder_path = site_data_path / "tiles"
    change_folder_path = site_data_path / "change"
    assert (
        tiles_geojson_path.exists()
    ), f"Provided tiles GeoJSON does not exists ({tiles_geojson_path})"
    assert (
        tiles_folder_path.exists()
    ), f"Provided tiles folder does not exists ({tiles_folder_path})"
    assert (
        change_folder_path.exists()
    ), f"Provided change folder does not exists ({change_folder_path})"

    if area_selection_path is not None:
        assert (
            area_selection_path.exists()
        ), f"Provided area selection GeoJSON does not exists ({area_selection_path})"

    assert output_path.exists(), f"Provided output path does not exists ({output_path})"
    # %%
    change_dirs = []
    if changes_selection is not None and len(changes_selection) > 0:
        for change_sel in changes_selection:
            change_sel_path = change_folder_path / change_sel
            assert change_sel_path.exists(), f"Selected change folder doesn't exists {change_sel} ({change_sel_path})"
            change_dirs.append(change_sel_path)
    else:
        for cs_dir in os.scandir(change_folder_path):
            if not cs_dir.is_dir():
                continue
            change_start_dir = cs_dir.name
            for ce_dir in os.scandir(change_folder_path / change_start_dir):
                if not ce_dir.is_dir():
                    continue
                change_end_dir = ce_dir.name
                change_dirs.append(Path(ce_dir))

    # %%
    tiles_gdf = gpd.read_file(tiles_geojson_path)
    if area_selection_path is not None:
        area_gdf = gpd.read_file(area_selection_path).to_crs(tiles_gdf.crs)
        area_geom = unary_union(area_gdf.geometry)
        tiles_gdf = tiles_gdf[tiles_gdf.covered_by(area_geom)]
        
    tiles_geom = unary_union(tiles_gdf.geometry)
    # %%
    vectorial_gdf = None
    _change_tiles_meta = []
    for change_dir in change_dirs:
        change_end_dir = change_dir.name
        change_start_dir = change_dir.parent.name
        
        vec_appeared = gpd.read_file(
            change_folder_path
            / change_start_dir
            / change_end_dir
            / "vectorial"
            / "pits.appeared.geojson"
        ).to_crs(tiles_gdf.crs)
        vec_appeared["change_kind"] = "pits.appeared"
        vec_appeared = vec_appeared[vec_appeared.within(tiles_geom)]

        vec_disappeared = gpd.read_file(
            change_folder_path
            / change_start_dir
            / change_end_dir
            / "vectorial"
            / "pits.disappeared.geojson"
        ).to_crs(tiles_gdf.crs)
        vec_disappeared["change_kind"] = "pits.disappeared"
        vec_disappeared = vec_disappeared[vec_disappeared.within(tiles_geom)]

        vec_ = gpd.GeoDataFrame(
            pd.concat([vec_appeared, vec_disappeared], ignore_index=True),
        )
        change = f"{change_start_dir}-{change_end_dir}"
        vec_["change"] = change

        if vectorial_gdf is None:
            vectorial_gdf = vec_
        else:
            vectorial_gdf = gpd.GeoDataFrame(
                pd.concat([vectorial_gdf, vec_], ignore_index=True)
            )

        change_kind = "pits.appeared"
        change_raster_dir = (
            change_folder_path / change_start_dir / change_end_dir / "raster"
        )
        raster_glob_string = str(
            change_raster_dir / change_kind / "tiles" / "*.tif"
        )
        # %%
        for appeared_tile_path in glob(raster_glob_string):
            # Input paths!
            appeared_tile_path = Path(appeared_tile_path)
            if not (tiles_gdf.tile_id == appeared_tile_path.stem).any():
                continue # The tile is not one of the selected tiles!
            dis_tile_path = (
                change_raster_dir
                / "pits.disappeared"
                / "tiles"
                / appeared_tile_path.name
            )
            im1_in_path = (
                tiles_folder_path / change_start_dir / appeared_tile_path.name
            )
            im2_in_path = (
                tiles_folder_path / change_end_dir / appeared_tile_path.name
            )
            # %%
            out_filename = "-".join(
                [change_start_dir, change_end_dir, appeared_tile_path.name]
            )

            _change_tiles_meta.append(
                {
                    "change": change,
                    "change_start": change_start_dir,
                    "change_end": change_end_dir,
                    "tile_name": out_filename,
                }
            )

            # Output paths
            im1_out_path = output_path / "im1" / out_filename
            im2_out_path = output_path / "im2" / out_filename
            app_out_path = output_path / "pits.appeared" / out_filename
            disapp_out_path = output_path / "pits.disappeared" / out_filename

            os.makedirs(im1_out_path.parent, exist_ok=True)
            shutil.copy2(im1_in_path, im1_out_path)

            os.makedirs(im2_out_path.parent, exist_ok=True)
            shutil.copy2(im2_in_path, im2_out_path)

            os.makedirs(app_out_path.parent, exist_ok=True)
            shutil.copy2(appeared_tile_path, app_out_path)

            os.makedirs(disapp_out_path.parent, exist_ok=True)
            shutil.copy2(dis_tile_path, disapp_out_path)
    vectorial_gdf.to_file(
        output_path / "vectorial.geojson", index=False, driver="GeoJSON"
    )
    pd.DataFrame(_change_tiles_meta).to_csv(output_path / "tiles_meta.csv", index=False)


if __name__ == "__main__":
    # %%
    if not in_notebook():
        parser = ArgumentParser(
            "pits_site_dataset.py",
            description="""
    Given the data folder of a site combines the tiles into the change detection dataset!
    """,
        )
        parser.add_argument(
            "-s",
            "--site_path",
            type=Path,
            help="Site data folder.",
            required=True,
        )
        parser.add_argument(
            "-o",
            "--output_path",
            type=Path,
            help="Output folder path.",
            required=True,
        )
        parser.add_argument(
            "-a",
            "--area_selection_path",
            type=Path,
            help="Path to geojson containing an area that contains all the selected tiles.",
            default=None,
        )
        parser.add_argument(
            "-c",
            "--changes_selection",
            nargs="+",
            help="Change folders to select inside of the SITE/change/--- when building the dataset. Providing none means take all.",
        )

        args = parser.parse_args()
        site_data_path = args.site_path
        output_path = args.output_path
        area_selection_path = args.area_selection_path
        changes_selection = args.changes_selection

    else:
        site_data_path = Path("/home/gsech/alceo/data/sites/ASWAN")
        output_path = Path("/home/gsech/alceo/dataset/pits/test_ASWAN")
        area_selection_path = Path(
            "/home/gsech/alceo/data/sites/ASWAN/test_area.geojson"
        )
        changes_selection = ["AS_04_08_2005/AS_05_10_2015"]
    # %%
    pits_site_dataset(
        site_data_path, output_path, area_selection_path, changes_selection
    )

# %%

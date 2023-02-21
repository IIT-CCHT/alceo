# %%
from argparse import ArgumentParser
import shutil
import os
from pathlib import Path
from glob import glob

from alceo.utils import in_notebook

# %%
def pits_site_dataset(site_data_path: Path, output_path: Path):
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

    assert output_path.exists(), f"Provided output path does not exists ({output_path})"

    # %%

    for cs_dir in os.scandir(change_folder_path):
        change_start_dir = cs_dir.name
        for ce_dir in os.scandir(change_folder_path / change_start_dir):
            change_end_dir = ce_dir.name
            change_kind = "pits.appeared"
            raster_glob_string = str(change_folder_path / change_start_dir / change_end_dir / "raster" / change_kind / "tiles" / "*.tif")
            for tile_path in glob(raster_glob_string):
                tile_path = Path(tile_path)
            # %%
                out_filename = "-".join([change_start_dir, change_end_dir, tile_path.name])

                im1_in_path = tiles_folder_path / change_start_dir / tile_path.name
                im1_out_path = output_path / "im1" / out_filename

                im2_in_path = tiles_folder_path / change_end_dir / tile_path.name
                im2_out_path = output_path / "im2" / out_filename

                app_out_path = output_path / "pits.appeared" / out_filename 
                disapp_out_path = output_path / "pits.disappeared" / out_filename 


                os.makedirs(im1_out_path.parent, exist_ok=True)
                shutil.copy2(im1_in_path, im1_out_path)

                os.makedirs(im2_out_path.parent, exist_ok=True)
                shutil.copy2(im2_in_path, im2_out_path)

                os.makedirs(app_out_path.parent, exist_ok=True)
                shutil.copy2(tile_path, app_out_path)

                os.makedirs(disapp_out_path.parent, exist_ok=True)
                shutil.copy2(tile_path, disapp_out_path)


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
        

        args = parser.parse_args()
        site_path = args.site_path
        output_path = args.output_path
    else:
        site_path = Path("/HDD1/gsech/source/alceo/data/sites/DURA_EUROPOS")
        output_path = Path("/HDD1/gsech/source/alceo/dataset/pits/DURA_EUROPOS")

    pits_site_dataset(site_path, output_path)
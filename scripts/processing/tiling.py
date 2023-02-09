# %%
from argparse import ArgumentParser
import os
from itertools import product
from pathlib import Path
import rasterio as rio
from rasterio import windows
from alceo.utils import in_notebook
from sorcery import unpack_keys
if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

#
parser = ArgumentParser("tiling.py", description="""
A utility script to take a raster saved in a GeoTIFF and subset it in various tiles.
""")
parser.add_argument(
    "-i",
    "--input_raster_path",
    type=Path,
    help="Input GeoTIFF that should be divided in tiles!",
    required=True,
)
parser.add_argument(
    "-o",
    "--output_directory",
    type=Path,
    help="The output directory where the tiles should be saved in. Defaults to current directory.",
    default='.',
)
parser.add_argument(
    "-h",
    "--tile_height",
    type=int,
    default=512,
    help="Tile height in number of pixels",
)
parser.add_argument(
    "-w",
    "--tile_width",
    type=int,
    default=512,
    help="Tile width in number of pixels",
)

args = parser.parse_args()
input_raster_path: Path = args.input_raster_path
output_directory: Path = args.output_directory
tile_width, tile_height = unpack_keys(args)
output_filename = input_raster_path.stem + '_{}-{}.tif'

if not output_directory.exists():
    os.makedirs(output_directory, exist_ok=True)

def get_tiles(ds):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, tile_width), range(0, nrows, tile_height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

with rio.open(input_raster_path) as inds:
    meta = inds.meta.copy()

    for window, transform in tqdm(get_tiles(inds)):
        meta['transform'] = transform
        meta['width'], meta['height'] = window.width, window.height
        tile_filename = output_filename.format(int(window.col_off), int(window.row_off))
        outpath = output_directory / tile_filename
        with rio.open(outpath, 'w', **meta) as outds:
            outds.write(inds.read(window=window))
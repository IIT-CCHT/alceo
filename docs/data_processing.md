# Dataset Processing
Most models work on raster data, we have the satellite images rasters as well as vectorial information concerning the looting pits.  
We need to process them because a satellite image is too big to be processes by our models and we cannot directly use vectorial information.

## Tilizing satelitte images

So, for a given set of site images `script/processing/produce_tiles.py` produces a GeoJSON containing vectorial features and pixel sizes of the tiles.

The output GeoJSON can be given with a single satellite image to `script/processing/rasterize_tiles.py` for producing GeoTIFFs of the tiles if needed.

## Computing change from site annotations

The dataset pipeline converged on annotating all the likely looting pits in each satellite image and then compute positive (pit appearance) and negative (pit disappearance) change given a couple of images annotations.
This work is done by `script/processing/change_from_annotations.py` script which taken the annotations GeoJSON for a site and the first and second dates of interest, computes pits `appearance`, `disappearance` and `permanence` features and saves them into a GeoJSON.  
These change features can be easily rasterized using [rasterio's CLI](https://rasterio.readthedocs.io/en/latest/cli.html) by providing the GeoJSON in the same CRS of a raster of which we want to produce a binary mask.
Example:
```
rio rasterize <...>/annotation/DURA/complete/DURA.appeared.geojson --like <...>/images/DURA_EUROPOS/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif --output <...>/annotation/DURA/complete/DURA.appeared.tif
```
The mask can be tilized using `script/processing/rasterize_tiles.py`.

## Project Data Structure
Consult DVC.md for information about the process of creating the project's pipelines in DVC.

```
data /
    annotations / # Contains the annotations used to produce datasets
        <SITE> /
            pits.geojson # annotations specific to the looting pits

dataset /
    pits_monitoring /
        <SITE>
```

## Dataset Structure

```
SITE, TILE_ID, START_DATE, END_DATE


# possible formats.
# <DATE> refers to either START or END programmatically.

/ <SITE> 
    / <DATE>
        / tiles / <TILE_ID>.tif
        / appeared / <TILE_ID>.tif # with respect of previous date.
        / disappeared / <TILE_ID>.tif # with respect of previous date.

/ <SITE> 
    / tiles / <DATE> / <TILE_ID>.tif
    / change / <START_DATE> / <END_DATE> / 
        / appeared / <TILE_ID>.tif
        / disappeared / <TILE_ID>.tif

```
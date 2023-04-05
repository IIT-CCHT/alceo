# Data Management Sub-system
# Introduction
Most machine learning models for change detection work directly on raster data, we have the satellite images rasters as well as vectorial information concerning the looting pits.   
Pre-processing is needed because a satellite image is too big to be processes directly by our models and models do now work with vectorial features.  
This pre-processing is done with a series of tools as well as some custom-made scripts that have to be executed in a precise flow, such flow is called a pipeline and is documented here.

A typical pipeline is composed of the following steps:
1. Georeferenced tilization which is the production of vectorial features documenting how a single site will be subdivided in tiles.
2. Splitting the site's images in tiles.
3. Computing from the annotations and for each couple of dates the pits that appeared and disappeared which will result in the change detection dataset ground truth features.
4. Rasterization of the ground truth features and their spliting in tiles.
5. Composing the dataset with relative metadata in the correct file-system structure.

# Pipeline

For each site a data pipeline has been implemented. 
The data pipeline main objective is the production of all intermediate artifacts for the final dataset, these artifacts depend on the site's images and annotations.

The creation of a site's pipeline starts with the following file system:

```
data
    /sites
        /<SITE NAME>
            /annotations # contains annotation files (i.e. pits.geojson).
            /images # contains images products.
            /dvc.yaml # configuration of the site level pipeline.
```




## Georeferenced tilization of satelitte images

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



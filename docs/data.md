# Data Management Sub-system
This sub-system is in charge of producing the datasets used to train and evaluate the project's models starting from satellite imagery data and human annotation of the looting activity. 
Most machine learning models for computer vision tasks, such as the ones developed for this project, work directly on raster data.
This creates a need to automatically pre-process the data in a scalable and configurable manner so that we can promptly update our datasets when additional annotations are or some potential issues are addressed.
This pre-processing is done with a series of tools as well as some custom-made scripts that have to be executed in a precise sequence, such sequence is called a pipeline.  

# Site configuration

For each site, a pre-processing pipeline has been configured. All the site data and the pipeline configuration are stored inside the `data/site/<SITE NAME>` folder.
The data pipeline's main objective is the production of all intermediate artifacts for the final dataset, these artifacts depend on the site's images and annotations.

The setup of a typical site's data directory starts with the following components:
1. Area of Interest GeoJSON (e.g. `area_of_interest.geojson`) describing the boundaries of the site that the sub-system should process.
2. Annotations GeoJSON containing the polygons delimiting the looting pits. Each pit should be marked with a
3. Training and Testing GeoJSON (e.g. `test_area.geojson` and `test_area.geojson` ) describing 

```
data
    /sites
        /<SITE NAME>
            /annotations # contains annotation files (i.e. pits.geojson).
            /images # contains images products.
            /dvc.yaml # configuration of the site level pipeline.
```

## Geo-referenced tilization of satellite images

So, for a given set of site images `script/processing/produce_tiles.py` produces a GeoJSON containing vectorial features and pixel sizes of the tiles.  
The output GeoJSON can be given with a single satellite image to `script/processing/rasterize_tiles.py` for producing GeoTIFFs of the tiles if needed.

## Computing change from site annotations

The dataset pipeline converged on annotating all the likely looting pits in each satellite image and then compute positive (pit appearance) and negative (pit disappearance) change given a couple of image annotations.
This work is done by `script/processing/change_from_annotations.py` script which taken the annotations GeoJSON for a site and the first and second dates of interest, computes pits `appearance`, `disappearance` and `permanence` features and saves them into a GeoJSON.  
These change features can be easily rasterized using [rasterio's CLI](https://rasterio.readthedocs.io/en/latest/cli.html) by providing the GeoJSON in the same CRS of a raster of which we want to produce a binary mask.
Example:
```
rio rasterize <...>/annotation/DURA/complete/DURA.appeared.geojson --like <...>/images/DURA_EUROPOS/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif --output <...>/annotation/DURA/complete/DURA.appeared.tif
```
The mask can be tilized using `script/processing/rasterize_tiles.py`.



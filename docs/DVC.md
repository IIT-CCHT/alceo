# DVC Journey
1. I've installed DVC using Poetry (pip) with `poetry add dvc`.
2. [Get Started with DVC](https://dvc.org/doc/start)
3. Then I've been following the [Get Started: Data Versioning](https://dvc.org/doc/start/data-management/data-versioning) tutorial.  
    I've encountered a somewhat major "roadblock" while integrating my current setup: I cannot `dvc add` files contained in a [symlinked directory](https://dvc.org/doc/user-guide/troubleshooting#add-symlink).  
    This means that I had to rename the symlink from my data folder on HDD1 (I'm currently working on boxx230) and create a real directory in the project folder.  
    After running `dvc add data/annotations/` I've tracked changes to `data/annotations.dvc` and `data/.gitignore`.
4. I've added a local remote for boxx230 in `/HDD1/dvc_remotes/alceo`
    ```
    mkdir /HDD1/dvc_remotes
    mkdir /HDD1/dvc_remotes/alceo
    dvc remote add -d boxx230_hdd1 /HDD1/dvc_remotes/alceo
    ```
    And I've experimented with `dvc push` and it saves a copy of `.dvc/cache/` content on the remote. 
    I've used `dvc pull` after deleting both `data/annotations` and `.dvc/cache`. It gets copied correctly from the remote.

5. I've made a pipeline stage that computes changes from pits annotations of DURA (Dura Europhos) site. As a reference the command from bash is:
    ``` 
    python scripts/processing/change_from_annotations.py -i data/annotations/DURA/pits.geojson -o data/change/DURA -f 26/5/2013 -s 19/09/2014
    ```
    The stage was made with:
    ```
    dvc stage add -n DURA_changes -d scripts/processing/change_from_annotations.py -d data/annotations/DURA/pits.geojson -o data/change/DURA python scripts/processing/change_from_annotations.py -i data/annotations/DURA/pits.geojson -o data/change/DURA -f 26/5/2013 -s 19/09/2014
    ```  

6. After thorough investigation DVC pull and push do not take into consideration sub-directories. This means that one cannot pull only a portion of a directory that has already been added.  
    With this consideration in mind I've changed the structure of the data folder and created a sub-dir called `sites` in which each site will get a directory that will be `dvc add`ed.

7. I've created the `data/sites/DURA_EUROPOS` directory and put inside the previously created `annotations` directory. I've also created the `images` directory and placed `DE_19_09_2014` and `DE_26_5_2013` images that were given to me by Maria Cristina.
8. I'm putting together the full pipeline using DVC
    Tile featurization:

    ```
    python src/alceo/processing/produce_tiles.py \
    -i data/sites/DURA_EUROPOS/images/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif \
    -i /HDD1/gsech/source/alceo/data/sites/DURA_EUROPOS/images/DE_26_5_2013/DE_26_5_2013_NN_diffuse.tif \
    -o data/sites/DURA_EUROPOS/tiles.json -p DURA_EUROPOS_
    ```
    Tile rasterization:

    ```
    python src/alceo/processing/rasterize_tiles.py \
        -t data/sites/DURA_EUROPOS/tiles.json \
        -i data/sites/DURA_EUROPOS/images/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif \
        -o data/sites/DURA_EUROPOS/tiles/DE_19_09_2014/DE_19_09_2014_NN_diffuse
    ```

9. I've integrated the area of interest in tile rasterization and was able to apply [foreach stages](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files#foreach-stages) to this pipeline stage obtaining the following `dvc.yaml` file:  

    ```  
    stages:
    produce_tiles_DURA_EUROPOS:
        cmd: python src/alceo/processing/produce_tiles.py 
        -i data/sites/DURA_EUROPOS/images/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif
        -i /HDD1/gsech/source/alceo/data/sites/DURA_EUROPOS/images/DE_26_5_2013/DE_26_5_2013_NN_diffuse.tif
        -o data/sites/DURA_EUROPOS/tiles.json -p DURA_EUROPOS_
        deps:
        - data/sites/DURA_EUROPOS/images/DE_19_09_2014/DE_19_09_2014_NN_diffuse.tif
        - data/sites/DURA_EUROPOS/images/DE_26_5_2013/DE_26_5_2013_NN_diffuse.tif
        - src/alceo/processing/produce_tiles.py
        outs:
        - data/sites/DURA_EUROPOS/tiles.json
    tilize_image:
        foreach:
        - site: DURA_EUROPOS
            image: DE_19_09_2014/DE_19_09_2014_NN_diffuse
        - site: DURA_EUROPOS
            image: DE_26_5_2013/DE_26_5_2013_NN_diffuse
        do:
        cmd: python src/alceo/processing/rasterize_tiles.py 
            -t data/sites/${item.site}/tiles.json
            -i data/sites/${item.site}/images/${item.image}.tif 
            -a data/sites/${item.site}/area_of_interest.geojson
            -o data/sites/${item.site}/tiles/${item.image}
        deps:
            - src/alceo/processing/rasterize_tiles.py
            - data/sites/${item.site}/tiles.json
            - data/sites/${item.site}/images/${item.image}.tif
            - data/sites/${item.site}/area_of_interest.geojson
    ```

    I was unable to parametrize the deps for the `produce_tiles` stage. This needs more studying as I'm not sure how to parametrize a list of parameters for a command plus parametrizing a list of dependencies.  

10. I've created the `change_annotations` stage using `foreach` so that we can produce automatically more stages by defining what changes we are interested in.  
    TODO: once the pipeline is done I should try to move all the various parametrizations outside of the loops itselfs. Maybe using a [parameters file](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files#parameters-files)?  

11. `rasterize_change` was also added to the pipeline and uses a foreach. This stage does not depend on my code, it uses `mkdir -p` to create the folder in which rasters will be generated (if needed) and then runs `rio rasterize ...`.  

12. Started working on full pipeline parametrization. It seems like I cannot unpack a list into `stage.deps`. This means that I have to re-think about `produce_tiles`. New strategy will be getting a single raster (a la `rio rasterize --like`), the `area_of_interest` GeoJSON and use them to produce the vectorial tiles.
    - Pro: rasterize_tile does not need to handle filtering.
    - Con: I cannot be sure that the area of interest covers ALL future images unless I "intersect" the area of interest with the images bounds before creating the tiles. This could be problematic because I would need to create an intermediate "true area of interest" file.

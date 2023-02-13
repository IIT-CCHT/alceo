# DVC Journey
0. I've installed DVC using Poetry (pip) with `poetry add dvc`.
1. [Get Started with DVC](https://dvc.org/doc/start)
2. Then I've been following the [Get Started: Data Versioning](https://dvc.org/doc/start/data-management/data-versioning) tutorial.  
    I've encountered a somewhat major "roadblock" while integrating my current setup: I cannot `dvc add` files contained in a [symlinked directory](https://dvc.org/doc/user-guide/troubleshooting#add-symlink).  
    This means that I had to rename the symlink from my data folder on HDD1 (I'm currently working on boxx230) and create a real directory in the project folder.  
    After running `dvc add data/annotations/` I've tracked changes to `data/annotations.dvc` and `data/.gitignore`.
3. I've added a local remote for boxx230 in `/HDD1/dvc_remotes/alceo`
    ```
    mkdir /HDD1/dvc_remotes
    mkdir /HDD1/dvc_remotes/alceo
    dvc remote add -d boxx230_hdd1 /HDD1/dvc_remotes/alceo
    ```
    And I've experimented with `dvc push` and it saves a copy of `.dvc/cache/` content on the remote. 
    I've used `dvc pull` after deleting both `data/annotations` and `.dvc/cache`. It gets copied correctly from the remote.

    <!-- TODO: investigate dvc pull by pipeline stage. --> I should investigate if by using `dvc pull <pipeline stage>` I get only the data needed by that stage or all the remote data.

4. I've made a pipeline stage that computes changes from pits annotations of DURA (Dura Europhos) site. As a reference the command from bash is:
    ``` 
    python scripts/processing/change_from_annotations.py -i data/annotations/DURA/pits.geojson -o data/change/DURA -f 26/5/2013 -s 19/09/2014
    ```
    The stage was made with:
    ```
    dvc stage add -n DURA_changes -d scripts/processing/change_from_annotations.py -d data/annotations/DURA/pits.geojson -o data/change/DURA python scripts/processing/change_from_annotations.py -i data/annotations/DURA/pits.geojson -o data/change/DURA -f 26/5/2013 -s 19/09/2014
    ```  

5. After thorough investigation DVC pull and push do not take into consideration sub-directories. This means that one cannot pull only a portion of a directory that has already been added.  
    With this consideration in mind I've changed the structure of the data folder and created a sub-dir called `sites` in which each site will get a directory that will be `dvc add`ed.
    
6. I've created the `data/sites/DURA_EUROPOS` directory and put inside the previously created `annotations` directory. I've also created the `images` directory and placed `DE_19_09_2014` and `DE_26_5_2013` images that were given to me by Maria Cristina.
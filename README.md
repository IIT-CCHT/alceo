# Automatic Looting Classification from Earth Observation (ALCEO)
![Documentation](https://github.com/github/docs/actions/workflows/pages_ci.yaml/badge.svg)  


This is the source code repository of the Python module developed for the ALCEO project. Reference of the Program-Contract: ALCEO Grant n. 4000136771, between European Space Agency (ESA) and Istituto Italiano di Tecnologia (IIT) for ALCEO: Automatic Looting Classification from Earth Observation.  

## Installing the ALCEO Python module
The first step to install the `alceo` python module is to clone the repository.
This project uses [Poetry](https://python-poetry.org/docs/) as a dependency manager.  
You can use `poetry install` to download all the dependencies and have the `alceo` module installed.  
The `[tool.poetry.dependencies]` section of `pyproject.toml` [metadata file](https://peps.python.org/pep-0621/) contains the project dependencies definition.   
By default, Poetry creates a virtual environment for the project. Refer to Poetry's [documentation to configure](https://python-poetry.org/docs/configuration/) this behavior.

## Serving the documentation locally
To host the documentation website on the local machine, use the `mkdocs serve` command after installing all the project dependencies.

## Retrieving Data and Models from a DVC Remote
[Data Version Control (DVC)](https://dvc.org/doc) has been extensively used in this project to create reproducible pipelines for handling data pre-processing, training, validating, testing, and performing inference of the machine learning models implemented during the project.

A DVC remote is needed for downloading and updating all experiments' data, logged metrics, and model checkpoints. As explained in DVC's documentation, one can add a new (default) remote by using the following command:
```
dvc remote add -d myremote /path/to/remote
```
Consult [DVC documentation](https://dvc.org/doc/user-guide/data-management/remote-storage) for more information about configuring remotes and a list of supported storage types. 

If the configured remote already contains project data, the `dvc pull` command will download them locally.

## Running a pipeline

The Data Management sub-system and Modelling sub-system pipelines are defined using [DVC](https://dvc.org/doc/user-guide/pipelines), for e detailed description of these sub-systems, refer to the corresponding sections in the documentation navigation menu.

The definition of the DVC pipeline stages happens inside various `dvc.yaml` files.  
To reproduce all the pipelines use the optional argument `-P` (or `--all-pipelines`).   
Specific stage targets can be selected for reproduction `dvc repro [targets [targets ...]]`. To run a stage provide to the `dvc repro` command the path to the `dvc.yaml` file followed by colon `:` and the stage name. For example, from the project root:
```
dvc repro pipelines/fit/dvc.yaml:fit_siam_diff
```

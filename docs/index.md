
## Installing the ALCEO Python module
This project uses [Poetry](https://python-poetry.org/docs/) as dependency manager. Once the repository is downloaded you can use `poetry install` to get all the dependencies downloaded and the `alceo` module installed. The full dependency definition can be found inside the `[tool.poetry.dependencies]` section of `pyproject.toml` [metadata file](https://peps.python.org/pep-0621/). By default Poetry creates a virtual environment for the project, refer to Poetry's [documentation to configure](https://python-poetry.org/docs/configuration/) this behaviour.

## Retrieving Data and Models from a DVC Remote
[Data Version Control (DVC)](https://dvc.org/doc) has been extensively used in this project to create reproducible pipelines for handling data pre-processing as well as training, validating, testing, and performing inference of the machine learning models implemented during the project.

A DVC remote is needed for downloading and updating all experiments' data, logged metrics, and model checkpoints. As explained in DVC's documentation one can add a new (default) remote by using the following command:
```
dvc remote add -d myremote /path/to/remote
```
Consult [DVC documentation](https://dvc.org/doc/user-guide/data-management/remote-storage) for more information about configuring remotes as well as a list of supported storage types. 

If the configured remote already contains project data and/or model checkpoints the `dvc pull` command will download them locally.

## Running a pipeline

The Data Management sub-system and Modelling sub-system pipelines are defined using [DVC](https://dvc.org/doc/user-guide/pipelines). For e detailed description of these sub-systems refer to the corresponding sections in the documentation navigation menu.

DVC pipeline stages are defined inside dvc.yaml files and the pipelines can be reproduced with the command `dvc repro`. To reproduce all the pipelines use the optional argument `-P` (or `--all-pipelines`).   
Specific stage targets can be selected for reproduction `dvc repro [targets [targets ...]]`. To run a stage provide to the `dvc repro` command the path to the `dvc.yaml` file followed by colon `:` and the stage name. For example, from the project root:
```
dvc repro pipelines/fit/dvc.yaml:fit_siam_diff
```

## Versioning Data and or Checkpoints

Once some data or checkpoint is added or modified the `dvc add` and `dvc push` commands can be used to update the remote state and share such results.


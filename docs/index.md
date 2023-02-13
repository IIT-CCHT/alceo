# ALCEO

This documentation is done using [mkdocs.org](https://www.mkdocs.org). Some useful commands are:

* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.

## Table of Contents
* [Data](data_processing): docs about data handling and how we go from annotations to dataset(s).

## Setting up the project

This project uses [Poetry](https://python-poetry.org/docs/) as dependency manager. Once the repository is downloaded you can use `poetry install` to get all the dependencies installed.  

### Optional setup
I'm using [Visual Studio Code](https://code.visualstudio.com/) to develop this project.  
The [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extension turns quite useful to connect to a remote host, I'm mainly working on Boxx230 but I've also used it to connect to Franklin instead of using a simple terminal.

### Data setup
All data will be pullable <!-- TODO when done integrating DVC fix this --> using DVC. Refer to the [Data](data_processing) section of the docs for more details.


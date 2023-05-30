# Modeling Sub-system

This sub-system trains, tests, and performs inference with Deep Learning-based change detection models.
The main tools used are the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework and [DVC](https://dvc.org/doc). 

PyTorch Lightning provides a convenient object-oriented abstraction between the model, data, and training-validation-test loop. This results in a reduction of boilerplate code and code standardization. This abstraction also allows for a degree of transparency between the experiments' developer/maintainer and compute infrastructure.

## CLI entry point
The `alceo` Python module is the entry point for the Modeling Sub-system operations. The entry point is a Command Line Interface implemented using the Lightning framework's CLI module: [Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).

If the project is installed correctly in the current Python environment, running:
```bash 
python -m alceo --help
```
Should yield:

```text
usage: __main__.py [-h] [-c CONFIG] [--print_config[=flags]] {fit,validate,test,predict} ...

pytorch-lightning trainer command line tool

optional arguments:
  -h, --help            Show this help message and exit.
  -c CONFIG, --config CONFIG
                        Path to a configuration file in json or yaml format.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords separated by comma. The supported flags are: comments, skip_default, skip_null.

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  {fit,validate,test,predict}
    fit                 Runs the full optimization routine.
    validate            Perform one evaluation epoch over the validation set.
    test                Perform one evaluation epoch over the test set.
    predict             Run inference on your data.
```

The help outputs the four verbs exposed by Lightning CLI: `fit`, `validate`, `test`, and `predict`.

### Training a model
Training a model using Lightning CLI requires an experiment configuration in JSON or YAML format. This kind of configuration requires the definition of some fundamental objects to run an experiment. These objects are:

1. The model's [`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html). It contains the fundamental parts of your experiment, for example: how to compute the model output, how to obtain the loss value for optimization, how to set up an optimizer, and how to log metrics.
2. The data's [`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). It encapsulates the data processing portion of your experiment, such as: loading data in PyTorch [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset), applying transforms, and producing training/validation/testing/prediction [DataLoaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
3. The Lightning's [`Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html) object. `Trainer` handles the training/validation/test/prediction loops and provides a callback interface for injecting custom logic without polluting the model's or data's code.

The `config/siam-diff.yaml` file is a configuration for an experiment with the architecture proposed by [Daudt et al.](https://ieeexplore.ieee.org/abstract/document/8451652). The following subsections describe the various sections of that configuration and how they can be modified for modifying most parts of the experiment.

#### Model

The `model` section describes how the model's LightningModule should be instantiated. In `alceo`'s CLI this LightningModule needs to be a specialization (so to extend) the `alceo.model.PhaseMetricModule` class. This is needed to allow for correctly receiving the labels needed for metric breakdowns of the multiple datasets that can compose a single phase (training, validation, test).

`alceo.model.AlceoChangeDetectionModule` is a helper class that extends `alceo.model.PhaseMetricModule` for change detection tasks. It takes as parameters two PyTorch modules: the change detection `network` that should output activations and the loss function (`loss_fn`). This helper computes the loss on the network's outputs, the metrics used to monitor an experiment, and the metrics breakdown for all the datasets. The monitored metrics are the Jaccard Index, F1 Score, Precision, Recall, mIoU, and a custom mIoU that ignores tiles with only background information.

An object is instantiated by providing its `class_path` (the class name in its correct namespace) and the `init_args` (the parameters' values needed for initialization). This comes from [`jsonargparse`](https://jsonargparse.readthedocs.io/en/stable/) which is a Python package that was used to build Lightning CLI.

```yaml
model:
  class_path: alceo.model.AlceoChangeDetectionModule
  init_args:
    network:
      class_path: alceo.model.SiamUnet_diff
      init_args:
        input_nbr: 4
        label_nbr: 2
    loss_fn:
      class_path: segmentation_models_pytorch.losses.JaccardLoss
      init_args:
        mode: multilabel
```

#### Data

The `data` section describes how the `LightningDataModule` should be instantiated. Much like the model's `LightningModule` this `LightningDataModule` should be a specialization of a provided class: `alceo.data_module.PhaseDataModule`. Also here a utility `LightningDataModule` is provided: `alceo.data_module.AlceoChangeDetectionDataModule`. This DataModule is compatible with datasets produced by the Data Management Sub-System pipelines. The user should just provide the datasets paths and labels and can parametrize the experiment batch size as well as the number of DataLoaders workers.

Because of the following issues ([9352](https://github.com/Lightning-AI/lightning/issues/9352) and [15688](https://github.com/Lightning-AI/lightning/issues/15688)) with training using multiple Datasets in 16 bit precision the `AlceoChangeDetectionDataModule` concatenates all the training datasets. This does not happen for the validation and test datasets so the metrics breakdowns are computed correctly.

This is an example of an `AlceoChangeDetectionDataModule` initialized with 3 datasets for training and 3 for validation and a 24 items batch:

```yaml 
data:
  class_path: alceo.data_module.AlceoChangeDetectionDataModule
  init_args: 
    batch_size: 24
    train_paths:
      - dataset/pits/train_DURA_EUROPOS
      - dataset/pits/train_ASWAN
      - dataset/pits/train_EBLA
    validation_paths:
      - dataset/pits/test_DURA_EUROPOS
      - dataset/pits/test_ASWAN
      - dataset/pits/test_EBLA
    validation_labels:
      - DURA_EUROPOS
      - ASWAN
      - EBLA
```
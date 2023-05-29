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

The `config/siam-diff.yaml` file is a configuration for an experiment with the architecture proposed by Daudt et al.. The following sub-sections describe how that configuration 
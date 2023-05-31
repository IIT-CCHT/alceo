# Modeling Sub-system

This sub-system trains, tests, and performs inference with Deep Learning-based change detection models.
The main tools used are the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework and [DVC](https://dvc.org/doc). 

PyTorch Lightning provides a convenient object-oriented abstraction between the model, data, and training-validation-test loop. This results in a reduction of boilerplate code and code standardization. This abstraction also allows for some transparency between the experiments' developer/maintainer and compute infrastructure.

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

The help outputs the four verbs exposed by Lightning CLI: `fit`, `validate`, `test`, and `predict`. The `fit`, `test`, and `predict` verbs are the most important ones. `validate` is somewhat redundant for this experiment as running the `fit` procedure logs all the metrics for the validation datasets.

The following section describes how to use the `fit` verb for training a model. 

### Training a model

To train a model, use the `fit` verb in the project root as follows:
```bash
python -m alceo fit -c config/siam-diff.yaml
```

Training a model using Lightning CLI requires an experiment configuration in JSON or YAML format. This kind of configuration requires the definition of some fundamental objects to run an experiment. These objects are:

1. The model's [`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html). It contains the fundamental parts of your experiment, for example: how to compute the model output, how to obtain the loss value for optimization, how to set up an optimizer, and how to log metrics.
2. The data's [`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). It encapsulates the data processing portion of your experiment, such as: loading data in PyTorch [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset), applying transforms, and producing training/validation/testing/prediction [DataLoaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
3. The Lightning's [`Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html) object. `Trainer` handles the training/validation/test/prediction loops and provides a callback interface for injecting custom logic without polluting the model's or data's code.

The `config/siam-diff.yaml` file is a configuration for an experiment with the architecture proposed by [Daudt et al.](https://ieeexplore.ieee.org/abstract/document/8451652). The following subsections describe the various sections of that experiment configuration.

#### Model

The `model` section describes how to instantiate the model's `LightningModule`. In `alceo`'s CLI, this LightningModule needs to be a specialization (so to extend) the `alceo.model.PhaseMetricModule` class. This is needed to allow for correctly receiving the labels needed for metric breakdowns of the multiple datasets that can compose a single phase (training, validation, test).

`alceo.model.AlceoChangeDetectionModule` is a helper class implementing change detection model experiments and extends `alceo.model.PhaseMetricModule`. It takes as parameters two PyTorch modules: the change detection `network` that should output activations and the loss function (`loss_fn`). This helper computes the loss on the network's outputs, the metrics used to monitor an experiment, and the metrics breakdown for all the datasets. The monitored metrics are the Jaccard Index, F1 Score, Precision, Recall, mIoU, and a custom mIoU that ignores tiles with only background information.

To instantiate an object `class_path` (the class name in its correct namespace) and the `init_args` (the parameters' values needed for initialization) must be provided. This syntax comes from [`jsonargparse`](https://jsonargparse.readthedocs.io/en/stable/), a Python package on which Lightning CLI depends on.

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

##### Optimizer and learning rate scheduler
The `optimizer` section is optional if the provided model's class implements the [`configure_optimizers`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers) method. `AlceoChangeDetectionModule` does not implement the said method by design to decouple the optimizer and the experiment.  

The `lr_scheduler` section defines the experiment learning rate scheduler precisely like the `optimizer` section. Otherwise, the model's class should implement the [`lr_schedulers`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lr-schedulers) method.   

```yaml 
optimizer:
  class_path: torch.optim.Adam
  init_args: 
    lr: 1e-4
    weight_decay: 1e-4
lr_scheduler:
  class_path: pytorch_lightning.cli.ReduceLROnPlateau
  init_args:
    mode: max
    monitor: validation/appeared/IoU
    factor: 0.95
```

#### Data

The `data` section describes how to instantiate the `LightningDataModule`. Much like the model's `LightningModule` this `LightningDataModule` should be a specialization of a provided class: `alceo.data_module.PhaseDataModule`.`alceo.data_module.AlceoChangeDetectionDataModule` is a utility `LightningDataModule` compatible with datasets produced by the Data Management Sub-System pipelines. The user should give the datasets paths and metric labels and can also parametrize the experiment batch size and the number of DataLoaders workers.

Because of the following issues ([9352](https://github.com/Lightning-AI/lightning/issues/9352) and [15688](https://github.com/Lightning-AI/lightning/issues/15688)) with training using multiple Datasets in 16-bit precision, the `AlceoChangeDetectionDataModule` concatenates all the training datasets. This behavior does not happen for the validation and test datasets.

Here is an example of an `AlceoChangeDetectionDataModule` initialized with three datasets for training, three datasets for validation, and a batch composed of 24 items:

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

#### Trainer
The `trainer` section contains the parameters used to instantiate the Lightning [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html#) object. Such parameters are called "Trainer flags" in the [PyTorch Lighting documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags).  
A parameter of general interest is [`logger`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#logger). It defines the Logger object that will persist the metric values logged during the experiment. For example, DVCLiveLogger allows for metric logging in DVC.  
Another critical parameter is [`callbacks`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#callbacks). It's a list of utilities that should not be part of a single experiment, but their logic is bound to the train/validation/test loops. For example, ModelCheckpoint configures a model checkpointing strategy and is agnostic to the underlying experiment. 
The [`accelerator`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#accelerator) flag specifies the kind of accelerator used in your infrastructure. When training in a distributed environment, ['strategy'](https://lightning.ai/docs/pytorch/stable/common/trainer.html#strategy) allows for choosing one of the implemented training strategies.  

To seed all the random number generators, Lightning exposes a section called `seed_everything` to specify a seed number.

```yaml
seed_everything: 1234
trainer:
  accelerator: gpu
  precision: 16
  max_time: "00:24:00:00"
  logger: 
    class_path: alceo.logger.DVCLiveLogger
    init_args:
      run_name: run
      dir: logs/siam_diff
  log_every_n_steps: 5
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
        monitor: validation/appeared/IoU
        save_last: True
        save_top_k: 1
        mode: max
        filename: "best_IoU"
        auto_insert_metric_name: False
    - class_path: pytorch_lightning.callbacks.LearningRateMonitor # This logs at the end of all epoch a metric containing the current learning rate (useful when learning rate schedulers are used)
      init_args:
        logging_interval: epoch
```

### Prediction from a model checkpoint

The `predict` verb runs a previously trained model on a dataset with the intent of producing the model activations. A [`BasePredictionWriter`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html#basepredictionwriter) callback persists activations on storage. `alceo.callback.TiffPredictionWriter` is a `BasePredictionWriter` that saves the network activation in a GeoTIFF with the same geo-reference of the input tile.  

The user should provide a configuration to run prediction on a new dataset. Lightning CLI allows the user to update and append to a configuration file. Here is an example on a dataset called "SITE_TIME0_TIME1":

```bash 
python -m alceo predict --config config/siam-diff.yaml \
    --ckpt_path logs/siam_diff/DvcLiveLogger/run/checkpoints/best_IoU.ckpt \
    --trainer.callbacks+=alceo.callback.TiffPredictionWriter \
    --trainer.callbacks.init_args.output_dir=inference/siam_diff/SITE_TIME0_TIME1 \
    --data.predict_paths+=dataset/pits/SITE_TIME0_TIME1 \
    --data.predict_labels+=SITE_TIME0_TIME1
```

The `ckpt_path` flag tells the CLI to initialize the model starting from a checkpoint. The `+=` syntax appends to the configuration values.  
In this example, `TiffPredictionWriter`'s `output_dir` parameter is defined as well as the `predict_path` and `predict_label` of the dataset, previously produced using the Data Management Sub-System pipelines.

## Modeling Sub-System Pipelines

The `pipelines` folder stores the Modeling Sub-System pipelines. The `fit` sub-folder contains a pipeline for fitting the Siam-Diff architecture. The `predict` sub-folder includes the configuration for doing model inference on a prediction dataset.
DVC will ensure to trigger model re-training and save new inferences when datasets or code changes.

The `predict` pipeline is composed of a single stage but runs several commands:

```yaml
stages:
  predict_SITE_TIME0_TIME1:
    wdir: ../../.
    cmd:
      - python -m alceo predict --config experiment_config.yaml --ckpt_path logs/siam_diff/DvcLiveLogger/run/checkpoints/best_IoU.ckpt --trainer.callbacks+=alceo.callback.TiffPredictionWriter --trainer.callbacks.init_args.output_dir=inference/siam_diff/SITE_TIME0_TIME1 --data.predict_paths+=dataset/pits/SITE_TIME0_TIME1 --data.predict_labels+=SITE_TIME0_TIME1
      - rio merge inference/siam_diff/SITE_TIME0_TIME1/activation/*.tif -o inference/siam_diff/SITE_TIME0_TIME1/activation.tif --overwrite
      - rio merge inference/siam_diff/SITE_TIME0_TIME1/mask/*.tif -o inference/siam_diff/SITE_TIME0_TIME1/mask.tif --overwrite
      - rio shapes --as-mask --bidx 1 inference/siam_diff/SITE_TIME0_TIME1/mask.tif -o inference/siam_diff/SITE_TIME0_TIME1/appeared.geojson
      - rio shapes --as-mask --bidx 2 inference/siam_diff/SITE_TIME0_TIME1/mask.tif -o inference/siam_diff/SITE_TIME0_TIME1/disappeared.geojson
    deps:
      - experiment_config.yaml 
      - logs/siam_diff/DvcLiveLogger/run/checkpoints/best_IoU.ckpt
      - dataset/pits/SITE_TIME0_TIME1
    outs:
      - inference/siam_diff/SITE_TIME0_TIME1 
```

This stage uses the `predict` verb with the `TiffPredictionWriter` and dataset introduced before. Then Rasterio's utility `merge` does a mosaic of the obtained GeoTIFFs into two rasters: `activation.tif` and `mask.tif`. Lastly, the utility `shapes` compute the vectorial representation from the `mask.tif` file for faster loading in GIS. 
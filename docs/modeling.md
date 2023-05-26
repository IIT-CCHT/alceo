# Modeling Sub-system

This sub-system trains, tests, and performs inference with Deep Learning-based change detection models.
The main tools used are the [PyTorch Lightning framework](https://lightning.ai/docs/pytorch/stable/) and [DVC](https://dvc.org/doc). 

PyTorch Lightning is used to reduce the training-validation-test loop boilerplate as it provides a convenient object-oriented abstraction. This abstraction also allows for a degree of transparency between the experiments' developer/maintainer and compute infrastructure.


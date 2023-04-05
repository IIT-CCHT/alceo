# Modelling Sub-system

This sub-system is tasked with training, testing and doing inference of Change Detection models based on Deep Learning.  

We're using the PyTorch Lightning framework so the model's main module is a LightningModule that, right now, implements all the functionalities in  `alceo.model.lightning.AlceoLightningModule`.
`AlceoLightningModule` uses any PyTorch Module through the `network` property.  


from model.lightning import AlceoLightningModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
if __name__ == "__main__":
    pl.seed_everything(1234, workers=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_false",
        devices=[0, 1, 2, 3],
        precision=16,
        max_epochs=10,
    )
    model = AlceoLightningModule()
    trainer.fit(model=model)
   # 

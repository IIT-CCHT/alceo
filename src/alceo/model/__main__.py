from alceo.model.pits import PitsLightningModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from dvclive.lightning import DVCLiveLogger
if __name__ == "__main__":
    pl.seed_everything(1234, workers=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=[0, 1],
        precision=16,
        max_epochs=100,
        logger=DVCLiveLogger(),
        log_every_n_steps=5,
    )
    model = PitsLightningModule()
    trainer.fit(model=model)
   # 

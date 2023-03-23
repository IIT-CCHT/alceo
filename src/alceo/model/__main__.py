from alceo.model.pits import PitsLightningModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from dvclive.lightning import DVCLiveLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    pl.seed_everything(1234, workers=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=[0, 1, 2, 3],
        precision=16,
        max_time="00:12:00:00",
        logger=DVCLiveLogger(run_name="pits_change_detection", dir="log"),
        log_every_n_steps=5,
        callbacks=[
            ModelCheckpoint(
                monitor="validation/DURAEUROPOS/appeared/iou-no-empty",
                save_last=True,
                save_top_k=2,
                mode="max",
                filename="DURAEUROPOS-{epoch:02d}-{validation/DURAEUROPOS/appeared/iou-no-empty:.5f}",
            ),
        ],
    )
    model = PitsLightningModule()
    trainer.fit(model=model)
#

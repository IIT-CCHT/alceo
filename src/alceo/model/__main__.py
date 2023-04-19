
from alceo.model.alceo_metric_module import AlceoMetricModule
from alceo.model.siam_diff import SiamUnet_diff
from alceo.data_module import AlceoChangeDetectionDataModule
import pytorch_lightning as pl
from dvclive.lightning import DVCLiveLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from segmentation_models_pytorch.losses import JaccardLoss




if __name__ == "__main__":
    pl.seed_everything(1234, workers=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_false",
        devices=[0, 1, 2, 3],
        precision=16,
        max_time="00:10:00:00",
        logger=DVCLiveLogger(run_name="pits_change_detection", dir="log"),
        log_every_n_steps=5,
        callbacks=[
            ModelCheckpoint(
                monitor="validation/appeared/IoU",
                save_last=True,
                save_top_k=2,
                mode="max",
                filename="epoch={epoch:02d}-IoU={validation/appeared/IoU:.5f}",
                auto_insert_metric_name=False,
            ),
        ],
    )
    loss_fn = JaccardLoss(mode="multilabel")
    datamodule = AlceoChangeDetectionDataModule()
    datasets_labels = ["DE", "AS", "EB"]
    network = SiamUnet_diff(input_nbr=4, label_nbr=2)
    model = AlceoMetricModule(
        network=network,
        loss_fn=loss_fn,
        training_labels=[],
        validation_labels=datasets_labels,
        test_labels=datasets_labels,
    )
    trainer.fit(model=model, datamodule=datamodule)
#

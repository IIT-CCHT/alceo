# %%
from pathlib import Path
from alceo.dataset.change_detection import AlceoChangeDetectionDataset
import pytorch_lightning as pl
from alceo.callback.pits_prediction_writer import TiffPredictionWriter
from alceo.model.alceo_change_detection_module import AlceoChangeDetectionModule
from torch.utils.data import DataLoader
from alceo.model.siam_diff import SiamUnet_diff
from segmentation_models_pytorch.losses import JaccardLoss
# %%
if __name__ == '__main__':
    # %%
    pred_writer = TiffPredictionWriter(
        output_dir=Path("/HDD1/gsech/source/alceo/inference/DURA_EUROPOS")
    )

    pl.seed_everything(1234, workers=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[
            pred_writer,
        ],
        precision=16,
    )

    pits_dataset_path = Path("/HDD1/gsech/source/alceo/dataset/pits/")
    dataset = AlceoChangeDetectionDataset(pits_dataset_path / "DURA_EUROPOS")
    pred_loader = DataLoader(dataset, batch_size=6, num_workers=5)
    datasets_labels = ["DE", "AS", "EB"]
    network = SiamUnet_diff(input_nbr=4, label_nbr=2)
    loss_fn = JaccardLoss(mode="multilabel")
    
    model = AlceoChangeDetectionModule.load_from_checkpoint("/HDD1/gsech/source/alceo/DvcLiveLogger/epoch=1017-IoU=0.34307.ckpt",
        network=network,
        loss_fn=loss_fn,
        training_labels=[],
        validation_labels=datasets_labels,
        test_labels=datasets_labels,
    )
    trainer.predict(
        dataloaders=pred_loader,
        model=model,
    )
    # %% rio merge data/sites/DURA_EUROPOS/inference/DE_26_5_2013-DE_19_09_2014/*.tif data/sites/DURA_EUROPOS/inference/DE_26_5_2013-DE_19_09_2014.tif --overwrite

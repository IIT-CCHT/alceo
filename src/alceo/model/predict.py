# %%
from pathlib import Path
from alceo.dataset.pits import PitsSiteDataset
import pytorch_lightning as pl
from alceo.callback.pits_prediction_writer import TiffPredictionWriter
from alceo.model.pits import PitsChangeDetectionNetwork, PitsLightningModule
from torch.utils.data import DataLoader

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
    dataset = PitsSiteDataset(pits_dataset_path / "DURA_EUROPOS")
    pred_loader = DataLoader(dataset, batch_size=6, num_workers=5)
    datasets_labels = ["DE", "AS", "EB"]
    network = PitsChangeDetectionNetwork()
    model = PitsLightningModule.load_from_checkpoint("/HDD1/gsech/source/alceo/DvcLiveLogger/pits_change_detection/checkpoints/last.ckpt",
        network=network,
        training_labels=[],
        validation_labels=datasets_labels,
        test_labels=datasets_labels,
    )
    trainer.predict(
        dataloaders=pred_loader,
        model=model,
    )
    # %% rio merge data/sites/DURA_EUROPOS/inference/DE_26_5_2013-DE_19_09_2014/*.tif data/sites/DURA_EUROPOS/inference/DE_26_5_2013-DE_19_09_2014.tif --overwrite

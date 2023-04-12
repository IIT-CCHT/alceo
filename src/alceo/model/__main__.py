from pathlib import Path
from typing import Any, Optional
from alceo.dataset.pits import PitsSiteDataset
from alceo.model.pits import PitsLightningModule, PitsChangeDetectionNetwork
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from dvclive.lightning import DVCLiveLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, Dataset, ConcatDataset, DataLoader
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    STEP_OUTPUT,
    EVAL_DATALOADERS,
    EPOCH_OUTPUT,
)


class PitsDataModule(pl.LightningDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        pits_dataset_path = Path("/HDD1/gsech/source/alceo/dataset/pits/")
        dataset = PitsSiteDataset(pits_dataset_path / "DURA_EUROPOS")
        self.datasets = [
            ("DURAEUROPOS", PitsSiteDataset(pits_dataset_path / "DURA_EUROPOS")),
            ("ASWAN", PitsSiteDataset(pits_dataset_path / "ASWAN")),
            ("EBLA", PitsSiteDataset(pits_dataset_path / "EBLA")),
        ]
        self.train_datasets, self.validation_datasets, self.test_datasets = [], [], []

        for label, dataset in self.datasets:
            train_dataset, validation_dataset, test_dataset = random_split(
                dataset, [0.8, 0.1, 0.1]
            )
            self.train_datasets.append(train_dataset)
            self.validation_datasets.append(validation_dataset)
            self.test_datasets.append(test_dataset)

        return super().setup(stage)

    def _dataloader_for_dataset(self, dataset: Dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=4,
            num_workers=5,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader_for_dataset(ConcatDataset(self.train_datasets))

    def val_dataloader(self) -> EVAL_DATALOADERS:
        _dataloaders = [
            self._dataloader_for_dataset(dataset)
            for dataset in self.validation_datasets
        ]
        return _dataloaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        _dataloaders = [
            self._dataloader_for_dataset(dataset) for dataset in self.test_datasets
        ]
        return _dataloaders


if __name__ == "__main__":
    pl.seed_everything(1234, workers=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=[0, 1, 2, 3],
        precision=16,
        max_time="00:10:00:00",
        logger=DVCLiveLogger(run_name="pits_change_detection", dir="log"),
        log_every_n_steps=5,
        callbacks=[
            ModelCheckpoint(
                monitor="validation/appeared/mIoU",
                save_last=True,
                save_top_k=2,
                mode="max",
                filename="epoch={epoch:02d}-mIoU={validation/appeared/mIoU:.5f}",
                auto_insert_metric_name=False,
            ),
        ],
    )
    datamodule = PitsDataModule()
    datasets_labels = ["DE", "AS", "EB"]
    network = PitsChangeDetectionNetwork()
    model = PitsLightningModule(
        network=network,
        training_labels=[],
        validation_labels=datasets_labels,
        test_labels=datasets_labels,
    )
    trainer.fit(model=model, datamodule=datamodule)
#

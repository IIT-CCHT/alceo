from pathlib import Path
from typing import Optional, List
from alceo.dataset.change_detection import AlceoChangeDetectionDataset

from torch.utils.data import random_split, Dataset, ConcatDataset, DataLoader
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)

import pytorch_lightning as pl


class AlceoChangeDetectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_paths: List[str] = [],
        train_labels: List[str] = [],
        validation_paths: List[str] = [],
        validation_labels: List[str] = [],
        test_paths: List[str] = [],
        test_labels: List[str] = [],
        predict_paths: List[str] = [],
        predict_labels: List[str] = [],
        batch_size: int = 16,
        num_workers: int = 5,
    ) -> None:
        """Data module for pits change detection. Labels will be used for creating tags for metric logging, except for training because they will be concatenated in a single training dataset.

        Args:
            train_paths (List[str]): Paths to a list of training datasets in ALCEO change detection format. All these will be concatenated.
            train_labels (List[str]): Labels for the training datasets!
            validation_paths (List[str]): Paths to a list of validation datasets in ALCEO change detection format.
            validation_labels (List[str]): Labels for the validation datasets!
            test_paths (List[str]): Paths to a list of test datasets in ALCEO change detection format.
            test_labels (List[str]): Labels for the test datasets!
            batch_size (int, optional): Batch size given to each dataloader. Defaults to 16.
            num_workers (int, optional): Number of workers for each dataloader. Defaults to 5.
        """
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        (
            self.train_datasets,
            self.validation_datasets,
            self.test_datasets,
            self.predict_datasets,
        ) = ([], [], [], [])

        for path in self.hparams.train_paths:
            dataset = AlceoChangeDetectionDataset(Path(path))
            self.train_datasets.append(dataset)

        for path in self.hparams.validation_paths:
            dataset = AlceoChangeDetectionDataset(Path(path))
            self.validation_datasets.append(dataset)

        for path in self.hparams.test_paths:
            dataset = AlceoChangeDetectionDataset(Path(path))
            self.test_datasets.append(dataset)
            
        for path in self.hparams.predict_paths:
            dataset = AlceoChangeDetectionDataset(Path(path))
            self.predict_datasets.append(dataset)
                

        return super().setup(stage)

    def _dataloader_for_dataset(self, dataset: Dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        _dataloaders = [
            self._dataloader_for_dataset(dataset)
            for dataset in self.predict_datasets
        ]
        return _dataloaders

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

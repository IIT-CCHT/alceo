from pathlib import Path
from typing import Optional, List
from alceo.dataset.segmentation import AlceoPitsImageSegmentationDataset
from .phase_data_module import PhaseDataModule
from torch.utils.data import random_split, Dataset, ConcatDataset, DataLoader
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)

import pytorch_lightning as pl


class AlceoSegmentationDataModule(PhaseDataModule):
    def __init__(
        self,
        train_datasets: List[AlceoPitsImageSegmentationDataset] = [],
        validation_datasets: List[AlceoPitsImageSegmentationDataset] = [],
        test_datasets: List[AlceoPitsImageSegmentationDataset] = [],
        predict_datasets: List[AlceoPitsImageSegmentationDataset] = [],
        train_labels: List[str] = [],
        validation_labels: List[str] = [],
        test_labels: List[str] = [],
        predict_labels: List[str] = [],
        batch_size: int = 16,
        num_workers: int = 5,
    ) -> None:
        """Data module for pits segmentation. Labels will be used for creating tags for metric logging, except for training because they will be concatenated in a single training dataset.

        Args:
            train_paths (List[str]): Paths to a list of training datasets in ALCEO segmentation format. All these will be concatenated.
            train_labels (List[str]): Labels for the training datasets!
            validation_paths (List[str]): Paths to a list of validation datasets in ALCEO segmentation format.
            validation_labels (List[str]): Labels for the validation datasets!
            test_paths (List[str]): Paths to a list of test datasets in ALCEO segmentation format.
            test_labels (List[str]): Labels for the test datasets!
            batch_size (int, optional): Batch size given to each dataloader. Defaults to 16.
            num_workers (int, optional): Number of workers for each dataloader. Defaults to 5.
        """
        super().__init__(
            train_labels,
            validation_labels,
            test_labels,
        )
        self.save_hyperparameters()

    def _dataloader_for_dataset(self, dataset: Dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        _dataloaders = [
            self._dataloader_for_dataset(dataset)
            for dataset in self.hparams.predict_datasets
        ]
        return _dataloaders

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader_for_dataset(ConcatDataset(self.hparams.train_datasets))

    def val_dataloader(self) -> EVAL_DATALOADERS:
        _dataloaders = [
            self._dataloader_for_dataset(dataset)
            for dataset in self.hparams.validation_datasets
        ]
        return _dataloaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        _dataloaders = [
            self._dataloader_for_dataset(dataset)
            for dataset in self.hparams.test_datasets
        ]
        return _dataloaders

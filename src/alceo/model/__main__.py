from alceo.model.pits import PitsLightningModule, PitsChangeDetectionNetwork
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from dvclive.lightning import DVCLiveLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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
            batch_size=16,
            num_workers=5,
        )
        
    def single_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch["im1"] = batch["im1"].float()
        batch["im2"] = batch["im2"].float()
        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int = -1) -> Any:
        if isinstance(batch, list):
            for i in range(len(batch)):
                batch[i] = self.single_batch_transfer(batch[i], dataloader_idx)
            return batch
        else:
            batch = self.single_batch_transfer(batch, dataloader_idx)

        return super().on_after_batch_transfer(batch, dataloader_idx)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader_for_dataset(ConcatDataset(self.train_datasets))

    def val_dataloader(self) -> EVAL_DATALOADERS:
        _dataloaders = [
            self._dataloader_for_dataset(dataset)
            for dataset in self.validation_datasets
        ]
        return _dataloaders

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
    network = PitsChangeDetectionNetwork()
    model = PitsLightningModule()
    trainer.fit(model=model)
#

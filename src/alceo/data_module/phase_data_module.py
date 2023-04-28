from typing import List
import pytorch_lightning as pl


class PhaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_labels: List[str],
        validation_labels: List[str],
        test_labels: List[str],
    ) -> None:
        super().__init__()

from typing import List
import pytorch_lightning as pl


class PhaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_labels: List[str],
        validation_labels: List[str],
        test_labels: List[str],
    ) -> None:
        """A base class for all data modules compatible with ALCEO CLI. This is a 
        specialization of LightningDataModule with some arguments that are 
        automatically bound with the model's LightningModule.

        Args:
            train_labels (List[str]): The list of labels used to identify the metrics associated with a specific training Dataset.
            validation_labels (List[str]): The list of metrics labels associated with a specific validation Dataset.
            test_labels (List[str]): The list of metrics labels associated with a specific testing Dataset.
        """
        
        super().__init__()

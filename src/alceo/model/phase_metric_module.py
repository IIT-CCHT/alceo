from typing import Any, List
import pytorch_lightning as pl

class PhaseMetricModule(pl.LightningModule):
    def __init__(self, train_labels: List[str],
        validation_labels: List[str],
        test_labels: List[str], *args: Any, **kwargs: Any) -> None:
        """A base class for all models compatible with ALCEO CLI. This is a 
        specialization of LightningModule with some arguments that are 
        automatically bound with the experiment's LightningDataModule.

        Args:
            train_labels (List[str]): The list of labels used to identify the metrics associated with a specific training Dataset.
            validation_labels (List[str]): The list of metrics labels associated with a specific validation Dataset.
            test_labels (List[str]): The list of metrics labels associated with a specific testing Dataset.
        """
        super().__init__(*args, **kwargs)
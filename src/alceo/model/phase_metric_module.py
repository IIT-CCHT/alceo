from typing import Any, List
import pytorch_lightning as pl

class PhaseMetricModule(pl.LightningModule):
    def __init__(self, train_labels: List[str],
        validation_labels: List[str],
        test_labels: List[str], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
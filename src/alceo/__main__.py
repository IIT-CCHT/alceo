from typing import Any
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from alceo.callback.tiff_prediction_writer import TiffPredictionWriter
from alceo.data_module import AlceoChangeDetectionDataModule, PhaseDataModule
from alceo.logger import DVCLiveLogger
from alceo.model import (
    AlceoChangeDetectionModule,
    PhaseMetricModule,
    AlceoSegmentationModule,
)
from pytorch_lightning import Trainer
from rasterio import logging

log = logging.getLogger()
log.setLevel(logging.ERROR)

class AlceoCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data.init_args.train_labels",
            "model.init_args.train_labels",
        )
        parser.link_arguments(
            "data.init_args.validation_labels",
            "model.init_args.validation_labels",
        )
        parser.link_arguments(
            "data.init_args.test_labels",
            "model.init_args.test_labels",
        )


def main():
    cli = AlceoCLI(
        PhaseMetricModule,
        PhaseDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()

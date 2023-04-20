from typing import Any
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from alceo.callback.pits_prediction_writer import TiffPredictionWriter
from alceo.data_module import AlceoChangeDetectionDataModule
from alceo.logger import DVCLiveLogger
from alceo.model import AlceoMetricModule
from pytorch_lightning import Trainer

class AlceoCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data.train_labels",
            "model.train_labels",
        )
        parser.link_arguments(
            "data.validation_labels",
            "model.validation_labels",
        )
        parser.link_arguments(
            "data.test_labels",
            "model.test_labels",
        )


def main():
    cli = AlceoCLI(
        model_class=AlceoMetricModule,
        datamodule_class=AlceoChangeDetectionDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()

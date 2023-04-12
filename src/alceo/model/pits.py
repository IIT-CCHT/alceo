# %%
from segmentation_models_pytorch import Unet
import torch


class PitsChangeDetectionNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.siamese_network = Unet(
            encoder_name="resnet18",
            in_channels=4,
            encoder_depth=3,
            decoder_channels=(256, 128, 64),
        )
        self.change_network = Unet(
            encoder_name="resnet18",
            encoder_depth=3,
            decoder_channels=(256, 128, 64),
            in_channels=128,
            classes=2,
        )

    def forward(
        self, im1, im2
    ) -> torch.Tensor:
        # Siamese feature extraction
        features1 = self.siamese_network.decoder(*self.siamese_network.encoder(im1))
        features2 = self.siamese_network.decoder(*self.siamese_network.encoder(im2))
        features = torch.cat((features1, features2), dim=1)
        change_pred = self.change_network(features)

        return change_pred
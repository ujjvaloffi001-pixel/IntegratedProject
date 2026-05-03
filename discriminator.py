from torch import nn


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for real/fake restored artwork classification."""

    def __init__(self, input_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int, normalize: bool = True) -> nn.Sequential:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(input_channels, base_channels, normalize=False),
            block(base_channels, base_channels * 2),
            block(base_channels * 2, base_channels * 4),
            block(base_channels * 4, base_channels * 8),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, padding=1),
        )

    def forward(self, image):
        return self.model(image)

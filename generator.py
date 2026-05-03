import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _build_resnet18(pretrained: bool) -> nn.Module:
    if not pretrained:
        return resnet18(weights=None)
    try:
        return resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as exc:
        print(f"Warning: pretrained ResNet-18 weights unavailable ({exc}). Using random initialization.")
        return resnet18(weights=None)


class DecoderBlock(nn.Module):
    """Upsample, fuse with an encoder skip feature, then refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))


class ResNetUNetGenerator(nn.Module):
    """
    Generator with a pretrained ResNet-18 encoder and U-Net decoder.

    Input and output tensors are expected in [-1, 1].
    """

    def __init__(self, pretrained: bool = True, freeze_encoder: bool = False) -> None:
        super().__init__()
        backbone = _build_resnet18(pretrained)

        self.input_block = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh(),
        )

        if freeze_encoder:
            for module in [self.input_block, self.encoder1, self.encoder2, self.encoder3, self.encoder4]:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.input_block(x)       # H/2
        pooled = self.maxpool(skip0)      # H/4
        skip1 = self.encoder1(pooled)     # H/4
        skip2 = self.encoder2(skip1)      # H/8
        skip3 = self.encoder3(skip2)      # H/16
        encoded = self.encoder4(skip3)    # H/32

        x = self.center(encoded)
        x = self.decoder4(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder2(x, skip1)
        x = self.decoder1(x, skip0)
        return self.final(x)

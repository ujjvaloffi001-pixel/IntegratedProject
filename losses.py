import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


class PerceptualLoss(nn.Module):
    """VGG feature loss for sharper restorations."""

    def __init__(self, layer_limit: int = 16) -> None:
        super().__init__()
        try:
            self.features = vgg16(weights=VGG16_Weights.DEFAULT).features[:layer_limit].eval()
        except Exception as exc:
            print(f"Warning: pretrained VGG-16 weights unavailable ({exc}). Perceptual loss uses random features.")
            self.features = vgg16(weights=None).features[:layer_limit].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = (prediction + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        self.features = self.features.to(prediction.device)
        return self.criterion(self.features(prediction), self.features(target))

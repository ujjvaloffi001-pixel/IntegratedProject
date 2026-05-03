import numpy as np
import torch
from skimage.metrics import structural_similarity


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().clamp(-1, 1)
    image = (image + 1.0) / 2.0
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((prediction.detach() - target.detach()) ** 2).item()


def psnr(prediction: torch.Tensor, target: torch.Tensor, max_value: float = 2.0) -> float:
    error = torch.mean((prediction.detach() - target.detach()) ** 2).item()
    if error == 0:
        return float("inf")
    return 20 * np.log10(max_value / np.sqrt(error))


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.ndim == 4:
        scores = [ssim(prediction[i], target[i]) for i in range(prediction.shape[0])]
        return float(np.mean(scores))

    pred_image = _to_numpy_image(prediction)
    target_image = _to_numpy_image(target)
    return float(structural_similarity(target_image, pred_image, channel_axis=2, data_range=1.0))

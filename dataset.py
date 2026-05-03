from pathlib import Path
import random
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.degradation import ArtworkDegrader


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ArtworkRestorationDataset(Dataset):
    """
    Dataset for paired or synthetic artwork restoration.

    Expected modes:
    - clean_dir only: degraded images are generated on the fly.
    - clean_dir + degraded_dir: file names are matched between folders.
    """

    def __init__(
        self,
        clean_dir: str,
        degraded_dir: str | None = None,
        image_size: int = 256,
        augment: bool = True,
        degrader: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.clean_dir = Path(clean_dir)
        self.degraded_dir = Path(degraded_dir) if degraded_dir else None
        self.clean_paths = sorted(path for path in self.clean_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.clean_paths:
            raise FileNotFoundError(f"No images found in {self.clean_dir}")

        self.degrader = degrader or ArtworkDegrader()
        self.resize = transforms.Resize((image_size, image_size), antialias=True)
        self.augment = augment
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, index: int) -> dict:
        clean_path = self.clean_paths[index]
        clean = Image.open(clean_path).convert("RGB")

        if self.degraded_dir:
            degraded_path = self.degraded_dir / clean_path.relative_to(self.clean_dir)
            if not degraded_path.exists():
                raise FileNotFoundError(f"Missing paired degraded image: {degraded_path}")
            degraded = Image.open(degraded_path).convert("RGB")
        else:
            degraded = self.degrader(clean)

        clean = self.resize(clean)
        degraded = self.resize(degraded)

        if self.augment and random.random() < 0.5:
            clean = transforms.functional.hflip(clean)
            degraded = transforms.functional.hflip(degraded)

        return {
            "degraded": self.to_tensor(degraded),
            "clean": self.to_tensor(clean),
            "filename": clean_path.name,
        }


def get_dataloader(
    clean_dir: str,
    degraded_dir: str | None = None,
    image_size: int = 256,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    augment: bool = True,
) -> DataLoader:
    dataset = ArtworkRestorationDataset(
        clean_dir=clean_dir,
        degraded_dir=degraded_dir,
        image_size=image_size,
        augment=augment,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

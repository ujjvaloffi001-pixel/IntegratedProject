import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.generator import ResNetUNetGenerator
from utils.dataset import IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore degraded artwork images with a trained generator.")
    parser.add_argument("--input_dir", type=str, default="dataset/degraded")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def load_generator(checkpoint_path: str, device: torch.device, pretrained: bool) -> ResNetUNetGenerator:
    generator = ResNetUNetGenerator(pretrained=pretrained).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["generator"] if "generator" in checkpoint else checkpoint
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    generator = load_generator(args.checkpoint, device, pretrained=not args.no_pretrained)
    image_paths = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No degraded images found in {input_dir}")

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Restoring"):
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            restored = generator(tensor).squeeze(0).cpu()
            save_image((restored + 1.0) / 2.0, output_dir / f"{image_path.stem}_restored.png")

    print(f"Saved restored images to {output_dir}")


if __name__ == "__main__":
    main()

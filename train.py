import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from models.discriminator import PatchDiscriminator
from models.generator import ResNetUNetGenerator
from utils.dataset import get_dataloader
from utils.losses import PerceptualLoss
from utils.metrics import mse, psnr, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-GAN artwork restoration model.")
    parser.add_argument("--clean_dir", type=str, default="dataset/clean", help="Folder with clean target artworks.")
    parser.add_argument("--degraded_dir", type=str, default=None, help="Optional paired degraded image folder.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_recon", type=float, default=100.0)
    parser.add_argument("--lambda_perceptual", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    return parser.parse_args()


def save_checkpoint(path: Path, generator, discriminator, optimizer_g, optimizer_d, epoch: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader(
        clean_dir=args.clean_dir,
        degraded_dir=args.degraded_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    generator = ResNetUNetGenerator(pretrained=not args.no_pretrained, freeze_encoder=args.freeze_encoder).to(device)
    discriminator = PatchDiscriminator().to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device) if args.lambda_perceptual > 0 else None

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint["epoch"] + 1

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_stats = {"g_loss": 0.0, "d_loss": 0.0, "psnr": 0.0, "ssim": 0.0, "mse": 0.0}

        for batch in progress:
            degraded = batch["degraded"].to(device)
            clean = batch["clean"].to(device)

            restored = generator(degraded)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(clean)
            fake_logits = discriminator(restored.detach())
            real_labels = torch.ones_like(real_logits, device=device)
            fake_labels = torch.zeros_like(fake_logits, device=device)
            d_loss = 0.5 * (
                adversarial_loss(real_logits, real_labels) + adversarial_loss(fake_logits, fake_labels)
            )
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            fake_logits = discriminator(restored)
            gan_loss = adversarial_loss(fake_logits, torch.ones_like(fake_logits, device=device))
            recon_loss = reconstruction_loss(restored, clean)
            g_loss = gan_loss + args.lambda_recon * recon_loss
            if perceptual_loss:
                g_loss = g_loss + args.lambda_perceptual * perceptual_loss(restored, clean)
            g_loss.backward()
            optimizer_g.step()

            batch_stats = {
                "g_loss": g_loss.item(),
                "d_loss": d_loss.item(),
                "psnr": psnr(restored, clean),
                "ssim": ssim(restored, clean),
                "mse": mse(restored, clean),
            }
            for key, value in batch_stats.items():
                epoch_stats[key] += value
            progress.set_postfix({key: f"{value:.4f}" for key, value in batch_stats.items() if key in ["g_loss", "d_loss", "psnr"]})

        epoch_stats = {key: value / len(dataloader) for key, value in epoch_stats.items()}
        epoch_stats["epoch"] = epoch
        history.append(epoch_stats)

        save_checkpoint(Path(args.checkpoint_dir) / "latest.pt", generator, discriminator, optimizer_g, optimizer_d, epoch, args)
        if epoch % 5 == 0 or epoch == args.epochs:
            save_checkpoint(Path(args.checkpoint_dir) / f"epoch_{epoch:03d}.pt", generator, discriminator, optimizer_g, optimizer_d, epoch, args)

        with open(Path(args.log_dir) / "training_history.json", "w", encoding="utf-8") as log_file:
            json.dump(history, log_file, indent=2)

    print(f"Training complete. Latest checkpoint: {Path(args.checkpoint_dir) / 'latest.pt'}")


if __name__ == "__main__":
    main()

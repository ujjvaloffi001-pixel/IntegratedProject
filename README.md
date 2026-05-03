# Deep Learning Based Digital Artwork Restoration using ResNet-GAN

This repository contains a complete PyTorch pipeline for restoring degraded digital artwork images. The model uses a pretrained ResNet encoder, a U-Net style decoder with skip connections, and a CNN discriminator trained with adversarial and reconstruction losses.

## Problem Statement

Old or poorly digitized artworks often contain noise, blur, cracks, scratches, color loss, and compression artifacts. The goal of this project is to learn a mapping from degraded artwork images to visually cleaner, high-quality restored images.

## Architecture

- **Generator:** ResNet-18 encoder initialized with ImageNet pretrained weights, followed by a U-Net decoder. Encoder features are reused through skip connections to preserve spatial structure.
- **Discriminator:** PatchGAN-style CNN binary classifier that judges whether image patches are real clean artwork or generated restored artwork.
- **Losses:** Binary cross entropy GAN loss, L1 reconstruction loss, and optional VGG perceptual loss.
- **Degradation module:** Synthetic image corruption with Gaussian noise, blur, scratches/cracks, and JPEG compression artifacts.

## Folder Structure

```text
project_root/
├── app/
│   ├── __init__.py
│   └── app.py
├── checkpoints/
│   └── .gitkeep
├── dataset/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── models/
│   ├── __init__.py
│   ├── discriminator.py
│   └── generator.py
├── results/
│   └── .gitkeep
├── testing/
│   ├── __init__.py
│   └── test.py
├── training/
│   ├── __init__.py
│   └── train.py
├── utils/
│   ├── __init__.py
│   ├── dataset.py
│   ├── degradation.py
│   ├── losses.py
│   └── metrics.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.sh
```

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
bash setup.sh
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir dataset\clean, dataset\degraded, results, checkpoints, logs
```

## Dataset Preparation

For synthetic degradation training, place clean artwork images in:

```text
dataset/clean/
```

For paired training, place clean and degraded images with matching relative names:

```text
dataset/clean/image_001.png
dataset/degraded/image_001.png
```

## Training

Train with synthetic degradation:

```bash
python -m training.train --clean_dir dataset/clean --epochs 50 --batch_size 4
```

Train with paired degraded images:

```bash
python -m training.train --clean_dir dataset/clean --degraded_dir dataset/degraded --epochs 50
```

Useful hyperparameters:

```bash
python -m training.train \
  --image_size 256 \
  --batch_size 8 \
  --lr 0.0002 \
  --lambda_recon 100 \
  --lambda_perceptual 0.05 \
  --freeze_encoder
```

Checkpoints are saved to `checkpoints/latest.pt` and periodic epoch files.

## Testing

Restore images from `dataset/degraded`:

```bash
python -m testing.test --checkpoint checkpoints/latest.pt --input_dir dataset/degraded --output_dir results
```

The restored images are written to `results/`.

## Streamlit Demo

```bash
streamlit run app/app.py
```

Upload an image, optionally simulate degradation, and compare the input with the restored output.

## Metrics

- **MSE:** Pixel-level mean squared error. Lower is better.
- **PSNR:** Peak signal-to-noise ratio. Higher is better.
- **SSIM:** Structural similarity score. Higher is better and usually aligns better with visual quality than MSE alone.

## Expected Results

After training on a meaningful artwork dataset, the generator should reduce visible scratches, noise, and compression artifacts while preserving the composition and color structure. Early checkpoints may look blurry; sharper results usually require more data, longer training, and perceptual loss tuning.

## Future Improvements

- Add color constancy and histogram matching.
- Train with larger ResNet backbones or Swin Transformer encoders.
- Add mixed precision training for faster GPU runs.
- Add experiment tracking with TensorBoard or Weights & Biases.
- Export the model to ONNX for deployment.

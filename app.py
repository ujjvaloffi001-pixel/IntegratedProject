from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from models.generator import ResNetUNetGenerator
from utils.degradation import ArtworkDegrader


st.set_page_config(page_title="Artwork Restoration", layout="wide")


@st.cache_resource
def load_model(checkpoint_path: str, use_pretrained_encoder: bool) -> tuple[ResNetUNetGenerator, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetUNetGenerator(pretrained=use_pretrained_encoder).to(device)
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["generator"] if "generator" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    model.eval()
    return model, device


def restore_image(model: ResNetUNetGenerator, device: torch.device, image: Image.Image, image_size: int) -> Image.Image:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    with torch.no_grad():
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        restored = model(tensor).squeeze(0).cpu().clamp(-1, 1)
        restored = (restored + 1.0) / 2.0
    return transforms.ToPILImage()(restored)


st.title("Deep Learning Based Digital Artwork Restoration")

checkpoint_path = st.sidebar.text_input("Checkpoint path", "checkpoints/latest.pt")
image_size = st.sidebar.slider("Image size", min_value=128, max_value=512, value=256, step=64)
use_pretrained = st.sidebar.checkbox("Use pretrained ResNet encoder", value=True)
simulate_degradation = st.sidebar.checkbox("Simulate degradation on upload", value=False)

uploaded_file = st.file_uploader("Upload degraded artwork", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    degraded_image = ArtworkDegrader()(original_image) if simulate_degradation else original_image
    model, device = load_model(checkpoint_path, use_pretrained)
    restored_image = restore_image(model, device, degraded_image, image_size)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(degraded_image, use_container_width=True)
    with col2:
        st.subheader("Restored")
        st.image(restored_image, use_container_width=True)
else:
    st.info("Upload an artwork image to run restoration.")

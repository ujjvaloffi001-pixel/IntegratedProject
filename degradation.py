import random
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class DegradationConfig:
    noise_std_range: tuple[float, float] = (4.0, 24.0)
    blur_radius_choices: tuple[int, ...] = (0, 3, 5)
    jpeg_quality_range: tuple[int, int] = (28, 75)
    scratch_count_range: tuple[int, int] = (2, 8)
    scratch_alpha_range: tuple[int, int] = (45, 130)


class ArtworkDegrader:
    """Creates synthetic degradation: noise, blur, scratches/cracks, and JPEG artifacts."""

    def __init__(self, config: DegradationConfig | None = None, apply_probability: float = 1.0) -> None:
        self.config = config or DegradationConfig()
        self.apply_probability = apply_probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.apply_probability:
            return image.copy()

        degraded = image.convert("RGB")
        operations = [self.add_gaussian_noise, self.add_blur, self.add_scratches, self.add_jpeg_artifacts]
        random.shuffle(operations)
        for operation in operations:
            if random.random() < 0.85:
                degraded = operation(degraded)
        return degraded

    def add_gaussian_noise(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image).astype(np.float32)
        std = random.uniform(*self.config.noise_std_range)
        noise = np.random.normal(0, std, array.shape)
        noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def add_blur(self, image: Image.Image) -> Image.Image:
        kernel = random.choice(self.config.blur_radius_choices)
        if kernel <= 1:
            return image
        array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(array, (kernel, kernel), 0)
        return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))

    def add_scratches(self, image: Image.Image) -> Image.Image:
        scratched = image.copy()
        overlay = Image.new("RGBA", scratched.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = scratched.size
        scratch_count = random.randint(*self.config.scratch_count_range)

        for _ in range(scratch_count):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = int(np.clip(x1 + random.gauss(0, width * 0.25), 0, width))
            y2 = int(np.clip(y1 + random.gauss(0, height * 0.25), 0, height))
            alpha = random.randint(*self.config.scratch_alpha_range)
            color = random.choice([(235, 235, 235, alpha), (35, 28, 20, alpha)])
            draw.line((x1, y1, x2, y2), fill=color, width=random.randint(1, 3))

        return Image.alpha_composite(scratched.convert("RGBA"), overlay).convert("RGB")

    def add_jpeg_artifacts(self, image: Image.Image) -> Image.Image:
        buffer = BytesIO()
        quality = random.randint(*self.config.jpeg_quality_range)
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

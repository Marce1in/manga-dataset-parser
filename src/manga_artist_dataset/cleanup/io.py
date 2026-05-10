"""Image loading and saving boundaries for cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
from PIL import Image, UnidentifiedImageError

from manga_artist_dataset.cleanup.models import UInt8Image
from manga_artist_dataset.errors import ImageProcessingError


def load_image_cv(path: Path) -> UInt8Image:
    """Load an image as an OpenCV BGR array.

    Example:
        `image_cv = load_image_cv(Path("page.png"))`.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageProcessingError(f"Could not read image at {path}; expected a valid raster image path.")
    return cast(UInt8Image, image)


def load_image_pil(path: Path) -> Image.Image:
    """Load an image as a Pillow RGB image.

    Example:
        `image_pil = load_image_pil(Path("page.png"))`.
    """
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as err:
        raise ImageProcessingError(f"Could not read image at {path}; expected a Pillow-supported image.") from err


def save_image_pil(image: Image.Image, path: Path) -> None:
    """Save a Pillow image, creating parent directories as needed.

    Example:
        `save_image_pil(image, Path("cleaned/page.png"))`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.save(path)
    except OSError as err:
        raise ImageProcessingError(f"Could not save image to {path}; expected a writable output path.") from err

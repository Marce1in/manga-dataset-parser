"""Final image sizing for cleaned manga pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from manga_artist_dataset.json_types import JsonObject


@dataclass(frozen=True)
class ImageTargetSize:
    """Final canvas dimensions for model-ready images.

    Example:
        `ImageTargetSize(width=512, height=768)`.
    """

    width: int
    height: int


DEFAULT_TARGET_SIZE = ImageTargetSize(width=512, height=768)
WHITE_RGB = (255, 255, 255)


def standardize_image_file(source_path: Path, destination: Path, target_size: ImageTargetSize) -> None:
    """Resize one image into a fixed white canvas without cropping.

    Example:
        `standardize_image_file(Path("in.png"), Path("out.png"), DEFAULT_TARGET_SIZE)`.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        canvas = standardize_image(image, target_size)
        canvas.save(destination, format="PNG", compress_level=6)


def standardize_image(image: Image.Image, target_size: ImageTargetSize) -> Image.Image:
    """Resize an image into a fixed white canvas while preserving aspect ratio.

    Example:
        `standardize_image(image, ImageTargetSize(width=512, height=768))`.
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    contained = ImageOps.contain(rgb, canvas_tuple(target_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", canvas_tuple(target_size), WHITE_RGB)
    canvas.paste(contained, centered_offset(contained, target_size))
    return canvas


def canvas_tuple(target_size: ImageTargetSize) -> tuple[int, int]:
    """Return Pillow's `(width, height)` size tuple.

    Example:
        `canvas_tuple(ImageTargetSize(width=512, height=768)) == (512, 768)`.
    """
    return target_size.width, target_size.height


def centered_offset(image: Image.Image, target_size: ImageTargetSize) -> tuple[int, int]:
    """Return the paste offset that centers an image in a target canvas.

    Example:
        `centered_offset(image, ImageTargetSize(width=512, height=768))`.
    """
    left = (target_size.width - image.width) // 2
    top = (target_size.height - image.height) // 2
    return left, top


def standardization_record(target_size: ImageTargetSize) -> JsonObject:
    """Render standardization settings for dataset metadata.

    Example:
        `standardization_record(DEFAULT_TARGET_SIZE)["width"] == 512`.
    """
    return {
        "width": target_size.width,
        "height": target_size.height,
        "format": "PNG",
        "resize": "contain",
        "padding": "white",
        "resampling": "lanczos",
    }

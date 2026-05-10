from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from manga_artist_dataset.cleanup.bubble_masks import extract_bubble_mask
from manga_artist_dataset.cleanup.config import CleanupConfig


def test_extract_bubble_mask_returns_full_size_binary_mask() -> None:
    image_cv = synthetic_bubble_cv()

    mask = extract_bubble_mask(image_cv, (10, 15, 90, 65), CleanupConfig())

    assert mask.shape == image_cv.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_extract_bubble_mask_tiny_bbox_returns_rectangle() -> None:
    image_cv = np.zeros((20, 20, 3), dtype=np.uint8)

    mask = extract_bubble_mask(image_cv, (2, 3, 7, 8), CleanupConfig())

    assert int(np.count_nonzero(mask)) == 25
    assert mask[3, 2] == 255


def test_extract_bubble_mask_invalid_region_does_not_crash() -> None:
    image_cv = np.zeros((20, 20, 3), dtype=np.uint8)

    mask = extract_bubble_mask(image_cv, (-10, -10, -1, -1), CleanupConfig())

    assert mask.shape == (20, 20)
    assert int(np.count_nonzero(mask)) == 0


def test_extract_bubble_mask_finds_synthetic_white_bubble() -> None:
    image_cv = synthetic_bubble_cv()

    mask = extract_bubble_mask(image_cv, (10, 15, 90, 65), CleanupConfig())

    assert mask[40, 50] == 255
    assert int(np.count_nonzero(mask)) > 2000


def synthetic_bubble_cv() -> np.ndarray:
    image = Image.new("RGB", (100, 80), (40, 40, 40))
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 15, 90, 65), fill=(255, 255, 255), outline=(0, 0, 0))
    return np.array(image)[:, :, ::-1].copy()

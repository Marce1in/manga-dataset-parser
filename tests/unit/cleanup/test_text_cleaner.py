from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.text_cleaner import clear_text_strokes


def test_clear_text_strokes_clears_dark_pixels_inside_mask() -> None:
    image = bubble_image(fill=(255, 255, 255))
    mask = rectangle_mask()

    clear_text_strokes(image, (10, 10, 90, 70), CleanupConfig(), mask=mask)

    assert image.getpixel((45, 38)) == (255, 255, 255)


def test_clear_text_strokes_keeps_pixels_outside_mask() -> None:
    image = bubble_image(fill=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 16, 16), fill=(0, 0, 0))
    mask = rectangle_mask()

    clear_text_strokes(image, (10, 10, 90, 70), CleanupConfig(), mask=mask)

    assert image.getpixel((13, 13)) == (0, 0, 0)


def test_clear_text_strokes_samples_gray_bubble_background() -> None:
    image = bubble_image(fill=(220, 220, 220))
    mask = rectangle_mask()

    clear_text_strokes(image, (10, 10, 90, 70), CleanupConfig(), mask=mask)

    pixel = image.getpixel((45, 38))
    assert isinstance(pixel, tuple)
    red, green, blue = pixel[:3]
    assert 215 <= red <= 225
    assert 215 <= green <= 225
    assert 215 <= blue <= 225


def test_clear_text_strokes_removes_light_gray_antialiased_text() -> None:
    image = bubble_image(fill=(255, 255, 255), text=(178, 178, 178))
    mask = rectangle_mask()
    config = CleanupConfig(dark_text_threshold=190)

    clear_text_strokes(image, (10, 10, 90, 70), config, mask=mask)

    assert image.getpixel((45, 38)) == (255, 255, 255)


def test_clear_text_strokes_leaves_empty_region_unchanged() -> None:
    image = Image.new("RGB", (100, 80), (255, 255, 255))
    before = image.tobytes()

    clear_text_strokes(image, (20, 20, 80, 60), CleanupConfig(), mask=rectangle_mask())

    assert image.tobytes() == before


def bubble_image(fill: tuple[int, int, int], text: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    image = Image.new("RGB", (100, 80), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 80, 60), fill=fill)
    draw.rectangle((40, 34, 60, 42), fill=text)
    return image


def rectangle_mask() -> np.ndarray:
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:61, 20:81] = 255
    return mask

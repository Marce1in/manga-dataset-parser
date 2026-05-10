"""Image inspection and page filtering helpers."""

from __future__ import annotations

import importlib
import struct
from io import BytesIO
from typing import Any, cast

from manga_artist_dataset.errors import DatasetError
from manga_artist_dataset.models import BuildConfig, PageOutput


def image_dimensions(content: bytes) -> tuple[int | None, int | None]:
    """Read image dimensions from common formats.

    Example:
        `width, height = image_dimensions(page_bytes)`.
    """
    if content.startswith(b"\x89PNG\r\n\x1a\n") and len(content) >= 24:
        return struct.unpack(">II", content[16:24])
    if content.startswith((b"GIF87a", b"GIF89a")) and len(content) >= 10:
        return struct.unpack("<HH", content[6:10])
    if content.startswith(b"BM") and len(content) >= 26:
        width, height = struct.unpack("<ii", content[18:26])
        return abs(width), abs(height)
    if content.startswith(b"\xff\xd8"):
        return jpeg_dimensions(content)
    return pillow_dimensions(content)


def pillow_dimensions(content: bytes) -> tuple[int | None, int | None]:
    try:
        image_module = importlib.import_module("PIL.Image")
        with image_module.open(BytesIO(content)) as image:
            return cast(tuple[int, int], image.size)
    except (ImportError, OSError):
        return None, None


def jpeg_dimensions(content: bytes) -> tuple[int | None, int | None]:
    """Read JPEG dimensions without decoding the full image.

    Example:
        `jpeg_dimensions(jpeg_bytes)`.
    """
    index = 2
    while index < len(content) - 9:
        result = jpeg_segment_dimensions(content, index)
        if result is not None:
            return result
        index = next_jpeg_segment_index(content, index)
    return None, None


def jpeg_segment_dimensions(content: bytes, index: int) -> tuple[int, int] | None:
    if content[index] != 0xFF:
        return None
    marker = content[index + 1]
    if marker not in jpeg_size_markers():
        return None
    segment_start = index + 2
    if segment_start + 7 > len(content):
        return None
    height, width = struct.unpack(">HH", content[segment_start + 3 : segment_start + 7])
    return width, height


def next_jpeg_segment_index(content: bytes, index: int) -> int:
    if content[index] != 0xFF:
        return index + 1
    marker = content[index + 1]
    index += 2
    if marker in {0xD8, 0xD9} or index + 2 > len(content):
        return index
    segment_length = int(struct.unpack(">H", content[index : index + 2])[0])
    return index + max(segment_length, 2)


def jpeg_size_markers() -> set[int]:
    return {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def is_probably_color_page(content: bytes) -> bool:
    """Detect pages that are likely full-color or colorized.

    Example:
        `is_probably_color_page(page_bytes)`.
    """
    image_module = require_pillow("Install Pillow to use color page filtering.")
    with image_module.open(BytesIO(content)) as image:
        image = image.convert("RGB")
        image.thumbnail((96, 96))
        raw_pixels = image.tobytes()
    return has_color_signal(raw_pixels)


def require_pillow(message: str) -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise DatasetError(message) from exc


def has_color_signal(raw_pixels: bytes) -> bool:
    if not raw_pixels:
        return False
    deltas, saturated = channel_deltas(raw_pixels)
    pixel_count = len(raw_pixels) // 3
    return (sum(deltas) / pixel_count) > 10 and (saturated / pixel_count) > 0.015


def channel_deltas(raw_pixels: bytes) -> tuple[list[int], int]:
    deltas: list[int] = []
    saturated = 0
    for index in range(0, len(raw_pixels), 3):
        red, green, blue = raw_pixels[index], raw_pixels[index + 1], raw_pixels[index + 2]
        delta = max(red, green, blue) - min(red, green, blue)
        deltas.append(delta)
        saturated += int(delta > 24)
    return deltas, saturated


def page_rejection_reason(content: bytes, config: BuildConfig) -> str | None:
    """Return why a page should be skipped, if any.

    Example:
        `page_rejection_reason(page_bytes, config)`.
    """
    width, height = image_dimensions(content)
    if is_filtered_double_spread(width, height, config):
        return "double_spread"
    if config.filter_color_pages and is_probably_color_page(content):
        return "color"
    return None


def is_filtered_double_spread(width: int | None, height: int | None, config: BuildConfig) -> bool:
    if not config.filter_double_spreads or config.split_double_spreads:
        return False
    return width is not None and height is not None and width > height * 1.15


def split_double_spread_bytes(content: bytes) -> list[PageOutput]:
    """Split a wide page into left and right PNG crops.

    Example:
        `split_double_spread_bytes(page_bytes)`.
    """
    image_module = require_pillow("Install Pillow to split double spreads.")
    with image_module.open(BytesIO(content)) as image:
        image = image.convert("RGB")
        width, height = image.size
        return split_image_halves(image, width, height)


def split_image_halves(image: Any, width: int, height: int) -> list[PageOutput]:
    midpoint = width // 2
    boxes = [("left", (0, 0, midpoint, height)), ("right", (midpoint, 0, width, height))]
    return [png_crop_output(image, name, box) for name, box in boxes]


def png_crop_output(image: Any, name: str, box: tuple[int, int, int, int]) -> PageOutput:
    buffer = BytesIO()
    image.crop(box).save(buffer, format="PNG")
    return PageOutput(buffer.getvalue(), name, ".png")


def accepted_page_outputs(
    content: bytes,
    config: BuildConfig,
    allow_split: bool = False,
) -> tuple[list[PageOutput], list[str], bool]:
    """Return accepted output bytes and rejection reasons for one page.

    Example:
        `outputs, rejected, did_split = accepted_page_outputs(page_bytes, config)`.
    """
    width, height = image_dimensions(content)
    if is_wide_page(width, height) and config.filter_double_spreads:
        return accepted_wide_page_outputs(content, config, allow_split)
    reason = page_rejection_reason(content, config)
    if reason is not None:
        return [], [reason], False
    return [PageOutput(content, None, None)], [], False


def is_wide_page(width: int | None, height: int | None) -> bool:
    return width is not None and height is not None and width > height * 1.15


def accepted_wide_page_outputs(
    content: bytes,
    config: BuildConfig,
    allow_split: bool,
) -> tuple[list[PageOutput], list[str], bool]:
    if not (config.split_double_spreads or allow_split):
        return [], ["double_spread"], False
    outputs, rejected = accepted_split_outputs(content, config)
    return outputs, rejected, True


def accepted_split_outputs(content: bytes, config: BuildConfig) -> tuple[list[PageOutput], list[str]]:
    outputs: list[PageOutput] = []
    rejected: list[str] = []
    for output in split_double_spread_bytes(content):
        if config.filter_color_pages and is_probably_color_page(output.content):
            rejected.append("color")
            continue
        outputs.append(output)
    return outputs, rejected

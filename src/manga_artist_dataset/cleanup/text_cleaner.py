"""Mask-aware speech bubble text cleanup."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.models import Bbox, UInt8Image

RgbColor = tuple[int, int, int]


def clear_text_strokes(
    image_pil: Image.Image,
    bbox: Bbox,
    config: CleanupConfig,
    mask: UInt8Image | None = None,
) -> None:
    """Clear dark text strokes inside one speech bubble bbox in place.

    Example:
        `clear_text_strokes(image, (20, 30, 120, 90), CleanupConfig(), mask=mask)`.
    """
    clipped = _clip_bbox_to_image(bbox, image_pil.size)
    if clipped is None:
        return
    image_array = np.array(image_pil.convert("RGB"))
    dark_mask = _dark_text_mask(image_array, clipped, config.dark_text_threshold)
    if mask is not None:
        dark_mask = _masked_dark_pixels(dark_mask, mask, clipped)
    clear_bbox = _clear_bbox_from_dark_pixels(dark_mask, clipped, config)
    if clear_bbox is None:
        return
    fill_color = _sample_fill_color(image_array, mask, clipped)
    _paint_clear_bbox(image_pil, image_array, clear_bbox, fill_color, mask)


def _clip_bbox_to_image(bbox: Bbox, image_size: tuple[int, int]) -> Bbox | None:
    width, height = image_size
    x1, x2 = sorted((bbox[0], bbox[2]))
    y1, y2 = sorted((bbox[1], bbox[3]))
    clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    return None if clipped[2] <= clipped[0] or clipped[3] <= clipped[1] else clipped


def _dark_text_mask(image_array: UInt8Image, bbox: Bbox, threshold: int) -> UInt8Image:
    x1, y1, x2, y2 = bbox
    roi = image_array[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    return (gray < threshold).astype(np.uint8) * 255


def _masked_dark_pixels(dark_mask: UInt8Image, mask: UInt8Image, bbox: Bbox) -> UInt8Image:
    mask_crop = _mask_crop(mask, bbox)
    if mask_crop is None or mask_crop.shape != dark_mask.shape:
        return np.zeros_like(dark_mask)
    return np.where(mask_crop > 0, dark_mask, 0).astype(np.uint8)


def _clear_bbox_from_dark_pixels(dark_mask: UInt8Image, bbox: Bbox, config: CleanupConfig) -> Bbox | None:
    connected = _connected_text_pixels(dark_mask, config)
    coords = cv2.findNonZero(connected)
    if coords is None:
        return None
    rx, ry, rw, rh = cv2.boundingRect(coords)
    return _expanded_roi_bbox(bbox, dark_mask.shape, (rx, ry, rw, rh), config.text_clear_margin)


def _connected_text_pixels(dark_mask: UInt8Image, config: CleanupConfig) -> UInt8Image:
    kernel_size = _odd_kernel_size(config.text_dilation_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    connected = cast(UInt8Image, cv2.dilate(dark_mask, kernel, iterations=config.text_dilation_iterations))
    return _without_edge_components(connected)


def _odd_kernel_size(value: int) -> int:
    if value < 1:
        raise ValueError(f"text_dilation_kernel_size must be >= 1; got {value}.")
    return value if value % 2 == 1 else value + 1


def _without_edge_components(connected: UInt8Image) -> UInt8Image:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)
    labels_array = cast(NDArray[np.int32], labels)
    stats_array = cast(NDArray[np.int32], stats)
    filtered = np.zeros_like(connected)
    for label in range(1, count):
        if _is_interior_component(labels_array, stats_array, label):
            filtered[labels_array == label] = 255
    return filtered if np.any(filtered) else connected


def _is_interior_component(labels: NDArray[np.int32], stats: NDArray[np.int32], label: int) -> bool:
    if int(stats[label, cv2.CC_STAT_AREA]) < 3:
        return False
    component = labels == label
    touches_edge = bool(
        component[0, :].any() or component[-1, :].any() or component[:, 0].any() or component[:, -1].any()
    )
    return not touches_edge


def _expanded_roi_bbox(bbox: Bbox, roi_shape: tuple[int, ...], rect: tuple[int, int, int, int], margin: int) -> Bbox:
    x1, y1, _x2, _y2 = bbox
    rx, ry, rw, rh = rect
    return (
        x1 + max(0, rx - margin),
        y1 + max(0, ry - margin),
        x1 + min(roi_shape[1], rx + rw + margin),
        y1 + min(roi_shape[0], ry + rh + margin),
    )


def _sample_fill_color(image_array: UInt8Image, mask: UInt8Image | None, bbox: Bbox) -> RgbColor:
    x1, y1, x2, y2 = bbox
    roi = image_array[y1:y2, x1:x2]
    candidates = _masked_fill_candidates(roi, mask, bbox) if mask is not None else roi.reshape(-1, 3)
    if candidates.size == 0:
        return (255, 255, 255)
    bright = candidates[np.mean(candidates, axis=1) > 180]
    source = bright if bright.size else candidates
    median = np.median(source, axis=0)
    return (int(median[0]), int(median[1]), int(median[2]))


def _masked_fill_candidates(roi: UInt8Image, mask: UInt8Image, bbox: Bbox) -> UInt8Image:
    mask_crop = _mask_crop(mask, bbox)
    if mask_crop is None or mask_crop.shape != roi.shape[:2]:
        return np.empty((0, 3), dtype=np.uint8)
    return roi[mask_crop > 0]


def _mask_crop(mask: UInt8Image, bbox: Bbox) -> UInt8Image | None:
    x1, y1, x2, y2 = bbox
    if y2 > mask.shape[0] or x2 > mask.shape[1]:
        return None
    return mask[y1:y2, x1:x2]


def _paint_clear_bbox(
    image_pil: Image.Image,
    image_array: UInt8Image,
    clear_bbox: Bbox,
    fill_color: RgbColor,
    mask: UInt8Image | None,
) -> None:
    x1, y1, x2, y2 = clear_bbox
    if mask is None:
        image_array[y1:y2, x1:x2] = fill_color
    else:
        _paint_masked_pixels(image_array, clear_bbox, fill_color, mask)
    image_pil.paste(Image.fromarray(image_array))


def _paint_masked_pixels(image_array: UInt8Image, clear_bbox: Bbox, fill_color: RgbColor, mask: UInt8Image) -> None:
    x1, y1, x2, y2 = clear_bbox
    mask_crop = _mask_crop(mask, clear_bbox)
    if mask_crop is None:
        return
    image_array[y1:y2, x1:x2][mask_crop > 0] = fill_color

"""Speech bubble mask extraction for manga cleanup."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.models import Bbox, UInt8Image


def extract_bubble_mask(image_cv: UInt8Image, bbox: Bbox, config: CleanupConfig) -> UInt8Image:
    """Return a full-page binary mask for a detected speech bubble.

    Example:
        `mask = extract_bubble_mask(image_cv, (10, 10, 80, 40), CleanupConfig())`.
    """
    height, width = image_cv.shape[:2]
    clipped = _clip_bbox_to_image(bbox, width, height)
    if clipped is None:
        return np.zeros((height, width), dtype=np.uint8)
    if _is_tiny_bbox(clipped):
        return _rectangle_mask(height, width, clipped)
    padded = _padded_bbox(clipped, width, height, config.bubble_mask_padding)
    labels, stats = _bright_components(image_cv, padded, config.bubble_mask_threshold)
    best_label = _best_component_label(labels, stats, clipped, padded)
    if best_label is None:
        return _rectangle_mask(height, width, clipped)
    return _mask_from_component(labels, best_label, padded, height, width, clipped)


def _clip_bbox_to_image(bbox: Bbox, width: int, height: int) -> Bbox | None:
    x1, x2 = sorted((bbox[0], bbox[2]))
    y1, y2 = sorted((bbox[1], bbox[3]))
    clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    return None if clipped[2] <= clipped[0] or clipped[3] <= clipped[1] else clipped


def _is_tiny_bbox(bbox: Bbox) -> bool:
    return bbox[2] - bbox[0] < 10 or bbox[3] - bbox[1] < 10


def _rectangle_mask(height: int, width: int, bbox: Bbox) -> UInt8Image:
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 255
    return mask


def _padded_bbox(bbox: Bbox, width: int, height: int, padding: int) -> Bbox:
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def _bright_components(
    image_cv: UInt8Image,
    padded_bbox: Bbox,
    threshold: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    x1, y1, x2, y2 = padded_bbox
    roi = image_cv[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    _unused_count, labels, stats, _unused_centroids = cv2.connectedComponentsWithStats(
        cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1],
        connectivity=8,
    )
    return cast(NDArray[np.int32], labels), cast(NDArray[np.int32], stats)


def _best_component_label(labels: NDArray[np.int32], stats: NDArray[np.int32], bbox: Bbox, padded: Bbox) -> int | None:
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    center = _center_in_padded_bbox(bbox, padded, labels.shape)
    centered = int(labels[center[1], center[0]])
    if _component_area_is_valid(stats, centered, bbox_area):
        return centered
    return _largest_valid_component(stats, bbox_area)


def _center_in_padded_bbox(bbox: Bbox, padded: Bbox, shape: tuple[int, ...]) -> tuple[int, int]:
    center_x = (bbox[0] + bbox[2]) // 2 - padded[0]
    center_y = (bbox[1] + bbox[3]) // 2 - padded[1]
    return (min(max(0, center_x), shape[1] - 1), min(max(0, center_y), shape[0] - 1))


def _component_area_is_valid(stats: NDArray[np.int32], label: int, bbox_area: int) -> bool:
    if label <= 0 or label >= stats.shape[0]:
        return False
    area = int(stats[label, cv2.CC_STAT_AREA])
    return bbox_area * 0.30 <= area <= bbox_area * 2.00


def _largest_valid_component(stats: NDArray[np.int32], bbox_area: int) -> int | None:
    candidates = [
        (int(stats[label, cv2.CC_STAT_AREA]), label)
        for label in range(1, stats.shape[0])
        if _component_area_is_valid(stats, label, bbox_area)
    ]
    return max(candidates)[1] if candidates else None


def _mask_from_component(
    labels: NDArray[np.int32],
    label: int,
    padded_bbox: Bbox,
    height: int,
    width: int,
    fallback_bbox: Bbox,
) -> UInt8Image:
    component_mask = (labels == label).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _rectangle_mask(height, width, fallback_bbox)
    shifted_contour = max(contours, key=cv2.contourArea) + np.array([padded_bbox[0], padded_bbox[1]])
    full_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(full_mask, [shifted_contour], -1, 255, -1)
    return full_mask

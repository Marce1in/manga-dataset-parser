"""Optional artwork text inpainting for cleanup."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from importlib import import_module
from typing import Protocol, cast

import numpy as np
from PIL import Image, ImageFilter

from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.models import Bbox


class Inpainter(Protocol):
    """Boundary for optional text-region inpainting.

    Example:
        `image = inpainter.inpaint_region(image, (10, 10, 40, 30))`.
    """

    def inpaint_region(self, image_pil: Image.Image, _bbox: Bbox) -> Image.Image:
        """Return an image with the requested region inpainted.

        Example:
            `image = inpainter.inpaint_region(image, bbox)`.
        """
        ...


class SimpleLamaCallable(Protocol):
    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image: ...


@dataclass(frozen=True)
class NoopInpainter:
    """Inpainter that leaves images unchanged.

    Example:
        `NoopInpainter().inpaint_region(image, bbox) is image`.
    """

    def inpaint_region(self, image_pil: Image.Image, _bbox: Bbox) -> Image.Image:
        """Return `image_pil` unchanged.

        Example:
            `same_image = NoopInpainter().inpaint_region(image, bbox)`.
        """
        return image_pil


@dataclass
class LamaInpainter:
    """Optional LaMa inpainter loaded only when requested.

    Example:
        `image = LamaInpainter(CleanupConfig(enable_artwork_inpainting=True)).inpaint_region(image, bbox)`.
    """

    config: CleanupConfig
    _load_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _infer_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _model: SimpleLamaCallable | None = field(default=None, init=False)
    _load_failed: bool = field(default=False, init=False)

    def inpaint_region(self, image_pil: Image.Image, bbox: Bbox) -> Image.Image:
        """Inpaint one artwork text bbox when LaMa is available.

        Example:
            `result = LamaInpainter(config).inpaint_region(image, (20, 20, 80, 60))`.
        """
        model = self._loaded_model()
        if model is None:
            return image_pil
        crop_box = _context_bbox(image_pil.size, bbox, self.config.inpaint_padding)
        image_crop = image_pil.crop(crop_box)
        mask_crop = _inpaint_mask_crop(crop_box, bbox)
        with self._infer_lock:
            result_crop = model(image_crop.convert("RGB"), mask_crop.convert("L")).convert("RGB")
        return _composited_inpaint_result(image_pil, crop_box, mask_crop, result_crop)

    def _loaded_model(self) -> SimpleLamaCallable | None:
        if self._model is not None or self._load_failed:
            return self._model
        with self._load_lock:
            if self._model is None and not self._load_failed:
                self._load_model()
        return self._model

    def _load_model(self) -> None:
        try:
            module = import_module("simple_lama_inpainting")
        except ImportError:
            self._load_failed = True
            return
        self._model = cast(SimpleLamaCallable, module.SimpleLama())


def _context_bbox(image_size: tuple[int, int], bbox: Bbox, padding: int) -> Bbox:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    padded = (max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding))
    return _pad_bbox_to_multiple_of_eight(padded, width, height)


def _pad_bbox_to_multiple_of_eight(bbox: Bbox, width: int, height: int) -> Bbox:
    x1, y1, x2, y2 = bbox
    pad_w = (8 - (x2 - x1) % 8) % 8
    pad_h = (8 - (y2 - y1) % 8) % 8
    return (x1, y1, min(width, x2 + pad_w), min(height, y2 + pad_h))


def _inpaint_mask_crop(crop_box: Bbox, bbox: Bbox) -> Image.Image:
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    mask_array = np.zeros((crop_height, crop_width), dtype=np.uint8)
    x1, y1, x2, y2 = _bbox_relative_to_crop(crop_box, bbox)
    mask_array[y1:y2, x1:x2] = 255
    return Image.fromarray(mask_array).filter(ImageFilter.MaxFilter(5))


def _bbox_relative_to_crop(crop_box: Bbox, bbox: Bbox) -> Bbox:
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    return (
        max(0, bbox[0] - crop_box[0] - 4),
        max(0, bbox[1] - crop_box[1] - 4),
        min(crop_width, bbox[2] - crop_box[0] + 4),
        min(crop_height, bbox[3] - crop_box[1] + 4),
    )


def _composited_inpaint_result(
    image_pil: Image.Image,
    crop_box: Bbox,
    mask_crop: Image.Image,
    result_crop: Image.Image,
) -> Image.Image:
    target = image_pil.copy()
    if result_crop.size != mask_crop.size:
        result_crop = result_crop.resize(mask_crop.size, Image.Resampling.LANCZOS)
    mask_array = np.array(mask_crop)
    result_array = np.array(result_crop)
    target_array = np.array(target.crop(crop_box))
    composite = np.where(mask_array[:, :, np.newaxis] > 0, result_array, target_array)
    target.paste(Image.fromarray(composite.astype(np.uint8)), (crop_box[0], crop_box[1]))
    return target

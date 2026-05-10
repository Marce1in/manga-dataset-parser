"""RT-DETR-v2 detector boundary for manga cleanup."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from PIL import Image

from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.models import Bbox, DetectedRegion, DetectedRegionKind, UInt8Image


@dataclass
class ComicTextBubbleDetector:
    """Detect manga speech bubbles and artwork text with RT-DETR-v2.

    Example:
        `regions = ComicTextBubbleDetector(CleanupConfig()).detect_regions(image_cv)`.
    """

    config: CleanupConfig
    _load_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _model: Any | None = field(default=None, init=False)
    _processor: Any | None = field(default=None, init=False)
    _device: str | None = field(default=None, init=False)

    def detect_regions(self, image_cv: UInt8Image) -> list[DetectedRegion]:
        """Return project-owned detections for one OpenCV BGR image.

        Example:
            `detector.detect_regions(image_cv)[0].kind is DetectedRegionKind.SPEECH_BUBBLE`.
        """
        model, processor, device = self._loaded_model()
        image_pil = _bgr_image_to_pil(image_cv)
        inputs = processor(images=image_pil, return_tensors="pt").to(device)
        torch = import_module("torch")
        with torch.no_grad():
            outputs = model(**inputs)
        return self._regions_from_outputs(model, processor, outputs, image_pil, device)

    def _loaded_model(self) -> tuple[Any, Any, str]:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._load_model()
        if self._model is None or self._processor is None or self._device is None:
            raise RuntimeError("RT-DETR model failed to load; expected model, processor, and device.")
        return self._model, self._processor, self._device

    def _load_model(self) -> None:
        transformers = import_module("transformers")
        torch = import_module("torch")
        processor_name = "RTDetrImageProcessor"
        model_name = "RTDetrV2ForObjectDetection"
        processor_class = getattr(transformers, processor_name)
        model_class = getattr(transformers, model_name)
        self._processor = processor_class.from_pretrained(self.config.detector_model_id)
        self._model = model_class.from_pretrained(self.config.detector_model_id)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

    def _regions_from_outputs(
        self,
        model: Any,
        processor: Any,
        outputs: Any,
        image_pil: Image.Image,
        device: str,
    ) -> list[DetectedRegion]:
        torch = import_module("torch")
        target_sizes = torch.tensor([(image_pil.height, image_pil.width)], device=device)
        processed = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.config.detector_confidence,
        )[0]
        regions = _build_detected_regions(model, processed, image_pil.size, self.config)
        return _deduplicate_regions(regions, self.config.deduplicate_overlap_ratio)


def _bgr_image_to_pil(image_cv: UInt8Image) -> Image.Image:
    rgb_image = image_cv[:, :, ::-1]
    return Image.fromarray(rgb_image)


def _build_detected_regions(
    model: Any,
    processed: dict[str, Any],
    image_size: tuple[int, int],
    config: CleanupConfig,
) -> list[DetectedRegion]:
    regions: list[DetectedRegion] = []
    values = zip(processed["scores"], processed["labels"], processed["boxes"], strict=True)
    for score_value, label_value, box_value in values:
        region = _region_from_model_values(model, score_value, label_value, box_value, image_size, config)
        if region is not None:
            regions.append(region)
    return sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))


def _region_from_model_values(
    model: Any,
    score_value: Any,
    label_value: Any,
    box_value: Any,
    image_size: tuple[int, int],
    config: CleanupConfig,
) -> DetectedRegion | None:
    score = float(score_value.item())
    label = str(model.config.id2label[int(label_value.item())])
    kind = _map_model_label_to_kind(label)
    if kind is DetectedRegionKind.ARTWORK_TEXT and score < config.artwork_text_min_confidence:
        return None
    bbox = _clamped_bbox(tuple(round(value) for value in box_value.tolist()), image_size)
    return None if bbox is None else DetectedRegion(bbox=bbox, kind=kind, score=score)


def _map_model_label_to_kind(label: str) -> DetectedRegionKind:
    if label in {"bubble", "text_bubble"}:
        return DetectedRegionKind.SPEECH_BUBBLE
    return DetectedRegionKind.ARTWORK_TEXT


def _clamped_bbox(raw_bbox: tuple[int, ...], image_size: tuple[int, int]) -> Bbox | None:
    width, height = image_size
    if len(raw_bbox) != 4:
        raise ValueError(f"Model bbox must have 4 values; got {raw_bbox!r}.")
    x1, y1, x2, y2 = raw_bbox
    clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    return None if clipped[2] <= clipped[0] or clipped[3] <= clipped[1] else clipped


def _bbox_overlap_ratio(first: Bbox, second: Bbox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    return intersection / first_area if first_area > 0 else 0.0


def _deduplicate_regions(regions: list[DetectedRegion], threshold: float) -> list[DetectedRegion]:
    kept: list[DetectedRegion] = []
    for region in sorted(regions, key=lambda value: value.score, reverse=True):
        if not _overlaps_kept_region(region, kept, threshold):
            kept.append(region)
    return sorted(kept, key=lambda value: (value.bbox[1], value.bbox[0]))


def _overlaps_kept_region(region: DetectedRegion, kept: list[DetectedRegion], threshold: float) -> bool:
    for existing in kept:
        forward = _bbox_overlap_ratio(region.bbox, existing.bbox)
        reverse = _bbox_overlap_ratio(existing.bbox, region.bbox)
        if forward > threshold or reverse > threshold:
            return True
    return False

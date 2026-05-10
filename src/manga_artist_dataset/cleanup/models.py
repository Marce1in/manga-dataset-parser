"""Typed values shared by the manga cleanup pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

Bbox = tuple[int, int, int, int]
UInt8Image = NDArray[np.uint8]


class DetectedRegionKind(Enum):
    SPEECH_BUBBLE = "speech_bubble"
    ARTWORK_TEXT = "artwork_text"


@dataclass(frozen=True)
class DetectedRegion:
    bbox: Bbox
    kind: DetectedRegionKind
    score: float


@dataclass
class BubbleRegion:
    bbox: Bbox
    score: float
    mask: UInt8Image | None


@dataclass
class CleanupPage:
    source_path: Path
    name: str
    image_cv: UInt8Image
    image_pil: Image.Image
    bubbles: list[BubbleRegion]
    artwork_regions: list[DetectedRegion]

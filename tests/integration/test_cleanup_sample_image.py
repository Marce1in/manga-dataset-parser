from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.models import DetectedRegion, DetectedRegionKind, UInt8Image
from manga_artist_dataset.cleanup.pipeline import MangaCleanupPipeline


class SampleBubbleDetector:
    def detect_regions(self, _image_cv: UInt8Image) -> list[DetectedRegion]:
        return [DetectedRegion((30, 25, 130, 80), DetectedRegionKind.SPEECH_BUBBLE, 0.99)]


def test_cleanup_sample_image_writes_same_size_changed_output(tmp_path: Path) -> None:
    source = tmp_path / "sample_page.png"
    output = tmp_path / "cleaned" / "sample_page.png"
    write_synthetic_manga_page(source)

    MangaCleanupPipeline(SampleBubbleDetector(), CleanupConfig()).clean_image(source, output)

    with Image.open(source) as source_image, Image.open(output) as output_image:
        assert output_image.size == source_image.size
        assert changed_pixel_count(source_image, output_image) > 0


def write_synthetic_manga_page(path: Path) -> None:
    image = Image.new("RGB", (160, 120), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 159, 119), outline=(0, 0, 0), width=2)
    draw.ellipse((30, 25, 130, 80), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.rectangle((62, 45, 98, 53), fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def changed_pixel_count(first: Image.Image, second: Image.Image) -> int:
    diff = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    return int(np.count_nonzero(np.array(diff)))

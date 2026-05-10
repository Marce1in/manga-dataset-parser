from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.models import DetectedRegion, DetectedRegionKind, UInt8Image
from manga_artist_dataset.cleanup.pipeline import MangaCleanupPipeline


class FakeBubbleDetector:
    def __init__(self, regions: list[DetectedRegion]) -> None:
        self.regions = regions
        self.calls = 0

    def detect_regions(self, _image_cv: UInt8Image) -> list[DetectedRegion]:
        self.calls += 1
        return self.regions


class FakeArtworkInpainter:
    def __init__(self) -> None:
        self.calls = 0

    def inpaint_region(self, image_pil: Image.Image, _bbox: tuple[int, int, int, int]) -> Image.Image:
        self.calls += 1
        return image_pil


def test_pipeline_calls_detector_and_saves_output(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    output = tmp_path / "cleaned" / "page.png"
    write_sample_page(source)
    detector = FakeBubbleDetector([DetectedRegion((20, 20, 80, 60), DetectedRegionKind.SPEECH_BUBBLE, 0.95)])

    saved = MangaCleanupPipeline(detector, CleanupConfig()).clean_image(source, output)

    assert saved == output
    assert output.exists()
    assert detector.calls == 1


def test_pipeline_does_not_overwrite_source_image(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    output = tmp_path / "cleaned" / "page.png"
    write_sample_page(source)
    before = source.read_bytes()
    detector = FakeBubbleDetector([DetectedRegion((20, 20, 80, 60), DetectedRegionKind.SPEECH_BUBBLE, 0.95)])

    MangaCleanupPipeline(detector, CleanupConfig()).clean_image(source, output)

    assert source.read_bytes() == before


def test_pipeline_skips_artwork_inpainting_when_disabled(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    output = tmp_path / "cleaned" / "page.png"
    write_sample_page(source)
    inpainter = FakeArtworkInpainter()
    detector = FakeBubbleDetector([DetectedRegion((20, 20, 80, 60), DetectedRegionKind.ARTWORK_TEXT, 0.95)])

    MangaCleanupPipeline(detector, CleanupConfig(), inpainter).clean_image(source, output)

    assert inpainter.calls == 0


def test_pipeline_cleans_directory_preserving_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "downloaded" / "class_a" / "page.png"
    output_dir = tmp_path / "cleaned"
    write_sample_page(source)
    detector = FakeBubbleDetector([DetectedRegion((20, 20, 80, 60), DetectedRegionKind.SPEECH_BUBBLE, 0.95)])

    saved = MangaCleanupPipeline(detector, CleanupConfig()).clean_directory(source.parent.parent, output_dir)

    assert saved == [output_dir / "class_a" / "page.png"]
    assert saved[0].exists()


def write_sample_page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (100, 80), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 80, 60), fill=(255, 255, 255))
    draw.rectangle((40, 34, 60, 42), fill=(0, 0, 0))
    image.save(path)

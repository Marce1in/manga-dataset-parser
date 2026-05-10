"""Orchestration for manga speech bubble cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

from manga_artist_dataset.cleanup.bubble_masks import extract_bubble_mask
from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.detector import ComicTextBubbleDetector
from manga_artist_dataset.cleanup.inpainter import Inpainter, LamaInpainter, NoopInpainter
from manga_artist_dataset.cleanup.io import load_image_cv, load_image_pil, save_image_pil
from manga_artist_dataset.cleanup.models import (
    BubbleRegion,
    CleanupPage,
    DetectedRegion,
    DetectedRegionKind,
    UInt8Image,
)


class RegionDetector(Protocol):
    """Boundary for speech bubble detectors used by the pipeline.

    Example:
        `regions = detector.detect_regions(image_cv)`.
    """

    def detect_regions(self, image_cv: UInt8Image) -> list[DetectedRegion]:
        """Return detected cleanup regions.

        Example:
            `detector.detect_regions(image_cv)`.
        """
        ...


class MangaCleanupPipeline:
    """Clean downloaded manga images without mutating the originals.

    Example:
        `MangaCleanupPipeline(detector, CleanupConfig()).clean_image(source, output)`.
    """

    def __init__(
        self,
        detector: RegionDetector,
        config: CleanupConfig,
        inpainter: Inpainter | None = None,
    ) -> None:
        self.detector = detector
        self.config = config
        self.inpainter = inpainter or NoopInpainter()

    def clean_image(self, input_path: Path, output_path: Path) -> Path:
        """Clean one image and return the saved path.

        Example:
            `saved = pipeline.clean_image(Path("page.png"), Path("cleaned/page.png"))`.
        """
        page = self._load_page(input_path)
        output_image = page.image_pil.copy()
        for bubble in page.bubbles:
            clear_masked_bubble(output_image, bubble, self.config)
        output_image = self._inpaint_artwork_text(output_image, page.artwork_regions)
        save_image_pil(output_image, output_path)
        return output_path

    def clean_directory(self, input_dir: Path, output_dir: Path) -> list[Path]:
        """Clean every supported image under `input_dir`.

        Example:
            `paths = pipeline.clean_directory(Path("downloaded"), Path("cleaned"))`.
        """
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Cleanup input must be a directory; got {input_dir}.")
        paths = list(iter_supported_image_paths(input_dir, self.config.supported_extensions))
        return [self.clean_image(path, output_dir / path.relative_to(input_dir)) for path in paths]

    def _load_page(self, input_path: Path) -> CleanupPage:
        image_cv = load_image_cv(input_path)
        image_pil = load_image_pil(input_path)
        regions = self.detector.detect_regions(image_cv)
        bubbles = _bubble_regions(image_cv, regions, self.config)
        artwork_regions = [region for region in regions if region.kind is DetectedRegionKind.ARTWORK_TEXT]
        return CleanupPage(input_path, input_path.stem, image_cv, image_pil, bubbles, artwork_regions)

    def _inpaint_artwork_text(self, image_pil: Image.Image, regions: list[DetectedRegion]) -> Image.Image:
        if not self.config.enable_artwork_inpainting:
            return image_pil
        for region in regions:
            image_pil = self.inpainter.inpaint_region(image_pil, region.bbox)
        return image_pil


def build_default_manga_cleanup_pipeline(config: CleanupConfig) -> MangaCleanupPipeline:
    """Build the production cleanup pipeline with the RT-DETR detector.

    Example:
        `pipeline = build_default_manga_cleanup_pipeline(CleanupConfig())`.
    """
    inpainter = LamaInpainter(config) if config.enable_artwork_inpainting else NoopInpainter()
    return MangaCleanupPipeline(ComicTextBubbleDetector(config), config, inpainter)


def clear_masked_bubble(image_pil: Image.Image, bubble: BubbleRegion, config: CleanupConfig) -> None:
    """Clear one bubble region using its extracted mask.

    Example:
        `clear_masked_bubble(image, bubble, CleanupConfig())`.
    """
    from manga_artist_dataset.cleanup.text_cleaner import clear_text_strokes

    clear_text_strokes(image_pil, bubble.bbox, config, mask=bubble.mask)


def iter_supported_image_paths(input_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Return supported image files under `input_dir` in stable order.

    Example:
        `paths = iter_supported_image_paths(Path("downloaded"), (".png",))`.
    """
    lowered = {extension.lower() for extension in extensions}
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in lowered)


def _bubble_regions(image_cv: UInt8Image, regions: list[DetectedRegion], config: CleanupConfig) -> list[BubbleRegion]:
    bubbles: list[BubbleRegion] = []
    for region in regions:
        if region.kind is DetectedRegionKind.SPEECH_BUBBLE:
            mask = extract_bubble_mask(image_cv, region.bbox, config)
            bubbles.append(BubbleRegion(region.bbox, region.score, mask))
    return bubbles

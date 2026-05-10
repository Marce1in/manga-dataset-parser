"""Dataset cleanup stage backed by the local manga cleanup pipeline."""

from __future__ import annotations

import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from PIL import Image

from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.image_standardize import (
    DEFAULT_TARGET_SIZE,
    ImageTargetSize,
    standardization_record,
    standardize_image_file,
)
from manga_artist_dataset.cleanup.pipeline import build_default_manga_cleanup_pipeline
from manga_artist_dataset.cleanup.scratch_png import ensure_under_root, prepare_scratch_png_records, resolve_record_path
from manga_artist_dataset.io.files import recreate_dir, sha256_file, workspace_path
from manga_artist_dataset.io.jsonl import load_jsonl, write_json_object, write_jsonl
from manga_artist_dataset.json_types import JsonObject

DEFAULT_INPUT_ROOT = Path("artifacts/datasets/polished_pages")
DEFAULT_OUTPUT_ROOT = Path("artifacts/datasets/panel_cleaned_pages")
DEFAULT_EXPECTED_TOTAL = 600
DEFAULT_EXPECTED_PER_CLASS = 60


@dataclass(frozen=True)
class DatasetCleanupConfig:
    """Configuration for the final cleaned dataset stage.

    Example:
        `DatasetCleanupConfig(Path("polished"), Path("cleaned"), True)`.
    """

    input_root: Path
    output_root: Path
    overwrite: bool
    target_size: ImageTargetSize = DEFAULT_TARGET_SIZE
    expected_total: int = DEFAULT_EXPECTED_TOTAL
    expected_per_class: int = DEFAULT_EXPECTED_PER_CLASS
    cleanup_config: CleanupConfig = field(default_factory=CleanupConfig)


class MangaPageCleanerRunner(Protocol):
    """Boundary for cleaning scratch PNG pages in production or tests.

    Example:
        `runner.clean(Path("png"), Path("raw_cleaned"))`.
    """

    def clean(self, input_root: Path, output_root: Path) -> list[Path]:
        """Clean source pages into `output_root`.

        Example:
            `saved = runner.clean(Path("png"), Path("cleaned"))`.
        """
        ...


@dataclass(frozen=True)
class LocalMangaPageCleanerRunner:
    """Runner backed by the RT-DETR/OpenCV/Pillow cleanup pipeline.

    Example:
        `LocalMangaPageCleanerRunner(CleanupConfig()).clean(Path("png"), Path("cleaned"))`.
    """

    cleanup_config: CleanupConfig

    def clean(self, input_root: Path, output_root: Path) -> list[Path]:
        """Clean a directory of scratch PNG pages.

        Example:
            `runner.clean(Path("png"), Path("raw_cleaned"))`.
        """
        pipeline = build_default_manga_cleanup_pipeline(self.cleanup_config)
        return pipeline.clean_directory(input_root, output_root)


def clean_panels(config: DatasetCleanupConfig, runner: MangaPageCleanerRunner | None = None) -> JsonObject:
    """Run local manga cleanup and write final dataset metadata.

    Example:
        `clean_panels(DatasetCleanupConfig(Path("polished"), Path("cleaned"), True))`.
    """
    records = load_source_records(config.input_root)
    validate_counts(records, config.expected_total, config.expected_per_class)
    recreate_dir(config.output_root, config.overwrite)
    active_runner = runner or LocalMangaPageCleanerRunner(config.cleanup_config)
    cleaned_records = run_cleanup_in_scratch(config, records, active_runner)
    validate_counts(cleaned_records, config.expected_total, config.expected_per_class)
    verify_final_dataset(config.output_root, cleaned_records)
    write_jsonl(config.output_root / "metadata.jsonl", cleaned_records)
    report = build_cleaned_report(config, cleaned_records)
    write_json_object(config.output_root / "dataset_report.json", report)
    return report


def run_cleanup_in_scratch(
    config: DatasetCleanupConfig,
    source_records: list[JsonObject],
    runner: MangaPageCleanerRunner,
) -> list[JsonObject]:
    """Convert sources to scratch PNGs, clean them, and standardize outputs.

    Example:
        `records = run_cleanup_in_scratch(config, source_records, runner)`.
    """
    with tempfile.TemporaryDirectory(prefix="manga_png_pages_") as scratch_name:
        scratch_root = Path(scratch_name)
        png_root = scratch_root / "png_pages"
        raw_cleaned_root = scratch_root / "cleaned_pages"
        png_records = prepare_scratch_png_records(source_records, config.input_root.resolve(), png_root.resolve())
        runner.clean(png_root, raw_cleaned_root)
        return build_cleaned_metadata(config, png_root, raw_cleaned_root, source_records, png_records)


def load_source_records(input_root: Path) -> list[JsonObject]:
    """Load source dataset records for cleanup preparation.

    Example:
        `records = load_source_records(Path("artifacts/datasets/polished_pages"))`.
    """
    metadata = input_root / "metadata.jsonl"
    if not metadata.exists():
        raise FileNotFoundError(f"Missing source metadata: {metadata}")
    return load_jsonl(metadata)


def build_cleaned_metadata(
    config: DatasetCleanupConfig,
    png_root: Path,
    raw_cleaned_root: Path,
    source_records: list[JsonObject],
    png_records: list[JsonObject],
) -> list[JsonObject]:
    """Build final metadata while standardizing each cleaned page.

    Example:
        `rows = build_cleaned_metadata(config, png_root, raw_root, source_rows, png_rows)`.
    """
    ensure_record_counts_match(source_records, png_records)
    return [
        cleaned_record(config, png_root.resolve(), raw_cleaned_root.resolve(), source, png)
        for source, png in zip(source_records, png_records, strict=True)
    ]


def ensure_record_counts_match(source_records: list[JsonObject], png_records: list[JsonObject]) -> None:
    """Validate scratch PNG metadata cardinality before final writes.

    Example:
        `ensure_record_counts_match(source_records, png_records)`.
    """
    if len(source_records) == len(png_records):
        return
    raise ValueError(f"Expected {len(source_records)} scratch PNG records, found {len(png_records)}.")


def cleaned_record(
    config: DatasetCleanupConfig,
    png_root: Path,
    raw_cleaned_root: Path,
    source: JsonObject,
    png: JsonObject,
) -> JsonObject:
    """Write one standardized cleaned image and return its metadata row.

    Example:
        `row = cleaned_record(config, png_root, raw_root, source, png)`.
    """
    png_path = resolve_record_path(png, "output_path")
    ensure_under_root(png_path, png_root)
    raw_cleaned_path = raw_cleaned_root / png_path.relative_to(png_root)
    cleaned_path = config.output_root.resolve() / raw_cleaned_path.relative_to(raw_cleaned_root)
    ensure_same_dimensions(png_path, raw_cleaned_path)
    standardize_image_file(raw_cleaned_path, cleaned_path, config.target_size)
    return finalized_record(config, source, cleaned_path)


def finalized_record(config: DatasetCleanupConfig, source: JsonObject, cleaned_path: Path) -> JsonObject:
    """Return final metadata without references to temporary PNG paths.

    Example:
        `row = finalized_record(config, source_record, Path("cleaned/page.png"))`.
    """
    cleaned = dict(source)
    cleaned["stage"] = "panel_cleaned_pages"
    cleaned["polished_output_path"] = source.get("output_path")
    cleaned["polished_sha256"] = source.get("sha256")
    cleaned["polished_width"] = source.get("width")
    cleaned["polished_height"] = source.get("height")
    cleaned["polished_bytes"] = source.get("bytes")
    cleaned["output_path"] = workspace_path(cleaned_path)
    cleaned["image_format"] = "PNG"
    cleaned["color_mode"] = "RGB"
    cleaned["cleanup"] = cleanup_record(config.cleanup_config)
    cleaned["standardization"] = standardization_record(config.target_size)
    cleaned["sha256"] = sha256_file(cleaned_path)
    cleaned["bytes"] = cleaned_path.stat().st_size
    cleaned["width"] = config.target_size.width
    cleaned["height"] = config.target_size.height
    return cleaned


def cleanup_record(config: CleanupConfig) -> JsonObject:
    """Describe the cleanup implementation used for the dataset.

    Example:
        `cleanup_record(CleanupConfig())["mode"] == "speech_bubble_cleanup"`.
    """
    return {
        "mode": "speech_bubble_cleanup",
        "detector_model_id": config.detector_model_id,
        "detector_confidence": config.detector_confidence,
        "dark_text_threshold": config.dark_text_threshold,
        "text_clear_margin": config.text_clear_margin,
        "text_dilation_kernel_size": config.text_dilation_kernel_size,
        "text_dilation_iterations": config.text_dilation_iterations,
        "bubble_mask_threshold": config.bubble_mask_threshold,
        "bubble_mask_padding": config.bubble_mask_padding,
        "artwork_inpainting": config.enable_artwork_inpainting,
    }


def image_size(path: Path) -> tuple[int, int]:
    """Return Pillow image dimensions for a path.

    Example:
        `size = image_size(Path("page.png"))`.
    """
    with Image.open(path) as image:
        return image.size


def ensure_same_dimensions(source: Path, cleaned: Path) -> None:
    """Validate that cleanup preserved source dimensions.

    Example:
        `ensure_same_dimensions(Path("source.png"), Path("cleaned.png"))`.
    """
    source_size = image_size(source)
    cleaned_size = image_size(cleaned)
    if source_size != cleaned_size:
        raise ValueError(f"Cleanup changed dimensions for {cleaned}: {source_size} -> {cleaned_size}.")


def validate_counts(records: list[JsonObject], expected_total: int, expected_per_class: int) -> None:
    """Validate total and per-class record counts.

    Example:
        `validate_counts(records, 600, 60)`.
    """
    if len(records) != expected_total:
        raise ValueError(f"Expected {expected_total} records, found {len(records)}.")
    bad = {name: count for name, count in class_counts(records).items() if int(count) != expected_per_class}
    if bad:
        raise ValueError(f"Expected {expected_per_class} records per class, found {bad}.")


def class_counts(records: list[JsonObject]) -> JsonObject:
    """Count records by class folder.

    Example:
        `counts = class_counts(records)`.
    """
    counts = Counter(record_class_slug(record) for record in records)
    return dict(sorted(counts.items()))


def record_class_slug(record: JsonObject) -> str:
    """Return the class folder recorded in an output path.

    Example:
        `slug = record_class_slug({"output_path": "out/01_artist/page.png"})`.
    """
    value = record.get("output_path")
    if not isinstance(value, str):
        raise ValueError(f"Record output_path must be a string; got {value!r}.")
    return Path(value).parent.name


def verify_final_dataset(output_root: Path, records: list[JsonObject]) -> None:
    """Validate final output file count and names.

    Example:
        `verify_final_dataset(Path("cleaned"), records)`.
    """
    png_files = sorted(output_root.glob("*/*.png"))
    if len(png_files) != len(records):
        raise ValueError(f"Cleaned file count mismatch: {len(png_files)} files vs {len(records)} rows.")


def build_cleaned_report(config: DatasetCleanupConfig, records: list[JsonObject]) -> JsonObject:
    """Build the final cleaned dataset report.

    Example:
        `report = build_cleaned_report(config, records)`.
    """
    return {
        "config": cleaned_report_config(config),
        "total_pages": len(records),
        "file_extension_counts": {".png": len(records)},
        "class_counts": class_counts(records),
        "targets": target_reports(records),
    }


def cleaned_report_config(config: DatasetCleanupConfig) -> JsonObject:
    """Render report configuration for the final cleaned dataset.

    Example:
        `cleaned_report_config(config)["stage"] == "panel_cleaned_pages"`.
    """
    return {
        "source_output": workspace_path(config.input_root),
        "output": workspace_path(config.output_root),
        "stage": "panel_cleaned_pages",
        "created_at": datetime.now(UTC).isoformat(),
        "cleanup": cleanup_record(config.cleanup_config),
        "standardization": standardization_record(config.target_size),
    }


def target_reports(records: list[JsonObject]) -> list[JsonObject]:
    """Build per-target counts for the dataset report.

    Example:
        `targets = target_reports(records)`.
    """
    targets: dict[str, JsonObject] = {}
    for record in records:
        slug = record_class_slug(record)
        targets.setdefault(slug, new_target_report(record, slug))
        targets[slug]["selected_page_count"] = int(targets[slug]["selected_page_count"]) + 1
    return [targets[key] for key in sorted(targets)]


def new_target_report(record: JsonObject, slug: str) -> JsonObject:
    """Create an empty target report row.

    Example:
        `row = new_target_report(record, "01_artist")`.
    """
    return {
        "label_id": record.get("label_id"),
        "artist": record.get("artist"),
        "series": record.get("series"),
        "class_slug": slug,
        "selected_page_count": 0,
    }

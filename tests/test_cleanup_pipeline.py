from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from manga_artist_dataset.cleanup.dataset_cleaner import DatasetCleanupConfig, clean_panels
from manga_artist_dataset.cleanup.image_standardize import (
    ImageTargetSize,
    canvas_tuple,
    centered_offset,
    standardization_record,
    standardize_image,
    standardize_image_file,
)
from manga_artist_dataset.cleanup.scratch_png import prepare_scratch_png_records


class CopyingMangaPageCleanerRunner:
    def __init__(self) -> None:
        self.last_input_root: Path | None = None

    def clean(self, input_root: Path, output_root: Path) -> list[Path]:
        self.last_input_root = input_root
        saved_paths: list[Path] = []
        for source_path in sorted(input_root.glob("*/*.png")):
            destination = output_root / source_path.parent.name / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            saved_paths.append(destination)
        return saved_paths


def test_prepare_scratch_png_records_preserves_class_and_writes_png(tmp_path: Path) -> None:
    input_root = tmp_path / "polished"
    source_path = write_source_image(input_root)
    write_source_metadata(input_root, source_path)

    output_root = tmp_path / "png"
    source_records = read_jsonl(input_root / "metadata.jsonl")
    records = prepare_scratch_png_records(source_records, input_root.resolve(), output_root.resolve())

    converted_path = output_root / "01_example_artist" / "sample.png"
    assert converted_path.exists()
    assert records[0]["output_path"] == str(converted_path)
    assert records[0]["stage"] == "scratch_png_pages"
    assert records[0]["image_format"] == "PNG"
    assert records[0]["width"] == 3
    assert records[0]["height"] == 5


def test_prepare_scratch_png_records_parallel_preserves_order(tmp_path: Path) -> None:
    input_root = tmp_path / "polished"
    source_paths = [write_source_image(input_root, f"sample_{index}.jpg") for index in range(2)]
    write_source_metadata(input_root, source_paths)

    output_root = tmp_path / "png"
    source_records = read_jsonl(input_root / "metadata.jsonl")
    records = prepare_scratch_png_records(source_records, input_root.resolve(), output_root.resolve(), worker_count=2)

    assert [Path(str(record["output_path"])).name for record in records] == ["sample_0.png", "sample_1.png"]
    assert all(Path(str(record["output_path"])).exists() for record in records)


def test_standardize_image_preserves_aspect_and_pads_white() -> None:
    target_size = ImageTargetSize(width=10, height=12)
    source = Image.new("RGB", (20, 10), "black")

    standardized = standardize_image(source, target_size)

    assert standardized.size == canvas_tuple(target_size)
    assert centered_offset(Image.new("RGB", (10, 5)), target_size) == (0, 3)
    assert standardized.getpixel((0, 0)) == (255, 255, 255)
    assert standardized.getpixel((5, 5)) == (0, 0, 0)


def test_standardize_image_file_writes_png_canvas(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    destination = tmp_path / "standardized.png"
    target_size = ImageTargetSize(width=8, height=12)
    Image.new("RGB", (4, 6), "black").save(source_path)

    standardize_image_file(source_path, destination, target_size)

    with Image.open(destination) as image:
        assert image.format == "PNG"
        assert image.size == (8, 12)


def test_standardization_record_describes_fixed_png_canvas() -> None:
    target_size = ImageTargetSize(width=512, height=768)

    record = standardization_record(target_size)

    assert record["width"] == 512
    assert record["height"] == 768
    assert record["resize"] == "contain"


def test_clean_panels_uses_scratch_png_and_standardizes_output(tmp_path: Path) -> None:
    input_root = tmp_path / "polished"
    source_paths = [write_source_image(input_root, f"sample_{index}.jpg") for index in range(2)]
    write_source_metadata(input_root, source_paths)
    output_root = tmp_path / "panel_cleaned_pages"
    runner = CopyingMangaPageCleanerRunner()

    report = clean_panels(cleanup_config_for_test(input_root, output_root), runner)

    records = read_jsonl(output_root / "metadata.jsonl")
    scratch_root = runner.last_input_root
    assert report["total_pages"] == 2
    assert scratch_root is not None
    assert not scratch_root.exists()
    assert [record["width"] for record in records] == [16, 16]
    assert [record["height"] for record in records] == [24, 24]
    assert [record["split"] for record in records] == ["test", "train"]
    assert [record["split_group"] for record in records] == ["20% anchor", "50% anchor"]
    assert output_split_dirs(output_root, records) == {"test", "train"}
    assert report["split_counts"] == {"test": 1, "train": 1}
    assert all("png_output_path" not in record for record in records)
    assert all(str(record["output_path"]).startswith(str(output_root)) for record in records)
    assert all(cleanup_mode(record) == "speech_bubble_cleanup" for record in records)


def cleanup_config_for_test(input_root: Path, output_root: Path) -> DatasetCleanupConfig:
    return DatasetCleanupConfig(
        input_root=input_root,
        output_root=output_root,
        overwrite=False,
        target_size=ImageTargetSize(width=16, height=24),
        expected_total=2,
        expected_per_class=2,
        scratch_workers=1,
        detector_workers=1,
        standardize_workers=2,
    )


def write_source_image(input_root: Path, filename: str = "sample.jpg") -> Path:
    class_dir = input_root / "01_example_artist"
    class_dir.mkdir(parents=True, exist_ok=True)
    source_path = class_dir / filename
    Image.new("RGB", (3, 5), "white").save(source_path)
    return source_path


def write_source_metadata(input_root: Path, source_paths: Path | list[Path]) -> None:
    paths = source_paths if isinstance(source_paths, list) else [source_paths]
    rows = [json.dumps(source_record(path)) for path in paths]
    (input_root / "metadata.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def source_record(source_path: Path) -> dict[str, object]:
    page_number = 1 if "0" in source_path.stem else 2
    anchor = "20% anchor" if page_number == 1 else "50% anchor"
    return {
        "label_id": 1,
        "artist": "Example Artist",
        "series": "Example Series",
        "chapter": f"{anchor}: Chapter {page_number}",
        "original_page_index": page_number,
        "output_path": str(source_path),
        "sha256": "old",
        "width": 3,
        "height": 5,
        "bytes": source_path.stat().st_size,
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def cleanup_mode(record: dict[str, object]) -> object:
    cleanup = record["cleanup"]
    assert isinstance(cleanup, dict)
    return cleanup["mode"]


def output_split_dirs(output_root: Path, records: list[dict[str, object]]) -> set[str]:
    return {Path(str(record["output_path"])).relative_to(output_root).parts[0] for record in records}

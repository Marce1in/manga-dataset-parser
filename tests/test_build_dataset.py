from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from manga_artist_dataset.build.chapter_sources import chapters_from_source
from manga_artist_dataset.build.pipeline import build_dataset
from manga_artist_dataset.errors import ManifestError
from manga_artist_dataset.models import BuildConfig, TargetSpec

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def write_pages(chapter_dir: Path, count: int) -> None:
    chapter_dir.mkdir(parents=True)
    for index in range(1, count + 1):
        (chapter_dir / f"{index:03d}.png").write_bytes(PNG_1X1)


def example_target(permission_note: str = "") -> TargetSpec:
    return TargetSpec(99, "Example Artist", "Example Series", [], permission_note)


def test_url_sources_require_download_flag_and_permission_note(tmp_path: Path) -> None:
    source = {"chapter": "1", "pages": ["https://example.test/001.png"]}
    with pytest.raises(ManifestError):
        chapters_from_source(example_target(), source, tmp_path, tmp_path, allow_downloads=False)
    with pytest.raises(ManifestError):
        chapters_from_source(example_target(), source, tmp_path, tmp_path, allow_downloads=True)


def test_url_sources_preserve_request_headers(tmp_path: Path) -> None:
    chapter = chapters_from_source(
        example_target(),
        {
            "chapter": "1",
            "pages": ["https://example.test/001.png"],
            "permission_note": "test source",
            "request_headers": {"Referer": "https://example.test/"},
        },
        tmp_path,
        tmp_path,
        allow_downloads=True,
    )
    assert chapter.pages[0].request_headers == {"Referer": "https://example.test/"}


def test_build_dataset_uses_mid_chapters_and_exact_page_count(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    for chapter in range(1, 6):
        write_pages(source_root / f"chapter-{chapter:03d}", 20)

    manifest_path = write_example_manifest(tmp_path, source_root)
    report = build_dataset(
        BuildConfig(
            sources_path=manifest_path,
            output_dir=tmp_path / "out",
            pages_per_artist=12,
            trim_start=5,
            trim_end=5,
            min_chapters=2,
            strict_targets=False,
        )
    )

    assert report["total_pages"] == 12
    assert report["targets"][0]["selected_chapters"] == ["Chapter 2", "Chapter 3"]
    assert_output_metadata(tmp_path / "out" / "metadata.jsonl")


def write_example_manifest(tmp_path: Path, source_root: Path) -> Path:
    manifest = {
        "targets": [
            {
                "label_id": 99,
                "artist": "Example Artist",
                "series": "Example Series",
                "sources": [
                    {"chapter": f"Chapter {chapter}", "path": str(source_root / f"chapter-{chapter:03d}")}
                    for chapter in range(1, 6)
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manga_sources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def assert_output_metadata(metadata_path: Path) -> None:
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 12
    assert {record["chapter"] for record in records} == {"Chapter 2", "Chapter 3"}
    assert all(6 <= record["original_page_index"] <= 15 for record in records)

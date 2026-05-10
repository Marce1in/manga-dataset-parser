from __future__ import annotations

import time
from pathlib import Path

import pytest

from manga_artist_dataset.build.page_work import PageProcessingResult, process_selected_pages
from manga_artist_dataset.build.writer import write_target_selection
from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.parallel import (
    CleanupChunkTask,
    clean_directory_parallel,
    clean_image_tasks_parallel,
)
from manga_artist_dataset.concurrency import bounded_worker_count, require_positive_worker_count
from manga_artist_dataset.io.http import HostDownloadLimiter
from manga_artist_dataset.models import BuildConfig, PageOutput, PageRef, TargetSelection, TargetSpec


class DelayedPageProcessor:
    def process(self, _target: TargetSpec, page: PageRef, _config: BuildConfig) -> PageProcessingResult:
        if page.page_index == 1:
            time.sleep(0.02)
        output = PageOutput(b"", None, ".png")
        return PageProcessingResult(page, [output], [], False)


def test_process_selected_pages_preserves_input_order(tmp_path: Path) -> None:
    target = target_spec()
    pages = page_refs(3)
    config = build_config(tmp_path, download_workers=2)

    results = list(process_selected_pages(target, pages, config, DelayedPageProcessor()))

    assert [result.page.page_index for result in results] == [1, 2, 3]


def test_write_target_selection_keeps_stable_output_numbering(tmp_path: Path) -> None:
    target = target_spec()
    selection = TargetSelection(target, page_refs(3), {})
    config = build_config(tmp_path, download_workers=2, pages_per_artist=3)

    records = write_target_selection(selection, config, DelayedPageProcessor())

    assert [record["original_page_index"] for record in records] == [1, 2, 3]
    assert [Path(str(record["output_path"])).stem[-4:] for record in records] == ["0001", "0002", "0003"]
    assert selection.report["considered_candidate_page_count"] == 3


def test_worker_count_validation_rejects_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="download_workers"):
        results = process_selected_pages(
            target_spec(),
            page_refs(1),
            build_config(tmp_path, download_workers=0),
            DelayedPageProcessor(),
        )
        list(results)
    with pytest.raises(ValueError, match="detector_workers"):
        clean_image_tasks_parallel(CleanupConfig(), [], 0)
    with pytest.raises(ValueError, match="custom_workers"):
        require_positive_worker_count(0, "custom_workers")
    with pytest.raises(ValueError, match="delay_seconds"):
        HostDownloadLimiter(-0.1)


def test_bounded_worker_count_caps_to_item_count() -> None:
    assert bounded_worker_count(8, 3, "scratch_workers") == 3
    assert bounded_worker_count(8, 0, "scratch_workers") == 1


def test_parallel_cleanup_returns_paths_in_stable_order(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    for name in ["003.png", "001.png", "002.png"]:
        write_placeholder_png(input_dir / "class_a" / name)

    saved = clean_directory_parallel(input_dir, output_dir, CleanupConfig(), 2, fake_clean_chunk)

    assert [path.name for path in saved] == ["001.png", "002.png", "003.png"]
    assert [path.read_text(encoding="utf-8") for path in saved] == ["001.png", "002.png", "003.png"]


def fake_clean_chunk(chunk: CleanupChunkTask) -> list[Path]:
    saved_paths: list[Path] = []
    for task in chunk.image_tasks:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        task.output_path.write_text(task.input_path.name, encoding="utf-8")
        saved_paths.append(task.output_path)
    return saved_paths


def target_spec() -> TargetSpec:
    return TargetSpec(1, "Example Artist", "Example Series", [])


def page_refs(count: int) -> list[PageRef]:
    return [PageRef("file", "Chapter 1", index, f"page-{index}.png", ".png") for index in range(1, count + 1)]


def build_config(tmp_path: Path, download_workers: int, pages_per_artist: int = 3) -> BuildConfig:
    return BuildConfig(
        sources_path=tmp_path / "sources.json",
        output_dir=tmp_path / "out",
        pages_per_artist=pages_per_artist,
        dry_run=True,
        strict_targets=False,
        download_workers=download_workers,
    )


def write_placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")

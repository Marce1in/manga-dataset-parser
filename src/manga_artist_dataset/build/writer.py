"""Write selected pages and build dataset reports."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from manga_artist_dataset.build.image_probe import image_dimensions
from manga_artist_dataset.build.page_work import (
    DefaultSelectedPageProcessor,
    PageProcessingResult,
    SelectedPageProcessor,
    process_selected_pages,
)
from manga_artist_dataset.errors import DatasetError
from manga_artist_dataset.image_formats import IMAGE_EXTENSIONS
from manga_artist_dataset.io.http import HostDownloadLimiter
from manga_artist_dataset.json_types import JsonObject
from manga_artist_dataset.models import BuildConfig, PageOutput, PageRef, TargetSelection, TargetSpec
from manga_artist_dataset.text_keys import slugify


def output_class_dir(output_dir: Path, target: TargetSpec) -> Path:
    return output_dir / f"{target.label_id:02d}_{slugify(target.artist)}"


def output_file_name(
    target: TargetSpec,
    page: PageRef,
    output_index: int,
    suffix_override: str | None = None,
    split_part: str | None = None,
) -> str:
    """Build a stable output filename for one selected page.

    Example:
        `output_file_name(target, page, 1)`.
    """
    suffix = normalized_suffix(suffix_override or page.suffix)
    part = f"__{slugify(split_part)}" if split_part else ""
    prefix = f"{target.label_id:02d}_{slugify(target.artist)}"
    chapter_slug = slugify(page.chapter_label, "chapter")
    return f"{prefix}__{chapter_slug}__p{page.page_index:04d}{part}__{output_index:04d}{suffix}"


def normalized_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return suffix
    return ".bin"


def prepare_output_dir(config: BuildConfig) -> None:
    if config.dry_run:
        return
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        if not config.overwrite:
            raise DatasetError(f"Output directory is not empty: {config.output_dir}. Use --overwrite.")
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)


def write_dataset(
    selections: list[TargetSelection],
    config: BuildConfig,
    page_processor: SelectedPageProcessor | None = None,
) -> JsonObject:
    """Write selected pages, metadata, and a report.

    Example:
        `report = write_dataset(selections, config)`.
    """
    prepare_output_dir(config)
    active_processor = page_processor or DefaultSelectedPageProcessor(
        HostDownloadLimiter(config.download_host_delay_seconds)
    )
    metadata_records: list[JsonObject] = []
    for selection in selections:
        metadata_records.extend(write_target_selection(selection, config, active_processor))
    report = dataset_report(selections, config, metadata_records)
    if not config.dry_run:
        write_metadata_and_report(config.output_dir, metadata_records, report)
    return report


def write_target_selection(
    selection: TargetSelection,
    config: BuildConfig,
    page_processor: SelectedPageProcessor,
) -> list[JsonObject]:
    target_dir = output_class_dir(config.output_dir, selection.target)
    if not config.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    state = TargetWriteState()
    records = collect_target_records(selection, config, target_dir, state, page_processor)
    update_selection_report(selection, state)
    ensure_target_page_count(selection, config, state.written_count)
    return records


class TargetWriteState:
    def __init__(self) -> None:
        self.output_index = 1
        self.skipped: dict[str, int] = {}
        self.considered_pages = 0
        self.split_pages = 0

    @property
    def written_count(self) -> int:
        return self.output_index - 1


def collect_target_records(
    selection: TargetSelection,
    config: BuildConfig,
    target_dir: Path,
    state: TargetWriteState,
    page_processor: SelectedPageProcessor,
) -> list[JsonObject]:
    records: list[JsonObject] = []
    results = process_selected_pages(selection.target, selection.selected_pages, config, page_processor)
    for result in results:
        state.considered_pages += 1
        records.extend(records_for_processed_page(selection.target, result, config, target_dir, state))
        if state.output_index > config.pages_per_artist:
            break
    return records


def records_for_processed_page(
    target: TargetSpec,
    result: PageProcessingResult,
    config: BuildConfig,
    target_dir: Path,
    state: TargetWriteState,
) -> list[JsonObject]:
    record_skips(state, result.rejected, result.did_split)
    return write_page_outputs(target, result.page, result.outputs, config, target_dir, state)


def record_skips(state: TargetWriteState, rejected: list[str], did_split: bool) -> None:
    state.split_pages += int(did_split)
    for reason in rejected:
        state.skipped[reason] = state.skipped.get(reason, 0) + 1


def write_page_outputs(
    target: TargetSpec,
    page: PageRef,
    outputs: list[PageOutput],
    config: BuildConfig,
    target_dir: Path,
    state: TargetWriteState,
) -> list[JsonObject]:
    records: list[JsonObject] = []
    for output in outputs:
        if state.output_index > config.pages_per_artist:
            break
        records.append(write_one_output(target, page, output, config, target_dir, state.output_index))
        state.output_index += 1
    return records


def write_one_output(
    target: TargetSpec,
    page: PageRef,
    output: PageOutput,
    config: BuildConfig,
    target_dir: Path,
    output_index: int,
) -> JsonObject:
    output_path = target_dir / output_file_name(target, page, output_index, output.suffix_override, output.split_part)
    if config.dry_run:
        return metadata_record(target, page, output_path, output.split_part, None, None, None, 0)
    output_path.write_bytes(output.content)
    width, height = image_dimensions(output.content)
    return metadata_record(
        target,
        page,
        output_path,
        output.split_part,
        hashlib.sha256(output.content).hexdigest(),
        width,
        height,
        len(output.content),
    )


def metadata_record(
    target: TargetSpec,
    page: PageRef,
    output_path: Path,
    split_part: str | None,
    sha256: str | None,
    width: int | None,
    height: int | None,
    bytes_written: int,
) -> JsonObject:
    return {
        "label_id": target.label_id,
        "artist": target.artist,
        "series": target.series,
        "chapter": page.chapter_label,
        "original_page_index": page.page_index,
        "split_part": split_part,
        "output_path": str(output_path),
        "source_ref": page.source_ref,
        "sha256": sha256,
        "width": width,
        "height": height,
        "bytes": bytes_written,
    }


def update_selection_report(selection: TargetSelection, state: TargetWriteState) -> None:
    selection.report["selected_page_count"] = state.written_count
    selection.report["considered_candidate_page_count"] = state.considered_pages
    selection.report["filter_skips"] = state.skipped
    selection.report["split_double_spread_source_pages"] = state.split_pages


def ensure_target_page_count(selection: TargetSelection, config: BuildConfig, written_count: int) -> None:
    if written_count >= config.pages_per_artist or config.allow_short:
        return
    prefix = f"{selection.target.artist} / {selection.target.series}"
    message = f"{prefix}: only wrote {written_count}; need {config.pages_per_artist}."
    raise DatasetError(message)


def dataset_report(
    selections: list[TargetSelection],
    config: BuildConfig,
    metadata_records: list[JsonObject],
) -> JsonObject:
    return {
        "config": build_report_config(config),
        "targets": [selection.report for selection in selections],
        "total_pages": len(metadata_records),
    }


def build_report_config(config: BuildConfig) -> JsonObject:
    return {
        "sources": str(config.sources_path),
        "output": str(config.output_dir),
        "pages_per_artist": config.pages_per_artist,
        "trim_start": config.trim_start,
        "trim_end": config.trim_end,
        "min_chapters": config.min_chapters,
        "seed": config.seed,
        "dry_run": config.dry_run,
        "allow_downloads": config.allow_downloads,
        "allow_short": config.allow_short,
        "use_all_sources": config.use_all_sources,
        "filter_color_pages": config.filter_color_pages,
        "filter_double_spreads": config.filter_double_spreads,
        "split_double_spreads": config.split_double_spreads,
        "split_double_spread_label_ids": sorted(config.split_double_spread_label_ids),
        "download_workers": config.download_workers,
        "download_host_delay_seconds": config.download_host_delay_seconds,
    }


def write_metadata_and_report(output_dir: Path, metadata_records: list[JsonObject], report: JsonObject) -> None:
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as metadata_file:
        for record in metadata_records:
            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

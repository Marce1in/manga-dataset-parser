"""Dataset build orchestration."""

from __future__ import annotations

import tempfile
from pathlib import Path

from manga_artist_dataset.build.chapter_sources import load_chapters_for_target
from manga_artist_dataset.build.manifest import (
    load_manifest,
    validate_expected_targets,
)
from manga_artist_dataset.build.selection import choose_pages
from manga_artist_dataset.build.writer import write_dataset
from manga_artist_dataset.errors import DatasetError
from manga_artist_dataset.json_types import JsonObject
from manga_artist_dataset.models import BuildConfig, TargetSelection, TargetSpec


def build_dataset(config: BuildConfig) -> JsonObject:
    """Build a page-level dataset from an explicit source manifest.

    Example:
        `report = build_dataset(BuildConfig(sources_path=path, output_dir=out))`.
    """
    targets = load_manifest(config.sources_path)
    validate_build_inputs(targets, config)
    with tempfile.TemporaryDirectory(prefix="manga_dataset_") as temp_name:
        selections = select_targets(targets, config.sources_path.parent, Path(temp_name), config)
        return write_dataset(selections, config)


def validate_build_inputs(targets: list[TargetSpec], config: BuildConfig) -> None:
    if config.strict_targets:
        validate_expected_targets(targets)


def select_targets(
    targets: list[TargetSpec],
    manifest_dir: Path,
    temp_dir: Path,
    config: BuildConfig,
) -> list[TargetSelection]:
    selections = [select_target(target, manifest_dir, temp_dir, config) for target in targets]
    raise_if_selection_errors(selections)
    return selections


def select_target(
    target: TargetSpec,
    manifest_dir: Path,
    temp_dir: Path,
    config: BuildConfig,
) -> TargetSelection:
    chapters = load_chapters_for_target(target, manifest_dir, temp_dir, config.allow_downloads)
    return choose_pages(target, chapters, config)


def raise_if_selection_errors(selections: list[TargetSelection]) -> None:
    errors = formatted_selection_errors(selections)
    if errors:
        raise DatasetError("Dataset cannot satisfy the requested counts:\n" + "\n".join(errors))


def formatted_selection_errors(selections: list[TargetSelection]) -> list[str]:
    errors: list[str] = []
    for selection in selections:
        for error in selection.errors:
            prefix = f"{selection.target.artist} / {selection.target.series}"
            errors.append(f"  - {prefix}: {error}")
    return errors

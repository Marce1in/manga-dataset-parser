"""Source manifest parsing and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from manga_artist_dataset.errors import ManifestError
from manga_artist_dataset.expected_targets import EXPECTED_TARGETS
from manga_artist_dataset.json_types import JsonObject, JsonValue
from manga_artist_dataset.models import TargetSpec
from manga_artist_dataset.text_keys import slugify


def load_manifest(path: Path) -> list[TargetSpec]:
    """Load typed target specs from a source manifest.

    Example:
        `targets = load_manifest(Path("manifests/manga_sources.polished.json"))`.
    """
    raw = read_manifest_json(path)
    raw_targets = manifest_targets(raw, path)
    return [target_from_json(index, item) for index, item in enumerate(raw_targets, start=1)]


def read_manifest_json(path: Path) -> JsonValue:
    try:
        return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path} is not valid JSON: {exc}") from exc


def manifest_targets(raw: JsonValue, path: Path) -> list[JsonObject]:
    if isinstance(raw, list):
        return ensure_object_list(raw, f"{path} top-level list")
    if isinstance(raw, dict) and isinstance(raw.get("targets"), list):
        return ensure_object_list(raw["targets"], f"{path} targets")
    raise ManifestError(f"{path} must be a JSON list or object with a 'targets' list.")


def ensure_object_list(values: list[JsonValue], context: str) -> list[JsonObject]:
    objects: list[JsonObject] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ManifestError(f"{context} item #{index} must be an object.")
        objects.append(value)
    return objects


def target_from_json(index: int, item: JsonObject) -> TargetSpec:
    missing = [name for name in ("label_id", "artist", "series", "sources") if name not in item]
    if missing:
        raise ManifestError(f"Target #{index} is missing required keys: {', '.join(missing)}.")
    if not isinstance(item["sources"], list):
        raise ManifestError(f"Target #{index} sources must be a list of source objects.")
    return TargetSpec(
        label_id=int(str(item["label_id"])),
        artist=str(item["artist"]),
        series=str(item["series"]),
        sources=ensure_object_list(item["sources"], f"target #{index} sources"),
        permission_note=str(item.get("permission_note") or ""),
    )


def validate_expected_targets(targets: list[TargetSpec]) -> None:
    """Validate the canonical ten manga artist labels.

    Example:
        `validate_expected_targets(load_manifest(path))`.
    """
    target_keys = {(target.label_id, slugify(target.artist), slugify(target.series)) for target in targets}
    missing = missing_expected_target_labels(target_keys)
    if missing:
        raise ManifestError("Manifest is missing expected targets:\n" + "\n".join(missing))


def missing_expected_target_labels(target_keys: set[tuple[int, str, str]]) -> list[str]:
    missing: list[str] = []
    for label_id, artist, series in EXPECTED_TARGETS:
        key = (label_id, slugify(artist), slugify(series))
        if key not in target_keys:
            missing.append(f"  - {label_id}: {artist} - {series}")
    return missing


def source_permission_note(target: TargetSpec, source: JsonObject) -> str:
    return str(source.get("permission_note") or target.permission_note or "").strip()

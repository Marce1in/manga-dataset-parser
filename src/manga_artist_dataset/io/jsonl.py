"""JSON and JSONL file boundary helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from manga_artist_dataset.json_types import JsonObject


def load_json_object(path: Path) -> JsonObject:
    """Load a JSON object from disk.

    Example:
        `load_json_object(Path("dataset_report.json"))`.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{path} must contain a JSON object."
        raise ValueError(msg)
    return cast(JsonObject, raw)


def write_json_object(path: Path, value: JsonObject) -> None:
    """Write a JSON object with stable UTF-8 formatting.

    Example:
        `write_json_object(Path("report.json"), {"total": 1})`.
    """
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[JsonObject]:
    """Load JSONL records from disk.

    Example:
        `records = load_jsonl(Path("metadata.jsonl"))`.
    """
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(_as_object(json.loads(line), path))
    return rows


def write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    """Write JSONL records with one object per line.

    Example:
        `write_jsonl(Path("metadata.jsonl"), rows)`.
    """
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _as_object(value: object, path: Path) -> JsonObject:
    if isinstance(value, dict):
        return cast(JsonObject, value)
    msg = f"{path} must contain only JSON objects in JSONL rows."
    raise ValueError(msg)

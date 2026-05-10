"""Deterministic train/test splitting for cleaned manga datasets."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from manga_artist_dataset.json_types import JsonObject

DatasetSplitName = Literal["train", "test"]
DEFAULT_TRAIN_FRACTION = 0.8


@dataclass(frozen=True)
class DatasetSplitAssignment:
    """Train/test decision for one metadata row.

    Example:
        `DatasetSplitAssignment("test", "20% anchor")`.
    """

    split_name: DatasetSplitName
    split_group: str


@dataclass(frozen=True)
class IndexedSplitRecord:
    """Metadata row with its original list position.

    Example:
        `IndexedSplitRecord(0, row)`.
    """

    index: int
    record: JsonObject


def assign_train_test_splits(records: list[JsonObject], train_fraction: float) -> list[DatasetSplitAssignment]:
    """Return deterministic split assignments in input order.

    Example:
        `assign_train_test_splits(records, 0.8)[0].split_name`.
    """
    validate_train_fraction(train_fraction)
    assignments = [DatasetSplitAssignment("train", split_group_for_record(record)) for record in records]
    for class_records in records_by_class(records).values():
        for index in selected_test_indices(class_records, train_fraction):
            group = split_group_for_record(records[index])
            assignments[index] = DatasetSplitAssignment("test", group)
    return assignments


def validate_train_fraction(value: float) -> None:
    """Validate the configured train fraction.

    Example:
        `validate_train_fraction(0.8)`.
    """
    if 0 < value < 1:
        return
    raise ValueError(f"train_fraction must be > 0 and < 1; got {value}.")


def records_by_class(records: list[JsonObject]) -> dict[str, list[IndexedSplitRecord]]:
    """Group records by class slug while preserving input positions.

    Example:
        `records_by_class(records)["01_artist"]`.
    """
    grouped: dict[str, list[IndexedSplitRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[class_slug_for_record(record)].append(IndexedSplitRecord(index, record))
    return dict(grouped)


def selected_test_indices(class_records: list[IndexedSplitRecord], train_fraction: float) -> set[int]:
    """Choose test rows for one class using balanced strategy groups.

    Example:
        `indices = selected_test_indices(records, 0.8)`.
    """
    test_total = len(class_records) - math.floor(len(class_records) * train_fraction)
    if test_total <= 0:
        return set()
    grouped = records_by_split_group(class_records)
    quotas = test_quotas_by_group(grouped, test_total)
    return {record.index for key, quota in quotas.items() for record in select_group_test_records(grouped[key], quota)}


def records_by_split_group(records: list[IndexedSplitRecord]) -> dict[str, list[IndexedSplitRecord]]:
    """Group class records by source strategy group.

    Example:
        `groups = records_by_split_group(records)`.
    """
    grouped: dict[str, list[IndexedSplitRecord]] = defaultdict(list)
    for record in records:
        grouped[split_group_for_record(record.record)].append(record)
    return dict(sorted(grouped.items(), key=lambda item: split_group_sort_key(item[0])))


def test_quotas_by_group(grouped: dict[str, list[IndexedSplitRecord]], test_total: int) -> dict[str, int]:
    """Allocate test quota equally across split groups, redistributing overflow.

    Example:
        `quotas = test_quotas_by_group(groups, 12)`.
    """
    keys = list(grouped)
    quotas = {key: min(len(grouped[key]), test_total // len(keys)) for key in keys}
    remaining = test_total - sum(quotas.values())
    while remaining > 0:
        key = next_group_with_capacity(keys, grouped, quotas)
        quotas[key] += 1
        remaining -= 1
    return quotas


def next_group_with_capacity(
    keys: list[str],
    grouped: dict[str, list[IndexedSplitRecord]],
    quotas: dict[str, int],
) -> str:
    candidates = [key for key in keys if quotas[key] < len(grouped[key])]
    if not candidates:
        total = sum(len(value) for value in grouped.values())
        raise ValueError(f"Cannot allocate test split; groups only contain {total} records.")
    return max(candidates, key=lambda key: (len(grouped[key]) - quotas[key], _reverse_sort_rank(key)))


def select_group_test_records(records: list[IndexedSplitRecord], quota: int) -> list[IndexedSplitRecord]:
    """Select representative rows across chapters and page positions.

    Example:
        `test_rows = select_group_test_records(records, 4)`.
    """
    if quota <= 0:
        return []
    ordered = round_robin_chapter_records(records)
    return ordered[:quota]


def round_robin_chapter_records(records: list[IndexedSplitRecord]) -> list[IndexedSplitRecord]:
    """Interleave representative rows from each chapter.

    Example:
        `ordered = round_robin_chapter_records(records)`.
    """
    queues = [spread_records(chapter_records) for chapter_records in records_by_chapter(records).values()]
    selected: list[IndexedSplitRecord] = []
    while any(queues):
        for queue in queues:
            if queue:
                selected.append(queue.pop(0))
    return selected


def records_by_chapter(records: list[IndexedSplitRecord]) -> dict[str, list[IndexedSplitRecord]]:
    """Group records by chapter label in stable order.

    Example:
        `chapters = records_by_chapter(records)`.
    """
    grouped: dict[str, list[IndexedSplitRecord]] = defaultdict(list)
    for record in sorted(records, key=record_sort_key):
        grouped[chapter_label(record.record)].append(record)
    return dict(grouped)


def spread_records(records: list[IndexedSplitRecord]) -> list[IndexedSplitRecord]:
    """Return records in middle-out order across page positions.

    Example:
        `spread_records(records)[0]`.
    """
    ordered = sorted(records, key=record_sort_key)
    return [ordered[index] for index in middle_out_indices(len(ordered))]


def middle_out_indices(count: int) -> list[int]:
    """Return list indices ordered from middle pages outward.

    Example:
        `middle_out_indices(5) == [2, 1, 3, 0, 4]`.
    """
    if count <= 0:
        return []
    midpoint = (count - 1) / 2
    return sorted(range(count), key=lambda index: (abs(index - midpoint), index))


def split_group_for_record(record: JsonObject) -> str:
    """Return the strategy group encoded in a chapter label.

    Example:
        `split_group_for_record({"chapter": "20% anchor: Chapter 1"})`.
    """
    chapter = chapter_label(record)
    for prefix in strategy_group_prefixes():
        if chapter.startswith(prefix):
            return prefix
    return "other"


def strategy_group_prefixes() -> tuple[str, ...]:
    return ("20% anchor", "50% anchor", "90% anchor", "Part 2 midpoint", "Part 4 midpoint", "Part 7 midpoint")


def split_group_sort_key(value: str) -> tuple[int, str]:
    order = {name: index for index, name in enumerate(strategy_group_prefixes())}
    return order.get(value, len(order)), value


def _reverse_sort_rank(value: str) -> int:
    return -split_group_sort_key(value)[0]


def class_slug_for_record(record: JsonObject) -> str:
    value = record.get("output_path")
    if not isinstance(value, str):
        raise ValueError(f"Record output_path must be a string; got {value!r}.")
    return Path(value).parent.name


def chapter_label(record: JsonObject) -> str:
    value = record.get("chapter")
    return str(value) if value else "unknown"


def record_sort_key(record: IndexedSplitRecord) -> tuple[str, int, str]:
    page_index = record.record.get("original_page_index", 0)
    return chapter_label(record.record), int(page_index), str(record.record.get("output_path", ""))


def split_counts(records: list[JsonObject]) -> JsonObject:
    """Count records by split name.

    Example:
        `split_counts(records)["train"]`.
    """
    counts = Counter(str(record.get("split")) for record in records)
    return {"test": counts.get("test", 0), "train": counts.get("train", 0)}


def class_split_counts(records: list[JsonObject]) -> JsonObject:
    """Count split names within each class.

    Example:
        `class_split_counts(records)["01_artist"]["test"]`.
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[class_slug_for_record(record)][str(record.get("split"))] += 1
    return {
        key: {"test": value.get("test", 0), "train": value.get("train", 0)} for key, value in sorted(counts.items())
    }


def expected_split_counts(total: int, train_fraction: float) -> dict[str, int]:
    """Return expected train/test counts for one class.

    Example:
        `expected_split_counts(60, 0.8) == {"test": 12, "train": 48}`.
    """
    validate_train_fraction(train_fraction)
    train_count = math.floor(total * train_fraction)
    return {"test": total - train_count, "train": train_count}

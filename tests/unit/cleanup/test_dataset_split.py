from __future__ import annotations

from collections import Counter

from manga_artist_dataset.cleanup.dataset_split import (
    DatasetSplitAssignment,
    assign_train_test_splits,
    class_split_counts,
    expected_split_counts,
    split_counts,
)
from manga_artist_dataset.json_types import JsonObject


def test_assign_train_test_splits_uses_equal_anchor_test_quota() -> None:
    records = split_records({"20% anchor": 20, "50% anchor": 20, "90% anchor": 20})

    assignments = assign_train_test_splits(records, 0.8)

    assert split_name_counts(assignments) == {"test": 12, "train": 48}
    assert assignment_test_group_counts(assignments) == {"20% anchor": 4, "50% anchor": 4, "90% anchor": 4}


def test_assign_train_test_splits_redistributes_when_anchor_is_small() -> None:
    records = split_records({"20% anchor": 2, "50% anchor": 29, "90% anchor": 29})

    assignments = assign_train_test_splits(records, 0.8)

    assert split_name_counts(assignments) == {"test": 12, "train": 48}
    assert assignment_test_group_counts(assignments) == {"20% anchor": 2, "50% anchor": 5, "90% anchor": 5}


def test_assign_train_test_splits_supports_jojo_midpoint_groups() -> None:
    records = split_records({"Part 2 midpoint": 21, "Part 4 midpoint": 21, "Part 7 midpoint": 18})

    assignments = assign_train_test_splits(records, 0.8)

    assert split_name_counts(assignments) == {"test": 12, "train": 48}
    assert assignment_test_group_counts(assignments) == {
        "Part 2 midpoint": 4,
        "Part 4 midpoint": 4,
        "Part 7 midpoint": 4,
    }


def test_assign_train_test_splits_is_deterministic() -> None:
    records = split_records({"20% anchor": 20, "50% anchor": 20, "90% anchor": 20})

    first = assign_train_test_splits(records, 0.8)
    second = assign_train_test_splits(records, 0.8)

    assert first == second


def test_split_count_helpers_include_train_and_test_keys() -> None:
    records = split_records({"20% anchor": 2})
    assignments = assign_train_test_splits(records, 0.8)
    assigned_records = [
        record | {"split": assignment.split_name} for record, assignment in zip(records, assignments, strict=True)
    ]

    assert expected_split_counts(2, 0.8) == {"test": 1, "train": 1}
    assert split_counts(assigned_records) == {"test": 1, "train": 1}
    assert class_split_counts(assigned_records) == {"01_example_artist": {"test": 1, "train": 1}}


def split_records(group_counts: dict[str, int]) -> list[JsonObject]:
    records: list[JsonObject] = []
    for group, count in group_counts.items():
        records.extend(split_group_records(group, count))
    return records


def split_group_records(group: str, count: int) -> list[JsonObject]:
    return [split_record(group, index) for index in range(1, count + 1)]


def split_record(group: str, index: int) -> JsonObject:
    chapter_number = 1 + (index % 3)
    return {
        "artist": "Example Artist",
        "chapter": f"{group}: Chapter {chapter_number}",
        "original_page_index": index,
        "output_path": f"artifacts/datasets/polished_pages/01_example_artist/page_{group}_{index:03d}.png",
    }


def split_name_counts(assignments: list[DatasetSplitAssignment]) -> dict[str, int]:
    return dict(Counter(assignment.split_name for assignment in assignments))


def assignment_test_group_counts(assignments: list[DatasetSplitAssignment]) -> dict[str, int]:
    counts = Counter(assignment.split_group for assignment in assignments if assignment.split_name == "test")
    return dict(sorted(counts.items()))

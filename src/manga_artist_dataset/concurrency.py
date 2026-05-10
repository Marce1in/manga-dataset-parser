"""Small concurrency validation helpers."""

from __future__ import annotations


def require_positive_worker_count(value: int, name: str) -> int:
    """Return a valid worker count or raise a value-specific error.

    Example:
        `require_positive_worker_count(4, "download_workers") == 4`.
    """
    if value >= 1:
        return value
    raise ValueError(f"{name} must be >= 1; got {value}.")


def bounded_worker_count(requested: int, item_count: int, name: str) -> int:
    """Cap workers to the amount of available work.

    Example:
        `bounded_worker_count(8, 3, "scratch_workers") == 3`.
    """
    require_positive_worker_count(requested, name)
    if item_count < 0:
        raise ValueError(f"item_count must be >= 0; got {item_count}.")
    return min(requested, max(item_count, 1))

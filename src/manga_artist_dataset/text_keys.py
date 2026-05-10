"""Stable text normalization and sorting helpers."""

from __future__ import annotations

import re
import unicodedata

type NaturalPart = tuple[int, float | str]


def normalize_text(value: str) -> str:
    """Return ASCII case-folded text for slugs and comparisons.

    Example:
        `normalize_text("JoJo's") == "jojo's"`.
    """
    normalized = unicodedata.normalize("NFKD", value.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'"))
    return normalized.encode("ascii", "ignore").decode("ascii").casefold()


def slugify(value: str, fallback: str = "item") -> str:
    """Return a filesystem-safe slug for a display label.

    Example:
        `slugify("Fullmetal Alchemist") == "fullmetal_alchemist"`.
    """
    normalized = normalize_text(value)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or fallback


def natural_key(value: str) -> tuple[NaturalPart, ...]:
    """Sort strings with embedded numbers in reading order.

    Example:
        `natural_key("page 10") > natural_key("page 2")`.
    """
    key: list[NaturalPart] = []
    for part in re.split(r"(\d+(?:\.\d+)?)", normalize_text(value)):
        if not part:
            continue
        key.append(natural_key_part(part))
    return tuple(key)


def natural_key_part(part: str) -> NaturalPart:
    if re.fullmatch(r"\d+(?:\.\d+)?", part):
        return (0, float(part))
    return (1, part)

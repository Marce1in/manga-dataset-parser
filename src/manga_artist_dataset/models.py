"""Typed domain models for dataset construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from manga_artist_dataset.json_types import JsonObject

PageKind = Literal["file", "zip", "url", "pdf"]


@dataclass(frozen=True)
class TargetSpec:
    label_id: int
    artist: str
    series: str
    sources: list[JsonObject]
    permission_note: str = ""


@dataclass(frozen=True)
class PageRef:
    kind: PageKind
    chapter_label: str
    page_index: int
    source_ref: str
    suffix: str
    path: Path | None = None
    archive_member: str | None = None
    url: str | None = None
    pdf_page_index: int | None = None
    request_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PageOutput:
    content: bytes
    split_part: str | None
    suffix_override: str | None


@dataclass
class Chapter:
    label: str
    source_ref: str
    pages: list[PageRef]
    sort_key: tuple[object, ...]
    sample_group: str | None = None

    def usable_pages(self, trim_start: int, trim_end: int) -> list[PageRef]:
        end = len(self.pages) - trim_end if trim_end else len(self.pages)
        if trim_start >= end:
            return []
        return self.pages[trim_start:end]


@dataclass(frozen=True)
class BuildConfig:
    sources_path: Path
    output_dir: Path
    pages_per_artist: int = 40
    trim_start: int = 5
    trim_end: int = 5
    min_chapters: int = 2
    seed: int = 42
    allow_downloads: bool = False
    allow_short: bool = False
    dry_run: bool = False
    overwrite: bool = False
    strict_targets: bool = True
    use_all_sources: bool = False
    filter_color_pages: bool = False
    filter_double_spreads: bool = False
    split_double_spreads: bool = False
    split_double_spread_label_ids: set[int] = field(default_factory=set)


@dataclass
class TargetSelection:
    target: TargetSpec
    selected_pages: list[PageRef]
    report: JsonObject
    errors: list[str] = field(default_factory=list)

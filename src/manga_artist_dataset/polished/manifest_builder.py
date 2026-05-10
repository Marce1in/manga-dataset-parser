#!/usr/bin/env python3
"""Generate the stricter 60-page manga artist source manifest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from manga_artist_dataset.polished.source_clients import (
    TRIM_END,
    TRIM_START,
    Anchor,
    anchor_chapters,
    chapter_number,
    cubari_sources_for_anchor,
    fetch_cubari_manifest,
    fetch_mangadex_chapters,
    github_zip_sources_for_anchor,
    mangadex_sources_for_anchor,
    mangapill_sources_for_naruto,
)

ANCHOR_LABELS = ["20% anchor", "90% anchor", "50% anchor"]


@dataclass(frozen=True)
class TargetEntry:
    label_id: int
    artist: str
    series: str
    total: int | str
    sources: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    note: str


def build_manifest(min_usable_pages: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the polished manifest and the chapter plan.

    Example:
        `manifest, plan = build_manifest(30)`.
    """
    entries = build_target_entries(min_usable_pages)
    manifest = {"targets": [entry_manifest(entry) for entry in entries]}
    plan = base_plan(min_usable_pages)
    plan["targets"] = [entry_plan(entry) for entry in entries]
    return manifest, plan


def build_target_entries(min_usable_pages: int) -> list[TargetEntry]:
    return [
        fullmetal_alchemist(min_usable_pages),
        monster(min_usable_pages),
        chainsaw_man(min_usable_pages),
        vagabond(min_usable_pages),
        berserk(min_usable_pages),
        naruto(min_usable_pages),
        one_piece(min_usable_pages),
        dragon_ball(min_usable_pages),
        jojo(min_usable_pages),
        jujutsu_kaisen(min_usable_pages),
    ]


def base_plan(min_usable_pages: int) -> dict[str, Any]:
    return {
        "rule": "ceil(20% * total), ceil(90% * total), ceil(50% * total); JoJo uses midpoint of parts 2, 4, and 7.",
        "trim_start": TRIM_START,
        "trim_end": TRIM_END,
        "min_usable_pages_per_anchor": min_usable_pages,
        "targets": [],
    }


def entry_manifest(entry: TargetEntry) -> dict[str, Any]:
    return {
        "label_id": entry.label_id,
        "artist": entry.artist,
        "series": entry.series,
        "permission_note": entry.note,
        "sources": entry.sources,
    }


def entry_plan(entry: TargetEntry) -> dict[str, Any]:
    return {
        "label_id": entry.label_id,
        "artist": entry.artist,
        "series": entry.series,
        "chapter_total_used": entry.total,
        "anchors": entry.rows,
        "source_count": len(entry.sources),
    }


def mangadex_target(
    label_id: int,
    artist: str,
    series: str,
    manga_id: str,
    language: str,
    total: int,
    min_usable_pages: int,
) -> TargetEntry:
    rows = fetch_mangadex_chapters(manga_id, language)
    sources, plan_rows = mangadex_anchor_sources(manga_id, language, rows, total, min_usable_pages, series)
    return TargetEntry(label_id, artist, series, total, sources, plan_rows, "MangaDex public API at-home pages.")


def mangadex_anchor_sources(
    manga_id: str,
    language: str,
    rows: dict[float, dict[str, Any]],
    total: int,
    min_usable_pages: int,
    series: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    for label, chapter in zip(ANCHOR_LABELS, anchor_chapters(total), strict=True):
        new_sources, row = mangadex_sources_for_anchor(
            manga_id=manga_id,
            language=language,
            rows=rows,
            anchor=Anchor(chapter, label),
            min_usable_pages=min_usable_pages,
            note=f"MangaDex public API at-home data-saver pages for {series}; language={language}.",
        )
        sources.extend(new_sources)
        plan_rows.append(row)
    return sources, plan_rows


def cubari_target(
    label_id: int,
    artist: str,
    series: str,
    manifest_url: str,
    total: int,
    min_usable_pages: int,
    note: str,
) -> TargetEntry:
    manifest = fetch_cubari_manifest(manifest_url)
    sources, rows = cubari_anchor_sources(manifest_url, manifest["chapters"], total, min_usable_pages, note)
    return TargetEntry(label_id, artist, series, total, sources, rows, "Public Cubari/GitHub manifest.")


def cubari_anchor_sources(
    manifest_url: str,
    chapters: dict[str, Any],
    total: int,
    min_usable_pages: int,
    note: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for label, chapter in zip(ANCHOR_LABELS, anchor_chapters(total), strict=True):
        new_sources, row = cubari_sources_for_anchor(
            manifest_url=manifest_url,
            chapters=chapters,
            anchor=Anchor(chapter, label),
            min_usable_pages=min_usable_pages,
            note=note,
        )
        sources.extend(new_sources)
        rows.append(row)
    return sources, rows


def fullmetal_alchemist(min_usable_pages: int) -> TargetEntry:
    return mangadex_target(
        1,
        "Hiromu Arakawa",
        "Fullmetal Alchemist",
        "dd8a907a-3850-4f95-ba03-ba201a8399e3",
        "en",
        108,
        min_usable_pages,
    )


def monster(min_usable_pages: int) -> TargetEntry:
    return cubari_target(
        2,
        "Naoki Urasawa",
        "Monster",
        "https://raw.githubusercontent.com/Dinis-CM/MonsterCubari/main/monster.json",
        162,
        min_usable_pages,
        "Public Cubari manifest from Dinis-CM/MonsterCubari with direct page URLs.",
    )


def chainsaw_man(min_usable_pages: int) -> TargetEntry:
    manga_id = "a77742b1-befd-49a4-bff5-1ad4e6b0ef7b"
    rows = fetch_mangadex_chapters(manga_id, "vi")
    total = max(int(number) for number in rows if float(number).is_integer())
    sources, plan_rows = mangadex_anchor_sources(manga_id, "vi", rows, total, min_usable_pages, "Chainsaw Man")
    return TargetEntry(
        3, "Tatsuki Fujimoto", "Chainsaw Man", total, sources, plan_rows, "MangaDex public API at-home pages."
    )


def vagabond(min_usable_pages: int) -> TargetEntry:
    return mangadex_target(
        4,
        "Takehiko Inoue",
        "Vagabond",
        "d1a9fdeb-f713-407f-960c-8326b586e6fd",
        "es-la",
        327,
        min_usable_pages,
    )


def berserk(min_usable_pages: int) -> TargetEntry:
    sources, rows = github_anchor_sources(
        363,
        min_usable_pages,
        "s1ddly/Berserk-DL",
        "Downloads/Images/berserk-chapter-{chapter:03d}.zip",
    )
    return TargetEntry(5, "Kentaro Miura", "Berserk", 363, sources, rows, "Public GitHub chapter ZIPs.")


def github_anchor_sources(
    total: int,
    min_usable_pages: int,
    repo: str,
    path_template: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for label, chapter in zip(ANCHOR_LABELS, anchor_chapters(total), strict=True):
        new_sources, row = github_zip_sources_for_anchor(
            repo=repo,
            path_template=path_template,
            anchor=Anchor(chapter, label),
            min_usable_pages=min_usable_pages,
            note=f"Public GitHub ZIP source from {repo}.",
        )
        sources.extend(new_sources)
        rows.append(row)
    return sources, rows


def naruto(min_usable_pages: int) -> TargetEntry:
    sources, rows = mangapill_sources_for_naruto(
        min_usable_pages=min_usable_pages,
        note="Mangapill chapter pages with direct CDN image URLs and referer header.",
    )
    return TargetEntry(6, "Masashi Kishimoto", "Naruto", 700, sources, rows, "Mangapill direct chapter page images.")


def one_piece(min_usable_pages: int) -> TargetEntry:
    manifest_url = "https://raw.githubusercontent.com/celsiusnarhwal/punk-records/main/cubari.json"
    manifest = fetch_cubari_manifest(manifest_url)
    total = max(int(chapter_number(key) or 0) for key in manifest["chapters"])
    sources, rows = cubari_anchor_sources(
        manifest_url,
        manifest["chapters"],
        total,
        min_usable_pages,
        "Public Cubari manifest from celsiusnarhwal/punk-records with direct page URLs.",
    )
    return TargetEntry(7, "Eiichiro Oda", "One Piece", total, sources, rows, "Public Cubari/GitHub manifest.")


def dragon_ball(min_usable_pages: int) -> TargetEntry:
    return mangadex_target(
        8,
        "Akira Toriyama",
        "Dragon Ball",
        "40bc649f-7b49-4645-859e-6cd94136e722",
        "vi",
        519,
        min_usable_pages,
    )


def jojo(min_usable_pages: int) -> TargetEntry:
    parts = [
        Anchor(35, "Part 2 midpoint", "61079efc-d1c4-4565-bbe6-de58e1d75fdf", "en", 69),
        Anchor(87, "Part 4 midpoint", "5ed1f8fc-a119-4cbc-aeae-26ce2bd3f838", "pt-br", 174),
        Anchor(48, "Part 7 midpoint", "b30dfee3-9d1d-4e8d-bfbe-8fcabc3c96f6", "en", 95),
    ]
    sources, rows = jojo_part_sources(parts, min_usable_pages)
    return TargetEntry(
        9,
        "Hirohiko Araki",
        "JoJo's Bizarre Adventure",
        "Part 2=69, Part 4=174, Part 7=95",
        sources,
        rows,
        "MangaDex public API at-home pages from non-colored part entries.",
    )


def jojo_part_sources(parts: list[Anchor], min_usable_pages: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for anchor in parts:
        new_sources, row = jojo_one_part(anchor, min_usable_pages)
        sources.extend(new_sources)
        rows.append(row)
    return sources, rows


def jojo_one_part(anchor: Anchor, min_usable_pages: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if anchor.manga_id is None or anchor.language is None:
        raise ValueError(f"JoJo anchor {anchor.label} must include manga_id and language.")
    rows = fetch_mangadex_chapters(anchor.manga_id, anchor.language)
    sources, row = mangadex_sources_for_anchor(
        manga_id=anchor.manga_id,
        language=anchor.language,
        rows=rows,
        anchor=anchor,
        min_usable_pages=min_usable_pages,
        note=f"MangaDex public API at-home data-saver pages for JoJo {anchor.label}; non-colored part entry.",
    )
    for source in sources:
        source["sample_group"] = anchor.label
    row["part_total"] = anchor.total_chapters
    return sources, row


def jujutsu_kaisen(min_usable_pages: int) -> TargetEntry:
    manifest_url = "https://raw.githubusercontent.com/mcradcliffe2490/hidden-inventory/main/cubari.json"
    sources, rows = cubari_anchor_sources(
        manifest_url,
        fetch_cubari_manifest(manifest_url)["chapters"],
        271,
        min_usable_pages,
        "Public Cubari GitHub manifest from mcradcliffe2490/hidden-inventory with TCB page URLs.",
    )
    add_sample_groups(sources)
    return TargetEntry(
        10,
        "Gege Akutami",
        "Jujutsu Kaisen",
        271,
        sources,
        rows,
        "Public Cubari/GitHub manifest with anchor-balanced sampling.",
    )


def add_sample_groups(sources: list[dict[str, Any]]) -> None:
    for source in sources:
        chapter = str(source.get("chapter") or "")
        source["sample_group"] = chapter.split(":", 1)[0]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the polished manga source manifest.")
    parser.add_argument("--output", type=Path, default=Path("manifests/manga_sources.polished.json"))
    parser.add_argument("--plan-output", type=Path, default=Path("artifacts/reports/chapter_plan.polished.json"))
    parser.add_argument("--min-usable-pages-per-anchor", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest, plan = build_manifest(args.min_usable_pages_per_anchor)
    write_manifest_outputs(args.output, args.plan_output, manifest, plan)
    print(
        json.dumps(
            {"manifest": str(args.output), "plan": str(args.plan_output), "targets": len(manifest["targets"])}, indent=2
        )
    )
    return 0


def write_manifest_outputs(output: Path, plan_output: Path, manifest: dict[str, Any], plan: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plan_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plan_output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

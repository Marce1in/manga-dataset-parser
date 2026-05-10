"""Chapter and page selection logic."""

from __future__ import annotations

import random

from manga_artist_dataset.json_types import JsonObject
from manga_artist_dataset.models import BuildConfig, Chapter, PageRef, TargetSelection, TargetSpec


def ranked_mid_chapters(chapters: list[Chapter]) -> list[Chapter]:
    """Rank chapters by closeness to the middle of the provided list.

    Example:
        `ranked_mid_chapters(chapters)[0]` is the midpoint chapter.
    """
    if not chapters:
        return []
    center = (len(chapters) - 1) / 2
    ranked = sorted(enumerate(chapters), key=lambda item: (abs(item[0] - center), item[0]))
    return [chapter for _, chapter in ranked]


def choose_pages(target: TargetSpec, chapters: list[Chapter], config: BuildConfig) -> TargetSelection:
    """Select candidate pages for one artist target.

    Example:
        `selection = choose_pages(target, chapters, config)`.
    """
    ranked = chapters if config.use_all_sources else ranked_mid_chapters(chapters)
    chapter_reports, selected_chapters, total_available = collect_usable_chapters(ranked, config)
    selected_pages = select_round_robin_pages(target, selected_chapters, total_available, config)
    errors = selection_errors(selected_chapters, selected_pages, config)
    return TargetSelection(
        target,
        selected_pages,
        selection_report(target, chapters, selected_chapters, chapter_reports, selected_pages),
        errors,
    )


def collect_usable_chapters(
    ranked: list[Chapter],
    config: BuildConfig,
) -> tuple[list[JsonObject], list[tuple[Chapter, list[PageRef]]], int]:
    reports: list[JsonObject] = []
    selected: list[tuple[Chapter, list[PageRef]]] = []
    total_available = 0
    for chapter in ranked:
        usable = chapter.usable_pages(config.trim_start, config.trim_end)
        reports.append(chapter_candidate_report(chapter, usable))
        if not usable:
            continue
        selected.append((chapter, usable))
        total_available += len(usable)
        if should_stop_chapter_selection(selected, total_available, config):
            break
    return reports, sorted(selected, key=lambda item: item[0].sort_key), total_available


def chapter_candidate_report(chapter: Chapter, usable: list[PageRef]) -> JsonObject:
    return {
        "chapter": chapter.label,
        "source": chapter.source_ref,
        "total_pages": len(chapter.pages),
        "usable_after_trim": len(usable),
    }


def should_stop_chapter_selection(
    selected: list[tuple[Chapter, list[PageRef]]],
    total_available: int,
    config: BuildConfig,
) -> bool:
    if config.use_all_sources:
        return False
    return len(selected) >= config.min_chapters and total_available >= config.pages_per_artist


def select_round_robin_pages(
    target: TargetSpec,
    selected_chapters: list[tuple[Chapter, list[PageRef]]],
    total_available: int,
    config: BuildConfig,
) -> list[PageRef]:
    limit = total_available if uses_post_selection_filters(config) else config.pages_per_artist
    grouped_pages = grouped_candidate_pages(selected_chapters)
    rng = random.Random(  # noqa: S311 - deterministic dataset sampling, not cryptographic randomness.
        f"{config.seed}:{target.label_id}:{target.artist}:{target.series}"
    )
    return round_robin_pages(list(grouped_pages.values()), limit, rng)


def uses_post_selection_filters(config: BuildConfig) -> bool:
    return config.filter_color_pages or config.filter_double_spreads


def grouped_candidate_pages(selected_chapters: list[tuple[Chapter, list[PageRef]]]) -> dict[str, list[PageRef]]:
    groups: dict[str, list[PageRef]] = {}
    for index, (chapter, pages) in enumerate(selected_chapters):
        group_key = chapter.sample_group or f"chapter:{index}:{chapter.label}"
        groups.setdefault(group_key, []).extend(pages)
    return groups


def grouped_candidate_labels(selected_chapters: list[tuple[Chapter, list[PageRef]]]) -> list[str]:
    return [chapter.sample_group or chapter.label for chapter, _ in selected_chapters]


def selection_errors(
    selected_chapters: list[tuple[Chapter, list[PageRef]]],
    selected_pages: list[PageRef],
    config: BuildConfig,
) -> list[str]:
    errors = raw_selection_errors(selected_chapters, selected_pages, config)
    return [] if config.allow_short else errors


def raw_selection_errors(
    selected_chapters: list[tuple[Chapter, list[PageRef]]],
    selected_pages: list[PageRef],
    config: BuildConfig,
) -> list[str]:
    errors: list[str] = []
    if len(selected_chapters) < config.min_chapters:
        errors.append(f"Only {len(selected_chapters)} usable chapter(s); need at least {config.min_chapters}.")
    if len(selected_pages) < config.pages_per_artist:
        errors.append(f"Only {len(selected_pages)} usable page(s); need {config.pages_per_artist}.")
    return errors


def selection_report(
    target: TargetSpec,
    chapters: list[Chapter],
    selected_chapters: list[tuple[Chapter, list[PageRef]]],
    chapter_reports: list[JsonObject],
    selected_pages: list[PageRef],
) -> JsonObject:
    return {
        "label_id": target.label_id,
        "artist": target.artist,
        "series": target.series,
        "chapter_count": len(chapters),
        "selected_chapters": [chapter.label for chapter, _ in selected_chapters],
        "selected_sample_groups": grouped_candidate_labels(selected_chapters),
        "selected_candidate_page_count": len(selected_pages),
        "chapter_candidates": chapter_reports,
    }


def round_robin_pages(chapter_pages: list[list[PageRef]], limit: int, rng: random.Random) -> list[PageRef]:
    """Shuffle pages per group and interleave groups until the limit is reached.

    Example:
        `round_robin_pages([[a], [b]], 2, random.Random(1))`.
    """
    queues = shuffled_queues(chapter_pages, rng)
    selected: list[PageRef] = []
    while queues and len(selected) < limit:
        selected.extend(pop_one_round(queues, limit - len(selected)))
        queues = [queue for queue in queues if queue]
    return selected


def shuffled_queues(chapter_pages: list[list[PageRef]], rng: random.Random) -> list[list[PageRef]]:
    queues: list[list[PageRef]] = []
    for pages in chapter_pages:
        queue = list(pages)
        rng.shuffle(queue)
        if queue:
            queues.append(queue)
    return queues


def pop_one_round(queues: list[list[PageRef]], remaining: int) -> list[PageRef]:
    selected: list[PageRef] = []
    for queue in queues:
        if len(selected) >= remaining:
            break
        selected.append(queue.pop(0))
    return selected

#!/usr/bin/env python3
"""Generate the stricter 60-page manga artist source manifest.

This keeps the source discovery logic out of the extraction script:

* compute the requested 20% / 90% / 50% chapter anchors
* use the special JoJo part 2 / 4 / 7 midpoint anchors
* expand MangaDex at-home and Cubari/Imgur sources into explicit page URLs
* add nearby chapters when a single chapter cannot provide enough pages after
  trimming the first and last five pages
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, cast

MANGADEX_API = "https://api.mangadex.org"
USER_AGENT = "pablo-ai-manga-school-dataset/0.2"
TRIM_START = 5
TRIM_END = 5
MANGAPILL_NARUTO_ID = 3069


@dataclass(frozen=True)
class Anchor:
    chapter: int
    label: str
    manga_id: str | None = None
    language: str | None = None
    total_chapters: int | None = None


def http_json(url: str, *, retries: int = 3, timeout: int = 45) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Could not fetch JSON from {url}: {last_error}")


def http_text(url: str, *, retries: int = 3, timeout: int = 45) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "User-Agent": USER_AGENT,
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return cast(bytes, response.read()).decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Could not fetch text from {url}: {last_error}")


def mangadex_get(path: str, params: dict[str, Any] | None = None) -> Any:
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params, doseq=True)
    return http_json(MANGADEX_API + path + query)


def chapter_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", text):
        return None
    return float(text)


def chapter_sort_key(value: str) -> tuple[int, float, str]:
    parsed = chapter_number(value)
    if parsed is None:
        return (1, 0.0, value)
    return (0, parsed, value)


def anchor_chapters(total: int) -> list[int]:
    return [
        max(1, math.ceil(total * 0.20)),
        max(1, math.ceil(total * 0.90)),
        max(1, math.ceil(total * 0.50)),
    ]


def usable_count(total_pages: int) -> int:
    return max(0, total_pages - TRIM_START - TRIM_END)


def fetch_mangadex_chapters(manga_id: str, language: str) -> dict[float, dict[str, Any]]:
    by_number: dict[float, dict[str, Any]] = {}
    offset = 0
    while True:
        payload = mangadex_get(
            "/chapter",
            {
                "manga": manga_id,
                "translatedLanguage[]": [language],
                "limit": 100,
                "offset": offset,
                "order[chapter]": "asc",
                "includeFutureUpdates": "0",
            },
        )
        data = payload.get("data", [])
        for item in data:
            attrs = item.get("attributes", {})
            number = chapter_number(attrs.get("chapter"))
            if number is None:
                continue
            if attrs.get("externalUrl") or attrs.get("isUnavailable"):
                continue
            pages = int(attrs.get("pages") or 0)
            if pages <= 0:
                continue
            existing = by_number.get(number)
            if existing is None or pages > int(existing["attributes"].get("pages") or 0):
                by_number[number] = item
        offset += len(data)
        if offset >= int(payload.get("total") or 0) or not data:
            break
        time.sleep(0.08)
    return by_number


def mangadex_page_urls(chapter_id: str) -> list[str]:
    payload = mangadex_get(f"/at-home/server/{chapter_id}")
    base_url = str(payload["baseUrl"]).rstrip("/")
    chapter = payload["chapter"]
    chapter_hash = chapter["hash"]
    files = chapter.get("dataSaver") or chapter.get("data") or []
    return [f"{base_url}/data-saver/{chapter_hash}/{filename}" for filename in files]


def select_nearby_rows(
    rows: dict[float, dict[str, Any]],
    anchor: int,
    min_usable_pages: int,
    max_chapters: int = 6,
) -> list[dict[str, Any]]:
    candidates = sorted(
        rows.values(),
        key=lambda item: (
            abs(float(item["attributes"]["chapter"]) - anchor),
            float(item["attributes"]["chapter"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    usable = 0
    for item in candidates:
        selected.append(item)
        usable += usable_count(int(item["attributes"].get("pages") or 0))
        if usable >= min_usable_pages or len(selected) >= max_chapters:
            break
    return sorted(selected, key=lambda item: chapter_sort_key(str(item["attributes"]["chapter"])))


def mangadex_sources_for_anchor(
    *,
    manga_id: str,
    language: str,
    rows: dict[float, dict[str, Any]],
    anchor: Anchor,
    min_usable_pages: int,
    note: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    selected = select_nearby_rows(rows, anchor.chapter, min_usable_pages)
    for item in selected:
        attrs = item["attributes"]
        chapter = str(attrs["chapter"])
        title = str(attrs.get("title") or "").strip()
        label = f"{anchor.label}: Chapter {chapter}"
        if title:
            label += f" - {title}"
        sources.append(
            {
                "chapter": label,
                "pages": mangadex_page_urls(item["id"]),
                "permission_note": note,
            }
        )
        time.sleep(0.08)
    return sources, {
        "anchor": anchor.chapter,
        "label": anchor.label,
        "selected_chapters": [str(item["attributes"]["chapter"]) for item in selected],
        "language": language,
        "source": "MangaDex",
        "manga_id": manga_id,
    }


def extract_cubari_pages(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item["url"] if isinstance(item, dict) else item) for item in value]
    if isinstance(value, str) and value.startswith("/proxy/api/imgur/chapter/"):
        album = value.rstrip("/").split("/")[-1]
        payload = http_json(f"https://cubari.moe/read/api/imgur/chapter/{album}/")
        return [str(item["src"]) for item in payload if isinstance(item, dict) and item.get("src")]
    return []


def cubari_chapter_pages(chapter: dict[str, Any]) -> list[str]:
    groups = chapter.get("groups") or {}
    for value in groups.values():
        pages = extract_cubari_pages(value)
        if pages:
            return pages
    return []


def fetch_cubari_manifest(url: str) -> dict[str, Any]:
    return cast(dict[str, Any], http_json(url))


def select_cubari_chapters(
    chapters: dict[str, Any],
    anchor: int,
    min_usable_pages: int,
    max_chapters: int = 7,
) -> list[tuple[str, dict[str, Any], list[str]]]:
    numeric = [(float(key), key, value) for key, value in chapters.items() if chapter_number(key) is not None]
    numeric.sort(key=lambda item: (abs(item[0] - anchor), item[0]))
    selected: list[tuple[str, dict[str, Any], list[str]]] = []
    usable = 0
    for _, key, chapter in numeric:
        pages = cubari_chapter_pages(chapter)
        if not pages:
            continue
        selected.append((key, chapter, pages))
        usable += usable_count(len(pages))
        if usable >= min_usable_pages or len(selected) >= max_chapters:
            break
    return sorted(selected, key=lambda item: chapter_sort_key(item[0]))


def cubari_sources_for_anchor(
    *,
    manifest_url: str,
    chapters: dict[str, Any],
    anchor: Anchor,
    min_usable_pages: int,
    note: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = select_cubari_chapters(chapters, anchor.chapter, min_usable_pages)
    sources: list[dict[str, Any]] = []
    for key, chapter, pages in selected:
        title = str(chapter.get("title") or "").strip()
        label = f"{anchor.label}: Chapter {key}"
        if title:
            label += f" - {title}"
        sources.append({"chapter": label, "pages": pages, "permission_note": note})
    return sources, {
        "anchor": anchor.chapter,
        "label": anchor.label,
        "selected_chapters": [key for key, _, _ in selected],
        "source": manifest_url,
    }


def mangapill_naruto_chapter_url(chapter: int) -> str:
    # Mangapill's Naruto chapter ids use the stable form observed as:
    # 3069-10140000 => Naruto chapter 140.
    return f"https://mangapill.com/chapters/{MANGAPILL_NARUTO_ID}-10{chapter:03d}000/naruto-chapter-{chapter}"


def mangapill_naruto_pages(chapter: int) -> list[str]:
    html = http_text(mangapill_naruto_chapter_url(chapter))
    urls = re.findall(r"<img[^>]+(?:data-src|src)=[\"']([^\"']+)[\"']", html, flags=re.IGNORECASE)
    pages = [url for url in urls if f"/file/mangap/{MANGAPILL_NARUTO_ID}/" in url]
    return pages


def mangapill_sources_for_naruto(
    *,
    min_usable_pages: int,
    note: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchors = [
        Anchor(140, "20% anchor"),
        Anchor(630, "90% anchor"),
        Anchor(350, "50% anchor"),
    ]
    sources: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    request_headers = {"Referer": "https://mangapill.com/"}
    for anchor in anchors:
        selected: list[tuple[int, list[str]]] = []
        usable = 0
        candidates = sorted(
            range(max(1, anchor.chapter - 4), anchor.chapter + 5),
            key=lambda chapter: (abs(chapter - anchor.chapter), chapter),
        )
        for chapter in candidates:
            pages = mangapill_naruto_pages(chapter)
            if not pages:
                continue
            selected.append((chapter, pages))
            usable += usable_count(len(pages))
            if usable >= min_usable_pages or len(selected) >= 7:
                break
            time.sleep(0.08)

        selected.sort(key=lambda item: item[0])
        for chapter, pages in selected:
            sources.append(
                {
                    "chapter": f"{anchor.label}: Chapter {chapter}",
                    "pages": pages,
                    "permission_note": note,
                    "request_headers": request_headers,
                    "sample_group": anchor.label,
                }
            )
        plan_rows.append(
            {
                "anchor": anchor.chapter,
                "label": anchor.label,
                "selected_chapters": [str(chapter) for chapter, _ in selected],
                "source": "https://mangapill.com",
            }
        )
    return sources, plan_rows


def github_zip_sources_for_anchor(
    *,
    repo: str,
    path_template: str,
    anchor: Anchor,
    min_usable_pages: int,
    note: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # ZIP page counts are only known after extraction, so take a compact window
    # around the anchor. Five Berserk chapters comfortably covers the 50-page
    # global target after trimming and filters.
    chapters = [anchor.chapter - 2, anchor.chapter - 1, anchor.chapter, anchor.chapter + 1, anchor.chapter + 2]
    chapters = [chapter for chapter in chapters if chapter > 0]
    sources = []
    for chapter in chapters:
        path = path_template.format(chapter=chapter)
        url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
        sources.append(
            {
                "chapter": f"{anchor.label}: Chapter {chapter}",
                "url": url,
                "permission_note": note,
            }
        )
    return sources, {
        "anchor": anchor.chapter,
        "label": anchor.label,
        "selected_chapters": [str(chapter) for chapter in chapters],
        "source": f"https://github.com/{repo}",
        "min_usable_pages": min_usable_pages,
    }


def github_pdf_sources_for_naruto(note: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Chapter-to-volume anchors are based on the original 700-chapter run.
    anchors = [
        (140, "20% anchor", 16),
        (630, "90% anchor", 66),
        (350, "50% anchor", 39),
    ]
    sources = []
    plan_rows = []
    for chapter, label, volume in anchors:
        sources.append(
            {
                "chapter": f"{label}: Chapter {chapter} / Volume {volume}",
                "url": f"https://raw.githubusercontent.com/Ghostasky/NarutoMangaPDF/main/{volume:02d}.pdf",
                "permission_note": note,
            }
        )
        plan_rows.append(
            {
                "anchor": chapter,
                "label": label,
                "selected_chapters": [f"volume {volume}"],
                "source": "https://github.com/Ghostasky/NarutoMangaPDF",
            }
        )
    return sources, plan_rows

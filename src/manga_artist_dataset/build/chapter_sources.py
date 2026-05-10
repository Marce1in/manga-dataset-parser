"""Expand manifest source entries into typed chapters."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from manga_artist_dataset.build.manifest import source_permission_note
from manga_artist_dataset.build.page_listers import (
    download_to_temp,
    list_archive_pages,
    list_folder_pages,
    list_pdf_pages,
    request_headers,
    url_page,
)
from manga_artist_dataset.build.source_paths import (
    chapter_label_for,
    is_archive_name,
    is_image_name,
    path_suffix,
    resolve_path,
    source_urls,
)
from manga_artist_dataset.errors import ManifestError
from manga_artist_dataset.image_formats import ARCHIVE_EXTENSIONS, IMAGE_EXTENSIONS
from manga_artist_dataset.json_types import JsonObject
from manga_artist_dataset.models import Chapter, PageRef, TargetSpec
from manga_artist_dataset.text_keys import natural_key


def chapters_from_source(
    target: TargetSpec,
    source: JsonObject,
    manifest_dir: Path,
    temp_dir: Path,
    allow_downloads: bool,
) -> Chapter:
    """Create one chapter from one manifest source entry.

    Example:
        `chapters_from_source(target, source, manifest_dir, temp_dir, True)`.
    """
    chapter_label = source_chapter_label(source)
    source_ref, pages = pages_from_source(target, source, chapter_label, manifest_dir, temp_dir, allow_downloads)
    return Chapter(chapter_label, source_ref, pages, natural_key(chapter_label), sample_group(source))


def source_chapter_label(source: JsonObject) -> str:
    fallback = "chapter"
    if source.get("path"):
        fallback = Path(str(source["path"])).stem
    if source.get("url"):
        fallback = Path(urllib.parse.urlparse(str(source["url"])).path).stem or "remote"
    return chapter_label_for(source, fallback)


def sample_group(source: JsonObject) -> str | None:
    value = source.get("sample_group")
    return str(value) if value else None


def pages_from_source(
    target: TargetSpec,
    source: JsonObject,
    chapter_label: str,
    manifest_dir: Path,
    temp_dir: Path,
    allow_downloads: bool,
) -> tuple[str, list[PageRef]]:
    if source.get("path"):
        return pages_from_path_source(source, chapter_label, manifest_dir)
    if source.get("url"):
        return pages_from_url_source(target, source, chapter_label, temp_dir, allow_downloads)
    return pages_from_explicit_urls(target, source, chapter_label, allow_downloads)


def pages_from_path_source(source: JsonObject, chapter_label: str, manifest_dir: Path) -> tuple[str, list[PageRef]]:
    path = resolve_path(str(source["path"]), manifest_dir)
    if not path.exists():
        raise ManifestError(f"Source path does not exist: {path}")
    return str(path), pages_for_existing_path(path, chapter_label)


def pages_for_existing_path(path: Path, chapter_label: str) -> list[PageRef]:
    if path.is_dir():
        return list_folder_pages(path, chapter_label)
    if path.suffix.lower() in ARCHIVE_EXTENSIONS:
        return list_archive_pages(path, chapter_label)
    if path.suffix.lower() == ".pdf":
        return list_pdf_pages(path, chapter_label)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return [PageRef("file", chapter_label, 1, str(path), path.suffix.lower(), path=path)]
    raise ManifestError(f"Unsupported source path type: {path}")


def pages_from_url_source(
    target: TargetSpec,
    source: JsonObject,
    chapter_label: str,
    temp_dir: Path,
    allow_downloads: bool,
) -> tuple[str, list[PageRef]]:
    url = str(source["url"])
    ensure_url_allowed(target, source, url, allow_downloads)
    if is_archive_name(url):
        return url, list_archive_pages(download_to_temp(url, temp_dir), chapter_label)
    if path_suffix(url) == ".pdf":
        return url, list_pdf_pages(download_to_temp(url, temp_dir), chapter_label)
    if is_image_name(url):
        return url, [url_page(1, chapter_label, url)]
    raise ManifestError(f"URL must point to an image, PDF, ZIP, or CBZ file: {url}")


def pages_from_explicit_urls(
    target: TargetSpec,
    source: JsonObject,
    chapter_label: str,
    allow_downloads: bool,
) -> tuple[str, list[PageRef]]:
    urls = source_urls(source)
    if not urls:
        raise ManifestError("Source must have 'path', 'url', 'pages', or 'urls'.")
    ensure_page_urls_allowed(target, source, allow_downloads)
    ensure_all_image_urls(urls)
    headers = request_headers(source)
    pages = [url_page(index, chapter_label, url, headers) for index, url in enumerate(urls, start=1)]
    return f"{len(urls)} explicit page URLs", pages


def ensure_url_allowed(target: TargetSpec, source: JsonObject, url: str, allow_downloads: bool) -> None:
    if not allow_downloads:
        raise ManifestError(f"URL source requires --allow-downloads: {url}")
    if not source_permission_note(target, source):
        raise ManifestError(f"URL source requires permission_note: {url}")


def ensure_page_urls_allowed(target: TargetSpec, source: JsonObject, allow_downloads: bool) -> None:
    if not allow_downloads:
        raise ManifestError(f"Page URL source for {target.artist} requires --allow-downloads.")
    if not source_permission_note(target, source):
        raise ManifestError(f"Page URL source for {target.artist} requires permission_note.")


def ensure_all_image_urls(urls: list[str]) -> None:
    bad_urls = [url for url in urls if not is_image_name(url)]
    if bad_urls:
        raise ManifestError(f"Explicit page URLs must be image files. First bad URL: {bad_urls[0]}")


def load_chapters_for_target(
    target: TargetSpec,
    manifest_dir: Path,
    temp_dir: Path,
    allow_downloads: bool,
) -> list[Chapter]:
    """Load all chapters for one target from its manifest sources.

    Example:
        `load_chapters_for_target(target, manifest_dir, temp_dir, True)`.
    """
    chapters = [
        chapters_from_source(target, source, manifest_dir, temp_dir, allow_downloads) for source in target.sources
    ]
    return sorted(chapters, key=lambda chapter: chapter.sort_key)

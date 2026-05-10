"""Page listing for local folders, archives, PDFs, and URL source entries."""

from __future__ import annotations

import hashlib
import importlib
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, cast

from manga_artist_dataset.build.source_paths import is_image_name, path_suffix
from manga_artist_dataset.errors import ManifestError
from manga_artist_dataset.image_formats import ARCHIVE_EXTENSIONS, IMAGE_EXTENSIONS
from manga_artist_dataset.io.http import fetch_url_bytes
from manga_artist_dataset.models import PageRef
from manga_artist_dataset.text_keys import natural_key


def list_folder_pages(path: Path, chapter_label: str) -> list[PageRef]:
    """List image pages inside a local chapter folder.

    Example:
        `list_folder_pages(Path("chapter"), "Chapter 1")`.
    """
    files = sorted(folder_image_files(path), key=lambda item: natural_key(str(item.relative_to(path))))
    return [file_page(index, chapter_label, file_path) for index, file_path in enumerate(files, start=1)]


def folder_image_files(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and not any(part.startswith(".") for part in candidate.relative_to(path).parts)
        and candidate.suffix.lower() in IMAGE_EXTENSIONS
    ]


def file_page(index: int, chapter_label: str, file_path: Path) -> PageRef:
    return PageRef("file", chapter_label, index, str(file_path), file_path.suffix.lower(), path=file_path)


def list_archive_pages(path: Path, chapter_label: str) -> list[PageRef]:
    """List image pages inside a ZIP or CBZ archive.

    Example:
        `list_archive_pages(Path("chapter.cbz"), "Chapter 1")`.
    """
    members = archive_image_members(path)
    return [archive_page(index, chapter_label, path, member) for index, member in enumerate(members, start=1)]


def archive_image_members(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if is_archive_image_member(name)]
    except zipfile.BadZipFile as exc:
        raise ManifestError(f"{path} is not a readable ZIP/CBZ archive.") from exc
    return sorted(names, key=natural_key)


def is_archive_image_member(name: str) -> bool:
    return (
        not name.endswith("/")
        and not any(part.startswith(".") for part in Path(name).parts)
        and Path(name).suffix.lower() in IMAGE_EXTENSIONS
    )


def archive_page(index: int, chapter_label: str, path: Path, member: str) -> PageRef:
    return PageRef(
        "zip", chapter_label, index, f"{path}!{member}", Path(member).suffix.lower(), path=path, archive_member=member
    )


def list_pdf_pages(path: Path, chapter_label: str) -> list[PageRef]:
    """List renderable pages inside a PDF source.

    Example:
        `list_pdf_pages(Path("chapter.pdf"), "Chapter 1")`.
    """
    fitz = import_fitz(path)
    with fitz.open(path) as document:
        count = int(document.page_count)
    return [pdf_page(index, chapter_label, path) for index in range(count)]


def import_fitz(path: Path) -> Any:
    try:
        return importlib.import_module("fitz")
    except ImportError as exc:
        message = f"{path} is a PDF. Install PyMuPDF to use PDF sources."
        raise ManifestError(message) from exc


def pdf_page(index: int, chapter_label: str, path: Path) -> PageRef:
    page_number = index + 1
    return PageRef(
        "pdf", chapter_label, page_number, f"{path}#page={page_number}", ".png", path=path, pdf_page_index=index
    )


def download_to_temp(url: str, temp_dir: Path) -> Path:
    """Download a URL source into a temporary path.

    Example:
        `download_to_temp("https://example.test/a.cbz", temp_dir)`.
    """
    suffix = safe_download_suffix(url)
    output_path = temp_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}{suffix}"
    request = urllib.request.Request(url, headers={"User-Agent": "manga-artist-dataset-builder/1.0"})
    output_path.write_bytes(fetch_url_bytes(request))
    return output_path


def safe_download_suffix(url: str) -> str:
    suffix = path_suffix(url)
    if suffix in IMAGE_EXTENSIONS | ARCHIVE_EXTENSIONS | {".pdf"}:
        return suffix
    return ".download"


def url_page(index: int, chapter_label: str, url: str, headers: dict[str, str] | None = None) -> PageRef:
    suffix = path_suffix(url) if is_image_name(url) else ".img"
    return PageRef("url", chapter_label, index, url, suffix, url=url, request_headers=headers or {})


def request_headers(source: dict[str, object]) -> dict[str, str]:
    headers = source.get("request_headers") or {}
    if not isinstance(headers, dict):
        return {}
    return {str(key): str(value) for key, value in cast(dict[object, object], headers).items()}

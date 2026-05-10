"""Read page bytes from typed page references."""

from __future__ import annotations

import importlib
import urllib.request
import zipfile
from typing import Any

from manga_artist_dataset.errors import DatasetError
from manga_artist_dataset.io.http import HostDownloadLimiter, fetch_url_bytes
from manga_artist_dataset.models import PageRef


def read_page_bytes(page: PageRef, download_limiter: HostDownloadLimiter | None = None) -> bytes:
    """Read raw image bytes for a page reference.

    Example:
        `content = read_page_bytes(page_ref)`.
    """
    if page.kind == "file":
        return read_file_page(page)
    if page.kind == "zip":
        return read_archive_page(page)
    if page.kind == "url":
        return read_url_page(page, download_limiter)
    if page.kind == "pdf":
        return read_pdf_page(page)
    raise DatasetError(f"Unsupported page kind: {page.kind}")


def read_file_page(page: PageRef) -> bytes:
    if page.path is None:
        raise DatasetError("File page is missing a path; expected PageRef.path.")
    return page.path.read_bytes()


def read_archive_page(page: PageRef) -> bytes:
    if page.path is None or page.archive_member is None:
        raise DatasetError("Archive page is missing path/member; expected path and archive_member.")
    with zipfile.ZipFile(page.path) as archive:
        return archive.read(page.archive_member)


def read_url_page(page: PageRef, download_limiter: HostDownloadLimiter | None = None) -> bytes:
    if page.url is None:
        raise DatasetError("URL page is missing a URL; expected PageRef.url.")
    if download_limiter is not None:
        download_limiter.wait_for(page.url)
    request = urllib.request.Request(
        page.url,
        headers={"User-Agent": "manga-artist-dataset-builder/1.0", **page.request_headers},
    )
    return fetch_url_bytes(request)


def read_pdf_page(page: PageRef) -> bytes:
    if page.path is None or page.pdf_page_index is None:
        raise DatasetError("PDF page is missing path/page index; expected path and pdf_page_index.")
    fitz = import_fitz()
    with fitz.open(page.path) as document:
        pdf_page = document.load_page(page.pdf_page_index)
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return bytes(pixmap.tobytes("png"))


def import_fitz() -> Any:
    try:
        return importlib.import_module("fitz")
    except ImportError as exc:
        raise DatasetError("Install PyMuPDF to read PDF sources.") from exc

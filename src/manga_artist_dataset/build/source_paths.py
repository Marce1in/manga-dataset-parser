"""Path and URL shape helpers for source manifests."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from manga_artist_dataset.errors import ManifestError
from manga_artist_dataset.image_formats import ARCHIVE_EXTENSIONS, IMAGE_EXTENSIONS
from manga_artist_dataset.json_types import JsonObject


def path_suffix(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    return Path(parsed.path).suffix.lower()


def is_image_name(value: str) -> bool:
    return path_suffix(value) in IMAGE_EXTENSIONS


def is_archive_name(value: str) -> bool:
    return path_suffix(value) in ARCHIVE_EXTENSIONS


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def chapter_label_for(source: JsonObject, fallback: str) -> str:
    return str(source.get("chapter") or source.get("chapter_label") or fallback)


def source_urls(source: JsonObject) -> list[str]:
    pages = source.get("pages", source.get("urls", []))
    if not isinstance(pages, list):
        raise ManifestError("'pages'/'urls' must be a list of image URLs.")
    return [source_url_from_item(item) for item in pages]


def source_url_from_item(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and item.get("url"):
        return str(item["url"])
    raise ManifestError("Each page URL must be a string or an object with a 'url' field.")

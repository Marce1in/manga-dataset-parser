"""HTTP boundary helpers for explicit source URLs."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import cast

from manga_artist_dataset.errors import DatasetError
from manga_artist_dataset.json_types import JsonObject


def fetch_url_bytes(request: urllib.request.Request, retries: int = 3) -> bytes:
    """Download bytes from an explicit manifest URL.

    Example:
        `fetch_url_bytes(urllib.request.Request("https://example.test/a.png"))`.
    """
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return cast(bytes, response.read())
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = exc
            _sleep_before_retry(attempt, retries)
    raise DatasetError(f"Could not download {request.full_url}: {last_error}") from last_error


def fetch_json(url: str, headers: dict[str, str], retries: int = 3) -> JsonObject:
    """Fetch a JSON object from a known API endpoint.

    Example:
        `fetch_json("https://api.example.test", {"Accept": "application/json"})`.
    """
    request = urllib.request.Request(url, headers=headers)
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return _read_json_response(request)
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            _sleep_before_retry(attempt, retries)
    raise DatasetError(f"Could not fetch JSON from {url}: {last_error}") from last_error


def fetch_text(url: str, headers: dict[str, str], retries: int = 3) -> str:
    """Fetch text from a known source endpoint.

    Example:
        `fetch_text("https://example.test", {"User-Agent": "tool"})`.
    """
    request = urllib.request.Request(url, headers=headers)
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return cast(bytes, response.read()).decode("utf-8", errors="replace")
        except (TimeoutError, urllib.error.URLError, UnicodeDecodeError) as exc:
            last_error = exc
            _sleep_before_retry(attempt, retries)
    raise DatasetError(f"Could not fetch text from {url}: {last_error}") from last_error


def _read_json_response(request: urllib.request.Request) -> JsonObject:
    with urllib.request.urlopen(request, timeout=45) as response:
        value = json.load(response)
    if isinstance(value, dict):
        return cast(JsonObject, value)
    msg = f"{request.full_url} returned non-object JSON; expected object."
    raise DatasetError(msg)


def _sleep_before_retry(attempt: int, retries: int) -> None:
    if attempt < retries:
        time.sleep(1.5 * attempt)

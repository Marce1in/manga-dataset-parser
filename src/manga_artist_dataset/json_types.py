"""JSON type aliases for dynamic boundary data."""

from __future__ import annotations

from typing import Any

type JsonPrimitive = str | int | float | bool | None
type JsonValue = Any
type JsonObject = dict[str, Any]

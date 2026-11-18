"""Marked, UTF-8 byte-bounded previews of request bodies, matching the Lens `lens_payloads::preview`."""

import json
from collections.abc import Callable
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

MIN_PREVIEW_BYTES: Final = 256
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def _marker(omitted: int) -> str:
    return f"…[Lens preview: {omitted} bytes omitted]"


def _cut(text: str, keep: int) -> str:
    encoded: Final = text.encode()
    if keep >= len(encoded):
        return text
    kept: Final = encoded[:keep].decode(errors="ignore")
    return kept + _marker(len(encoded) - len(kept.encode()))


def _dumps(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _longest_leaf(value: JsonValue) -> int:
    if isinstance(value, str):
        return len(value.encode())
    if isinstance(value, list):
        return max((_longest_leaf(item) for item in value), default=0)
    if isinstance(value, dict):
        return max((_longest_leaf(item) for item in value.values()), default=0)
    return 0


def _capped(value: JsonValue, keep: int) -> JsonValue:
    if isinstance(value, str):
        return _cut(value, keep)
    if isinstance(value, list):
        return [_capped(item, keep) for item in value]
    if isinstance(value, dict):
        return {name: _capped(item, keep) for name, item in value.items()}
    return value


def _largest_fitting(budget: int, low: int, high: int, render: Callable[[int], str]) -> int:
    if low >= high:
        return low
    middle: Final = low + (high - low + 1) // 2
    if len(render(middle).encode()) <= budget:
        return _largest_fitting(budget, middle, high, render)
    return _largest_fitting(budget, low, middle - 1, render)


def _bounded(budget: int, upper: int, render: Callable[[int], str]) -> str | None:
    if len(render(0).encode()) > budget:
        return None
    return render(_largest_fitting(budget, 0, upper, render))


def preview(content: str, budget: int) -> str:
    limit: Final = max(budget, MIN_PREVIEW_BYTES)
    size: Final = len(content.encode())
    if size <= limit:
        return content
    try:
        value: Final = _JSON.validate_json(content)
    except ValidationError:
        return _bounded(limit, size, lambda keep: _cut(content, keep)) or ""
    structured: Final = _bounded(limit, _longest_leaf(value), lambda keep: _dumps(_capped(value, keep)))
    if structured is not None:
        return structured
    return _bounded(limit, size, lambda keep: _dumps(_cut(content, keep))) or ""

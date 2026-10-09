"""Per-field caps that keep huge payloads out of Langfuse's ClickHouse rows.

Every string bound for a span attribute is capped before export, mirroring the
agent-runtime capture discipline: 32 KiB per field, cut on a code-point boundary,
marked with ``...(truncated)``. The budget is the ``ensure_ascii``-escaped width,
which is what ``safe_dumps`` actually emits, so CJK and emoji content cannot
inflate a capped field past the limit. The 413 backstop in ``langfuse_sdk`` stays
as the last resort; this makes it rare.
"""

import json
import os
from collections.abc import Mapping
from itertools import accumulate
from typing import (
    Final,
    cast,  # noqa: TID251  # identity-preserving type laundering for isinstance-narrowed user payloads
)

from litellm._logging import verbose_logger

DEFAULT_MAX_FIELD_BYTES: Final = 32 * 1024
MAX_FIELD_BYTES_METADATA_KEY: Final = "langfuse_max_field_bytes"
_MAX_FIELD_BYTES_ENV: Final = "LANGFUSE_MAX_FIELD_BYTES"
_TRUNCATION_SUFFIX: Final = "...(truncated)"


def resolve_max_field_bytes(override: object = None) -> int:
    """Bytes allowed per string field; a value of ``0`` or less disables capping.

    Precedence: the per-request ``langfuse_max_field_bytes`` metadata value, then
    ``LANGFUSE_MAX_FIELD_BYTES``, then the default.
    """
    if override is None:
        override = os.environ.get(_MAX_FIELD_BYTES_ENV)
    parsed: Final = _parse_byte_count(override)
    if parsed is not None:
        return parsed
    if override is not None:
        verbose_logger.warning(
            "Invalid langfuse max field bytes %r; falling back to %d", override, DEFAULT_MAX_FIELD_BYTES
        )
    return DEFAULT_MAX_FIELD_BYTES


def _parse_byte_count(raw: object) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            return None
    return None


def cap_payload(value: object, max_field_bytes: int) -> object:
    """Return ``value`` with every string capped at ``max_field_bytes`` UTF-8 bytes.

    Plain containers are rebuilt capped; anything else passes through untouched. A
    disabled limit (``max_field_bytes <= 0``) returns ``value`` itself.
    """
    if max_field_bytes <= 0:
        return value
    return _cap(value, max_field_bytes)


def _cap(value: object, limit: int) -> object:
    if isinstance(value, str):
        if _serialized_width(value) <= limit:
            return value
        budget: Final = limit - len(_TRUNCATION_SUFFIX) - 2
        kept: Final = "" if budget <= 0 else _serialized_prefix(value, budget)
        return kept + _TRUNCATION_SUFFIX
    rebuilt: Final = _capped_container(value, limit)
    return value if rebuilt is None else rebuilt


def _serialized_width(value: str) -> int:
    """Bytes ``safe_dumps`` emits for this string: the quotes plus the ``ensure_ascii`` escapes."""
    return len(json.dumps(value))


def _serialized_prefix(value: str, budget: int) -> str:
    """Longest head of ``value`` whose ``ensure_ascii``-escaped form fits ``budget`` bytes.

    ``safe_dumps`` escapes non-ASCII as ``\\uXXXX`` (6 bytes, 12 per astral code point), so a raw
    byte cap understates what reaches ClickHouse; the cut budgets the escaped form instead.
    Escapes are self-contained per code point, so cutting between code points is always safe.
    """
    running_widths: Final = accumulate(_escaped_width(char) for char in value)
    cut: Final = next((index for index, total in enumerate(running_widths) if total > budget), len(value))
    return value[:cut]


def _escaped_width(char: str) -> int:
    """Bytes ``json.dumps`` emits for one code point under ``ensure_ascii``."""
    point: Final[int] = ord(char)
    if point in (0x08, 0x09, 0x0A, 0x0C, 0x0D) or point in (0x22, 0x5C):
        return 2
    if point < 0x20:
        return 6
    if point < 0x7F:
        return 1
    if point <= 0xFFFF:
        return 6
    return 12


def _capped_container(value: object, limit: int) -> object | None:
    """``value`` rebuilt with every string capped, or None when it is not a plain container."""
    mapping: Final = _as_mapping(value)
    if mapping is not None:
        return {key: _cap(item, limit) for key, item in mapping.items()}
    items: Final = _as_object_list(value)
    if items is not None:
        return [_cap(item, limit) for item in items]
    members: Final = _as_object_tuple(value)
    if members is not None:
        return tuple(_cap(item, limit) for item in members)
    return None


def _as_mapping(value: object) -> Mapping[object, object] | None:
    """``value`` as a plain mapping, or None."""
    if isinstance(value, Mapping):
        return cast(Mapping[object, object], value)
    return None


def _as_object_list(value: object) -> list[object] | None:
    """``value`` as a list, or None."""
    if isinstance(value, list):
        return cast(list[object], value)
    return None


def _as_object_tuple(value: object) -> tuple[object, ...] | None:
    """``value`` as a tuple, or None."""
    if isinstance(value, tuple):
        return cast(tuple[object, ...], value)
    return None

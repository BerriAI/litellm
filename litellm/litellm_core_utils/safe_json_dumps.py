import json
from collections.abc import Callable
from typing import Any, Final

from pydantic import BaseModel

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH


def strip_null_bytes(value: str) -> str:
    """Strip NUL bytes, which PostgreSQL text/jsonb columns reject (error 22P05)."""
    return value.replace("\x00", "")


def safe_dumps(
    data: Any,
    max_depth: int = DEFAULT_MAX_RECURSE_DEPTH,
    value_transform: Callable[[str | None, str], str] | None = None,
) -> str:
    """
    Recursively serialize data while detecting circular references.
    If a circular reference is detected then a marker string is returned.
    NUL bytes are stripped from strings to prevent PostgreSQL 22P05 errors.

    value_transform, when given, is applied to every string leaf (and to the
    str() fallback for non-serializable objects) with the mapping key the leaf
    was reached under, so callers can rewrite values without touching structure.
    """

    def _transform(key: str | None, value: str) -> str:
        return value if value_transform is None else value_transform(key, value)

    def _serialize(obj: Any, seen: set, depth: int, key: str | None = None) -> Any:
        # Check for maximum depth.
        if depth > max_depth:
            return "MaxDepthExceeded"
        # Base-case: if it is a primitive, simply return it.
        if isinstance(obj, str):
            cleaned = obj.replace("\x00", "") if "\x00" in obj else obj
            return _transform(key, cleaned)
        if isinstance(obj, (int, float, bool, type(None))):
            return obj
        # Check for circular reference.
        if id(obj) in seen:
            return "CircularReference Detected"
        seen.add(id(obj))
        result: dict | list | tuple | set | str
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if isinstance(k, (str)):
                    clean_k = k.replace("\x00", "") if "\x00" in k else k
                    result[clean_k] = _serialize(v, seen, depth + 1, clean_k)
            seen.remove(id(obj))
            return result
        elif isinstance(obj, list):
            result = [_serialize(item, seen, depth + 1, key) for item in obj]
            seen.remove(id(obj))
            return result
        elif isinstance(obj, tuple):
            result = tuple(_serialize(item, seen, depth + 1, key) for item in obj)
            seen.remove(id(obj))
            return result
        elif isinstance(obj, set):
            result = sorted([_serialize(item, seen, depth + 1, key) for item in obj])
            seen.remove(id(obj))
            return result
        elif isinstance(obj, BaseModel):
            dumped: Final = obj.model_dump()
            result = _serialize(dumped, seen, depth + 1, key)
            seen.remove(id(obj))
            return result
        else:
            # Fall back to string conversion for non-serializable objects.
            try:
                return _transform(key, strip_null_bytes(str(obj)))
            except Exception:
                return "Unserializable Object"

    safe_data: Final = _serialize(data, set(), 0)
    return json.dumps(safe_data, default=str)

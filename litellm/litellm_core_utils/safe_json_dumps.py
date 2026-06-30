import json
from typing import Any, Union

from pydantic import BaseModel

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

# json renders a NUL byte as this 6-character escape; its presence in the output
# means a string value held a NUL, so we hand the payload to the sanitizer.
_NUL_ESCAPE = "\\u0000"


def strip_null_bytes(value: str) -> str:
    """Strip NUL bytes, which PostgreSQL text/jsonb columns reject (error 22P05)."""
    return value.replace("\x00", "")


def _stringify_unserializable(obj: object) -> str:
    try:
        return strip_null_bytes(str(obj))
    except Exception:
        return "Unserializable Object"


def _json_default(obj: object) -> object:
    """Encode the values json can't, mirroring the sanitizer's leaf handling."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return _stringify_unserializable(obj)


def _sanitize(obj: Any, seen: set, depth: int, max_depth: int) -> Any:
    """Recursively rebuild ``obj`` into a JSON-safe value: strip NUL bytes, replace
    circular references with a marker, and truncate past ``max_depth``."""
    if depth > max_depth:
        return "MaxDepthExceeded"
    if isinstance(obj, str):
        return obj.replace("\x00", "") if "\x00" in obj else obj
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    if id(obj) in seen:
        return "CircularReference Detected"
    seen.add(id(obj))
    result: Union[dict, list, tuple, set, str]
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(k, (str)):
                clean_k = k.replace("\x00", "") if "\x00" in k else k
                result[clean_k] = _sanitize(v, seen, depth + 1, max_depth)
        seen.remove(id(obj))
        return result
    elif isinstance(obj, list):
        result = [_sanitize(item, seen, depth + 1, max_depth) for item in obj]
        seen.remove(id(obj))
        return result
    elif isinstance(obj, tuple):
        result = tuple(_sanitize(item, seen, depth + 1, max_depth) for item in obj)
        seen.remove(id(obj))
        return result
    elif isinstance(obj, set):
        result = sorted([_sanitize(item, seen, depth + 1, max_depth) for item in obj])
        seen.remove(id(obj))
        return result
    elif isinstance(obj, BaseModel):
        dumped = obj.model_dump()
        result = _sanitize(dumped, seen, depth + 1, max_depth)
        seen.remove(id(obj))
        return result
    else:
        # Fall back to string conversion for non-serializable objects.
        return _stringify_unserializable(obj)


def safe_dumps(data: Any, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH) -> str:
    """
    Recursively serialize data while detecting circular references.
    If a circular reference is detected then a marker string is returned.
    NUL bytes are stripped from strings to prevent PostgreSQL 22P05 errors.

    With the default depth budget the common case is encoded in a single pass; a NUL
    byte in the output, a circular reference, or a key json cannot encode falls back
    to the recursive sanitizer. A caller-tightened ``max_depth`` always uses the
    sanitizer so its truncation stays exact.
    """
    if max_depth >= DEFAULT_MAX_RECURSE_DEPTH:
        try:
            encoded = json.dumps(data, default=_json_default)
        except (ValueError, TypeError, RecursionError):
            pass
        else:
            if _NUL_ESCAPE not in encoded:
                return encoded
    return json.dumps(_sanitize(data, set(), 0, max_depth), default=str)

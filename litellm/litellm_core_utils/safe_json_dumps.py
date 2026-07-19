import json
from typing import Any, Union

from pydantic import BaseModel

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH


def strip_null_bytes(value: str) -> str:
    """Strip NUL bytes, which PostgreSQL text/jsonb columns reject (error 22P05)."""
    return value.replace("\x00", "")


def safe_dumps(data: Any, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH) -> str:
    """
    Recursively serialize data while detecting circular references.
    If a circular reference is detected then a marker string is returned.
    NUL bytes are stripped from strings to prevent PostgreSQL 22P05 errors.
    """

    def _serialize(obj: Any, seen: set, depth: int) -> Any:
        # Check for maximum depth.
        if depth > max_depth:
            return "MaxDepthExceeded"
        # Base-case: if it is a primitive, simply return it.
        if isinstance(obj, str):
            return obj.replace("\x00", "") if "\x00" in obj else obj
        if isinstance(obj, (int, float, bool, type(None))):
            return obj
        # Check for circular reference.
        if id(obj) in seen:
            return "CircularReference Detected"
        seen.add(id(obj))
        result: Union[dict, list, tuple, set, str]
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if isinstance(k, str):
                    clean_k = k.replace("\x00", "") if "\x00" in k else k
                else:
                    # JSON only allows string keys; convert non-str keys (e.g. tuples
                    # used by the OTel integration as dedup keys) rather than dropping
                    # them silently, which caused data loss in spend-log payloads.
                    # Mirror the value fallback (lines 68-71): if __str__ raises, use
                    # a safe placeholder so safe_dumps never propagates an exception.
                    try:
                        str_k = str(k)
                        clean_k = str_k.replace("\x00", "") if "\x00" in str_k else str_k
                    except Exception:  # noqa: BLE001 - key __str__ must never propagate
                        clean_k = "UnserializableKey"
                result[clean_k] = _serialize(v, seen, depth + 1)
            seen.remove(id(obj))
            return result
        elif isinstance(obj, list):
            result = [_serialize(item, seen, depth + 1) for item in obj]
            seen.remove(id(obj))
            return result
        elif isinstance(obj, tuple):
            result = tuple(_serialize(item, seen, depth + 1) for item in obj)
            seen.remove(id(obj))
            return result
        elif isinstance(obj, set):
            result = sorted([_serialize(item, seen, depth + 1) for item in obj])
            seen.remove(id(obj))
            return result
        elif isinstance(obj, BaseModel):
            dumped = obj.model_dump()
            result = _serialize(dumped, seen, depth + 1)
            seen.remove(id(obj))
            return result
        else:
            # Fall back to string conversion for non-serializable objects.
            try:
                return strip_null_bytes(str(obj))
            except Exception:
                return "Unserializable Object"

    safe_data = _serialize(data, set(), 0)
    return json.dumps(safe_data, default=str)

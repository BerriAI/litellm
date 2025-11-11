import json
from typing import Any, Union

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH


def safe_dumps(data: Any, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH) -> str:
    """
    Recursively serialize data while detecting circular references.
    If a circular reference is detected then a marker string is returned.
    """

    def _serialize(obj: Any, seen: set, depth: int) -> Any:
        # Check for maximum depth.
        if depth > max_depth:
            return "MaxDepthExceeded"
        # Base-case: if it is a primitive, simply return it.
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        # Check for circular reference.
        if id(obj) in seen:
            return "CircularReference Detected"
        seen.add(id(obj))
        result: Union[dict, list, tuple, set, str]
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if isinstance(k, (str)):
                    result[k] = _serialize(v, seen, depth + 1)
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
        else:
            # Fall back to string conversion for non-serializable objects.
            try:
                return str(obj)
            except Exception:
                return "Unserializable Object"

    safe_data = _serialize(data, set(), 0)
    return json.dumps(safe_data, default=str)
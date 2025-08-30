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


def filter_json_serializable(
    data: Any, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH
) -> Any:
    """
    Recursively filter data to only include JSON serializable items.
    Non-serializable items are completely skipped (not included in the result).
    """

    def _is_json_serializable(obj: Any) -> bool:
        """Test if an object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    def _filter(obj: Any, seen: set, depth: int) -> Any:
        # Check for maximum depth.
        if depth > max_depth:
            return None

        # Base-case: if it is a primitive, test if it's serializable
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj if _is_json_serializable(obj) else None

        # Check for circular reference.
        if id(obj) in seen:
            return None

        seen.add(id(obj))

        try:
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    # Only include keys that are strings and values that are serializable
                    if isinstance(k, str):
                        filtered_value = _filter(v, seen, depth + 1)
                        # Only add the key-value pair if the value is serializable
                        if filtered_value is not None or v is None:
                            if _is_json_serializable(filtered_value):
                                result[k] = filtered_value
                seen.remove(id(obj))
                return result

            elif isinstance(obj, list):
                result = []
                for item in obj:
                    filtered_item = _filter(item, seen, depth + 1)
                    # Only include items that are serializable
                    if filtered_item is not None or item is None:
                        if _is_json_serializable(filtered_item):
                            result.append(filtered_item)
                seen.remove(id(obj))
                return result

            elif isinstance(obj, tuple):
                filtered_items = []
                for item in obj:
                    filtered_item = _filter(item, seen, depth + 1)
                    # Only include items that are serializable
                    if filtered_item is not None or item is None:
                        if _is_json_serializable(filtered_item):
                            filtered_items.append(filtered_item)
                seen.remove(id(obj))
                return tuple(filtered_items)

            elif isinstance(obj, set):
                filtered_items = []
                for item in obj:
                    filtered_item = _filter(item, seen, depth + 1)
                    # Only include items that are serializable
                    if filtered_item is not None or item is None:
                        if _is_json_serializable(filtered_item):
                            filtered_items.append(filtered_item)
                seen.remove(id(obj))
                return sorted(filtered_items)

            else:
                # Test if the object is directly serializable
                seen.remove(id(obj))
                return obj if _is_json_serializable(obj) else None

        except Exception:
            if id(obj) in seen:
                seen.remove(id(obj))
            return None

    return _filter(data, set(), 0)

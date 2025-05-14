"""
Helper for safe JSON loading in LiteLLM.
"""
from typing import Any
import json
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

def safe_json_loads(data: str, default: Any = None, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH) -> Any:
    """
    Safely parse a JSON string. If parsing fails, return the default value (None by default).
    Recursively checks nested structures for excessive depth.
    """
    def _check_depth(obj: Any, depth: int) -> Any:
        if depth > max_depth:
            return "MaxDepthExceeded"
        if isinstance(obj, dict):
            return {k: _check_depth(v, depth + 1) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_check_depth(item, depth + 1) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_check_depth(item, depth + 1) for item in obj)
        elif isinstance(obj, set):
            return set(_check_depth(item, depth + 1) for item in obj)
        else:
            return obj

    try:
        loaded = json.loads(data)
        return _check_depth(loaded, 0)
    except Exception:
        return default 
"""
Helper for safe JSON loading in LiteLLM.
"""
from typing import Any
import orjson

def safe_json_loads(data: str, default: Any = None) -> Any:
    """
    Safely parse a JSON string. If parsing fails, return the default value (None by default).
    """
    try:
        return orjson.loads(data)
    except Exception:
        return default 
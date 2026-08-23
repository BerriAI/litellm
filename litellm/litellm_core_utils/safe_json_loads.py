"""
Helper for safe JSON loading in LiteLLM.
"""

import json
from typing import Any


def safe_json_loads(data: str, default: Any = None) -> Any:
    """
    Safely parse a JSON string. If parsing fails, return the default value (None by default).
    """
    try:
        return json.loads(data)
    except Exception:
        return default

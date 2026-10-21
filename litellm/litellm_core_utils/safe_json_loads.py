"""
Helper for safe JSON loading in LiteLLM.
"""

import json
from typing import Any, Final

from pydantic import TypeAdapter

_DECODED_JSON: Final = TypeAdapter(object)


def safe_json_loads(data: str, default: object = None) -> Any:
    """
    Safely parse a JSON string. If parsing fails, return the default value (None by default).
    """
    try:
        return _DECODED_JSON.validate_python(json.loads(data))
    except Exception:
        return default

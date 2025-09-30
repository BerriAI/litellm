"""Small logging utilities used by cache helpers and scenarios."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

__all__ = [
    "truncate_large_value",
    "log_safe_results",
    "log_api_request",
    "log_api_response",
]

logger = logging.getLogger(__name__)
_BASE64_IMAGE_PATTERN = re.compile(r"^(data:image/[a-zA-Z+.-]+;base64,)")


def truncate_large_value(value: Any, max_str_len: int = 100, max_list_elements_shown: int = 10) -> Any:
    """Return a log-safe representation of ``value`` by truncating large payloads."""

    if isinstance(value, str):
        match = _BASE64_IMAGE_PATTERN.match(value)
        if match:
            header = match.group(1)
            data = value[len(header) :]
            if len(data) > max_str_len:
                half = max(max_str_len // 2, 1)
                return header + f"{data[:half]}...{data[-half:]}"
            return value
        if len(value) > max_str_len:
            half = max(max_str_len // 2, 1)
            return f"{value[:half]}...{value[-half:]}"
        return value

    if isinstance(value, list):
        if len(value) > max_list_elements_shown:
            element_type = type(value[0]).__name__ if value else "elements"
            return f"[<{len(value)} {element_type} elements>]"
        return [
            truncate_large_value(item, max_str_len, max_list_elements_shown)
            if isinstance(item, (dict, list))
            else item
            for item in value
        ]

    if isinstance(value, dict):
        return {
            k: truncate_large_value(v, max_str_len, max_list_elements_shown)
            for k, v in value.items()
        }

    return value


def log_safe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a log-friendly copy of ``results`` with large values truncated."""

    safe: List[Dict[str, Any]] = []
    for item in results:
        safe.append({k: truncate_large_value(v) for k, v in item.items()})
    return safe


def log_api_request(service_name: str, request_data: Dict[str, Any], *, truncate: bool = True) -> None:
    payload = truncate_large_value(request_data) if truncate else request_data
    logger.debug("%s API Request: %s", service_name, payload)


def log_api_response(service_name: str, response_data: Any, *, truncate: bool = True) -> None:
    payload = truncate_large_value(response_data) if truncate else response_data
    logger.debug("%s API Response: %s", service_name, payload)

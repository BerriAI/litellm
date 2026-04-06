"""Utility helpers for A2A agent endpoints."""

from typing import Dict, Mapping, Optional


def merge_agent_headers(
    *,
    dynamic_headers: Optional[Mapping[str, str]] = None,
    static_headers: Optional[Mapping[str, str]] = None,
) -> Optional[Dict[str, str]]:
    """Merge outbound HTTP headers for A2A agent calls.

    Merge rules:
    - Start with ``dynamic_headers`` (values extracted from the incoming client request).
    - Overlay ``static_headers`` (admin-configured per agent).

    If both contain the same key, ``static_headers`` wins.
    """
    merged: Dict[str, str] = {}

    if dynamic_headers:
        merged.update({str(k): str(v) for k, v in dynamic_headers.items()})

    if static_headers:
        merged.update({str(k): str(v) for k, v in static_headers.items()})

    return merged or None

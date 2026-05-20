"""Utility helpers for A2A agent endpoints."""

from typing import Dict, Mapping, Optional

# Re-export from the canonical SDK location so the proxy and SDK always
# share the same provider-config lookup logic.
from litellm.interactions.agents.utils import (  # noqa: F401
    get_provider_agents_api_config,
)


def merge_agent_headers(
    *,
    dynamic_headers: Optional[Mapping[str, str]] = None,
    static_headers: Optional[Mapping[str, str]] = None,
) -> Optional[Dict[str, str]]:
    """Merge outbound HTTP headers for A2A agent calls.

    Merge rules:
    - Start with ``dynamic_headers`` (values extracted from the incoming client request).
    - Overlay ``static_headers`` (admin-configured per agent).
    - Comparison is case-insensitive (HTTP headers are case-insensitive), so a
      static ``Authorization`` strips any dynamic ``authorization`` before the
      static value is written. The static side's casing is preserved.

    If both contain the same header (case-insensitively), ``static_headers`` wins.
    """
    merged: Dict[str, str] = {}

    if dynamic_headers:
        merged.update({str(k): str(v) for k, v in dynamic_headers.items()})

    if static_headers:
        static_lower = {str(k).lower() for k in static_headers}
        merged = {k: v for k, v in merged.items() if k.lower() not in static_lower}
        merged.update({str(k): str(v) for k, v in static_headers.items()})

    return merged or None

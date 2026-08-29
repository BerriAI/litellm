"""Thin Python wrapper for the Rust pipeline reverse bridge.

The Rust core owns the full request processing pipeline for supported routes.
This module provides a clean Python API for dispatching requests through Rust,
with automatic fallback to Python when the bridge is unavailable or the route
is not supported.
"""

from __future__ import annotations

import json
import os
from typing import Any, Final

from litellm._logging import verbose_logger
from litellm.rust_bridge.loader import get_native_bridge

_TRUTHY_ENV_VALUES: Final = frozenset({"1", "true", "yes", "on"})


def _env_enables_rust_pipeline() -> bool:
    return (
        os.getenv("LITELLM_RUST_PIPELINE", "").strip().lower()
        in _TRUTHY_ENV_VALUES
    )


def _get_process_request():
    """Return the native process_request function, or None when unavailable."""
    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return getattr(native_bridge, "process_request", None)


def process_request(route: str, request_body: dict[str, Any]) -> dict[str, Any] | None:
    """Process a request through the Rust pipeline.

    Args:
        route: The route path (e.g., "/v1/chat/completions")
        request_body: The request body as a dict

    Returns:
        The response body as a dict, or None if the Rust path is unavailable
        or the route is not supported.
    """
    if not _env_enables_rust_pipeline():
        return None

    process_func = _get_process_request()
    if process_func is None:
        return None

    try:
        request_json = json.dumps(request_body)
        response_json = process_func(route, request_json)
        return json.loads(response_json)
    except Exception:
        verbose_logger.debug(
            "Rust pipeline failed for route %s, falling back to Python",
            route,
            exc_info=True,
        )
        return None

"""
Register Prometheus-related ASGI middleware only when prometheus is configured
as a LiteLLM callback.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from litellm._logging import verbose_proxy_logger

_PROMETHEUS_MIDDLEWARES_REGISTERED = False


def _normalize_callback_list(value: Union[str, List[Any], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def prometheus_callbacks_enabled(litellm_settings: Optional[dict]) -> bool:
    """
    Return True when ``prometheus`` appears in any LiteLLM callback list in config.

    Matches the check used for PROMETHEUS_MULTIPROC_DIR setup in proxy_cli.
    """
    if not litellm_settings:
        return False

    callbacks = _normalize_callback_list(litellm_settings.get("callbacks"))
    success_callbacks = _normalize_callback_list(
        litellm_settings.get("success_callback")
    )
    failure_callbacks = _normalize_callback_list(
        litellm_settings.get("failure_callback")
    )
    all_callbacks = callbacks + success_callbacks + failure_callbacks
    return any("prometheus" in callback for callback in all_callbacks)


def maybe_register_prometheus_middlewares(
    app: Any,
    litellm_settings: Optional[dict] = None,
) -> bool:
    """
    Add PrometheusAuthMiddleware and InFlightRequestsMiddleware when prometheus
    is enabled. Idempotent per process.
    """
    global _PROMETHEUS_MIDDLEWARES_REGISTERED

    if _PROMETHEUS_MIDDLEWARES_REGISTERED:
        return False

    if not prometheus_callbacks_enabled(litellm_settings):
        return False

    from litellm.proxy.middleware.in_flight_requests_middleware import (
        InFlightRequestsMiddleware,
    )
    from litellm.proxy.middleware.prometheus_auth_middleware import (
        PrometheusAuthMiddleware,
    )

    app.add_middleware(PrometheusAuthMiddleware)
    app.add_middleware(InFlightRequestsMiddleware)
    _PROMETHEUS_MIDDLEWARES_REGISTERED = True
    verbose_proxy_logger.info(
        "Registered Prometheus ASGI middleware (prometheus callback enabled)"
    )
    return True

"""
Register Prometheus-related ASGI middleware only when prometheus is configured
as a LiteLLM callback.

Middleware must be registered before the ASGI server starts handling traffic
(at app creation / import time). Registration during lifespan/startup is too
late because Starlette may have already built the middleware stack.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Union

from litellm._logging import verbose_proxy_logger

# Per-process guard: idempotent registration on the same app instance.
_PROMETHEUS_MIDDLEWARES_REGISTERED = False


def _normalize_callback_list(value: Union[str, List[Any], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _load_litellm_settings_from_yaml(config_file_path: str) -> Optional[dict]:
    try:
        import yaml

        with open(config_file_path, "r") as config_file:
            config = yaml.safe_load(config_file) or {}
        litellm_settings = config.get("litellm_settings")
        return litellm_settings if isinstance(litellm_settings, dict) else None
    except Exception:
        return None


def load_litellm_settings_from_env() -> Optional[dict]:
    """
    Best-effort sync load of ``litellm_settings`` for app-init middleware registration.

    Used when worker processes import ``proxy_server:app`` with ``WORKER_CONFIG`` or
    ``CONFIG_FILE_PATH`` already set in the environment.
    """
    config_file_path = os.getenv("CONFIG_FILE_PATH")
    if config_file_path and os.path.isfile(config_file_path):
        return _load_litellm_settings_from_yaml(config_file_path)

    worker_config_raw = os.getenv("WORKER_CONFIG")
    if not worker_config_raw:
        return None

    try:
        worker_config = json.loads(worker_config_raw)
    except json.JSONDecodeError:
        return None

    if isinstance(worker_config, str) and os.path.isfile(worker_config):
        return _load_litellm_settings_from_yaml(worker_config)

    if isinstance(worker_config, dict):
        config_path = worker_config.get("config")
        if isinstance(config_path, str) and os.path.isfile(config_path):
            return _load_litellm_settings_from_yaml(config_path)

    return None


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

    Call before ``uvicorn.run`` / gunicorn ``serve()`` — not from lifespan hooks.
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

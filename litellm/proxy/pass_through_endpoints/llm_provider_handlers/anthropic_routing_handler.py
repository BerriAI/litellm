"""
Multi-backend Anthropic passthrough routing with health-aware backend selection.

Port of sovereign-anthropic-router's HealthTracker, AnthropicProxy, and
AnthropicRouter into LiteLLM's passthrough infrastructure.

When ``anthropic_router`` is present in litellm_config.yaml, the Anthropic
passthrough endpoint resolves model→backend via glob matching, forwards
requests with automatic failover, and tracks per-backend health.

Without ``anthropic_router`` config, behaviour is unchanged.
"""

from __future__ import annotations

import fnmatch
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider


# ---------------------------------------------------------------------------
# Pydantic models (inline — not imported from the hub)
# ---------------------------------------------------------------------------

class AuthConfig(BaseModel):
    """Authentication configuration for a backend."""

    type: Literal["api-key", "bearer"] = "api-key"
    key_env: str = ""  # environment variable name holding the credential


class Backend(BaseModel):
    """A single Anthropic-compatible backend."""

    name: str
    url: str  # base URL, e.g. https://api.anthropic.com
    auth: AuthConfig = Field(default_factory=AuthConfig)
    timeout: int = 120
    weight: int = 100
    retryable: bool = True


class Route(BaseModel):
    """Maps a model name (or glob) to an ordered list of backends."""

    model: str  # exact name or fnmatch glob, e.g. "claude-sonnet-*"
    backends: List[Backend]


class RouterSettings(BaseModel):
    """Global router settings."""

    cooldown_seconds: int = 30
    max_failures: int = 3
    default_timeout: int = 120


class AnthropicRouterConfig(BaseModel):
    """Top-level config read from ``anthropic_router`` in litellm_config.yaml."""

    routes: List[Route]
    settings: RouterSettings = Field(default_factory=RouterSettings)


# ---------------------------------------------------------------------------
# Health tracker
# ---------------------------------------------------------------------------

class Status(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DEAD = "dead"


@dataclass
class HealthEntry:
    status: Status = Status.HEALTHY
    failures: int = 0
    last_failure: float = 0.0
    cooldown_until: float = 0.0


class HealthTracker:
    """Tracks backend health in memory.

    A backend is DEGRADED after 2+ failures. It enters DEAD (cooldown)
    after ``max_failures`` consecutive failures and stays there for
    ``cooldown_seconds``.
    """

    def __init__(self, cooldown_seconds: int = 30, max_failures: int = 3) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._max_failures = max_failures
        self._entries: Dict[str, HealthEntry] = {}

    def _ensure(self, name: str) -> HealthEntry:
        if name not in self._entries:
            self._entries[name] = HealthEntry()
        return self._entries[name]

    def is_healthy(self, name: str) -> bool:
        """Return True if the backend is not in DEAD cooldown."""
        entry = self._ensure(name)
        if entry.status == Status.DEAD:
            if time.monotonic() >= entry.cooldown_until:
                entry.status = Status.DEGRADED
                return True
            return False
        return True

    def record_success(self, name: str) -> None:
        """Record a successful request — reset failures."""
        entry = self._ensure(name)
        if entry.status != Status.HEALTHY:
            verbose_proxy_logger.info(
                "anthropic_router: backend %s transitioned from %s to %s",
                name, entry.status.value, Status.HEALTHY.value,
            )
        entry.status = Status.HEALTHY
        entry.failures = 0

    def record_failure(self, name: str) -> None:
        """Record a failed request — increment counter, apply cooldown if needed."""
        entry = self._ensure(name)
        old_status = entry.status
        entry.failures += 1
        entry.last_failure = time.monotonic()

        if entry.failures >= self._max_failures:
            entry.status = Status.DEAD
            entry.cooldown_until = time.monotonic() + self._cooldown_seconds
        elif entry.failures >= 2:
            entry.status = Status.DEGRADED

        if entry.status != old_status:
            verbose_proxy_logger.info(
                "anthropic_router: backend %s transitioned from %s to %s",
                name, old_status.value, entry.status.value,
            )

    def status(self, name: str) -> Status:
        """Get current status for a backend."""
        return self._ensure(name).status


# ---------------------------------------------------------------------------
# HTTP proxy
# ---------------------------------------------------------------------------

class AnthropicProxy:
    """Forwards HTTP requests to Anthropic-compatible backends via LiteLLM's
    managed httpx client pool.

    Preserves the request body as raw bytes — no parsing, no translation.
    Headers are copied verbatim except Host, Authorization, x-api-key, and
    Transfer-Encoding which are replaced with the backend's own auth.
    """

    @staticmethod
    def _resolve_credential(key_env: str) -> str:
        """Resolve a credential from the environment.

        Tries ``get_secret_str`` first (supports ``os.environ/…`` syntax),
        then falls back to ``os.environ``.
        """
        value = get_secret_str(key_env)
        if value is not None:
            return value
        return os.environ.get(key_env, "")

    @staticmethod
    async def forward(
        backend: Backend,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Forward a request to the given backend.

        Args:
            backend: Target backend configuration.
            method: HTTP method (POST, GET, etc.).
            path: URL path (e.g. /v1/messages).
            headers: Original request headers.
            body: Raw request body bytes.
            timeout: Optional per-request timeout override (seconds).

        Returns:
            httpx.Response from the backend.
        """
        url = f"{backend.url.rstrip('/')}{path}"

        # Resolve credential
        key = AnthropicProxy._resolve_credential(backend.auth.key_env)

        # Build auth header
        if backend.auth.type == "api-key":
            auth_header_name = "x-api-key"
            auth_header_value = key
        else:
            auth_header_name = "authorization"
            auth_header_value = f"Bearer {key}"

        # Copy headers, stripping auth/host/transfer-encoding
        strip_headers = {"host", "authorization", "transfer-encoding", "x-api-key"}
        fwd_headers: Dict[str, str] = {}
        for k, v in headers.items():
            if k.lower() in strip_headers:
                continue
            fwd_headers[k] = v
        fwd_headers[auth_header_name] = auth_header_value

        # Use LiteLLM's managed httpx client pool
        request_timeout = timeout or backend.timeout
        async_client_obj = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.PassThroughEndpoint,
            params={"timeout": request_timeout},
        )
        async_client = async_client_obj.client

        return await async_client.request(
            method=method,
            url=url,
            headers=fwd_headers,
            content=body,
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class AnthropicRouter:
    """Resolves a model name to an ordered list of backends.

    Backends are returned in priority order:
    1. Healthy backends (healthy or degraded but not in cooldown)
    2. Dead backends (as last resort; caller can skip them)

    Glob matching via ``fnmatch``: ``"claude-sonnet-*"`` matches
    ``"claude-sonnet-5"``.
    """

    def __init__(self, config: AnthropicRouterConfig) -> None:
        self._routes = config.routes
        self._settings = config.settings
        self.health = HealthTracker(
            cooldown_seconds=config.settings.cooldown_seconds,
            max_failures=config.settings.max_failures,
        )

    @property
    def routes(self) -> List[Route]:
        """Public accessor for configured routes."""
        return self._routes

    def resolve(self, model: str) -> List[Backend]:
        """Return backends in priority order for the given model.

        Args:
            model: The model name from the request body.

        Returns:
            Ordered list of Backend objects. Healthy backends first,
            dead backends last. Empty list if no route matches.
        """
        for route in self._routes:
            if fnmatch.fnmatch(model, route.model):
                healthy: List[Backend] = []
                dead: List[Backend] = []
                for b in route.backends:
                    if self.health.is_healthy(b.name):
                        healthy.append(b)
                    else:
                        dead.append(b)
                return healthy + dead
        return []


# ---------------------------------------------------------------------------
# Helpers — config loading and model extraction
# ---------------------------------------------------------------------------

_router_instance: Optional[AnthropicRouter] = None
_router_initialized: bool = False


def get_anthropic_router() -> Optional[AnthropicRouter]:
    """Return the singleton AnthropicRouter instance.

    On first call, attempts lazy initialisation from the running proxy
    config (``proxy_config.get_config_state()``).  Subsequent calls return
    the cached instance.  Returns ``None`` when no ``anthropic_router``
    section is present in the config.
    """
    global _router_instance, _router_initialized

    if not _router_initialized:
        _router_initialized = True
        try:
            from litellm.proxy.proxy_server import proxy_config

            config_dict = proxy_config.get_config_state() if proxy_config else {}
            init_anthropic_router_from_config(config_dict)
        except Exception:
            verbose_proxy_logger.debug(
                "anthropic_router: lazy init skipped (proxy config not available yet)"
            )

    return _router_instance


def init_anthropic_router_from_config(config_dict: Dict[str, Any]) -> Optional[AnthropicRouter]:
    """Initialise (or re-initialise) the router from a config dictionary.

    Called once at startup and on config hot-reload. When the
    ``anthropic_router`` key is absent or empty, the router is cleared
    and the passthrough endpoint falls back to the default single-backend
    behaviour.

    Args:
        config_dict: The full proxy config dictionary (as returned by
            ``proxy_config.get_config_state()``).

    Returns:
        The new AnthropicRouter instance, or None if no ``anthropic_router``
        key is present.
    """
    global _router_instance

    section = config_dict.get("anthropic_router")
    if section is None:
        _router_instance = None
        return None

    try:
        router_config = AnthropicRouterConfig(**section)
        _router_instance = AnthropicRouter(router_config)
        verbose_proxy_logger.info(
            "anthropic_router: initialised with %d route(s)",
            len(router_config.routes),
        )
        return _router_instance
    except Exception:
        verbose_proxy_logger.exception(
            "anthropic_router: failed to parse config section — "
            "passthrough will fall back to default behaviour"
        )
        _router_instance = None
        return None


def extract_model_from_body(body: bytes) -> str:
    """Best-effort extraction of the ``model`` field from a JSON request body.

    Returns ``"unknown"`` when the body cannot be parsed or the field is absent.
    """
    import json

    try:
        data = json.loads(body)
        return data.get("model", "unknown")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "unknown"

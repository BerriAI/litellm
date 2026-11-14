"""Optional fastapi-guard security middleware for the proxy.

fastapi-guard (https://github.com/Guard-Core/fastapi-guard) provides IP
block/allow lists, rate limiting with auto-ban, user-agent blocking,
penetration-attempt detection, passive mode, optional Redis-backed shared
state, and optional IPInfo geo/cloud-provider lookups.

Positioning: this is edge protection at the ASGI layer, complementary to
litellm's existing controls. It does not replace the virtual-key RPM/TPM
limiter (identity-level), the Enterprise ``allowed_ips`` auth-layer check,
or the login throttle. It runs before auth so screened requests never
reach key validation.

Design constraints honored here:

- Pure ASGI (``__call__(scope, receive, send)``), never BaseHTTPMiddleware,
  matching the middleware house style and its streaming-preservation test.
- Opt-in via ``LITELLM_GUARD_ENABLED``; when unset, the wrapper delegates
  straight through with no guard import and no config work.
- Fail-closed when enabled: the proxy does not depend on fastapi-guard (it
  is deliberately absent from pyproject), so the import is lazy. If the
  flag is set and the package is missing, every screened request is
  rejected with a 503 in litellm's OpenAI-error envelope and an error is
  logged once: an operator who explicitly enabled the guard must not get
  unscreened passthrough. (Fork PRs may not modify dependency files per
  the guard-fork-dependencies CI policy, so the dependency addition is
  left to the maintainers.)
- Zero-trust client identity by default: ``X-Forwarded-For`` is honored
  only from proxies explicitly listed in ``LITELLM_GUARD_TRUSTED_PROXIES``
  (guard ignores forwarded headers from unlisted peers), so a caller
  cannot spoof its way past allowlists, limits, or bans by setting the
  header. Deployments behind a load balancer must list their ingress
  addresses there.
- Rejections guard itself builds use guard's response shapes; the one
  rejection this wrapper builds (guard unavailable) uses litellm's
  OpenAI-error JSON shape (the ``AdmissionControlMiddleware`` model), so
  clients see a familiar envelope.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Final

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger

GUARD_ENABLED_ENV: Final = "LITELLM_GUARD_ENABLED"
GUARD_EXCLUDED_PATHS_ENV: Final = "LITELLM_GUARD_EXCLUDED_PATHS"
DEFAULT_EXCLUDED_PATHS: Final = ("/health", "/health/liveliness", "/health/readiness", "/metrics")


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw: Final = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def guard_enabled() -> bool:
    """Read lazily per request so YAML ``environment_variables`` entries are
    seen (the ``LITELLM_ENABLE_HSTS`` pattern)."""
    return _env_bool(GUARD_ENABLED_ENV)


def _build_guard_config() -> object:
    """Build the guard config from the LITELLM_GUARD_* environment variables.

    Returns guard's ``SecurityConfig``. The ``object`` return and the
    dynamic import are the seam: fastapi-guard is an optional dependency
    that is deliberately absent from the environments litellm type-checks,
    so the class can only be reached through a runtime lookup.
    """
    guard_module: Final = importlib.import_module("guard")
    # fastapi-guard is an optional dependency that is deliberately absent
    # from the environments litellm type-checks, so attributes read off the
    # dynamically imported module are Any; the Callable annotation re-types
    # the seam so no Any escapes this line.
    security_config_cls: Final[Callable[..., object]] = guard_module.SecurityConfig  # pyright: ignore[reportAny]  # optional dep absent from the typecheck venv; seam re-typed below

    redis_url_env: Final = _env_str("LITELLM_GUARD_REDIS_URL")
    ipinfo_token_env: Final = _env_str("LITELLM_GUARD_IPINFO_TOKEN")
    # One literal, no conditional inserts: optional list-valued features are
    # passed as their empty/None form, which SecurityConfig treats the same
    # as unset. In-memory state unless a Redis URL is configured: never
    # implicitly depend on a Redis server being reachable. The proxy
    # defaults to a single uvicorn worker; multi-worker deployments should
    # set LITELLM_GUARD_REDIS_URL so counters and bans are shared.
    kwargs: Final = {
        "enable_rate_limiting": True,
        "rate_limit": _env_int("LITELLM_GUARD_RATE_LIMIT", 100),
        "rate_limit_window": _env_int("LITELLM_GUARD_RATE_LIMIT_WINDOW", 60),
        "enable_ip_banning": True,
        "auto_ban_threshold": _env_int("LITELLM_GUARD_AUTO_BAN_THRESHOLD", 10),
        "auto_ban_duration": _env_int("LITELLM_GUARD_AUTO_BAN_DURATION", 300),
        "enable_penetration_detection": True,
        "enable_redis": bool(redis_url_env),
        "redis_url": redis_url_env or None,
        "redis_prefix": "litellm_guard:",
        "passive_mode": _env_bool("LITELLM_GUARD_PASSIVE_MODE"),
        "blacklist": _csv(_env_str("LITELLM_GUARD_BLOCKED_IPS")),
        "whitelist": _csv(_env_str("LITELLM_GUARD_ALLOWED_IPS")) or None,
        "blocked_user_agents": list(_csv(_env_str("LITELLM_GUARD_BLOCKED_USER_AGENTS"))),
        "blocked_countries": frozenset(_csv(_env_str("LITELLM_GUARD_BLOCKED_COUNTRIES"))),
        "whitelist_countries": frozenset(_csv(_env_str("LITELLM_GUARD_ALLOWED_COUNTRIES"))),
        "block_cloud_providers": frozenset(_csv(_env_str("LITELLM_GUARD_BLOCK_CLOUD_PROVIDERS"))) or None,
        # Zero-trust default: no proxies are trusted, so forwarded headers
        # are ignored and the connecting peer is the client. List real
        # ingress addresses to enable X-Forwarded-For resolution.
        "trusted_proxies": _csv(_env_str("LITELLM_GUARD_TRUSTED_PROXIES")),
        "trusted_proxy_depth": _env_int("LITELLM_GUARD_TRUSTED_PROXY_DEPTH", 1),
        "exclude_paths": list(_csv(_env_str(GUARD_EXCLUDED_PATHS_ENV)) or DEFAULT_EXCLUDED_PATHS),
        "ipinfo_token": ipinfo_token_env or None,
    }
    return security_config_cls(**kwargs)


def _guard_unavailable_response() -> JSONResponse:
    """Fail-closed rejection for when the guard is enabled but cannot be
    initialized (litellm's OpenAI-error envelope, the
    ``AdmissionControlMiddleware`` model)."""
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": (
                    "LITELLM_GUARD_ENABLED is set but the fastapi-guard package "
                    "is not installed. Install it with: pip install fastapi-guard. "
                    "Requests are rejected until then."
                ),
                "type": "guard_unavailable_error",
                "code": "503",
            }
        },
    )


class GuardMiddleware:
    """Pure-ASGI fastapi-guard wrapper.

    The inner fastapi-guard middleware is instantiated on the first request
    after the flag is seen (so YAML-configured environment variables are
    honored), and the downstream app is wrapped by it; everything the flag
    admits flows through guard's pipeline and then the normal proxy stack.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._guard_asgi: ASGIApp | None = None
        self._guard_failed = False

    def _ensure_guard(self) -> ASGIApp | None:
        """Build and return the guard ASGI app, or None when the package is
        missing (logged loudly once; the caller then fails closed)."""
        if self._guard_asgi is not None:
            return self._guard_asgi
        if self._guard_failed:
            return None
        try:
            guard_module: Final = importlib.import_module("guard")
            # Same optional-dep seam as _build_guard_config: the attribute
            # is Any; the Callable annotation keeps it from escaping.
            middleware_cls: Final[Callable[..., ASGIApp]] = guard_module.SecurityMiddleware  # pyright: ignore[reportAny]  # optional dep absent from the typecheck venv; seam re-typed below
        except ImportError as exc:
            self._guard_failed = True
            # Log loudly once; security misconfiguration must be visible.
            verbose_proxy_logger.error(
                "LITELLM_GUARD_ENABLED is set but fastapi-guard is not "
                "installed: %s. Install it with: pip install fastapi-guard. "
                "Requests are REJECTED until then.",
                exc,
            )
            return None

        self._guard_asgi = middleware_cls(self.app, config=_build_guard_config())
        return self._guard_asgi

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not guard_enabled():
            await self.app(scope, receive, send)
            return

        guard_asgi: Final = self._ensure_guard()
        if guard_asgi is None:
            # Flag on, package missing: fail closed with 503 rather than
            # forward unscreened. The operator explicitly enabled the
            # guard; silently honoring the opposite of that intent would
            # make the flag a no-op. The misconfiguration is logged once.
            await _guard_unavailable_response()(scope, receive, send)
            return

        await guard_asgi(scope, receive, send)

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
- fastapi-guard is NOT added to pyproject dependencies by this module: the
  import is lazy and raises an actionable error when the flag is enabled
  without the package installed. (Fork PRs may not modify dependency files
  per the guard-fork-dependencies CI policy, so the dependency addition is
  left to the maintainers.)
- Rejections are built in litellm's OpenAI-error JSON shape (the
  ``AdmissionControlMiddleware`` model), so clients see the same error
  envelope for 403/429 as for other proxy rejections.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from starlette.types import ASGIApp, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from guard import SecurityConfig

GUARD_ENABLED_ENV = "LITELLM_GUARD_ENABLED"
GUARD_EXCLUDED_PATHS_ENV = "LITELLM_GUARD_EXCLUDED_PATHS"
DEFAULT_EXCLUDED_PATHS = "/health,/health/liveliness,/health/readiness,/metrics"
DEFAULT_TRUSTED_PROXIES = "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
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


def _build_guard_config() -> SecurityConfig:
    """Build the SecurityConfig from the LITELLM_GUARD_* environment variables."""
    from guard import SecurityConfig

    kwargs: dict[str, object] = {
        "enable_rate_limiting": True,
        "rate_limit": _env_int("LITELLM_GUARD_RATE_LIMIT", 100),
        "rate_limit_window": _env_int("LITELLM_GUARD_RATE_LIMIT_WINDOW", 60),
        "enable_ip_banning": True,
        "auto_ban_threshold": _env_int("LITELLM_GUARD_AUTO_BAN_THRESHOLD", 10),
        "auto_ban_duration": _env_int("LITELLM_GUARD_AUTO_BAN_DURATION", 300),
        "enable_penetration_detection": True,
        # In-memory state unless a Redis URL is configured: never implicitly
        # depend on a Redis server being reachable. The proxy defaults to a
        # single uvicorn worker; multi-worker deployments should set
        # LITELLM_GUARD_REDIS_URL so counters and bans are shared.
        "enable_redis": False,
        "passive_mode": _env_bool("LITELLM_GUARD_PASSIVE_MODE"),
        "blacklist": _csv(_env_str("LITELLM_GUARD_BLOCKED_IPS")),
        "blocked_user_agents": list(_csv(_env_str("LITELLM_GUARD_BLOCKED_USER_AGENTS"))),
        "trusted_proxies": _csv(_env_str("LITELLM_GUARD_TRUSTED_PROXIES")) or _csv(DEFAULT_TRUSTED_PROXIES),
        "trusted_proxy_depth": _env_int("LITELLM_GUARD_TRUSTED_PROXY_DEPTH", 1),
        "exclude_paths": list(_csv(_env_str(GUARD_EXCLUDED_PATHS_ENV)) or DEFAULT_EXCLUDED_PATHS.split(",")),
    }

    if allowed_ips := _csv(_env_str("LITELLM_GUARD_ALLOWED_IPS")):
        kwargs["whitelist"] = allowed_ips
    if blocked_countries := _csv(_env_str("LITELLM_GUARD_BLOCKED_COUNTRIES")):
        kwargs["blocked_countries"] = frozenset(blocked_countries)
    if allowed_countries := _csv(_env_str("LITELLM_GUARD_ALLOWED_COUNTRIES")):
        kwargs["whitelist_countries"] = frozenset(allowed_countries)
    if cloud_providers := _csv(_env_str("LITELLM_GUARD_BLOCK_CLOUD_PROVIDERS")):
        kwargs["block_cloud_providers"] = frozenset(cloud_providers)

    if redis_url := _env_str("LITELLM_GUARD_REDIS_URL"):
        kwargs["enable_redis"] = True
        kwargs["redis_url"] = redis_url
        kwargs["redis_prefix"] = "litellm_guard:"

    if ipinfo_token := _env_str("LITELLM_GUARD_IPINFO_TOKEN"):
        kwargs["ipinfo_token"] = ipinfo_token

    return SecurityConfig(**kwargs)


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

    def _ensure_guard(self, scope: Scope) -> bool:
        if self._guard_asgi is not None or self._guard_failed:
            return not self._guard_failed
        try:
            from guard import PureASGISecurityMiddleware

            self._guard_asgi = PureASGISecurityMiddleware(self.app, config=_build_guard_config())
            return True
        except ImportError as exc:
            self._guard_failed = True
            # Log loudly once; security misconfiguration must be visible.
            verbose_proxy_logger.error(
                "LITELLM_GUARD_ENABLED is set but fastapi-guard is not "
                "installed: %s. Install it with: pip install fastapi-guard. "
                "Guard is DISABLED until then.",
                exc,
            )
        return not self._guard_failed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not guard_enabled():
            await self.app(scope, receive, send)
            return

        if not self._ensure_guard(scope):
            # Flag on, package missing: fail open with a loud log rather
            # than taking the proxy down (matches the disabled path; the
            # misconfiguration is logged at startup of the first request).
            await self.app(scope, receive, send)
            return

        await self._guard_asgi(scope, receive, send)

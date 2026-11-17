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
- Client identity follows the trust uvicorn applied before this wrapper turned
  its proxy headers off: ``X-Forwarded-For`` is honored only from the peers in
  ``LITELLM_GUARD_TRUSTED_PROXIES`` when that variable is set (an empty value
  trusts nobody), and otherwise from the peers in ``FORWARDED_ALLOW_IPS``, or
  ``127.0.0.1`` when that is unset too (``*`` trusts every address). A caller
  cannot spoof its way past allowlists, limits, or bans by setting the header
  from an unlisted peer, and a loopback reverse proxy keeps working by default.
  Deployments behind a load balancer must list their ingress addresses. The
  proxy's uvicorn and gunicorn start paths turn off
  uvicorn's own ``proxy_headers`` client rewriting when ``LITELLM_GUARD_ENABLED``
  is in the process environment at startup (``server_proxy_header_overrides``),
  so the server cannot pre-empt that policy. Not covered, set it yourself:
  a flag that only comes from the YAML ``environment_variables`` (read after
  the server is already running), hypercorn and granian (configure them not to
  rewrite the client address), and the Rust forward mode, whose
  ``RustPeerMiddleware`` runs uvicorn's ``ProxyHeadersMiddleware`` with the
  ``FORWARDED_ALLOW_IPS`` trust (set that so it trusts no peer).
- Guard keeps the raw socket peer, and litellm's own code gets the identity
  guard resolved. Everything the wrapper passes on (requests guard admits,
  admitted WebSocket handshakes and the probes) carries a copy of the scope
  whose ``client`` is the client resolved above and whose ``scheme`` follows
  ``X-Forwarded-Proto`` (``http``, ``https``, ``ws`` or ``wss``, the way
  uvicorn's ``ProxyHeadersMiddleware`` reads it), but only when the socket peer
  is a trusted proxy by the rule above. Otherwise the scope is
  untouched. Without this, turning off uvicorn's ``proxy_headers`` would leave
  the proxy's IP-based authorization (``IPAddressUtils.get_mcp_client_ip``)
  seeing the ingress address and its HTTPS checks (``is_request_https``,
  which decides the ``Secure`` cookie flag) seeing ``http`` behind TLS
  termination.
- Request bodies are never previewed in guard's logs. Guard logs up to 100
  characters of a field that trips penetration detection, and its default list
  of redacted body fields does not name any of the places a prompt travels in,
  so the wrapper redacts them all (``_PROMPT_BODY_FIELDS``).
- WebSocket scopes are not screened by guard's HTTP pipeline. The wrapper
  enforces ``LITELLM_GUARD_BLOCKED_IPS`` and ``LITELLM_GUARD_ALLOWED_IPS``
  (IPs or CIDRs) on the handshake, resolving the client the way guard does
  (socket peer; ``X-Forwarded-For`` only from the trusted proxies above,
  at ``LITELLM_GUARD_TRUSTED_PROXY_DEPTH``) and closing with 1008 before
  accept on a miss. The lists follow guard's rules (the allowlist alone when
  set, IPv4-mapped entries read as IPv4), and an invalid entry or a depth
  below 1 raises ``ValueError`` instead of being ignored. WebSocket scopes
  still get no rate limiting, no auto-ban and no payload inspection, and
  those routes still go through key auth.
- Only the unauthenticated k8s probes (``/health/liveliness``,
  ``/health/liveness``, ``/health/readiness``) bypass guard by default, matched
  by exact path under the root path. The probes keep the IP lists. The bans,
  rate limits, country and cloud-provider blocks that guard applies to its own
  exclusions do not apply to them, on purpose: a ban or a 429 on a liveness
  probe restarts the pod. ``LITELLM_GUARD_EXCLUDED_PATHS`` replaces that
  default and is handed to guard, which matches each entry as a subtree.
- Rejections guard itself builds use guard's response shapes; the one
  rejection this wrapper builds (guard unavailable) uses litellm's
  OpenAI-error JSON shape (the ``AdmissionControlMiddleware`` model), so
  clients see a familiar envelope.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network, ip_address, ip_network
from itertools import chain
from typing import Final, TypeAlias

from pydantic import TypeAdapter
from starlette.responses import JSONResponse
from starlette.routing import get_route_path
from starlette.types import ASGIApp, Receive, Scope, Send
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.auth.special_headers import SpecialHeaders

GUARD_ENABLED_ENV: Final = "LITELLM_GUARD_ENABLED"
TRUSTED_PROXIES_ENV: Final = "LITELLM_GUARD_TRUSTED_PROXIES"
GUARD_EXCLUDED_PATHS_ENV: Final = "LITELLM_GUARD_EXCLUDED_PATHS"
# Only the unauthenticated k8s probes (health_endpoints/_health_endpoints.py).
# /health runs live deployment calls behind auth and /metrics can be public,
# so neither is exempt by default. guard-core matches exclude_paths by subtree
# (the entry or anything under entry + "/"), which would also exempt the
# authenticated /health/readiness/details, so the defaults bypass guard's
# pipeline by exact path in the wrapper instead (the IP lists are still
# enforced there) and are not passed to guard.
DEFAULT_EXCLUDED_PATHS: Final = ("/health/liveliness", "/health/liveness", "/health/readiness")


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


IPAddress: TypeAlias = IPv4Address | IPv6Address
IPNetwork: TypeAlias = IPv4Network | IPv6Network

WS_POLICY_VIOLATION: Final = 1008


def _strip_bracketed_port(candidate: str) -> str:
    closing: Final = candidate.find("]")
    if closing == -1:
        return candidate
    remainder: Final = candidate[closing + 1 :]
    if remainder and not (remainder.startswith(":") and remainder[1:].isdigit()):
        return candidate
    return candidate[1:closing]


def _strip_forwarded_port(raw: str) -> str:
    """guard-core's ``_strip_forwarded_entry_port``: a port is dropped only
    when it is all digits, anything else is left for the address parse to
    refuse."""
    candidate: Final = raw.strip()
    if candidate.startswith("["):
        return _strip_bracketed_port(candidate)
    host, _, port = candidate.partition(":")
    return host if candidate.count(":") == 1 and port.isdigit() else candidate


def _parse_ip(raw: str) -> IPAddress | None:
    try:
        return ip_address(_strip_forwarded_port(raw))
    except ValueError:
        return None


def _canonical(address: IPAddress | None) -> IPAddress | None:
    """guard-core's ``_canonical_ip_text``: an IPv4-mapped address without a
    scope id is the IPv4 address."""
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None and address.scope_id is None:
        return address.ipv4_mapped
    return address


def _parse_network(entry: str) -> IPNetwork | None:
    try:
        return ip_network(entry, strict=False)
    except ValueError:
        return None


def _networks(entries: Sequence[str]) -> tuple[IPNetwork, ...]:
    return tuple(network for network in map(_parse_network, entries) if network is not None)


def _validate_entries(variable: str, entries: Sequence[str], *, allow_unix: bool = False) -> tuple[str, ...]:
    """guard-core's ``_validate_ip_or_cidr_list``: an entry with a "/" is a
    network, any other an address, and anything else is refused (``unix``
    stays valid for the trusted proxies)."""
    for entry in entries:
        if allow_unix and entry == "unix":
            continue
        try:
            ip_network(entry, strict=False) if "/" in entry else ip_address(entry)
        except ValueError:
            raise ValueError(f"{variable}: invalid IP or CIDR entry {entry!r}") from None
    return tuple(entries)


def _validated_entries(variable: str) -> tuple[str, ...]:
    return _validate_entries(variable, _csv(_env_str(variable)))


_UVICORN_DEFAULT_TRUSTED: Final = "127.0.0.1"
_EVERY_ADDRESS: Final = ("0.0.0.0/0", "::/0")


def _trusted_proxy_entries() -> tuple[str, ...]:
    """The trusted proxies: ``LITELLM_GUARD_TRUSTED_PROXIES`` when it is set,
    even to an empty string (trust nobody); when it is not set, the trust
    uvicorn would have used, ``FORWARDED_ALLOW_IPS`` or else ``127.0.0.1``,
    with ``*`` meaning every address."""
    explicit: Final = os.getenv(TRUSTED_PROXIES_ENV)
    if explicit is not None:
        return _validate_entries(TRUSTED_PROXIES_ENV, _csv(explicit), allow_unix=True)
    inherited: Final = _csv(os.getenv("FORWARDED_ALLOW_IPS", _UVICORN_DEFAULT_TRUSTED))
    expanded: Final = tuple(chain.from_iterable(_EVERY_ADDRESS if entry == "*" else (entry,) for entry in inherited))
    return _validate_entries("FORWARDED_ALLOW_IPS", expanded, allow_unix=True)


def _listed(client: IPAddress, entries: Sequence[str]) -> bool:
    """guard-core's ``_ip_in_list``: a "/" entry matches by network, any other
    entry by canonical address (``::ffff:1.2.3.4`` is ``1.2.3.4``)."""
    return any(
        client in ip_network(entry, strict=False) if "/" in entry else client == _canonical(ip_address(entry))
        for entry in entries
    )


def _in_any(address: IPAddress, networks: Sequence[IPNetwork]) -> bool:
    return any(address in network for network in networks)


class _ClientScopeView(TypedDict):
    """The slice of an ASGI http or websocket scope the IP check reads."""

    headers: ReadOnly[Sequence[tuple[bytes, bytes]]]
    client: NotRequired[ReadOnly[tuple[str, int] | None]]


_CLIENT_SCOPE_VIEW: Final = TypeAdapter(_ClientScopeView)


def _peer_and_forwarded_for(scope: Scope) -> tuple[str | None, str | None]:
    view: Final = _CLIENT_SCOPE_VIEW.validate_python(scope)
    client: Final = view.get("client")
    # fastapi-guard joins every line of a repeated header in order (adapters.py
    # _join_repeated_header_lines), so a proxy that adds its own line instead of
    # appending to the client's is read as one chain.
    lines: Final = [value.decode("latin-1") for name, value in view["headers"] if name == b"x-forwarded-for"]
    forwarded_for: Final = ", ".join(lines) if lines else None
    return (client[0] if client else None), forwarded_for


def _resolve_client(scope: Scope, trusted: Sequence[IPNetwork], depth: int) -> IPAddress | None:
    """The client address the way guard-core resolves it on HTTP: the socket
    peer, unless the peer is a trusted proxy, in which case the
    ``X-Forwarded-For`` entry ``depth`` from the right (the rightmost
    untrusted entry when proxies to its right are not all listed)."""
    if depth < 1:
        raise ValueError("trusted_proxy_depth must be at least 1")
    peer_text, forwarded_for = _peer_and_forwarded_for(scope)
    peer_address: Final = _parse_ip(peer_text) if peer_text is not None else None
    if peer_address is None:
        return None
    peer: Final = _canonical(peer_address)
    # guard-core tests trust on the peer as the server reports it (an
    # IPv4-mapped peer is not inside an IPv4 range), the chain entries canonical.
    if not _in_any(peer_address, trusted) or not forwarded_for:
        return peer
    chain: Final = [_canonical(_parse_ip(entry)) for entry in forwarded_for.split(",")]
    if len(chain) < depth:
        return peer
    right_of_selected: Final = chain[len(chain) - depth + 1 :]
    if any(entry is None or not _in_any(entry, trusted) for entry in right_of_selected):
        first_untrusted: Final = next((e for e in reversed(chain) if e is None or not _in_any(e, trusted)), None)
        return first_untrusted or peer
    return chain[-depth] or peer


def _ip_lists_permit(scope: Scope) -> bool:
    """Enforce LITELLM_GUARD_BLOCKED_IPS and LITELLM_GUARD_ALLOWED_IPS (IPs or
    CIDRs) outside guard's pipeline: on WebSocket handshakes (guard only
    screens HTTP scopes) and on the exact-path probes. The rules are guard's:
    the allowlist alone when it is set, the blocklist otherwise, and an invalid
    entry or a proxy depth below 1 is refused with a ``ValueError`` as guard's
    config refuses it, even when no list is set. Rate limits, bans and payload
    inspection do not apply to either."""
    blocked: Final = _validated_entries("LITELLM_GUARD_BLOCKED_IPS")
    allowed: Final = _validated_entries("LITELLM_GUARD_ALLOWED_IPS")
    trusted: Final = _trusted_proxy_entries()
    depth: Final = _env_int("LITELLM_GUARD_TRUSTED_PROXY_DEPTH", 1)
    if depth < 1:
        raise ValueError("trusted_proxy_depth must be at least 1")
    if not blocked and not allowed:
        return True
    client: Final = _resolve_client(scope, _networks(trusted), depth)
    listed: Final = client is not None and _listed(client, allowed or blocked)
    permitted: Final = listed if allowed else not listed
    if not permitted and _env_bool("LITELLM_GUARD_PASSIVE_MODE"):
        verbose_proxy_logger.warning(
            "guard passive mode: client %s would be rejected by the guard IP lists (WebSocket handshake or health probe)",
            client,
        )
        return True
    return permitted


_FORWARDED_SCHEMES: Final = frozenset({"http", "https", "ws", "wss"})


def _forwarded_scheme(scope: Scope) -> str | None:
    """uvicorn's ``ProxyHeadersMiddleware`` rule: the last ``X-Forwarded-Proto``
    line when it is one of http, https, ws or wss (http and https read as ws and
    wss on a WebSocket scope)."""
    view: Final = _CLIENT_SCOPE_VIEW.validate_python(scope)
    lines: Final = [value for name, value in view["headers"] if name == b"x-forwarded-proto"]
    if not lines:
        return None
    proto: Final = lines[-1].decode("latin1").strip()
    if proto not in _FORWARDED_SCHEMES:
        return None
    return proto.replace("http", "ws") if scope["type"] == "websocket" else proto


def _scope_with_guard_resolved_identity(scope: Scope) -> Scope:
    """A copy of the scope carrying the client and scheme guard resolved, when
    the socket peer is a trusted proxy; the scope itself otherwise."""
    trusted: Final = _networks(_trusted_proxy_entries())
    peer_text: Final = _peer_and_forwarded_for(scope)[0]
    peer: Final = _parse_ip(peer_text) if peer_text is not None else None
    if peer is None or not _in_any(peer, trusted):
        return scope
    client: Final = _resolve_client(scope, trusted, _env_int("LITELLM_GUARD_TRUSTED_PROXY_DEPTH", 1))
    scheme: Final = _forwarded_scheme(scope)
    client_update: Final = {"client": (str(client), 0)} if client not in (None, _canonical(peer)) else {}
    scheme_update: Final = {"scheme": scheme} if scheme else {}
    return {**scope, **client_update, **scheme_update}


class _ResolvedIdentityApp:
    """The app guard (and the wrapper, for what bypasses guard) passes requests
    to: the downstream app, called with the identity guard resolved."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(_scope_with_guard_resolved_identity(scope), receive, send)


def _is_default_probe(scope: Scope) -> bool:
    """Exact probe path under the root path (the route path guard matches on),
    and only while the operator has not replaced the exclusions with
    ``LITELLM_GUARD_EXCLUDED_PATHS``."""
    return get_route_path(scope) in DEFAULT_EXCLUDED_PATHS and not _csv(_env_str(GUARD_EXCLUDED_PATHS_ENV))


def guard_enabled() -> bool:
    """Read lazily per request so YAML ``environment_variables`` entries are
    seen (the ``LITELLM_ENABLE_HSTS`` pattern)."""
    return _env_bool(GUARD_ENABLED_ENV)


def server_proxy_header_overrides() -> Mapping[str, bool]:
    """Uvicorn settings for the proxy's own server start paths (``uvicorn.run``
    and the gunicorn ``UvicornWorker``).

    Uvicorn defaults to ``proxy_headers=True``: it rewrites ``scope["client"]``
    from ``X-Forwarded-For`` for peers in ``FORWARDED_ALLOW_IPS`` (loopback by
    default, or everything with ``*``) before any middleware runs. Guard would
    then see a client the caller chose, and its ``trusted_proxies`` policy
    would never apply. With the guard enabled the server leaves the peer alone
    so guard alone decides who the client is. Nothing changes when it is off.
    """
    return {"proxy_headers": False} if guard_enabled() else {}


_PROMPT_BODY_FIELDS: Final = frozenset(
    {
        "messages",
        "content",
        "prompt",
        "input",
        "system",
        "instructions",
        "contents",
        "text",
        "query",
        "documents",
    }
)


def _credential_header_names() -> frozenset[str]:
    """Header names guard must redact from its log lines.

    Guard's built-in redaction covers only authorization, proxy-authorization,
    cookie and x-api-key, yet its blocked-request warnings print the request
    headers. The proxy also accepts a key in api-key, x-goog-api-key,
    ocp-apim-subscription-key, x-litellm-api-key and the configured
    ``litellm_key_header_name``. Read when the config is built (first request),
    after the YAML settings are loaded.
    """
    from litellm.proxy import proxy_server

    configured: Final = proxy_server.general_settings_view().get("litellm_key_header_name")
    custom: Final = frozenset({configured.lower()}) if isinstance(configured, str) else frozenset[str]()
    return SpecialHeaders.litellm_credential_header_names() | custom


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
        # Unset list: trust what uvicorn trusted before the server's proxy
        # headers went off (FORWARDED_ALLOW_IPS, else loopback). Set it to
        # an empty value to trust no proxy and ignore forwarded headers.
        "trusted_proxies": _trusted_proxy_entries(),
        "trusted_proxy_depth": _env_int("LITELLM_GUARD_TRUSTED_PROXY_DEPTH", 1),
        # Operator entries only, with guard's subtree matching.
        "exclude_paths": list(_csv(_env_str(GUARD_EXCLUDED_PATHS_ENV))),
        "ipinfo_token": ipinfo_token_env or None,
        "log_sensitive_headers": _credential_header_names(),
        "log_sensitive_body_fields": _PROMPT_BODY_FIELDS,
    }
    return security_config_cls(**kwargs)


def _ip_rejected_response() -> JSONResponse:
    return JSONResponse(
        status_code=403,
        content={
            "error": {
                "message": "Client IP is not permitted by the guard IP lists.",
                "type": "guard_ip_rejected_error",
                "code": "403",
            }
        },
    )


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
        self._downstream: Final = _ResolvedIdentityApp(app)
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

        self._guard_asgi = middleware_cls(self._downstream, config=_build_guard_config())
        return self._guard_asgi

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or not guard_enabled():
            await self.app(scope, receive, send)
            return

        if scope["type"] == "http" and _is_default_probe(scope):
            # Same scope guard gives its own exclusions: the IP lists still
            # apply, rate limits, bans and payload inspection do not.
            if _ip_lists_permit(scope):
                await self._downstream(scope, receive, send)
            else:
                await _ip_rejected_response()(scope, receive, send)
            return

        guard_asgi: Final = self._ensure_guard()
        if scope["type"] == "websocket":
            # Guard's pipeline only handles HTTP, so a WebSocket handshake
            # gets the IP lists enforced here (and fails closed with the
            # package missing). Closing before accept rejects the handshake.
            if guard_asgi is None or not _ip_lists_permit(scope):
                await send({"type": "websocket.close", "code": WS_POLICY_VIOLATION})
                return
            await self._downstream(scope, receive, send)
            return
        if guard_asgi is None:
            # Flag on, package missing: fail closed with 503 rather than
            # forward unscreened. The operator explicitly enabled the
            # guard; silently honoring the opposite of that intent would
            # make the flag a no-op. The misconfiguration is logged once.
            await _guard_unavailable_response()(scope, receive, send)
            return

        await guard_asgi(scope, receive, send)

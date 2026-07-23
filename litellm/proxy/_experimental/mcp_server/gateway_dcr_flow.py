"""The gateway-level DCR flow for the aggregate ``/mcp`` endpoint (``mcp_gateway_dcr``).

An OAuth-only DCR client (Claude Desktop, Claude Code, MCP Inspector) pointed at the
aggregate ``/mcp`` endpoint discovers the gateway as its authorization server (PR 1 of
this track) and then walks the flow implemented here:

1. ``POST /register``: stateless dynamic client registration. The ``client_id`` IS the
   registration: the client's redirect URIs are sealed into it with the repo's
   authenticated symmetric helper, so nothing is persisted and a forged or tampered
   client_id simply fails to open. Clients are always public (``token_endpoint_auth_method
   "none"``); PKCE S256 is what protects the code.
2. ``GET /authorize``: validates the client and redirect URI, requires S256 PKCE, and
   interposes LiteLLM sign-in. Without a session cookie the browser is sent through
   ``/sso/key/generate`` with a same-origin ``return_to`` so it lands back here after
   login. With a session, the flow parameters and the SSO user are sealed into a per-flow
   HttpOnly cookie (the same pattern as the upstream OAuth state relay) and the browser is
   sent to the connect page, where the user authorizes individual servers (vaulting those
   tokens server-side) before finishing.
3. ``POST /authorize/complete``: the deliberate finish step. A POST (not GET) bound to the
   SameSite=Lax flow cookie, so a cross-site link cannot silently mint a code with the
   victim's session, and the signed-in user must match the user sealed into the flow.
   Mints a short-lived, single-use, gateway-sealed authorization code and redirects to the
   client's registered redirect URI.
4. ``POST /token``: exchanges the code (PKCE-verified, client- and redirect-bound,
   single-use) for the identity-only session tokens of
   :mod:`.outbound_credentials.session_token`, re-validating that the litellm user is
   still active first; the ``refresh_token`` grant rotates the pair the same way.

Nothing here stores state server-side except the single-use code guard (a TTL cache
entry). Every sealed value is authenticated encryption over the proxy salt/master key
family, opened totally (bad input maps to an OAuth error, never a raise), and every
identity is a stable reference re-validated live at mint, refresh, and (in the admission
PR) tool-call time. Upstream server credentials never appear anywhere in this flow; they
are vaulted per user by the existing ``/v1/mcp`` authorize endpoints and resolved at
egress by user id.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from base64 import urlsafe_b64encode
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Awaitable, Callable, Literal, TypeVar
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.caching.caching import DualCache
from litellm.proxy._experimental.mcp_server.oauth_utils import (
    TOKEN_NO_CACHE_HEADERS,
    get_request_base_url,
    is_loopback_redirect_host,
    validate_redirect_uri_shape,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
    SessionRefreshOpened,
    open_session_refresh_bearer,
    session_keys_from_master_key,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    SESSION_REFRESH_TTL_SECONDS,
    MintedSessionToken,
    SessionKeys,
    SessionPrincipal,
    mint_session_refresh_token,
    mint_session_token,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)

GATEWAY_DCR_CLIENT_ID_PREFIX = "llm_dcrc_"
"""Marker prefix on every gateway-issued DCR client_id so the root authorize/token
endpoints can route an aggregate-flow request without decrypting, and existing per-server
flows (whose client_ids are upstream-issued) are never captured by the aggregate arm."""

GATEWAY_AUTH_CODE_PREFIX = "llm_gcode_"
"""Marker prefix on the gateway-sealed authorization code, distinct from the bridge
``llm_bcode_`` so neither flow can consume the other's codes."""

CONNECT_FLOW_COOKIE_PREFIX = "mcp_connect_flow_"
"""Per-flow HttpOnly cookie holding the sealed connect flow, keyed by a short random
handle carried in the connect-page URL (the same handle-plus-cookie pattern as the
``mcp_oauth_state_`` upstream relay, for the same reasons: replica-safe with no
server-side session store, and the sealed value never appears in a URL)."""

CONNECT_FLOW_TTL_SECONDS = 600
GATEWAY_AUTH_CODE_TTL_SECONDS = 120
_CLAIM_TTL_BUFFER_SECONDS = 60
_USED_CODE_CACHE_PREFIX = "mcp_gateway_dcr_code_used:"
_USED_FLOW_CACHE_PREFIX = "mcp_gateway_dcr_flow_used:"
_USED_REFRESH_CACHE_PREFIX = "mcp_gateway_dcr_refresh_used:"

MAX_REDIRECT_URIS = 3
MAX_REDIRECT_URI_LENGTH = 256
MAX_CLIENT_ID_LENGTH = 2048
"""Registration bounds. They exist to bound the sealed client_id, which rides inside
every session-token claim set: 3 URIs of 256 bytes seal to roughly 1.2KB, comfortably
under this cap and under the session token's own 4KB ceiling. Claude Desktop and MCP
Inspector register one or two redirect URIs."""

MAX_STATE_LENGTH = 1024
"""Bound on the client ``state`` sealed into the flow cookie and echoed on the auth-code
redirect. An unbounded ``state`` can push the sealed cookie past the browser's ~4KB cap
(silently dropped, breaking the flow); spec clients send a short opaque value."""

MIN_CODE_VERIFIER_LENGTH = 43
MAX_CODE_VERIFIER_LENGTH = 128
"""RFC 7636 section 4.1 bounds for the PKCE ``code_verifier``. Enforced so an out-of-range
verifier gets a clean ``invalid_request`` instead of an opaque PKCE-mismatch."""

_UNPREFIXED = ""
"""Prefix for a sealed value that carries no wire marker because it is never routed by
prefix (the connect flow lives only in its own per-handle cookie, opened by that one
handle). Named so the empty-string argument to ``_seal`` / ``_open_sealed`` reads as
deliberate rather than a typo."""

_CLIENT_RECORD_DEBUG_KEY = "gateway_dcr_client"
_CONNECT_FLOW_DEBUG_KEY = "gateway_connect_flow"
_AUTH_CODE_DEBUG_KEY = "gateway_authorization_code"

ReloadUserFailure = Literal["unresolvable", "unavailable", "no_active_key"]
ReloadUser = Callable[[str], Awaitable[ReloadUserFailure | None]]
"""Injected live-user revalidation (the token endpoint's mirror of admission):
``None`` means the user is active; ``unavailable`` is a retryable DB outage; anything
else fails the grant closed."""


class GatewayDcrClient(BaseModel):
    """The registration record sealed into a gateway DCR ``client_id``.

    ``extra="forbid"`` so a sealed value of another type (an auth code, a connect flow)
    that happened to decrypt under the shared key can never validate as a client record:
    cross-type confusion is rejected at the model boundary, not left to differing required
    fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    redirect_uris: tuple[str, ...] = Field(min_length=1, max_length=MAX_REDIRECT_URIS)
    iat: int


class _ConnectFlow(BaseModel):
    """One in-flight authorize: the SSO user it belongs to and the client parameters
    needed to mint the code at the finish step. Sealed into the per-flow cookie. ``jti``
    makes the flow single-use at complete; ``extra="forbid"`` rejects cross-type
    confusion."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    user_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    redirect_uri: str = Field(min_length=1)
    state: str
    code_challenge: str = Field(min_length=1)
    jti: str = Field(min_length=1)
    exp: int


class _GatewayAuthCode(BaseModel):
    """The gateway-sealed authorization code: the user consent it represents and the
    bindings the token endpoint must verify (client, redirect URI, PKCE challenge),
    plus a ``jti`` for the single-use guard. ``extra="forbid"`` rejects cross-type
    confusion."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    user_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    redirect_uri: str = Field(min_length=1)
    code_challenge: str = Field(min_length=1)
    jti: str = Field(min_length=1)
    iat: int
    exp: int


def is_gateway_dcr_client_id(client_id: str | None) -> bool:
    """Cheap prefix routing test so the root endpoints only enter the aggregate arm for
    clients this flow registered; every other client_id keeps today's behavior."""
    return client_id is not None and client_id.startswith(GATEWAY_DCR_CLIENT_ID_PREFIX)


def _oauth_error(status_code: int, error: str, description: str) -> JSONResponse:
    """RFC 6749 section 5.2 / RFC 7591 section 3.2.2 error body. Descriptions carry no
    token, code, or URL material so they are safe to relay to any client."""
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "error_description": description},
        headers=TOKEN_NO_CACHE_HEADERS,
    )


def _seal(prefix: str, payload: BaseModel) -> str:
    return prefix + encrypt_value_helper(payload.model_dump_json())


_SealedModelT = TypeVar("_SealedModelT", bound=BaseModel)


def _open_sealed(value: str, prefix: str, model: type[_SealedModelT], debug_key: str) -> _SealedModelT | None:
    """Open a sealed value totally: anything that is not prefix-shaped, does not decrypt,
    or does not validate returns ``None`` for the caller to map onto an OAuth error."""
    if not value.startswith(prefix):
        return None
    decrypted = decrypt_value_helper(value[len(prefix) :], debug_key, return_original_value=False)
    if not isinstance(decrypted, str):
        return None
    try:
        return model.model_validate_json(decrypted)
    except ValidationError:
        return None


def open_gateway_dcr_client(client_id: str) -> GatewayDcrClient | None:
    return _open_sealed(client_id, GATEWAY_DCR_CLIENT_ID_PREFIX, GatewayDcrClient, _CLIENT_RECORD_DEBUG_KEY)


async def register_aggregate_client(request: Request, request_body: Mapping[str, object]) -> Response:
    """RFC 7591 dynamic registration against the gateway itself, statelessly.

    Only ``redirect_uris`` is authoritative; every client is registered as a public
    ``token_endpoint_auth_method "none"`` client regardless of what it asked for (RFC
    7591 lets the server override metadata), because the gateway never issues client
    secrets: possession of a secret would add nothing over the mandatory S256 PKCE, and a
    stateless registration has nowhere to keep one. Nothing is persisted, so open
    registration cannot be used to fill storage.

    Redirect-URI *hygiene* is not decided here: :func:`validate_redirect_uri_shape` is
    the single owner of that rule across the MCP OAuth surface, so allowlisted native
    callbacks (``cursor://``) are accepted and fragments, missing hosts, userinfo
    (``https://claude.ai@attacker.example/cb``) and backslash hosts are rejected exactly
    as they are on /authorize and /callback.

    What this endpoint does decide is its own trust policy, which is deliberately wider
    than :func:`validate_trusted_redirect_uri`'s: registration is *public*, so any https
    client may register (that is what lets a hosted MCP client register at all), and the
    controls are mandatory S256 PKCE plus the consent screen showing the client origin.
    http is confined to loopback per RFC 8252 section 7.3.
    """
    raw_uris = request_body.get("redirect_uris")
    if not isinstance(raw_uris, list) or not raw_uris or len(raw_uris) > MAX_REDIRECT_URIS:
        return _oauth_error(
            400,
            "invalid_redirect_uri",
            f"redirect_uris must be a list of 1 to {MAX_REDIRECT_URIS} URIs",
        )
    if not all(isinstance(uri, str) and len(uri) <= MAX_REDIRECT_URI_LENGTH for uri in raw_uris):
        return _oauth_error(
            400,
            "invalid_redirect_uri",
            f"each redirect URI must be a string of at most {MAX_REDIRECT_URI_LENGTH} characters",
        )
    for uri in raw_uris:
        parsed = urlparse(uri)
        try:
            if validate_redirect_uri_shape(parsed):
                continue  # allowlisted native callback, e.g. cursor://
        except HTTPException as exc:
            # The shared validator speaks HTTP; RFC 7591 registration answers with an OAuth
            # error object, so translate the shape without re-deciding the rule.
            return _oauth_error(400, "invalid_redirect_uri", str(exc.detail))
        if parsed.scheme == "https" or (parsed.scheme == "http" and is_loopback_redirect_host(parsed)):
            continue
        return _oauth_error(
            400,
            "invalid_redirect_uri",
            "each redirect URI must be https, http on a loopback host, or a registered native callback",
        )
    now = datetime.now(timezone.utc)
    client_id = _seal(
        GATEWAY_DCR_CLIENT_ID_PREFIX, GatewayDcrClient(redirect_uris=tuple(raw_uris), iat=int(now.timestamp()))
    )
    if len(client_id) > MAX_CLIENT_ID_LENGTH:
        return _oauth_error(400, "invalid_client_metadata", "registered metadata is too large")
    return JSONResponse(
        status_code=201,
        content={
            "client_id": client_id,
            "client_id_issued_at": int(now.timestamp()),
            "redirect_uris": list(raw_uris),
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        },
    )


def _flow_cookie_name(handle: str) -> str:
    return f"{CONNECT_FLOW_COOKIE_PREFIX}{handle}"


def _cookie_path_and_secure(request: Request) -> tuple[str, bool]:
    parsed = urlparse(get_request_base_url(request))
    return parsed.path or "/", parsed.scheme == "https"


def _append_query_params(url: str, params: dict[str, str]) -> str:
    parsed = urlparse(url)
    query = parse_qsl(parsed.query, keep_blank_values=True) + list(params.items())
    return urlunparse(parsed._replace(query=urlencode(query)))


def relative_request_url(request: Request) -> str:
    """The request's own path and query as a same-origin ``return_to`` target for the
    login round-trip; relative by construction, so it can never leave the gateway."""
    path = request.url.path
    return f"{path}?{request.url.query}" if request.url.query else path


def aggregate_authorize(
    request: Request,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str | None,
    code_challenge_method: str | None,
    response_type: str | None,
    session_user_id: str | None,
) -> Response:
    """The aggregate authorize verb: validate the client, require S256 PKCE, interpose
    LiteLLM sign-in, and hand the browser to the connect page with the flow sealed into a
    per-flow cookie.

    Validation failures respond directly with 400 and never redirect: per RFC 6749
    section 4.1.2.1 an unvalidated redirect URI must not receive an error redirect, and
    once the client is at fault there is no trusted place to send the browser.
    """
    client = open_gateway_dcr_client(client_id)
    if client is None:
        return _oauth_error(400, "invalid_client", "unknown or malformed client_id")
    if redirect_uri not in client.redirect_uris:
        return _oauth_error(400, "invalid_request", "redirect_uri is not registered for this client")
    if response_type != "code":
        return _oauth_error(400, "unsupported_response_type", "response_type must be 'code'")
    if not code_challenge or code_challenge_method != "S256":
        return _oauth_error(
            400,
            "invalid_request",
            "PKCE is required: send code_challenge with code_challenge_method=S256",
        )
    if len(state) > MAX_STATE_LENGTH:
        return _oauth_error(400, "invalid_request", f"state must be at most {MAX_STATE_LENGTH} characters")
    base_url = get_request_base_url(request)
    if session_user_id is None:
        login_url = f"{base_url}/sso/key/generate?{urlencode({'return_to': relative_request_url(request)})}"
        return RedirectResponse(login_url, status_code=303)
    now = datetime.now(timezone.utc)
    handle = secrets.token_urlsafe(24)
    flow = _ConnectFlow(
        user_id=session_user_id,
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        jti=secrets.token_urlsafe(24),
        exp=int(now.timestamp()) + CONNECT_FLOW_TTL_SECONDS,
    )
    connect_url = _append_query_params(
        f"{base_url}/ui/chat/integrations",
        {"connect_flow": handle, "connect_client": _origin_only(redirect_uri)},
    )
    response = RedirectResponse(connect_url, status_code=303)
    path, secure = _cookie_path_and_secure(request)
    response.set_cookie(
        key=_flow_cookie_name(handle),
        value=_seal(_UNPREFIXED, flow),
        max_age=CONNECT_FLOW_TTL_SECONDS,
        path=path,
        secure=secure,
        httponly=True,
        samesite="lax",
    )
    return response


def _origin_only(url: str) -> str:
    """Scheme+host for display on the connect page; never the full redirect URI, whose
    path or query could carry values that do not belong in a page URL or logs."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else ""


async def complete_connect_flow(
    request: Request,
    flow_handle: str,
    session_user_id: str | None,
    cache: DualCache,
) -> Response:
    """The deliberate finish step of the connect flow: mint the gateway authorization
    code and send the browser back to the client.

    Reached by POST so a cross-site GET cannot trigger it, and bound to the HttpOnly
    per-flow cookie plus an exact match between the signed-in user and the user sealed
    into the flow: a link crafted by another party dies here with ``access_denied``
    instead of minting a code for the victim's identity. The flow is single-use (an atomic
    claim on its ``jti``), so a double-submit cannot mint two codes from one sign-in.
    """
    sealed_flow = request.cookies.get(_flow_cookie_name(flow_handle))
    if sealed_flow is None:
        return _oauth_error(400, "invalid_request", "unknown or expired connect flow")
    flow = _open_sealed(sealed_flow, _UNPREFIXED, _ConnectFlow, _CONNECT_FLOW_DEBUG_KEY)
    if flow is None:
        return _oauth_error(400, "invalid_request", "unknown or expired connect flow")
    now = datetime.now(timezone.utc)
    if now.timestamp() >= flow.exp:
        return _oauth_error(400, "invalid_request", "the connect flow has expired; restart the connection")
    if session_user_id is None:
        return _oauth_error(401, "login_required", "sign in to LiteLLM to finish connecting")
    if session_user_id != flow.user_id:
        return _oauth_error(403, "access_denied", "the signed-in user does not match this connect flow")
    if not await _SingleUseGuard(cache).claim(
        f"{_USED_FLOW_CACHE_PREFIX}{flow.jti}", CONNECT_FLOW_TTL_SECONDS + _CLAIM_TTL_BUFFER_SECONDS
    ):
        return _oauth_error(400, "invalid_request", "this connect flow was already completed; restart the connection")
    code = _seal(
        GATEWAY_AUTH_CODE_PREFIX,
        _GatewayAuthCode(
            user_id=flow.user_id,
            client_id=flow.client_id,
            redirect_uri=flow.redirect_uri,
            code_challenge=flow.code_challenge,
            jti=secrets.token_urlsafe(24),
            iat=int(now.timestamp()),
            exp=int(now.timestamp()) + GATEWAY_AUTH_CODE_TTL_SECONDS,
        ),
    )
    params = {"code": code, **({"state": flow.state} if flow.state else {})}
    response = RedirectResponse(_append_query_params(flow.redirect_uri, params), status_code=303)
    path, secure = _cookie_path_and_secure(request)
    response.delete_cookie(key=_flow_cookie_name(flow_handle), path=path, secure=secure, httponly=True, samesite="lax")
    return response


def _pkce_verifier_matches(code_verifier: str, code_challenge: str) -> bool:
    """RFC 7636 S256 verification, total over hostile input. The comparison is over bytes
    so a non-ASCII ``code_challenge`` (which reaches here unvalidated from the client's
    authorize request) simply fails to match instead of raising ``TypeError`` the way
    ``hmac.compare_digest`` does on two ``str`` with non-ASCII content. The verifier is
    ASCII per spec; a compliant client's challenge is base64url and matches."""
    digest = hashlib.sha256(code_verifier.encode("ascii", "replace")).digest()
    computed = urlsafe_b64encode(digest).rstrip(b"=")
    return hmac.compare_digest(computed, code_challenge.encode("utf-8"))


class _SingleUseGuard:
    """Atomic single-use claim for a one-time id (an auth-code, connect-flow ``jti``, or refresh-token
    ``jti``) over the injected proxy cache.

    Uses an atomic increment rather than a get-then-set: two concurrent redemptions of the same id
    cannot both observe "unused", because exactly one increment returns 1. The claim IS the gate, so it
    fails closed. Crucially, the increment must be recorded in a backend SHARED across replicas, or the
    single-use property is per-worker only (each replica's in-memory counter returns 1, so a captured
    id replays through a different worker):

    - When a Redis backend is configured it is the SOLE authority: the claim goes straight to Redis
      (``INCR`` is atomic across replicas), and any Redis fault fails the claim CLOSED — it never falls
      back to the per-worker in-memory count (``DualCache.async_increment_cache`` does fall back, which
      is exactly the replay window this avoids).
    - With no Redis configured (single-replica) the in-memory increment is authoritative within the one
      process. A multi-worker deployment must run Redis for the guarantee to hold across workers.

    The id's own TTL is the outer bound. For the auth code, PKCE binding is the primary defense against
    interception; this makes the RFC 6749 4.1.2 single-use property reliable on top of it."""

    def __init__(self, cache: DualCache) -> None:
        self._cache = cache

    async def claim(self, key: str, ttl_seconds: int) -> bool:
        """Atomically claim ``key``. ``True`` iff this caller is the first (increment to 1); ``False``
        on a replay (>1) or when the claim could not be recorded in the shared backend (fail closed)."""
        from litellm.proxy.proxy_server import redis_usage_cache  # noqa: PLC0415  # circular import at module load

        # Resolve the shared authority HERE rather than trusting the injected cache: callers pass
        # user_api_key_cache, which only carries a redis_cache when enable_redis_auth_cache is set
        # (off by default), so a guard that read its injected cache silently degraded every claim to
        # a per-worker count on a stock multi-worker deployment. redis_usage_cache is the store the
        # proxy already treats as cross-worker, so no call site can wire the guarantee away.
        redis_cache = redis_usage_cache or getattr(self._cache, "redis_cache", None)
        if redis_cache is not None:
            # Shared, atomic authority for multi-replica deployments. Claim ONLY against Redis and fail
            # CLOSED on any Redis fault (async_increment re-raises) rather than fall back to the
            # per-worker in-memory count, which would let each replica observe count==1 and replay the id.
            try:
                count = await redis_cache.async_increment(key, 1, ttl=ttl_seconds)
            except Exception as e:  # noqa: BLE001  # ANY Redis fault fails the single-use claim closed
                verbose_logger.warning(
                    "mcp gateway single-use claim: shared cache backend unavailable, failing closed: %s", e
                )
                return False
            return count == 1
        # No shared backend configured (single-replica): the in-memory increment is authoritative.
        count = await self._cache.async_increment_cache(key, 1, ttl=ttl_seconds, local_only=True)
        return count == 1


def _session_token_pair(principal: SessionPrincipal, keys: SessionKeys, now: datetime) -> Response:
    access = mint_session_token(principal, keys, now)
    refresh = mint_session_refresh_token(principal, keys, now)
    if not isinstance(access, MintedSessionToken) or not isinstance(refresh, MintedSessionToken):
        return _oauth_error(500, "server_error", "failed to mint the session credential")
    return JSONResponse(
        status_code=200,
        content={
            "access_token": access.token.get_secret_value(),
            "token_type": "Bearer",
            "expires_in": int((access.expires_at - now).total_seconds()),
            "refresh_token": refresh.token.get_secret_value(),
        },
        headers=TOKEN_NO_CACHE_HEADERS,
    )


def _reload_failure_response(failure: ReloadUserFailure) -> Response:
    """Map the live-user revalidation failure onto its OAuth error, exhaustively, so a new
    ``ReloadUserFailure`` member is a type error here rather than silently 400ing."""
    match failure:
        case "unavailable":
            return _oauth_error(503, "temporarily_unavailable", "the gateway database is unavailable; retry")
        case "unresolvable":
            return _oauth_error(500, "server_error", "the gateway is not configured to resolve users")
        case "no_active_key":
            return _oauth_error(400, "invalid_grant", "the user for this grant is no longer active")
        case _:
            assert_never(failure)


async def aggregate_token(
    request: Request,
    grant_type: str,
    code: str | None,
    redirect_uri: str | None,
    client_id: str,
    code_verifier: str | None,
    refresh_token: str | None,
    master_key: str | None,
    reload_user: ReloadUser,
    cache: DualCache,
) -> Response:
    """The aggregate token verb: authorization_code and refresh_token grants for the
    identity-only session pair. Every path re-validates the litellm user live before
    minting, so a deactivated user cannot obtain or renew a session."""
    if master_key is None:
        verbose_logger.error("mcp_gateway_dcr token grant rejected: no master_key configured")
        return _oauth_error(500, "server_error", "the gateway has no master key configured")
    keys = session_keys_from_master_key(master_key)
    now = datetime.now(timezone.utc)
    if grant_type == "authorization_code":
        return await _authorization_code_grant(
            code=code,
            redirect_uri=redirect_uri,
            client_id=client_id,
            code_verifier=code_verifier,
            keys=keys,
            now=now,
            reload_user=reload_user,
            guard=_SingleUseGuard(cache),
        )
    if grant_type == "refresh_token":
        return await _refresh_token_grant(
            refresh_token=refresh_token,
            client_id=client_id,
            keys=keys,
            now=now,
            reload_user=reload_user,
            guard=_SingleUseGuard(cache),
        )
    return _oauth_error(400, "unsupported_grant_type", "grant_type must be authorization_code or refresh_token")


async def _authorization_code_grant(
    code: str | None,
    redirect_uri: str | None,
    client_id: str,
    code_verifier: str | None,
    keys: SessionKeys,
    now: datetime,
    reload_user: ReloadUser,
    guard: _SingleUseGuard,
) -> Response:
    if not code or not redirect_uri or not code_verifier:
        return _oauth_error(400, "invalid_request", "code, redirect_uri, and code_verifier are required")
    if not MIN_CODE_VERIFIER_LENGTH <= len(code_verifier) <= MAX_CODE_VERIFIER_LENGTH:
        return _oauth_error(400, "invalid_request", "code_verifier must be 43 to 128 characters (RFC 7636)")
    parsed = _open_sealed(code, GATEWAY_AUTH_CODE_PREFIX, _GatewayAuthCode, _AUTH_CODE_DEBUG_KEY)
    if parsed is None:
        return _oauth_error(400, "invalid_grant", "the authorization code is invalid")
    if now.timestamp() >= parsed.exp:
        return _oauth_error(400, "invalid_grant", "the authorization code has expired")
    if client_id != parsed.client_id or redirect_uri != parsed.redirect_uri:
        return _oauth_error(400, "invalid_grant", "the authorization code was issued to a different client")
    if not _pkce_verifier_matches(code_verifier, parsed.code_challenge):
        return _oauth_error(400, "invalid_grant", "PKCE verification failed")
    # Revalidate the user BEFORE claiming the code, so a transient DB outage (a retryable
    # 503) does not consume a still-valid code and force the client to restart sign-in.
    failure = await reload_user(parsed.user_id)
    if failure is not None:
        return _reload_failure_response(failure)
    # Atomic single-use claim is the gate: on a concurrent double-redeem exactly one caller
    # wins, and a claim that cannot be recorded fails closed.
    if not await guard.claim(
        f"{_USED_CODE_CACHE_PREFIX}{parsed.jti}", GATEWAY_AUTH_CODE_TTL_SECONDS + _CLAIM_TTL_BUFFER_SECONDS
    ):
        return _oauth_error(400, "invalid_grant", "the authorization code was already used")
    return _session_token_pair(SessionPrincipal(user_id=parsed.user_id, client_id=client_id), keys, now)


async def _refresh_token_grant(
    refresh_token: str | None,
    client_id: str,
    keys: SessionKeys,
    now: datetime,
    reload_user: ReloadUser,
    guard: _SingleUseGuard,
) -> Response:
    if not refresh_token:
        return _oauth_error(400, "invalid_request", "refresh_token is required")
    opened = open_session_refresh_bearer(refresh_token, keys, now, expected_client_id=client_id)
    if not isinstance(opened, SessionRefreshOpened):
        return _oauth_error(400, "invalid_grant", "the refresh token is invalid for this client")
    failure = await reload_user(opened.principal.user_id)
    if failure is not None:
        return _reload_failure_response(failure)
    # Refresh-token rotation (OAuth 2.0 Security BCP section 4.13): the presented refresh token is
    # single-use. Claim its jti before issuing the replacement pair, so a captured or replayed
    # refresh token cannot mint a second pair after the legitimate holder rotated. Claimed AFTER
    # user revalidation so a transient DB 503 does not burn a still-valid token; a claim that
    # cannot be recorded fails closed, exactly like the authorization-code path.
    if not await guard.claim(
        f"{_USED_REFRESH_CACHE_PREFIX}{opened.jti}", SESSION_REFRESH_TTL_SECONDS + _CLAIM_TTL_BUFFER_SECONDS
    ):
        return _oauth_error(400, "invalid_grant", "the refresh token was already used")
    return _session_token_pair(opened.principal, keys, now)

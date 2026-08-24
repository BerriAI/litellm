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
import html
import secrets
from base64 import urlsafe_b64encode
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeVar
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import ReadOnly, TypedDict, assert_never

from litellm._logging import verbose_logger
from litellm.caching.caching import DualCache
from litellm.proxy._experimental.mcp_server.oauth_utils import (
    TOKEN_NO_CACHE_HEADERS,
    canonical_resource_uri,
    canonicalize_url_identity,
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
    SessionAudience,
    SessionKeys,
    SessionPrincipal,
    mint_session_refresh_token,
    mint_session_token,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.common_utils.html_forms.native_client_consent import (
    render_native_client_consent_page,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer

GATEWAY_DCR_CLIENT_ID_PREFIX: Final = "llm_dcrc_"
"""Marker prefix on every gateway-issued DCR client_id so the root authorize/token
endpoints can route an aggregate-flow request without decrypting, and existing per-server
flows (whose client_ids are upstream-issued) are never captured by the aggregate arm."""

GATEWAY_AUTH_CODE_PREFIX: Final = "llm_gcode_"
"""Marker prefix on the gateway-sealed authorization code, distinct from the bridge
``llm_bcode_`` so neither flow can consume the other's codes."""

CONNECT_FLOW_COOKIE_PREFIX: Final = "mcp_connect_flow_"
"""Per-flow HttpOnly cookie holding the sealed connect flow, keyed by a short random
handle carried in the connect-page URL (the same handle-plus-cookie pattern as the
``mcp_oauth_state_`` upstream relay, for the same reasons: replica-safe with no
server-side session store, and the sealed value never appears in a URL)."""

CONNECT_FLOW_TTL_SECONDS: Final = 600
GATEWAY_AUTH_CODE_TTL_SECONDS: Final = 120
MANUAL_DELIVERY_AUTH_CODE_TTL_SECONDS: Final = 300
"""Lifetime of a code the user delivers by hand (headless/remote client, LIT-4863 class):
copy-pasting a callback URL from a laptop browser to an SSH session is slower than a
browser redirect, so manual-delivery codes get 5 minutes instead of 2, still well under
the 10-minute ceiling RFC 6749 section 4.1.2 recommends. Single-use and PKCE binding are
unchanged, so the longer window only extends how long the legitimate holder has to paste
it, not what an observer could do with it."""
_CLAIM_TTL_BUFFER_SECONDS: Final = 60
_USED_CODE_CACHE_PREFIX: Final = "mcp_gateway_dcr_code_used:"
_USED_FLOW_CACHE_PREFIX: Final = "mcp_gateway_dcr_flow_used:"
_USED_REFRESH_CACHE_PREFIX: Final = "mcp_gateway_dcr_refresh_used:"

MAX_REDIRECT_URIS: Final = 3
MAX_REDIRECT_URI_LENGTH: Final = 256
MAX_CLIENT_ID_LENGTH: Final = 2048
"""Registration bounds. They exist to bound the sealed client_id, which rides inside
every session-token claim set: 3 URIs of 256 bytes seal to roughly 1.2KB, comfortably
under this cap and under the session token's own 4KB ceiling. Claude Desktop and MCP
Inspector register one or two redirect URIs."""

MAX_STATE_LENGTH: Final = 1024
"""Bound on the client ``state`` sealed into the flow cookie and echoed on the auth-code
redirect. An unbounded ``state`` can push the sealed cookie past the browser's ~4KB cap
(silently dropped, breaking the flow); spec clients send a short opaque value."""

MIN_CODE_VERIFIER_LENGTH: Final = 43
MAX_CODE_VERIFIER_LENGTH: Final = 128
"""RFC 7636 section 4.1 bounds for the PKCE ``code_verifier``. Enforced so an out-of-range
verifier gets a clean ``invalid_request`` instead of an opaque PKCE-mismatch."""

_UNPREFIXED: Final = ""
"""Prefix for a sealed value that carries no wire marker because it is never routed by
prefix (the connect flow lives only in its own per-handle cookie, opened by that one
handle). Named so the empty-string argument to ``_seal`` / ``_open_sealed`` reads as
deliberate rather than a typo."""

_CLIENT_RECORD_DEBUG_KEY: Final = "gateway_dcr_client"
_CONNECT_FLOW_DEBUG_KEY: Final = "gateway_connect_flow"
_AUTH_CODE_DEBUG_KEY: Final = "gateway_authorization_code"

ReloadUserFailure = Literal["unresolvable", "unavailable", "no_active_key"]
ReloadUser = Callable[[str], Awaitable[ReloadUserFailure | None]]
"""Injected live-user revalidation (the token endpoint's mirror of admission):
``None`` means the user is active; ``unavailable`` is a retryable DB outage; anything
else fails the grant closed."""

PROXY_API_AUDIENCE: Final[SessionAudience] = "proxy_api"
"""The audience a native client (``lite login --pkce``, a Go CLI) asks for by sending the
proxy base URL itself as the RFC 8707 ``resource``: the grant then mints the proxy-API CLI
credential that LLM routes accept, instead of the MCP-only session pair."""

ProxyCredentialMintFailure = Literal[ReloadUserFailure, "not_a_member", "team_required"]


class MintedProxyCredential(BaseModel):
    model_config = ConfigDict(frozen=True)
    key: str = Field(min_length=1)
    expires_in: int = Field(gt=0)
    user_id: str = Field(min_length=1)
    team_id: str | None = None


class MintProxyCredential(Protocol):
    """Injected proxy-API credential minter ``(user_id, team_id)``: reloads the user live,
    checks team membership, refuses a teamless grant for a user who has teams to pick from,
    and mints the same credential ``lite login`` mints."""

    def __call__(
        self, user_id: str, team_id: str | None, /
    ) -> Awaitable[MintedProxyCredential | ProxyCredentialMintFailure]: ...


class ConsentTeam(BaseModel):
    model_config = ConfigDict(frozen=True)
    team_id: str = Field(min_length=1)
    team_alias: str | None = None


class LookupConsentTeams(Protocol):
    """Injected lookup of the teams a signed-in user may bind a proxy-API credential to."""

    def __call__(self, user_id: str, /) -> Awaitable[tuple[ConsentTeam, ...] | ReloadUserFailure]: ...


async def _refuse_proxy_credential(user_id: str, team_id: str | None) -> ProxyCredentialMintFailure:
    return "unresolvable"


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
    resource_server_id: str | None = None
    audience: SessionAudience | None = None


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
    resource_server_id: str | None = None
    audience: SessionAudience | None = None
    team_id: str | None = None


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
    """Serialized ``exclude_none`` for the same reason session JWTs are minted that way: an
    optional claim that is unset never reaches the wire, so during a rolling deploy a blob
    sealed by a new pod without the new claim set stays byte-compatible with predating pods
    whose strict models forbid unknown keys. This holds for every sealed artifact and every
    future optional claim by construction; it requires each optional field to default to
    ``None`` so reopening restores exactly what was sealed."""
    return prefix + encrypt_value_helper(payload.model_dump_json(exclude_none=True))


_SealedModelT = TypeVar("_SealedModelT", bound=BaseModel)


def _open_sealed(value: str, prefix: str, model: type[_SealedModelT], debug_key: str) -> _SealedModelT | None:
    """Open a sealed value totally: anything that is not prefix-shaped, does not decrypt,
    or does not validate returns ``None`` for the caller to map onto an OAuth error."""
    if not value.startswith(prefix):
        return None
    decrypted: Final = decrypt_value_helper(value[len(prefix) :], debug_key, return_original_value=False)
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
    raw_uris: Final = request_body.get("redirect_uris")
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
    now: Final = datetime.now(timezone.utc)
    client_id: Final = _seal(
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
    parsed: Final = urlparse(get_request_base_url(request))
    return parsed.path or "/", parsed.scheme == "https"


def _append_query_params(url: str, params: Iterable[tuple[str, str]]) -> str:
    parsed: Final = urlparse(url)
    query: Final = (*parse_qsl(parsed.query, keep_blank_values=True), *params)
    return urlunparse(parsed._replace(query=urlencode(query)))


def relative_request_url(request: Request) -> str:
    """The request's own path and query as a same-origin ``return_to`` target for the
    login round-trip; relative by construction, so it can never leave the gateway."""
    path: Final = request.url.path
    return f"{path}?{request.url.query}" if request.url.query else path


def resolve_scoped_resource_server(request: Request, resource: str | None) -> MCPServer | None:
    """Resolve an RFC 8707 ``resource`` value to the single gateway-managed oauth2 server it
    names, or ``None`` for every other shape: absent, the aggregate resource, a foreign
    host, an unparseable value, a multi-server path, an unknown name, or any server mode the
    keyless gateway flow does not serve (whose protected-resource metadata never directs a
    client here). ``None`` means the flow stays unscoped and byte-identical to today, so a
    hostile or confused ``resource`` can never widen anything; a resolved server only ever
    NARROWS the session via the sealed scope.

    Resolution is an IDENTITY question, deliberately free of the per-IP visibility filter:
    access is enforced where it belongs (grant intersection at admission, IP checks on the
    MCP routes), while filtering here would mint an entitlement-wide UNSCOPED bearer exactly
    when the caller asked to narrow, and would let authorize-time vs token-time IP drift
    turn a matching redemption into a spurious ``invalid_target``."""
    if resource is None:
        return None
    canonical: Final = canonical_resource_uri(resource)
    if canonical is None:
        return None
    base: Final = canonicalize_url_identity(get_request_base_url(request))
    if canonical == f"{base}/mcp" or not canonical.startswith(f"{base}/"):
        return None
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (  # noqa: PLC0415  # proxy import cycle
        MCPRequestHandler,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415  # proxy import cycle
        global_mcp_server_manager,
    )

    names: Final = MCPRequestHandler.extract_target_server_names_from_path(canonical[len(base) :])
    if len(names) != 1:
        return None
    server: Final = global_mcp_server_manager.get_mcp_server_by_name(names[0])
    if server is None or not server.is_gateway_managed_oauth2:
        return None
    return server


def aggregate_authorize(
    request: Request,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str | None,
    code_challenge_method: str | None,
    response_type: str | None,
    session_user_id: str | None,
    resource: str | None = None,
) -> Response:
    """The aggregate authorize verb: validate the client, require S256 PKCE, interpose
    LiteLLM sign-in, and hand the browser to the connect page with the flow sealed into a
    per-flow cookie.

    A per-server RFC 8707 ``resource`` naming a gateway-managed oauth2 server scopes the
    flow to that one server: the scope is sealed into the flow, carried into the code, and
    bound into the session token, while the connect page interlude runs exactly as before.

    Validation failures respond directly with 400 and never redirect: per RFC 6749
    section 4.1.2.1 an unvalidated redirect URI must not receive an error redirect, and
    once the client is at fault there is no trusted place to send the browser.
    """
    rejected: Final = _rejected_authorize_request(
        client_id, redirect_uri, state, code_challenge, code_challenge_method, response_type
    )
    if rejected is not None:
        return rejected
    base_url: Final = get_request_base_url(request)
    if session_user_id is None:
        return _login_redirect(base_url, request)
    scoped_server: Final = resolve_scoped_resource_server(request, resource)
    handle: Final = secrets.token_urlsafe(24)
    flow: Final = _new_connect_flow(
        session_user_id=session_user_id,
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge or "",
        resource_server_id=scoped_server.server_id if scoped_server is not None else None,
        audience=None,
    )
    connect_url: Final = _append_query_params(
        f"{base_url}/ui/connect",
        (("connect_flow", handle), ("connect_client", _origin_only(redirect_uri))),
    )
    response: Final = RedirectResponse(connect_url, status_code=303)
    _set_flow_cookie(response, request, handle, flow)
    return response


async def native_client_authorize(
    request: Request,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str | None,
    code_challenge_method: str | None,
    response_type: str | None,
    session_user_id: str | None,
    lookup_consent_teams: LookupConsentTeams,
) -> Response:
    """The authorize verb for a native client that named the proxy API itself as its
    RFC 8707 ``resource``: the same client, redirect, PKCE, and sign-in checks as the
    aggregate verb plus a loopback-only redirect (the credential this grant mints is the
    user's personal proxy key, which belongs on their own machine and never behind a hosted
    callback), then the consent page rendered right here (no connect-page interlude, since
    there is no per-server vaulting to do) with the flow sealed into the per-flow cookie
    and its handle carried only in the form, never in a URL."""
    rejected: Final = _rejected_authorize_request(
        client_id, redirect_uri, state, code_challenge, code_challenge_method, response_type
    )
    if rejected is not None:
        return rejected
    if not is_loopback_redirect_host(urlparse(redirect_uri)):
        return _oauth_error(400, "invalid_request", "a proxy-API grant may only redirect to a loopback address")
    base_url: Final = get_request_base_url(request)
    if session_user_id is None:
        return _login_redirect(base_url, request)
    teams: Final = await lookup_consent_teams(session_user_id)
    if not isinstance(teams, tuple):
        return _consent_lookup_failure_response(teams)
    handle: Final = secrets.token_urlsafe(24)
    flow: Final = _new_connect_flow(
        session_user_id=session_user_id,
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge or "",
        resource_server_id=None,
        audience=PROXY_API_AUDIENCE,
    )
    page: Final = render_native_client_consent_page(
        client_origin=_origin_only(redirect_uri),
        user_id=session_user_id,
        teams=tuple((team.team_id, team.team_alias or team.team_id) for team in teams),
        flow_handle=handle,
        complete_url=f"{base_url}/authorize/complete",
    )
    response: Final = HTMLResponse(page, headers=_CONSENT_PAGE_HEADERS)
    _set_flow_cookie(response, request, handle, flow)
    return response


_CONSENT_PAGE_HEADERS: Final = MappingProxyType(
    {
        **TOKEN_NO_CACHE_HEADERS,
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": "frame-ancestors 'none'",
    }
)

NATIVE_CLIENT_AUTH_CONTRACT_VERSION: Final = 1
"""The version a native client checks before trusting the rest of the discovery document.
Bump it only when an existing field changes meaning or goes away; adding fields is free."""


class NativeClientAuthContract(TypedDict):
    contract_version: ReadOnly[int]
    issuer: ReadOnly[str]
    authorization_endpoint: ReadOnly[str]
    token_endpoint: ReadOnly[str]
    registration_endpoint: ReadOnly[str]
    revocation_endpoint: ReadOnly[str]
    resource: ReadOnly[str]
    response_types_supported: ReadOnly[tuple[str, ...]]
    grant_types_supported: ReadOnly[tuple[str, ...]]
    code_challenge_methods_supported: ReadOnly[tuple[str, ...]]
    token_endpoint_auth_methods_supported: ReadOnly[tuple[str, ...]]
    revocation_endpoint_auth_methods_supported: ReadOnly[tuple[str, ...]]


def native_client_auth_contract(request: Request) -> NativeClientAuthContract:
    """The versioned discovery document at ``/.well-known/litellm-cli-auth``: everything a
    native client (in any language) needs to run the sign-in without reading LiteLLM
    source. ``resource`` is the exact value to send as the RFC 8707 ``resource`` parameter
    on authorize and token requests so the grant is issued for the proxy API."""
    base_url: Final = get_request_base_url(request)
    contract: Final[NativeClientAuthContract] = {
        "contract_version": NATIVE_CLIENT_AUTH_CONTRACT_VERSION,
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        "registration_endpoint": f"{base_url}/register",
        "revocation_endpoint": f"{base_url}/revoke",
        "resource": base_url,
        "response_types_supported": ("code",),
        "grant_types_supported": ("authorization_code", "refresh_token"),
        "code_challenge_methods_supported": ("S256",),
        "token_endpoint_auth_methods_supported": ("none",),
        "revocation_endpoint_auth_methods_supported": ("none",),
    }
    return contract


def is_proxy_api_resource(request: Request, resource: str | None) -> bool:
    """True when the RFC 8707 ``resource`` names the proxy itself (its base URL), which is
    how a native client asks for the proxy-API audience rather than an MCP session."""
    if resource is None:
        return False
    canonical: Final = canonical_resource_uri(resource)
    return canonical is not None and canonical == canonicalize_url_identity(get_request_base_url(request))


def _rejected_authorize_request(
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str | None,
    code_challenge_method: str | None,
    response_type: str | None,
) -> Response | None:
    client: Final = open_gateway_dcr_client(client_id)
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
    return None


def _login_redirect(base_url: str, request: Request) -> Response:
    return_to: Final = urlencode((("return_to", relative_request_url(request)),))
    return RedirectResponse(f"{base_url}/sso/key/generate?{return_to}", status_code=303)


def _new_connect_flow(
    session_user_id: str,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    resource_server_id: str | None,
    audience: SessionAudience | None,
) -> _ConnectFlow:
    now: Final = datetime.now(timezone.utc)
    return _ConnectFlow(
        user_id=session_user_id,
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        jti=secrets.token_urlsafe(24),
        exp=int(now.timestamp()) + CONNECT_FLOW_TTL_SECONDS,
        resource_server_id=resource_server_id,
        audience=audience,
    )


def _set_flow_cookie(response: Response, request: Request, handle: str, flow: _ConnectFlow) -> None:
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


def _consent_lookup_failure_response(failure: ReloadUserFailure) -> Response:
    match failure:
        case "unavailable":
            return _oauth_error(503, "temporarily_unavailable", "the gateway database is unavailable; retry")
        case "unresolvable":
            return _oauth_error(500, "server_error", "the gateway is not configured to resolve users")
        case "no_active_key":
            return _oauth_error(403, "access_denied", "the signed-in user is not active")
        case _:
            assert_never(failure)


def _origin_only(url: str) -> str:
    """Scheme+host for display on the connect page; never the full redirect URI, whose
    path or query could carry values that do not belong in a page URL or logs."""
    parsed: Final = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else ""


async def complete_connect_flow(
    request: Request,
    flow_handle: str,
    session_user_id: str | None,
    cache: DualCache,
    delivery: str | None = None,
    team_id: str | None = None,
    decision: str | None = None,
) -> Response:
    """The deliberate finish step of the connect flow: mint the gateway authorization
    code and send the browser back to the client.

    Reached by POST so a cross-site GET cannot trigger it, and bound to the HttpOnly
    per-flow cookie plus an exact match between the signed-in user and the user sealed
    into the flow: a link crafted by another party dies here with ``access_denied``
    instead of minting a code for the victim's identity. The flow is single-use (an atomic
    claim on its ``jti``), so a double-submit cannot mint two codes from one sign-in.

    ``delivery`` chooses how the code reaches the client. Default (absent or
    ``"redirect"``) is the 303 to the client's registered redirect URI. ``"manual"``
    renders the callback URL on a page instead, for a client whose redirect URI is a
    loopback host but which runs on a DIFFERENT machine than the browser (EC2/SSH box,
    container): the 303 would dereference the browser machine's loopback and the code
    would never arrive, so the user carries it over by pasting the URL into the client or
    fetching it from the client machine's terminal. Manual delivery is honored only for
    loopback redirect URIs; a routable redirect URI works from any browser by
    construction, so those flows always redirect. The user who sees the page is exactly
    the user the 303 would have carried the code to, and the same user already sees the
    code today in the dead redirect's address bar, so the page exposes the code to no new
    party. Unknown ``delivery`` values are rejected rather than defaulted: a client that
    asked for manual delivery and got a dead redirect instead would silently lose its
    code.

    ``decision`` and ``team_id`` come from the native-client consent page. ``"deny"``
    burns the flow and sends the client ``error=access_denied`` so it stops waiting;
    ``team_id`` is sealed into the code only for proxy-API flows, where it picks which of
    the user's teams the minted credential is attributed to.
    """
    if delivery not in (None, "redirect", "manual"):
        return _oauth_error(400, "invalid_request", "delivery must be 'redirect' or 'manual'")
    if decision not in (None, "approve", "deny"):
        return _oauth_error(400, "invalid_request", "decision must be 'approve' or 'deny'")
    sealed_flow: Final = request.cookies.get(_flow_cookie_name(flow_handle))
    if sealed_flow is None:
        return _oauth_error(400, "invalid_request", "unknown or expired connect flow")
    flow: Final = _open_sealed(sealed_flow, _UNPREFIXED, _ConnectFlow, _CONNECT_FLOW_DEBUG_KEY)
    if flow is None:
        return _oauth_error(400, "invalid_request", "unknown or expired connect flow")
    now: Final = datetime.now(timezone.utc)
    if now.timestamp() >= flow.exp:
        return _oauth_error(400, "invalid_request", "the connect flow has expired; restart the connection")
    if session_user_id is None:
        return _oauth_error(401, "login_required", "sign in to LiteLLM to finish connecting")
    if session_user_id != flow.user_id:
        return _oauth_error(403, "access_denied", "the signed-in user does not match this connect flow")
    flow_refusal: Final = _claim_refusal(
        await _SingleUseGuard(cache).claim(
            f"{_USED_FLOW_CACHE_PREFIX}{flow.jti}", CONNECT_FLOW_TTL_SECONDS + _CLAIM_TTL_BUFFER_SECONDS
        ),
        replayed=_oauth_error(
            400, "invalid_request", "this connect flow was already completed; restart the connection"
        ),
    )
    if flow_refusal is not None:
        return flow_refusal
    response: Final = (
        _denied_flow_response(flow) if decision == "deny" else _approved_flow_response(flow, delivery, team_id, now)
    )
    path, secure = _cookie_path_and_secure(request)
    response.delete_cookie(key=_flow_cookie_name(flow_handle), path=path, secure=secure, httponly=True, samesite="lax")
    return response


def _state_param(flow: _ConnectFlow) -> tuple[tuple[str, str], ...]:
    return (("state", flow.state),) if flow.state else ()


def _denied_flow_response(flow: _ConnectFlow) -> Response:
    params: Final = (("error", "access_denied"), *_state_param(flow))
    return RedirectResponse(_append_query_params(flow.redirect_uri, params), status_code=303)


def _approved_flow_response(flow: _ConnectFlow, delivery: str | None, team_id: str | None, now: datetime) -> Response:
    manual_delivery: Final = delivery == "manual" and is_loopback_redirect_host(urlparse(flow.redirect_uri))
    code_ttl: Final = MANUAL_DELIVERY_AUTH_CODE_TTL_SECONDS if manual_delivery else GATEWAY_AUTH_CODE_TTL_SECONDS
    code: Final = _seal(
        GATEWAY_AUTH_CODE_PREFIX,
        _GatewayAuthCode(
            user_id=flow.user_id,
            client_id=flow.client_id,
            redirect_uri=flow.redirect_uri,
            code_challenge=flow.code_challenge,
            jti=secrets.token_urlsafe(24),
            iat=int(now.timestamp()),
            exp=int(now.timestamp()) + code_ttl,
            resource_server_id=flow.resource_server_id,
            audience=flow.audience,
            team_id=(team_id or None) if flow.audience == PROXY_API_AUDIENCE else None,
        ),
    )
    callback_url: Final = _append_query_params(flow.redirect_uri, (("code", code), *_state_param(flow)))
    if manual_delivery:
        return _manual_delivery_response(callback_url)
    return RedirectResponse(callback_url, status_code=303)


def _manual_delivery_response(callback_url: str) -> Response:
    """The manual code-delivery page: the callback URL the 303 would have followed,
    rendered for the user to carry to the machine the client actually runs on (paste into
    the client's prompt, or fetch with curl from that machine's terminal). Served
    no-store because the body holds a live single-use code, and the URL is HTML-escaped
    because it is client-influenced. The page renders the URL as data only, never as a
    ready-to-paste shell command: no single quoting of an attacker-influenced string is
    correct across POSIX shells, cmd.exe, and PowerShell (cmd.exe ignores single quotes
    and percent-expands inside double quotes), so any command string this page suggested
    would be wrong for some shell the user might paste it into."""
    safe_url: Final = html.escape(callback_url, quote=True)
    minutes: Final = MANUAL_DELIVERY_AUTH_CODE_TTL_SECONDS // 60
    body: Final = (
        "<html><head><title>Finish connecting</title></head><body>"
        "<h2>Almost done</h2>"
        "<p>Your MCP client runs on a different machine, so this browser cannot deliver the"
        " authorization code to it. On the machine where the client runs, paste this URL into"
        " the client's prompt (Claude Code accepts the pasted callback URL), or pass it as the"
        " quoted argument of a curl command from that machine's terminal:</p>"
        f'<p><input type="text" value="{safe_url}" readonly size="100" onclick="this.select()"></p>'
        f"<p>The code is single-use and expires in {minutes} minutes. You can close this window"
        " once the client confirms it is connected.</p>"
        "</body></html>"
    )
    return HTMLResponse(body, headers=TOKEN_NO_CACHE_HEADERS)


def _pkce_verifier_matches(code_verifier: str, code_challenge: str) -> bool:
    """RFC 7636 S256 verification, total over hostile input. The comparison is over bytes
    so a non-ASCII ``code_challenge`` (which reaches here unvalidated from the client's
    authorize request) simply fails to match instead of raising ``TypeError`` the way
    ``hmac.compare_digest`` does on two ``str`` with non-ASCII content. The verifier is
    ASCII per spec; a compliant client's challenge is base64url and matches."""
    digest: Final = hashlib.sha256(code_verifier.encode("ascii", "replace")).digest()
    computed: Final = urlsafe_b64encode(digest).rstrip(b"=")
    return hmac.compare_digest(computed, code_challenge.encode("utf-8"))


ClaimOutcome = Literal["first", "replayed", "unavailable"]

_CLAIM_UNAVAILABLE_DESCRIPTION: Final = "the single-use record is unavailable right now; try again shortly"


def _claim_refusal(outcome: ClaimOutcome, replayed: Response) -> Response | None:
    """A claim that is not the first caller's is refused, but the two reasons must stay apart on the
    wire: a replay is the grant's own 4xx, while a shared backend that could not record the claim is
    a 503 (RFC 7009 section 2.2.1, RFC 6749 section 5.2 ``temporarily_unavailable``), so the client
    keeps the still-valid token and retries instead of being told it was already used."""
    match outcome:
        case "first":
            return None
        case "replayed":
            return replayed
        case "unavailable":
            return _oauth_error(503, "temporarily_unavailable", _CLAIM_UNAVAILABLE_DESCRIPTION)
        case _:
            assert_never(outcome)


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

    async def claim(self, key: str, ttl_seconds: int) -> ClaimOutcome:
        """Atomically claim ``key``. ``"first"`` iff this caller is the first (increment to 1),
        ``"replayed"`` on a replay (>1), and ``"unavailable"`` when the claim could not be recorded in
        the shared backend, which every caller treats as a refusal (fail closed)."""
        from litellm.proxy.proxy_server import redis_usage_cache  # noqa: PLC0415  # circular import at module load

        # Resolve the shared authority HERE rather than trusting the injected cache: callers pass
        # user_api_key_cache, which only carries a redis_cache when enable_redis_auth_cache is set
        # (off by default), so a guard that read its injected cache silently degraded every claim to
        # a per-worker count on a stock multi-worker deployment. redis_usage_cache is the store the
        # proxy already treats as cross-worker, so no call site can wire the guarantee away.
        redis_cache: Final = redis_usage_cache or getattr(self._cache, "redis_cache", None)
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
                return "unavailable"
            return "first" if count == 1 else "replayed"
        # No shared backend configured (single-replica): the in-memory increment is authoritative.
        count = await self._cache.async_increment_cache(key, 1, ttl=ttl_seconds, local_only=True)
        return "first" if count == 1 else "replayed"


def _session_token_pair(principal: SessionPrincipal, keys: SessionKeys, now: datetime) -> Response:
    access: Final = mint_session_token(principal, keys, now)
    refresh: Final = mint_session_refresh_token(principal, keys, now)
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


class _ProxyCredentialTokenResponse(TypedDict):
    access_token: ReadOnly[str]
    token_type: ReadOnly[Literal["Bearer"]]
    expires_in: ReadOnly[int]
    refresh_token: ReadOnly[str]
    user_id: ReadOnly[str]
    team_id: ReadOnly[str | None]


def _proxy_credential_response(
    minted: MintedProxyCredential, principal: SessionPrincipal, keys: SessionKeys, now: datetime
) -> Response:
    """The proxy-API token response: the access token is the very credential ``lite
    login`` stores (accepted on every proxy route with user and team attribution), and
    the refresh token is a gateway-sealed rotating token bound to the team the credential
    was minted for, so a renewal keeps the team the user consented to."""
    bound_principal: Final = principal.model_copy(update=MappingProxyType({"team_id": minted.team_id}))
    refresh: Final = mint_session_refresh_token(bound_principal, keys, now)
    if not isinstance(refresh, MintedSessionToken):
        return _oauth_error(500, "server_error", "failed to mint the session credential")
    body: Final[_ProxyCredentialTokenResponse] = {
        "access_token": minted.key,
        "token_type": "Bearer",
        "expires_in": minted.expires_in,
        "refresh_token": refresh.token.get_secret_value(),
        "user_id": minted.user_id,
        "team_id": minted.team_id,
    }
    return JSONResponse(status_code=200, content=body, headers=TOKEN_NO_CACHE_HEADERS)


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


def _mint_failure_response(failure: ProxyCredentialMintFailure) -> Response:
    match failure:
        case "not_a_member":
            return _oauth_error(
                400, "invalid_grant", "the user is no longer a member of the team this grant was issued for"
            )
        case "team_required":
            return _oauth_error(
                400, "invalid_grant", "this user belongs to a team; sign in again and pick the team for this credential"
            )
        case "unavailable" | "unresolvable" | "no_active_key":
            return _reload_failure_response(failure)
        case _:
            assert_never(failure)


def _resource_conflicts_with_scope(
    request: Request, resource: str | None, sealed_resource_server_id: str | None
) -> bool:
    """True when a scoped grant is being redeemed for a DIFFERENT resource than the one
    sealed into it (RFC 8707 section 2.2: reject with ``invalid_target``). An absent
    ``resource`` never conflicts (the sealed scope still binds the minted session), and an
    unscoped grant ignores the parameter entirely, exactly as the endpoint always has, so
    no pre-existing client breaks."""
    if sealed_resource_server_id is None or resource is None:
        return False
    resolved: Final = resolve_scoped_resource_server(request, resource)
    return resolved is None or resolved.server_id != sealed_resource_server_id


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
    resource: str | None = None,
    mint_proxy_credential: MintProxyCredential = _refuse_proxy_credential,
) -> Response:
    """The aggregate token verb: authorization_code and refresh_token grants for the
    identity-only session pair, or for the proxy-API credential when the grant was issued
    with that audience. Every path re-validates the litellm user live before minting, so a
    deactivated user cannot obtain or renew a session."""
    if master_key is None:
        verbose_logger.error("mcp_gateway_dcr token grant rejected: no master_key configured")
        return _oauth_error(500, "server_error", "the gateway has no master key configured")
    keys: Final = session_keys_from_master_key(master_key)
    now: Final = datetime.now(timezone.utc)
    issue: Final = _GrantIssuer(
        request=request,
        resource=resource,
        keys=keys,
        now=now,
        reload_user=reload_user,
        mint_proxy_credential=mint_proxy_credential,
        guard=_SingleUseGuard(cache),
    )
    if grant_type == "authorization_code":
        return await _authorization_code_grant(
            request=request,
            code=code,
            redirect_uri=redirect_uri,
            client_id=client_id,
            code_verifier=code_verifier,
            resource=resource,
            now=now,
            issue=issue,
        )
    if grant_type == "refresh_token":
        return await _refresh_token_grant(
            request=request,
            refresh_token=refresh_token,
            client_id=client_id,
            resource=resource,
            keys=keys,
            now=now,
            issue=issue,
        )
    return _oauth_error(400, "unsupported_grant_type", "grant_type must be authorization_code or refresh_token")


class _GrantIssuer:
    """The tail every grant shares once its own proof (code + PKCE, or a refresh token)
    has checked out: revalidate the user live, claim the single-use marker, mint. The
    claim comes AFTER revalidation and minting so a transient DB 503 never burns a
    still-valid code or refresh token, and fails closed when it cannot be recorded."""

    def __init__(
        self,
        request: Request,
        resource: str | None,
        keys: SessionKeys,
        now: datetime,
        reload_user: ReloadUser,
        mint_proxy_credential: MintProxyCredential,
        guard: _SingleUseGuard,
    ) -> None:
        self._request: Final = request
        self._resource: Final = resource
        self._keys: Final = keys
        self._now: Final = now
        self._reload_user: Final = reload_user
        self._mint_proxy_credential: Final = mint_proxy_credential
        self._guard: Final = guard

    async def __call__(
        self, principal: SessionPrincipal, claim_key: str, claim_ttl_seconds: int, replayed: str
    ) -> Response:
        match principal.audience:
            case None:
                return await self._issue_session_pair(principal, claim_key, claim_ttl_seconds, replayed)
            case "proxy_api":
                return await self._issue_proxy_credential(principal, claim_key, claim_ttl_seconds, replayed)
            case _:
                assert_never(principal.audience)

    async def _issue_session_pair(
        self, principal: SessionPrincipal, claim_key: str, claim_ttl_seconds: int, replayed: str
    ) -> Response:
        failure: Final = await self._reload_user(principal.user_id)
        if failure is not None:
            return _reload_failure_response(failure)
        refusal: Final = await self._claim_refusal(claim_key, claim_ttl_seconds, replayed)
        if refusal is not None:
            return refusal
        return _session_token_pair(principal, self._keys, self._now)

    async def _issue_proxy_credential(
        self, principal: SessionPrincipal, claim_key: str, claim_ttl_seconds: int, replayed: str
    ) -> Response:
        if self._resource is not None and not is_proxy_api_resource(self._request, self._resource):
            return _oauth_error(
                400, "invalid_target", "resource does not match the proxy API this grant was issued for"
            )
        minted: Final = await self._mint_proxy_credential(principal.user_id, principal.team_id)
        if not isinstance(minted, MintedProxyCredential):
            return _mint_failure_response(minted)
        refusal: Final = await self._claim_refusal(claim_key, claim_ttl_seconds, replayed)
        if refusal is not None:
            return refusal
        return _proxy_credential_response(minted, principal, self._keys, self._now)

    async def _claim_refusal(self, claim_key: str, claim_ttl_seconds: int, replayed: str) -> Response | None:
        return _claim_refusal(
            await self._guard.claim(claim_key, claim_ttl_seconds), replayed=_oauth_error(400, "invalid_grant", replayed)
        )


async def _authorization_code_grant(
    request: Request,
    code: str | None,
    redirect_uri: str | None,
    client_id: str,
    code_verifier: str | None,
    resource: str | None,
    now: datetime,
    issue: _GrantIssuer,
) -> Response:
    if not code or not redirect_uri or not code_verifier:
        return _oauth_error(400, "invalid_request", "code, redirect_uri, and code_verifier are required")
    if not MIN_CODE_VERIFIER_LENGTH <= len(code_verifier) <= MAX_CODE_VERIFIER_LENGTH:
        return _oauth_error(400, "invalid_request", "code_verifier must be 43 to 128 characters (RFC 7636)")
    parsed: Final = _open_sealed(code, GATEWAY_AUTH_CODE_PREFIX, _GatewayAuthCode, _AUTH_CODE_DEBUG_KEY)
    if parsed is None:
        return _oauth_error(400, "invalid_grant", "the authorization code is invalid")
    if now.timestamp() >= parsed.exp:
        return _oauth_error(400, "invalid_grant", "the authorization code has expired")
    if client_id != parsed.client_id or redirect_uri != parsed.redirect_uri:
        return _oauth_error(400, "invalid_grant", "the authorization code was issued to a different client")
    if _resource_conflicts_with_scope(request, resource, parsed.resource_server_id):
        return _oauth_error(400, "invalid_target", "resource does not match the scope this code was issued for")
    if not _pkce_verifier_matches(code_verifier, parsed.code_challenge):
        return _oauth_error(400, "invalid_grant", "PKCE verification failed")
    # The marker's TTL derives from the code's own remaining lifetime so it outlives
    # whichever lifetime the code was minted with.
    return await issue(
        SessionPrincipal(
            user_id=parsed.user_id,
            client_id=client_id,
            resource_server_id=parsed.resource_server_id,
            audience=parsed.audience,
            team_id=parsed.team_id,
        ),
        claim_key=f"{_USED_CODE_CACHE_PREFIX}{parsed.jti}",
        claim_ttl_seconds=parsed.exp - int(now.timestamp()) + _CLAIM_TTL_BUFFER_SECONDS,
        replayed="the authorization code was already used",
    )


async def _refresh_token_grant(
    request: Request,
    refresh_token: str | None,
    client_id: str,
    resource: str | None,
    keys: SessionKeys,
    now: datetime,
    issue: _GrantIssuer,
) -> Response:
    if not refresh_token:
        return _oauth_error(400, "invalid_request", "refresh_token is required")
    opened: Final = open_session_refresh_bearer(refresh_token, keys, now, expected_client_id=client_id)
    if not isinstance(opened, SessionRefreshOpened):
        return _oauth_error(400, "invalid_grant", "the refresh token is invalid for this client")
    if _resource_conflicts_with_scope(request, resource, opened.principal.resource_server_id):
        return _oauth_error(400, "invalid_target", "resource does not match the scope this token was issued for")
    # Refresh-token rotation (OAuth 2.0 Security BCP section 4.13): the presented refresh token is
    # single-use, so a captured or replayed refresh token cannot mint a second pair after the
    # legitimate holder rotated.
    return await issue(
        opened.principal,
        claim_key=f"{_USED_REFRESH_CACHE_PREFIX}{opened.jti}",
        claim_ttl_seconds=SESSION_REFRESH_TTL_SECONDS + _CLAIM_TTL_BUFFER_SECONDS,
        replayed="the refresh token was already used",
    )


async def revoke_refresh_token(token: str, client_id: str, master_key: str | None, cache: DualCache) -> Response:
    """RFC 7009 revocation for the gateway's refresh tokens: burn the presented token's
    ``jti`` so neither the holder nor a thief can rotate it again. Access tokens are
    stateless and expire on their own (the proxy-API credential within
    ``CLI_JWT_EXPIRATION_HOURS``), so per RFC 7009 section 2.2 an unrecognized or already
    dead token still answers 200; only an unknown client is refused. A live token whose
    burn could not be recorded in the shared backend answers 503 (section 2.2.1), so the
    client knows the token still stands and retries instead of reporting a logout that
    never happened."""
    if not is_gateway_dcr_client_id(client_id) or open_gateway_dcr_client(client_id) is None:
        return _oauth_error(401, "invalid_client", "unknown or malformed client_id")
    if master_key is None:
        verbose_logger.error("mcp_gateway_dcr revoke rejected: no master_key configured")
        return _oauth_error(500, "server_error", "the gateway has no master key configured")
    keys: Final = session_keys_from_master_key(master_key)
    now: Final = datetime.now(timezone.utc)
    opened: Final = open_session_refresh_bearer(token, keys, now, expected_client_id=client_id)
    if isinstance(opened, SessionRefreshOpened):
        burned: Final = await _SingleUseGuard(cache).claim(
            f"{_USED_REFRESH_CACHE_PREFIX}{opened.jti}", SESSION_REFRESH_TTL_SECONDS + _CLAIM_TTL_BUFFER_SECONDS
        )
        if burned == "unavailable":
            return _oauth_error(503, "temporarily_unavailable", _CLAIM_UNAVAILABLE_DESCRIPTION)
    return Response(content="{}", media_type="application/json", headers=TOKEN_NO_CACHE_HEADERS)

from __future__ import annotations

import hmac
import html
import re
import secrets
import time
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal
from urllib.parse import quote, urlparse

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from mcp.shared.auth import OAuthMetadata, OAuthToken, ProtectedResourceMetadata
from pydantic import BaseModel, ConfigDict, Field

from litellm.proxy._experimental.mcp_server.bridge_token_flow import (
    _litellm_key_from_request,  # pyright: ignore[reportPrivateUsage]  # reuse the bridge credential parser
)
from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
    _open_sealed,  # pyright: ignore[reportPrivateUsage]  # reuse authenticated gateway artifact decoding
    _seal,  # pyright: ignore[reportPrivateUsage]  # reuse authenticated gateway artifact encoding
)
from litellm.proxy._experimental.mcp_server.oauth_utils import TOKEN_NO_CACHE_HEADERS, get_request_base_url
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
    _strip_bearer,  # pyright: ignore[reportPrivateUsage]  # share admission credential normalization
)
from litellm.proxy._types import UserAPIKeyAuth, hash_token

KEYED_FLOW_PREFIX: Final = "llm_kflow_"
KEYED_FLOW_TTL_SECONDS: Final = 600


class KeyedOAuthFlow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    key_hash: str = Field(min_length=64, max_length=64)
    user_id: str = Field(min_length=1)
    server_id: str = Field(min_length=1)
    resource: str = Field(min_length=1)
    jti: str = Field(min_length=1)
    exp: int


def start_keyed_oauth_flow(request: Request, auth: UserAPIKeyAuth, server_id: str) -> str:
    key: Final = _litellm_key_from_request(request)
    if not key or not auth.user_id:
        raise HTTPException(
            status_code=403,
            detail="Interactive MCP OAuth requires a virtual key assigned to a user. Contact your gateway administrator.",
        )
    return _seal(
        KEYED_FLOW_PREFIX,
        KeyedOAuthFlow(
            key_hash=hash_token(key),
            user_id=auth.user_id,
            server_id=server_id,
            resource=f"{get_request_base_url(request)}/mcp",
            jti=secrets.token_urlsafe(24),
            exp=int(time.time()) + KEYED_FLOW_TTL_SECONDS,
        ),
    )


def open_keyed_oauth_flow(request: Request, value: str, *, allow_expired: bool = False) -> KeyedOAuthFlow:
    flow: Final = _open_sealed(value, KEYED_FLOW_PREFIX, KeyedOAuthFlow, "keyed_oauth_flow")
    if (
        flow is None
        or (not allow_expired and flow.exp <= int(time.time()))
        or flow.resource != f"{get_request_base_url(request)}/mcp"
    ):
        raise HTTPException(
            status_code=400, detail="OAuth connection expired or is invalid. Reconnect your MCP client."
        )
    return flow


def keyed_resource_metadata(request: Request, value: str) -> JSONResponse:
    flow: Final = open_keyed_oauth_flow(request, value, allow_expired=True)
    metadata: Final = ProtectedResourceMetadata.model_validate(
        MappingProxyType(
            {
                "resource": flow.resource,
                "authorization_servers": (f"{get_request_base_url(request)}/mcp/keyed/{quote(value, safe='')}",),
            }
        )
    )
    return JSONResponse(metadata.model_dump(mode="json", exclude_none=True), headers=TOKEN_NO_CACHE_HEADERS)


def keyed_server_metadata(request: Request, value: str) -> JSONResponse:
    open_keyed_oauth_flow(request, value, allow_expired=True)
    base: Final = get_request_base_url(request)
    issuer: Final = f"{base}/mcp/keyed/{quote(value, safe='')}"
    metadata: Final = OAuthMetadata.model_validate(
        MappingProxyType(
            {
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "registration_endpoint": f"{base}/register",
                "token_endpoint": f"{base}/token",
                "response_types_supported": ("code",),
                "grant_types_supported": ("authorization_code", "refresh_token"),
                "token_endpoint_auth_methods_supported": ("none",),
                "code_challenge_methods_supported": ("S256",),
            }
        )
    )
    return JSONResponse(metadata.model_dump(mode="json", exclude_none=True), headers=TOKEN_NO_CACHE_HEADERS)


async def keyed_authorize(
    request: Request,
    value: str,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str | None,
    code_challenge_method: str | None,
    response_type: str | None,
    resource: str | None,
) -> Response:
    from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
        _rejected_authorize_request,  # pyright: ignore[reportPrivateUsage]  # apply the existing DCR redirect and PKCE policy
    )

    flow: Final = open_keyed_oauth_flow(request, value)
    refusal: Final = _rejected_authorize_request(
        client_id,
        redirect_uri,
        state,
        code_challenge,
        code_challenge_method,
        response_type,
    )
    if refusal is not None:
        return refusal
    if resource is not None and resource != flow.resource:
        raise HTTPException(status_code=400, detail="OAuth resource does not match this connection")
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", code_challenge or ""):
        raise HTTPException(status_code=400, detail="A valid S256 PKCE challenge is required")
    base: Final = get_request_base_url(request)
    parsed: Final = urlparse(base)
    if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise HTTPException(status_code=400, detail="Key confirmation requires HTTPS outside localhost")
    binding: Final = KeyedAuthorization(
        flow=flow, client_id=client_id, redirect_uri=redirect_uri, state=state, code_challenge=code_challenge or ""
    )
    await _store().begin(flow.jti, _seal(KEYED_BINDING_PREFIX, binding), datetime.fromtimestamp(flow.exp, timezone.utc))
    stored: Final = await _binding(flow.jti, request)
    if stored != binding:
        raise HTTPException(status_code=400, detail="This connection has already started. Reconnect your MCP client.")
    browser: Final = KeyedBrowser(grant_id=flow.jti, csrf=secrets.token_urlsafe(32), exp=flow.exp)
    response: Final = HTMLResponse(
        "<!doctype html><html><head><title>Connect MCP server</title></head><body>"
        "<h1>Confirm your LiteLLM key</h1>"
        "<p>Enter the same virtual key configured in your MCP client to continue to the upstream sign-in.</p>"
        f"<p>Gateway: {html.escape(base)}</p>"
        f"<p>Client callback: {html.escape(urlparse(redirect_uri).netloc or urlparse(redirect_uri).scheme)}</p>"
        f'<form method="post" action="{html.escape(base)}/mcp/keyed/confirm" autocomplete="off">'
        f'<input type="hidden" name="grant_id" value="{html.escape(flow.jti)}">'
        f'<input type="hidden" name="csrf" value="{html.escape(browser.csrf)}">'
        '<label>Virtual key <input type="password" name="api_key" autocomplete="off" required></label>'
        '<button type="submit">Continue to sign-in</button></form>'
        "<p>To cancel, close this tab.</p></body></html>",
        headers=MappingProxyType(
            {
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": "default-src 'none'; form-action 'self'; frame-ancestors 'none'",
            }
        ),
    )
    response.set_cookie(
        _browser_cookie(flow.jti),
        _seal(KEYED_BROWSER_PREFIX, browser),
        max_age=KEYED_FLOW_TTL_SECONDS,
        httponly=True,
        secure=parsed.scheme == "https",
        samesite="lax",
        path=parsed.path or "/",
    )
    return response


KEYED_BROWSER_PREFIX: Final = "llm_kbrowser_"
KEYED_BINDING_PREFIX: Final = "llm_kbinding_"
KEYED_CODE_PREFIX: Final = "llm_kcode_"
KEYED_ACCESS_PREFIX: Final = "llm_kaccess_"
KEYED_REFRESH_PREFIX: Final = "llm_krefresh_"


class KeyedAuthorization(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    flow: KeyedOAuthFlow
    client_id: str
    redirect_uri: str
    state: str
    code_challenge: str


class KeyedBrowser(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    grant_id: str
    csrf: str
    exp: int


class KeyedCode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["code"] = "code"
    grant_id: str
    upstream_code: str
    exp: int


class KeyedToken(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["access", "refresh"]
    grant_id: str
    jti: str
    exp: int


def _browser_cookie(grant_id: str) -> str:
    return f"mcp_keyed_{grant_id}"


def _store() -> KeyedOAuthGrantStore:
    from litellm.proxy._experimental.mcp_server.db import KeyedOAuthGrantStore
    from litellm.proxy.utils import get_prisma_client_or_throw

    return KeyedOAuthGrantStore(get_prisma_client_or_throw("OAuth authorization storage is unavailable"))


def _expires(seconds: int) -> datetime:
    return datetime.fromtimestamp(time.time() + seconds, timezone.utc)


async def _binding(grant_id: str, request: Request, *, require_active: bool = False) -> KeyedAuthorization:
    try:
        row: Final = await _store().get(grant_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="OAuth authorization storage is unavailable") from exc
    if require_active and (row is None or row.status != "active"):
        raise HTTPException(status_code=401, detail="MCP authorization grant is revoked")
    if row is None or row.status == "revoked" or row.expires_at <= datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="OAuth grant is expired or revoked. Reconnect your client.")
    binding: Final = _open_sealed(row.binding_b64, KEYED_BINDING_PREFIX, KeyedAuthorization, "keyed_binding")
    if binding is None or binding.flow.resource != f"{get_request_base_url(request)}/mcp":
        raise HTTPException(status_code=400, detail="OAuth grant does not match this resource")
    return binding


async def _active_key(request: Request, binding: KeyedAuthorization) -> UserAPIKeyAuth:
    from litellm.proxy._experimental.mcp_server.bridge_token_flow import (
        _key_owner_scim_deactivated,  # pyright: ignore[reportPrivateUsage]  # preserve SCIM revocation policy
        _reload_active_key_by_hash,  # pyright: ignore[reportPrivateUsage]  # preserve original-key revocation policy
        can_store_oauth_credential,
    )

    resolved: Final = await _reload_active_key_by_hash(binding.flow.key_hash)
    if isinstance(resolved, str):
        raise HTTPException(
            status_code=403 if resolved == "no_active_key" else 503,
            detail="The original virtual key cannot authorize this connection",
        )
    if resolved.key.user_id != binding.flow.user_id or await _key_owner_scim_deactivated(resolved.key):
        raise HTTPException(status_code=403, detail="The virtual key owner has changed")
    if not await can_store_oauth_credential(request, resolved.key, binding.flow.server_id):
        raise HTTPException(status_code=403, detail="The virtual key cannot authorize this MCP server")
    return resolved.key


async def confirm_keyed_authorization(request: Request) -> Response:
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import authorize_with_server
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager

    form: Final = await request.form(max_fields=3, max_files=0)
    grant_id: Final = form.get("grant_id")
    csrf: Final = form.get("csrf")
    key: Final = form.get("api_key")
    if not isinstance(grant_id, str) or not isinstance(csrf, str) or not isinstance(key, str):
        raise HTTPException(status_code=400, detail="Missing key confirmation fields")
    cookie: Final = request.cookies.get(_browser_cookie(grant_id), "")
    browser: Final = _open_sealed(cookie, KEYED_BROWSER_PREFIX, KeyedBrowser, "keyed_browser")
    base: Final = get_request_base_url(request)
    origin: Final = request.headers.get("origin")
    expected_origin: Final = urlparse(base)
    if origin is not None and origin != f"{expected_origin.scheme}://{expected_origin.netloc}":
        raise HTTPException(status_code=403, detail="Key confirmation must originate from this gateway")
    if (
        browser is None
        or browser.grant_id != grant_id
        or browser.exp <= time.time()
        or not hmac.compare_digest(csrf.encode(), browser.csrf.encode())
    ):
        raise HTTPException(status_code=403, detail="Key confirmation expired. Reconnect your client.")
    binding: Final = await _binding(grant_id, request)
    store: Final = _store()
    if not await store.attempt(grant_id):
        raise HTTPException(status_code=403, detail="Key confirmation expired. Reconnect your client.")
    supplied: Final = _strip_bearer(key)
    if not hmac.compare_digest(hash_token(supplied), binding.flow.key_hash):
        raise HTTPException(status_code=403, detail="Enter the same virtual key configured in your MCP client")
    auth: Final = await _active_key(request, binding)
    server: Final = global_mcp_server_manager.get_mcp_server_by_id(binding.flow.server_id)
    if server is None or not (server.is_gateway_managed_oauth2 and server.needs_user_oauth_token):
        raise HTTPException(status_code=403, detail="This MCP server no longer supports this authorization flow")
    if not await store.transition(grant_id, "pending", "authorizing", _expires(KEYED_FLOW_TTL_SECONDS)):
        raise HTTPException(status_code=400, detail="This authorization has already started")
    response: Final = await authorize_with_server(
        request=request,
        mcp_server=server,
        client_id=server.client_id or "",
        redirect_uri=binding.redirect_uri,
        state=binding.state,
        code_challenge=binding.code_challenge,
        code_challenge_method="S256",
        response_type="code",
        keyed_auth=auth,
        keyed_grant_id=grant_id,
    )
    response.delete_cookie(_browser_cookie(grant_id), path=expected_origin.path or "/")
    return response


async def complete_keyed_callback(request: Request, grant_id: str, upstream_code: str) -> str:
    binding: Final = await _binding(grant_id, request)
    await _active_key(request, binding)
    sealed: Final = _seal(
        KEYED_CODE_PREFIX, KeyedCode(grant_id=grant_id, upstream_code=upstream_code, exp=int(time.time()) + 120)
    )
    if not await _store().transition(grant_id, "authorizing", "code", _expires(120), code_hash=hash_token(sealed)):
        raise HTTPException(status_code=400, detail="OAuth callback has already been used or expired")
    return sealed


if TYPE_CHECKING:
    from litellm.proxy._experimental.mcp_server.db import KeyedOAuthGrantStore


def is_keyed_bearer_shaped(value: object) -> bool:
    return isinstance(value, str) and _strip_bearer(value).startswith(
        (KEYED_ACCESS_PREFIX, KEYED_REFRESH_PREFIX, KEYED_CODE_PREFIX)
    )


def _token_pair(grant_id: str) -> tuple[JSONResponse, str, datetime]:
    from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
        SESSION_REFRESH_TTL_SECONDS,
        SESSION_TTL_SECONDS,
    )

    expiry: Final = _expires(SESSION_REFRESH_TTL_SECONDS)
    access: Final = _seal(
        KEYED_ACCESS_PREFIX,
        KeyedToken(
            kind="access", grant_id=grant_id, jti=secrets.token_urlsafe(24), exp=int(time.time()) + SESSION_TTL_SECONDS
        ),
    )
    refresh: Final = _seal(
        KEYED_REFRESH_PREFIX,
        KeyedToken(kind="refresh", grant_id=grant_id, jti=secrets.token_urlsafe(24), exp=int(expiry.timestamp())),
    )
    return (
        JSONResponse(
            OAuthToken(
                access_token=access, refresh_token=refresh, token_type="Bearer", expires_in=SESSION_TTL_SECONDS
            ).model_dump(mode="json", exclude_none=True),
            headers=TOKEN_NO_CACHE_HEADERS,
        ),
        hash_token(refresh),
        expiry,
    )


async def exchange_keyed_token(
    request: Request,
    grant_type: str,
    code: str | None,
    redirect_uri: str | None,
    client_id: str,
    code_verifier: str | None,
    refresh_token: str | None,
    resource: str | None,
    server_name: str | None,
) -> Response:
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import exchange_token_with_server
    from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
        _oauth_error,  # pyright: ignore[reportPrivateUsage]  # reuse OAuth error serialization
        _pkce_verifier_matches,  # pyright: ignore[reportPrivateUsage]  # reuse constant-time PKCE verification
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager

    try:
        authorization: Final = (
            _open_sealed(code or "", KEYED_CODE_PREFIX, KeyedCode, "keyed_code")
            if grant_type == "authorization_code"
            else None
        )
        refreshed: Final = (
            _open_sealed(refresh_token or "", KEYED_REFRESH_PREFIX, KeyedToken, "keyed_refresh")
            if grant_type == "refresh_token"
            else None
        )
        grant: Final = authorization if authorization is not None else refreshed
        if grant is None or (refreshed is not None and refreshed.kind != "refresh"):
            return _oauth_error(400, "invalid_grant", "Invalid gateway authorization grant")
        grant_id: Final = grant.grant_id
        if grant.exp <= time.time():
            return _oauth_error(400, "invalid_grant", "Gateway authorization grant expired")
        binding: Final = await _binding(grant_id, request)
        if client_id != binding.client_id or (resource is not None and resource != binding.flow.resource):
            return _oauth_error(400, "invalid_grant", "Authorization grant belongs to a different client or resource")
        auth: Final = await _active_key(request, binding)
        server: Final = global_mcp_server_manager.get_mcp_server_by_id(binding.flow.server_id)
        if server is None or not (server.is_gateway_managed_oauth2 and server.needs_user_oauth_token):
            return _oauth_error(400, "invalid_grant", "MCP server no longer supports this authorization grant")
        if server_name is not None:
            selected: Final = global_mcp_server_manager.get_mcp_server_by_name(server_name)
            if selected is None or selected.server_id != server.server_id:
                return _oauth_error(400, "invalid_grant", "Authorization grant belongs to a different MCP server")
        store: Final = _store()
        if authorization is not None:
            if (
                redirect_uri != binding.redirect_uri
                or not code_verifier
                or not 43 <= len(code_verifier) <= 128
                or not _pkce_verifier_matches(code_verifier, binding.code_challenge)
            ):
                return _oauth_error(400, "invalid_grant", "Authorization code redirect or PKCE verification failed")
            if not await store.transition(
                grant_id, "code", "exchanging", _expires(120), expected_hash=hash_token(code or "")
            ):
                await store.revoke(grant_id, expected_status="active")
                return _oauth_error(400, "invalid_grant", "Authorization code already used or expired")
            upstream: Final = await exchange_token_with_server(
                request=request,
                mcp_server=server,
                grant_type="authorization_code",
                code=authorization.upstream_code,
                redirect_uri=redirect_uri,
                client_id=server.client_id or "",
                client_secret=None,
                code_verifier=code_verifier,
                keyed_auth=auth,
            )
            if upstream.status_code != 200:
                await store.revoke(grant_id)
                return upstream
        elif not await global_mcp_server_manager.has_user_oauth_token(server, auth):
            await store.revoke(grant_id)
            return _oauth_error(
                400, "invalid_grant", "Upstream authorization is unavailable. Reconnect your MCP client."
            )
        response, refresh_hash, refresh_expiry = _token_pair(grant_id)
        if not await store.transition(
            grant_id,
            "exchanging" if authorization is not None else "active",
            "active",
            refresh_expiry,
            expected_hash=None if authorization is not None else hash_token(refresh_token or ""),
            refresh_hash=refresh_hash,
        ):
            await store.revoke(grant_id)
            return _oauth_error(400, "invalid_grant", "Authorization grant was already used or revoked")
        return response
    except HTTPException as exc:
        return _oauth_error(
            exc.status_code, "server_error" if exc.status_code >= 500 else "invalid_grant", str(exc.detail)
        )

    except Exception:
        return _oauth_error(503, "server_error", "OAuth authorization storage is unavailable. Retry the connection.")


async def validate_keyed_bearer(request: Request, auth: UserAPIKeyAuth) -> UserAPIKeyAuth:
    value: Final = _strip_bearer(request.headers.get("authorization", ""))
    if not is_keyed_bearer_shaped(value):
        return auth
    token: Final = _open_sealed(value, KEYED_ACCESS_PREFIX, KeyedToken, "keyed_access")
    if token is None or token.kind != "access" or token.exp <= time.time():
        raise HTTPException(status_code=401, detail="MCP authorization token is invalid or expired")
    try:
        binding: Final = await _binding(token.grant_id, request, require_active=True)
    except HTTPException as exc:
        if exc.status_code == 400:
            raise HTTPException(status_code=401, detail="MCP authorization grant is invalid or expired") from exc
        raise
    presented_key: Final = _litellm_key_from_request(request)
    if (
        not auth.via_virtual_key
        or not presented_key
        or auth.user_id != binding.flow.user_id
        or not hmac.compare_digest(hash_token(presented_key), binding.flow.key_hash)
    ):
        raise HTTPException(status_code=403, detail="MCP authorization requires its original virtual key")
    return auth.model_copy(update=MappingProxyType({"mcp_session_resource_server_id": binding.flow.server_id}))


async def revoke_keyed_token(request: Request, value: str, client_id: str) -> Response:
    from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
        _oauth_error,  # pyright: ignore[reportPrivateUsage]  # reuse OAuth error serialization
    )

    token: Final = _open_sealed(value, KEYED_REFRESH_PREFIX, KeyedToken, "keyed_revoke")
    if token is None or token.kind != "refresh":
        return Response(status_code=200, headers=TOKEN_NO_CACHE_HEADERS)
    row: Final = await _store().get(token.grant_id)
    if row is None:
        return Response(status_code=200, headers=TOKEN_NO_CACHE_HEADERS)
    binding: Final = _open_sealed(row.binding_b64, KEYED_BINDING_PREFIX, KeyedAuthorization, "keyed_binding")
    if (
        binding is None
        or binding.client_id != client_id
        or binding.flow.resource != f"{get_request_base_url(request)}/mcp"
    ):
        return _oauth_error(400, "invalid_client", "Authorization grant belongs to another client")
    await _store().revoke(token.grant_id)
    return Response(status_code=200, headers=TOKEN_NO_CACHE_HEADERS)

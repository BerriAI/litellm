"""v2-native refresher for the ``authorization_code`` mode: the refresh_token grant, then persist.

Mints a fresh access token from a stored refresh_token by POSTing the RFC 6749 refresh_token grant to
the server's token endpoint, persists the rotated triple, and returns the new typed ``OAuthToken`` for
``RefreshingTokenStore`` to cache. The HTTP post and the persist are injected, so the orchestration
and the (untyped) response parsing stay testable without a live IdP or DB. Replaces v1's
``refresh_user_oauth_token`` as part of step 1b; rotation safety - one refresh per (user, server)
across replicas - is the wrapping store's distributed single-flight, not this refresher's concern.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Final, Protocol

import httpx
from fastapi import HTTPException

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import loads_json_object, single_log_line
from litellm.proxy.mcp_server.auth.token_endpoint_auth import (
    TokenEndpointAuthConfigError,
)
from litellm.proxy.mcp_server.oauth_identity_binding import (
    BindingValidator,
    RefreshTokenPresented,
    enforce_oauth_identity_binding,
)
from litellm.proxy.mcp_server.oauth_utils import build_upstream_oauth2_token_request, get_cimd_client_id
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

ServerLookup = Callable[[str], "MCPServer | None"]
TokenEndpointPost = Callable[[str, dict[str, str], dict[str, str]], Awaitable["dict[str, object] | None"]]
_NOT_A_REFUSAL: Final = frozenset(
    {"invalid_client", "unauthorized_client", "unsupported_grant_type", "server_error", "temporarily_unavailable"}
)


def refusal_body(response: httpx.Response) -> Mapping[str, object] | None:
    if response.status_code not in (400, 401):
        return None
    body: Final = loads_json_object(response.content) or {}
    error: Final = body.get("error")
    return body if isinstance(error, str) and error else {**body, "error": f"HTTP {response.status_code}"}


async def _ignore_refusal(user_id: str, server_id: str, refused_refresh_token: str) -> None:
    return None


def refused_refresh_error(body: Mapping[str, object]) -> str | None:
    access_token: Final = body.get("access_token")
    if isinstance(access_token, str) and access_token:
        return None
    error: Final = body.get("error")
    return error if isinstance(error, str) and error and error not in _NOT_A_REFUSAL else None


class CredentialPersist(Protocol):
    async def __call__(
        self,
        user_id: str,
        server_id: str,
        access_token: str,
        refresh_token: str | None,
        expires_in: int | None,
        scopes: tuple[str, ...] | None,
        identity_binding_proof: str | None = None,
        cimd_client_id: str | None = None,
        refreshed_from: str | None = None,
    ) -> bool | None: ...


def _parse_expires_in(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _parse_scopes(raw: object) -> tuple[str, ...] | None:
    return tuple(raw.split()) if isinstance(raw, str) else None


class AuthorizationCodeRefresher:
    """``TokenRefresher`` for authorization_code: refresh_token grant against the server, then persist.

    ``token_endpoint`` POSTs the OAuth form and returns the parsed JSON body (``None`` on any
    transport/HTTP failure, mirroring v1: a failed refresh is a miss, not a 500). ``persist`` writes
    the rotated triple for ``(user, server)`` - the v1 ``store_user_oauth_credential`` write, which
    stays. Returns ``None`` (the arm challenges) when there is no refresh_token, the server lacks a
    token endpoint, or the grant fails; never a stale or partial token. A rotated refresh_token from
    the response replaces the old one; an omitted one is carried forward, as are the recorded scopes
    when the response omits ``scope``.
    """

    def __init__(
        self,
        server_lookup: ServerLookup,
        token_endpoint: TokenEndpointPost,
        persist: CredentialPersist,
        *,
        record_refusal: Callable[[str, str, str], Awaitable[None]] = _ignore_refusal,
        clock: Callable[[], float] = time.time,
        identity_validator: BindingValidator = enforce_oauth_identity_binding,
    ) -> None:
        self._server_lookup = server_lookup
        self._token_endpoint = token_endpoint
        self._persist = persist
        self._record_refusal = record_refusal
        self._clock = clock
        self._identity_validator = identity_validator

    async def refresh(self, user_id: str, server_id: str, token: OAuthToken) -> OAuthToken | None:
        try:
            return await self._refresh(user_id, server_id, token)
        except HTTPException as exc:
            if exc.status_code != 403:
                raise
            return None

    async def _refresh(self, user_id: str, server_id: str, token: OAuthToken) -> OAuthToken | None:
        if token.refresh_token is None:
            return None
        server: Final = self._server_lookup(server_id)
        if server is None:
            return None
        token_url: Final = server.effective_token_url
        if not token_url:
            return None

        cimd_client_id: Final = token.cimd_client_id or get_cimd_client_id(server)
        try:
            token_request: Final = build_upstream_oauth2_token_request(
                server,
                auth_method=server.token_endpoint_auth_method,
                client_id=server.client_id,
                client_secret=server.client_secret,
                cimd_client_id=cimd_client_id,
            )
        except TokenEndpointAuthConfigError as exc:
            verbose_logger.warning("MCP OAuth refresh misconfigured for server %s: %s", server_id, exc)
            return None
        binding: Final = server.oauth_identity_binding
        if binding is not None and binding.mode == "enforce":
            await self._identity_validator(
                server=server,
                token_response={},
                litellm_user_id=user_id,
                grant_type="refresh_token",
                refresh_ownership=RefreshTokenPresented(token.refresh_token),
            )
        form: Final = {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            **token_request.body,
        }
        body: Final = await self._token_endpoint(token_url, form, token_request.headers)
        if body is None:
            return None
        refused: Final = refused_refresh_error(body)
        if refused is not None:
            verbose_logger.warning("MCP OAuth refresh refused for server %s: %s", server_id, single_log_line(refused))
            await self._record_refusal(user_id, server_id, token.refresh_token)
            return None
        access_token: Final = body.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return None

        binding_proof: Final = await self._identity_validator(
            server=server,
            token_response=body,
            litellm_user_id=user_id,
            grant_type="refresh_token",
            refresh_ownership=RefreshTokenPresented(token.refresh_token),
        )
        rotated: Final = body.get("refresh_token")
        new_refresh: Final = rotated if isinstance(rotated, str) and rotated else token.refresh_token
        expires_in: Final = _parse_expires_in(body.get("expires_in"))
        granted_scopes: Final = _parse_scopes(body.get("scope"))
        scopes: Final = granted_scopes if granted_scopes is not None else token.scopes

        stored: Final = await self._persist(
            user_id,
            server_id,
            access_token,
            new_refresh,
            expires_in,
            scopes,
            **({"identity_binding_proof": binding_proof} if binding_proof is not None else {}),
            **({"cimd_client_id": cimd_client_id} if cimd_client_id is not None else {}),
            refreshed_from=token.refresh_token,
        )
        if stored is False:
            return None
        return OAuthToken(
            access_token=access_token,
            expires_at=self._clock() + expires_in if expires_in is not None else None,
            refresh_token=new_refresh,
            scopes=scopes,
            identity_binding_proof=binding_proof,
            cimd_client_id=cimd_client_id,
        )

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
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from fastapi import HTTPException
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.auth.token_endpoint_auth import (
    TokenEndpointAuthConfigError,
)
from litellm.proxy._experimental.mcp_server.oauth_identity_binding import (
    BindingValidator,
    RefreshTokenPresented,
    enforce_oauth_identity_binding,
)
from litellm.proxy._experimental.mcp_server.oauth_utils import build_upstream_oauth2_token_request, get_cimd_client_id
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error, Result
from litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_refresher import (
    RefreshFailure,
)

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

ServerLookup = Callable[[str], "MCPServer | None"]
TokenEndpointPost: TypeAlias = Callable[
    [str, dict[str, str], dict[str, str]], Awaitable[Result[Mapping[str, object], RefreshFailure]]
]
CredentialPurge: TypeAlias = Callable[[str, str, str], Awaitable[None]]

_REVOKED_GRANT_ERRORS: Final = frozenset({"invalid_grant", "invalid_token"})


def classify_token_endpoint_refusal(status_code: int, error: str | None) -> RefreshFailure:
    if 400 <= status_code < 500 and error in _REVOKED_GRANT_ERRORS:
        return RefreshFailure.of_rejected(f"the token endpoint answered {status_code} {error}")
    return RefreshFailure.of_unavailable(
        f"the token endpoint answered {status_code} {error or 'without an error code'}"
    )


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
    ) -> None: ...


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

    ``token_endpoint`` POSTs the OAuth form and returns the parsed JSON body or a classified
    ``RefreshFailure``. ``persist`` writes the rotated triple for ``(user, server)`` - the v1
    ``store_user_oauth_credential`` write, which stays. A ``rejected`` failure (the IdP answered
    ``invalid_grant`` / ``invalid_token``) means the stored refresh token is dead, so ``purge`` drops
    the credential that still holds it and the user is challenged to re-authorize; an ``unavailable``
    one keeps the credential so the next request retries. Returns ``None`` (the arm challenges) when
    there is no refresh_token, the server lacks a token endpoint, or the grant fails; never a stale or
    partial token. A rotated refresh_token from the response replaces the old one; an omitted one is
    carried forward, as are the recorded scopes when the response omits ``scope``.
    """

    def __init__(
        self,
        server_lookup: ServerLookup,
        token_endpoint: TokenEndpointPost,
        persist: CredentialPersist,
        *,
        purge: CredentialPurge,
        clock: Callable[[], float] = time.time,
        identity_validator: BindingValidator = enforce_oauth_identity_binding,
    ) -> None:
        self._server_lookup = server_lookup
        self._token_endpoint = token_endpoint
        self._persist = persist
        self._purge = purge
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
        outcome: Final = await self._token_endpoint(token_url, form, token_request.headers)
        if isinstance(outcome, Error):
            await self._on_failure(user_id, server_id, token.refresh_token, outcome.error)
            return None
        body: Final = outcome.ok
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

        await self._persist(
            user_id,
            server_id,
            access_token,
            new_refresh,
            expires_in,
            scopes,
            **({"identity_binding_proof": binding_proof} if binding_proof is not None else {}),
            **({"cimd_client_id": cimd_client_id} if cimd_client_id is not None else {}),
        )
        return OAuthToken(
            access_token=access_token,
            expires_at=self._clock() + expires_in if expires_in is not None else None,
            refresh_token=new_refresh,
            scopes=scopes,
            identity_binding_proof=binding_proof,
            cimd_client_id=cimd_client_id,
        )

    async def _on_failure(self, user_id: str, server_id: str, refresh_token: str, failure: RefreshFailure) -> None:
        match failure.kind:
            case "rejected":
                verbose_logger.warning(
                    "MCP OAuth refresh rejected for user=%s server=%s (%s); removing the stored credential so "
                    "the user re-authorizes",
                    user_id,
                    server_id,
                    failure.detail,
                )
                await self._purge(user_id, server_id, refresh_token)
            case "unavailable":
                verbose_logger.warning(
                    "MCP OAuth refresh failed for user=%s server=%s (%s); keeping the stored credential",
                    user_id,
                    server_id,
                    failure.detail,
                )
            case _:
                assert_never(failure.kind)

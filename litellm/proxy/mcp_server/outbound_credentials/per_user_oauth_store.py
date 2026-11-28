"""Composition root for the v2-native authorization_code per-user OAuth token store (step 1b).

Assembles ``Cached(Refreshing(V2PerUserTokenStore))`` and replaces ``V1PerUserTokenStore`` in the
resolver. The runtime collaborators (DB, HTTP, the shared cache, Redis) are LiteLLM globals not ready
at import time, so the chain is built lazily on first use. When Redis is wired it uses the
cross-replica path (DualCache-backed cache + ``SET NX PX`` coordinator); otherwise it falls back to
the foundation's in-process defaults (correct for a single replica). The DB read/refresh-grant/persist
collaborators acquire their globals per call, mirroring v1's lazy-import pattern.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Final, TypeAlias

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.proxy.mcp_server.oauth_identity_binding import credential_binding_matches
from litellm.proxy.mcp_server.outbound_credentials.authz_code_refresher import (
    AuthorizationCodeRefresher,
    AuthorizationCodeTokenDenied,
    AuthorizationCodeTokenSuccess,
    AuthorizationCodeTokenUnreachable,
)
from litellm.proxy.mcp_server.outbound_credentials.dual_cache_token_backend import (
    DualCacheTokenCacheBackend,
)
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    CachedOAuthTokenStore,
    OAuthToken,
    ReactiveOAuthTokenStore,
    RefreshCoordinator,
    RefreshingTokenStore,
    TokenCacheBackend,
    TokenStoreUnavailable,
    fetch_token_result,
)
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy.mcp_server.outbound_credentials.runtime_refresh_coordinator import (
    runtime_refresh_coordinator,
)
from litellm.proxy.mcp_server.outbound_credentials.token_cache_codec import (
    OAuthTokenCacheCodec,
)
from litellm.proxy.mcp_server.outbound_credentials.types import CredError
from litellm.proxy.mcp_server.outbound_credentials.v2_token_store import (
    V2PerUserTokenStore,
)

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

# A token with no declared expiry is cached for this long; one with an expiry is cached until then.
_DEFAULT_TTL_SECONDS: Final = 300.0

ServerLookup: TypeAlias = Callable[[str], "MCPServer | None"]
StoreBuilder: TypeAlias = Callable[[ServerLookup], tuple[ReactiveOAuthTokenStore, bool]]
_TOKEN_BODY_ADAPTER: Final = TypeAdapter(dict[str, object])


async def _read_credential(user_id: str, server_id: str) -> Mapping[str, object] | None:
    from litellm.proxy.mcp_server.db import (  # noqa: PLC0415
        get_user_oauth_credential,
    )
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global is wired after module import

    if prisma_client is None:
        raise TokenStoreUnavailable("Database not connected")
    return await get_user_oauth_credential(prisma_client, user_id, server_id, use_writer=True)


async def _persist_credential(
    user_id: str,
    server_id: str,
    access_token: str,
    refresh_token: str | None,
    expires_in: int | None,
    scopes: tuple[str, ...] | None,
    identity_binding_proof: str | None = None,
    cimd_client_id: str | None = None,
    refreshed_from: str | None = None,
) -> bool | None:
    from litellm.proxy.mcp_server.db import (  # noqa: PLC0415
        store_user_oauth_credential,
    )
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global is wired after module import

    if prisma_client is None:
        return
    return await store_user_oauth_credential(
        prisma_client=prisma_client,
        user_id=user_id,
        server_id=server_id,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        scopes=list(scopes) if scopes is not None else None,
        skip_byok_guard=True,
        identity_binding_proof=identity_binding_proof,
        **({"cimd_client_id": cimd_client_id} if cimd_client_id is not None else {}),
        **({"refreshed_from": refreshed_from} if refreshed_from is not None else {}),
    )


async def _post_token_endpoint(
    url: str, form: dict[str, str], headers: dict[str, str]
) -> AuthorizationCodeTokenSuccess | AuthorizationCodeTokenDenied | AuthorizationCodeTokenUnreachable:
    from litellm.llms.custom_httpx.http_handler import (  # noqa: PLC0415
        get_async_httpx_client,  # pyright: ignore
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider  # noqa: PLC0415

    provider: Final = httpxSpecialProvider.Oauth2Check
    request_headers: Final = {"Accept": "application/json", **headers}
    try:
        client: Final = get_async_httpx_client(llm_provider=provider)  # pyright: ignore
        response: Final = await client.post(url, headers=request_headers, data=form)  # pyright: ignore
        response.raise_for_status()  # pyright: ignore
    except httpx.HTTPStatusError as exc:
        verbose_logger.warning("MCP OAuth refresh token endpoint returned HTTP %s", exc.response.status_code)
        try:
            error_body: Final = _TOKEN_BODY_ADAPTER.validate_json(exc.response.content)
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
            return AuthorizationCodeTokenDenied(
                status_code=exc.response.status_code,
                body={},
                detail=f"token endpoint returned HTTP {exc.response.status_code}",
            )
        return AuthorizationCodeTokenDenied(
            status_code=exc.response.status_code,
            body=error_body,
            detail=f"token endpoint returned HTTP {exc.response.status_code}",
        )
    except Exception as exc:  # noqa: BLE001
        verbose_logger.warning("MCP OAuth refresh request failed: %s", type(exc).__name__)
        return AuthorizationCodeTokenUnreachable(detail=type(exc).__name__)
    try:
        body: Final = _TOKEN_BODY_ADAPTER.validate_json(response.content)
    except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
        verbose_logger.warning(
            "MCP OAuth refresh token endpoint returned HTTP %s with non-JSON-object body",
            response.status_code,
        )
        return AuthorizationCodeTokenDenied(
            status_code=response.status_code,
            body={},
            detail="token endpoint returned a non-JSON-object body",
        )
    return AuthorizationCodeTokenSuccess(body=body)


async def _revoke_credential(user_id: str, server_id: str, access_token: str, refresh_token: str | None) -> None:
    from litellm.proxy.mcp_server.db import (  # noqa: PLC0415  # runtime global avoids a package import cycle
        invalidate_user_oauth_credential_if_grant_matches,
    )
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global is wired after module import

    if prisma_client is None:
        return
    await invalidate_user_oauth_credential_if_grant_matches(
        prisma_client, user_id, server_id, access_token, refresh_token
    )


def _redis_cache_is_available() -> bool:
    from litellm.proxy.proxy_server import redis_usage_cache  # noqa: PLC0415  # runtime global

    return redis_usage_cache is not None


def _cred_error_detail(err: CredError) -> str:
    match err.tag:
        case "unauthorized":
            return err.unauthorized.detail
        case "misconfigured":
            return err.misconfigured
        case "precondition_required":
            return err.precondition_required
        case "upstream_unavailable":
            return err.upstream_unavailable
        case _:
            return err.summary


def _encode_refresh_error(outcome: Result[OAuthToken, CredError]) -> str | None:
    if not isinstance(outcome, Error):
        return None
    err: Final = outcome.error
    return json.dumps({"tag": err.tag, "detail": _cred_error_detail(err)})


def _decode_refresh_error(payload: str) -> Result[OAuthToken, CredError] | None:
    try:
        payload_map: Final = _TOKEN_BODY_ADAPTER.validate_json(payload)
    except ValidationError:
        return None
    tag: Final = payload_map.get("tag")
    detail: Final = payload_map.get("detail")
    if not isinstance(tag, str) or not isinstance(detail, str):
        return None
    match tag:
        case "unauthorized":
            return Error(CredError.of_unauthorized(detail))
        case "misconfigured":
            return Error(CredError.of_misconfigured(detail))
        case "precondition_required":
            return Error(CredError.of_precondition_required(detail))
        case _:
            return Error(CredError.of_upstream_unavailable(detail))


def _runtime_result_coordinator() -> RefreshCoordinator[Result[OAuthToken, CredError]] | None:
    return runtime_refresh_coordinator(encode_error=_encode_refresh_error, decode_error=_decode_refresh_error)


def _runtime_backend_and_coordinator() -> tuple[
    TokenCacheBackend | None, RefreshCoordinator[Result[OAuthToken, CredError]] | None, bool
]:
    """The cross-replica cache + coordinator when Redis is wired, else ``(None, None, False)`` so the
    foundation's in-process defaults are used (a single replica needs no shared cache or lock).
    """
    from litellm.proxy.common_utils.encrypt_decrypt_utils import (  # noqa: PLC0415  # runtime global
        decrypt_value_helper,
        encrypt_value_helper,
    )
    from litellm.proxy.proxy_server import redis_usage_cache  # noqa: PLC0415  # runtime global

    coordinator: Final = _runtime_result_coordinator()
    if coordinator is None or redis_usage_cache is None:
        return None, None, False
    codec: Final = OAuthTokenCacheCodec(
        encrypt_value_helper,
        lambda blob: decrypt_value_helper(blob, "mcp_per_user_token", exception_type="debug"),
    )
    backend: Final = DualCacheTokenCacheBackend(redis_usage_cache, codec)
    return backend, coordinator, True


async def _read_bound_credential(
    server_lookup: ServerLookup, user_id: str, server_id: str
) -> Mapping[str, object] | None:
    credential: Final = await _read_credential(user_id, server_id)
    server: Final = server_lookup(server_id)
    binding: Final = server.oauth_identity_binding if server else None
    if credential is not None and binding is not None and binding.mode == "enforce":
        if not await credential_binding_matches(binding, user_id, server_id, credential):
            return None
    return credential


def _build_per_user_oauth_token_store(
    server_lookup: ServerLookup,
) -> tuple[CachedOAuthTokenStore, bool]:
    backend, coordinator, uses_redis = _runtime_backend_and_coordinator()
    refresher: Final = AuthorizationCodeRefresher(
        server_lookup, _post_token_endpoint, _persist_credential, revoke=_revoke_credential
    )
    refreshing: Final = RefreshingTokenStore(
        V2PerUserTokenStore(partial(_read_bound_credential, server_lookup)), refresher, coordinator=coordinator
    )
    return CachedOAuthTokenStore(refreshing, default_ttl_seconds=_DEFAULT_TTL_SECONDS, backend=backend), uses_redis


def build_per_user_oauth_token_store(
    server_lookup: ServerLookup,
) -> CachedOAuthTokenStore:
    store, _uses_redis = _build_per_user_oauth_token_store(server_lookup)
    return store


class LazyPerUserOAuthTokenStore:
    """``OAuthTokenStore`` that builds the v2-native chain on first ``fetch``.

    The chain's cache/lock collaborators are LiteLLM runtime globals not available when the resolver
    is constructed at import time, so construction is deferred to the first request (by when they are
    wired). A no-Redis chain is replaced once Redis becomes available.
    """

    def __init__(
        self,
        server_lookup: ServerLookup,
        *,
        store_builder: StoreBuilder = _build_per_user_oauth_token_store,
        redis_available: Callable[[], bool] = _redis_cache_is_available,
    ) -> None:
        self._server_lookup = server_lookup
        self._store_builder = store_builder
        self._redis_available = redis_available
        self._store: ReactiveOAuthTokenStore | None = None
        self._uses_redis = False
        self._fetch_lock = asyncio.Condition()
        self._local_fetches = 0

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        outcome: Final = await self.fetch_result(user_id, server_id)
        return outcome.ok if isinstance(outcome, Ok) else None

    async def _enforce_identity_binding(self, user_id: str, server_id: str, token: OAuthToken) -> bool:
        server: Final = self._server_lookup(server_id)
        binding: Final = server.oauth_identity_binding if server else None
        if binding is not None and binding.mode == "enforce":
            if not await credential_binding_matches(
                binding, user_id, server_id, {"identity_binding_proof": token.identity_binding_proof}
            ):
                await self.invalidate(user_id, server_id)
                return False
        return True

    async def fetch_result(self, user_id: str, server_id: str) -> Result[OAuthToken, CredError]:
        outcome: Final = await self._fetch_token_result(user_id, server_id)
        if isinstance(outcome, Error):
            return outcome
        if not await self._enforce_identity_binding(user_id, server_id, outcome.ok):
            return Error(CredError.of_unauthorized("Authorization required: complete the OAuth flow for this server."))
        return outcome

    async def _fetch_token_result(self, user_id: str, server_id: str) -> Result[OAuthToken, CredError]:
        if self._uses_redis:
            cached_store: Final = self._store
            if cached_store is not None:
                return await fetch_token_result(cached_store, user_id, server_id)

        store, uses_redis = await self._store_for_fetch()
        return await asyncio.shield(self._fetch_then_finish(store, uses_redis, user_id, server_id))

    async def _fetch_then_finish(
        self, store: ReactiveOAuthTokenStore, uses_redis: bool, user_id: str, server_id: str
    ) -> Result[OAuthToken, CredError]:
        try:
            return await fetch_token_result(store, user_id, server_id)
        finally:
            if not uses_redis:
                await self._finish_local_fetch()

    async def peek(self, user_id: str, server_id: str) -> OAuthToken | None:
        token: Final = await self._peek_token(user_id, server_id)
        if token is not None and not await self._enforce_identity_binding(user_id, server_id, token):
            return None
        return token

    async def _peek_token(self, user_id: str, server_id: str) -> OAuthToken | None:
        if self._uses_redis:
            cached_store: Final = self._store
            if cached_store is not None:
                return await cached_store.peek(user_id, server_id)

        store, uses_redis = await self._store_for_fetch()
        return await asyncio.shield(self._peek_then_finish(store, uses_redis, user_id, server_id))

    async def _peek_then_finish(
        self, store: ReactiveOAuthTokenStore, uses_redis: bool, user_id: str, server_id: str
    ) -> OAuthToken | None:
        try:
            return await store.peek(user_id, server_id)
        finally:
            if not uses_redis:
                await self._finish_local_fetch()

    async def refresh_now(
        self, user_id: str, server_id: str, failed_access_token: str
    ) -> Result[OAuthToken, CredError]:
        if self._uses_redis:
            cached_store: Final = self._store
            if cached_store is not None:
                return await cached_store.refresh_now(user_id, server_id, failed_access_token)

        store, uses_redis = await self._store_for_fetch()
        return await asyncio.shield(
            self._refresh_then_finish(store, uses_redis, user_id, server_id, failed_access_token)
        )

    async def _refresh_then_finish(
        self, store: ReactiveOAuthTokenStore, uses_redis: bool, user_id: str, server_id: str, failed_access_token: str
    ) -> Result[OAuthToken, CredError]:
        try:
            return await store.refresh_now(user_id, server_id, failed_access_token)
        finally:
            if not uses_redis:
                await self._finish_local_fetch()

    async def invalidate(self, user_id: str, server_id: str) -> None:
        """Drop the chain's cached entry for ``(user_id, server_id)`` after the credential row
        changes (re-auth, revoke). Builds the chain if no fetch has run yet, so a shared (Redis)
        cache entry written by another worker is dropped too; the in-process case is then a no-op
        on an empty cache.
        """
        if self._uses_redis:
            cached_store: Final = self._store
            if cached_store is not None:
                await cached_store.invalidate(user_id, server_id)
                return

        store, uses_redis = await self._store_for_fetch()
        try:
            await store.invalidate(user_id, server_id)
        finally:
            if not uses_redis:
                await self._finish_local_fetch()

    async def _store_for_fetch(self) -> tuple[ReactiveOAuthTokenStore, bool]:
        async with self._fetch_lock:
            while (
                self._store is not None and not self._uses_redis and self._redis_available() and self._local_fetches > 0
            ):
                await self._fetch_lock.wait()
            store = self._store
            if store is None or (not self._uses_redis and self._redis_available()):
                store, self._uses_redis = self._store_builder(self._server_lookup)
                self._store = store
            uses_redis: Final = self._uses_redis
            if not uses_redis:
                self._local_fetches += 1
            return store, uses_redis

    async def _finish_local_fetch(self) -> None:
        async with self._fetch_lock:
            self._local_fetches -= 1
            if self._local_fetches == 0:
                self._fetch_lock.notify_all()

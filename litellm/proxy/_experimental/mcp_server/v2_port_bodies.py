"""Real (imperative-shell) bodies for the v2 outbound-credential ports.

These adapters bridge the clean-room v2 ports to real infrastructure (litellm's HTTP client, v1
storage, ...). They live on the v1/integration side so the v2 core keeps its no-v1-imports
invariant; the graft composition root injects them and v2 unit tests fake them.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Awaitable, Callable, Mapping, Optional

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from litellm.constants import MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy.gateway.mcp.outbound_credentials.clock import Clock, SystemClock
from litellm.proxy.gateway.mcp.outbound_credentials.credential_store import (
    CredentialKey,
)
from litellm.proxy.gateway.mcp.outbound_credentials.token_store import (
    StoredToken,
    TokenKey,
)
from litellm.proxy.gateway.mcp.outbound_credentials.types import (
    AssumeRole,
    AwsSigV4Config,
    ClientCredentialsConfig,
    CredError,
    StaticKeys,
    TokenExchangeConfig,
)
from litellm.proxy.gateway.mcp.result import Error, Ok, Result
from litellm.types.llms.custom_http import httpxSpecialProvider


class _TokenResponse(BaseModel):
    """The slice of an OAuth2 token response we consume; extra fields are ignored."""

    model_config = ConfigDict(extra="ignore")
    access_token: str
    expires_in: Optional[int] = None


async def request_token(
    endpoint: str, data: dict[str, str], clock: Clock
) -> Result[StoredToken, CredError]:
    """POST an OAuth token request and parse the response into a StoredToken.

    Shared by the client_credentials grant and the RFC 8693 exchange (both hit a token endpoint
    and read access_token / expires_in). Errors-as-values: network / 5xx -> upstream_unavailable,
    4xx -> misconfigured, missing access_token -> misconfigured. The raw lifetime is stored; the
    resolver arm's _REFRESH_BUFFER (= v1's default buffer) handles proactive re-mint.
    """
    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)
    try:
        response = await client.post(endpoint, data=data)
    except Exception as e:  # network / timeout / DNS
        return Error(
            CredError.of_upstream_unavailable(f"token endpoint unreachable: {e}")
        )
    if response is None:
        return Error(
            CredError.of_upstream_unavailable("token endpoint returned no response")
        )
    if response.status_code >= 500:
        return Error(
            CredError.of_upstream_unavailable(
                f"token endpoint returned {response.status_code}"
            )
        )
    if response.status_code >= 400:
        return Error(
            CredError.of_misconfigured(
                f"token request rejected ({response.status_code})"
            )
        )
    try:
        parsed = _TokenResponse.model_validate(response.json())
    except ValidationError:
        return Error(
            CredError.of_misconfigured("token response missing a valid access_token")
        )
    ttl = timedelta(
        seconds=(
            parsed.expires_in
            if parsed.expires_in is not None
            else MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL
        )
    )
    return Ok(
        StoredToken(
            access_token=SecretStr(parsed.access_token),
            expires_at=clock.now() + ttl,
        )
    )


class HttpxClientCredentialsFetcher:
    """RFC 6749 client_credentials grant via litellm's configured httpx client.

    Phase-1 graft body, mirroring v1's grant (`oauth2_token_cache._fetch_token`) for parity:
    `client_id` / `client_secret` / `scope` in the POST body, parse `access_token` /
    `expires_in`. Phase 2, when v2 owns the upstream MCP transport, swaps to the SDK's
    `ClientCredentialsOAuthProvider` (a connection-session-coupled `httpx.Auth`, not a fetcher).
    """

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._clock: Clock = clock or SystemClock()

    async def fetch(
        self, config: ClientCredentialsConfig
    ) -> Result[StoredToken, CredError]:
        data = {
            "grant_type": "client_credentials",
            "client_id": config.client_id,
            "client_secret": config.client_secret.get_secret_value(),
        }
        if config.scopes:
            data["scope"] = " ".join(config.scopes)

        return await request_token(config.token_url, data, self._clock)


_TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"


class HttpxTokenExchanger:
    """RFC 8693 token exchange (OBO) via litellm's configured httpx client.

    Swaps the caller's inbound subject_token for a token bound to the upstream (audience=resource,
    RFC 8707), mirroring v1's exchange grant: the gateway authenticates as an OAuth client via
    client_id / client_secret. The inbound token is sent only to the exchange endpoint, never to
    the upstream.
    """

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._clock: Clock = clock or SystemClock()

    async def exchange(
        self, config: TokenExchangeConfig, subject_token: SecretStr, resource: str
    ) -> Result[StoredToken, CredError]:
        if not config.token_exchange_endpoint:
            return Error(
                CredError.of_misconfigured(
                    "token_exchange: no exchange endpoint configured"
                )
            )
        data = {
            "grant_type": _TOKEN_EXCHANGE_GRANT_TYPE,
            "subject_token": subject_token.get_secret_value(),
            "subject_token_type": config.subject_token_type,
        }
        if config.client_id:
            data["client_id"] = config.client_id
        if config.client_secret is not None:
            data["client_secret"] = config.client_secret.get_secret_value()
        if resource:
            data["audience"] = resource
        if config.scopes:
            data["scope"] = " ".join(config.scopes)
        return await request_token(config.token_exchange_endpoint, data, self._clock)


class HttpxSigV4Signer:
    """AWS SigV4 signer for the `aws_sigv4` arm, backed by botocore.

    Phase-1 graft body: maps the typed credential source onto v1's proven `MCPSigV4Auth`, so the
    signed headers stay byte-identical to v1 (the signer relocates into the v2 package when v1 is
    retired). Credentials resolve eagerly at build time -- an unassumable role or a missing
    ambient chain fails closed here, not mid-request -- and botocore refreshes temporary STS
    credentials at sign time.
    """

    async def build(self, config: AwsSigV4Config) -> Result[httpx.Auth, CredError]:
        try:
            auth = await asyncio.to_thread(_build_sigv4_auth, config)
        except Exception as e:  # classified into a CredError below
            return Error(_classify_sigv4_error(e))
        return Ok(auth)


def _build_sigv4_auth(config: AwsSigV4Config) -> httpx.Auth:
    from litellm.experimental_mcp_client.client import MCPSigV4Auth

    creds = config.credentials
    if isinstance(creds, StaticKeys):
        return MCPSigV4Auth(
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key.get_secret_value(),
            aws_session_token=(
                creds.session_token.get_secret_value() if creds.session_token else None
            ),
            aws_region_name=config.region,
            aws_service_name=config.service,
        )
    if isinstance(creds, AssumeRole):
        return MCPSigV4Auth(
            aws_role_name=creds.role_arn,
            aws_session_name=creds.session_name,
            aws_region_name=config.region,
            aws_service_name=config.service,
        )
    return MCPSigV4Auth(aws_region_name=config.region, aws_service_name=config.service)


async def _read_v1_user_credential(subject_id: str, server_id: str) -> Optional[str]:
    from litellm.proxy._experimental.mcp_server.db import get_user_credential
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise RuntimeError("no DB client available for BYOK credential lookup")
    return await get_user_credential(prisma_client, subject_id, server_id)


class V1ByokCredentialStore:
    """Per-user BYOK credential store backed by v1's LiteLLM_MCPUserCredentials.

    Bridges the v2 CredentialStore port to v1's existing read (`db.get_user_credential`, which
    does the find_unique + credential_b64 decrypt), keyed by (subject_id == user_id, server_id).
    A missing row is `Ok(None)` (the arm turns that into a 401); a DB outage is
    `upstream_unavailable`. The reader is injected so the arm stays unit-testable without a DB.
    """

    def __init__(
        self,
        reader: Callable[
            [str, str], Awaitable[Optional[str]]
        ] = _read_v1_user_credential,
    ) -> None:
        self._reader = reader

    async def get(self, key: CredentialKey) -> Result[Optional[str], CredError]:
        if not key.subject_id:
            # No authenticated identity -> no per-user credential; never share one slot.
            return Ok(None)
        try:
            value = await self._reader(key.subject_id, key.server_id)
        except Exception as e:
            return Error(
                CredError.of_upstream_unavailable(f"BYOK credential lookup failed: {e}")
            )
        return Ok(value)


def _classify_sigv4_error(error: Exception) -> CredError:
    # botocore is an optional dependency without type stubs; match its connection-error classes
    # by name rather than importing it just to isinstance-check.
    if type(error).__name__ in (
        "EndpointConnectionError",
        "ConnectTimeoutError",
        "ConnectionError",
    ):
        return CredError.of_upstream_unavailable(
            f"STS unreachable while resolving aws_sigv4 credentials: {error}"
        )
    return CredError.of_misconfigured(
        f"aws_sigv4 credentials could not be resolved: {error}"
    )


# v1 refreshes the per-user token on every read, so the token this store returns is always
# currently valid; expiry is set beyond the resolver's refresh buffer to keep v2's proactive
# refresh inert (the OAuth dance and refresh stay on v1 until it retires).
_BRIDGE_TOKEN_TTL = timedelta(hours=1)


class _OAuthPayload(BaseModel):
    """The slice of v1's stored OAuth2 payload we consume; extra fields are ignored."""

    model_config = ConfigDict(extra="ignore")
    access_token: str


async def _read_valid_user_oauth_token(
    subject_id: str, server_id: str
) -> Optional[Mapping[str, object]]:
    from litellm.proxy._experimental.mcp_server.db import (
        get_user_oauth_credential,
        resolve_valid_user_oauth_token,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise RuntimeError("no DB client available for OAuth token lookup")
    server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
    if server is None:
        return None
    cred = await get_user_oauth_credential(prisma_client, subject_id, server_id)
    return await resolve_valid_user_oauth_token(subject_id, server, cred, prisma_client)


class V1OAuthTokenStore:
    """Per-user authorization_code TokenStore backed by v1's stored OAuth tokens.

    Read-path graft: ``get`` returns the user's currently-valid upstream token, reusing v1's
    ``resolve_valid_user_oauth_token`` (which reads the stored credential and refreshes it via the
    server's token_url when near expiry). Because v1 refreshes on every read, the returned token is
    always currently valid, so ``expires_at`` is set beyond the resolver's refresh buffer to keep
    v2's proactive-refresh path inert; the dance and refresh stay on v1 until it retires, at which
    point a v2-owned store returns real expiry and the resolver owns refresh. A missing/invalid
    token is ``Ok(None)`` (the arm turns that into a 401 that starts the OAuth flow); a DB outage is
    ``upstream_unavailable``. The reader is injected so the arm stays unit-testable without a DB.
    """

    def __init__(
        self,
        reader: Callable[
            [str, str], Awaitable[Optional[Mapping[str, object]]]
        ] = _read_valid_user_oauth_token,
        clock: Optional[Clock] = None,
    ) -> None:
        self._reader = reader
        self._clock: Clock = clock or SystemClock()

    async def get(self, key: TokenKey) -> Result[Optional[StoredToken], CredError]:
        if not key.subject_id:
            # No authenticated identity -> no per-user token; never share one slot.
            return Ok(None)
        try:
            payload = await self._reader(key.subject_id, key.server_id)
        except Exception as e:
            return Error(
                CredError.of_upstream_unavailable(f"OAuth token lookup failed: {e}")
            )
        if payload is None:
            return Ok(None)
        try:
            parsed = _OAuthPayload.model_validate(payload)
        except ValidationError:
            return Ok(None)  # no usable access_token -> drives the OAuth dance
        return Ok(
            StoredToken(
                access_token=SecretStr(parsed.access_token),
                expires_at=self._clock.now() + _BRIDGE_TOKEN_TTL,
                refresh_token=None,
            )
        )

    async def put(self, key: TokenKey, token: StoredToken) -> Result[None, CredError]:
        # Unreached during the bridge: get returns non-near-expiry tokens, so the resolver never
        # refreshes/persists through this port; v1 persists inside resolve_valid_user_oauth_token.
        _ = (key, token)
        return Ok(None)

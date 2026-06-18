"""Real (imperative-shell) bodies for the v2 outbound-credential ports.

These adapters bridge the clean-room v2 ports to real infrastructure (litellm's HTTP client, v1
storage, ...). They live on the v1/integration side so the v2 core keeps its no-v1-imports
invariant; the graft composition root injects them and v2 unit tests fake them.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from litellm.constants import MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy.gateway.mcp.outbound_credentials.clock import Clock, SystemClock
from litellm.proxy.gateway.mcp.outbound_credentials.token_store import StoredToken
from litellm.proxy.gateway.mcp.outbound_credentials.types import (
    ClientCredentialsConfig,
    CredError,
)
from litellm.proxy.gateway.mcp.result import Error, Ok, Result
from litellm.types.llms.custom_http import httpxSpecialProvider


class _TokenResponse(BaseModel):
    """The slice of an OAuth2 token response we consume; extra fields are ignored."""

    model_config = ConfigDict(extra="ignore")
    access_token: str
    expires_in: Optional[int] = None


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

        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)
        try:
            response = await client.post(config.token_url, data=data)
        except Exception as e:  # network / timeout / DNS
            return Error(
                CredError.of_upstream_unavailable(
                    f"client_credentials token endpoint unreachable: {e}"
                )
            )

        if response is None:
            return Error(
                CredError.of_upstream_unavailable(
                    "client_credentials token endpoint returned no response"
                )
            )
        if response.status_code >= 500:
            return Error(
                CredError.of_upstream_unavailable(
                    f"client_credentials token endpoint returned {response.status_code}"
                )
            )
        if response.status_code >= 400:
            return Error(
                CredError.of_misconfigured(
                    f"client_credentials grant rejected ({response.status_code})"
                )
            )

        try:
            parsed = _TokenResponse.model_validate(response.json())
        except ValidationError:
            return Error(
                CredError.of_misconfigured(
                    "client_credentials response missing a valid access_token"
                )
            )
        # Store the raw lifetime; the resolver arm's _REFRESH_BUFFER (60s, = v1's default
        # buffer) handles proactive re-mint, so no buffer is subtracted here.
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
                expires_at=self._clock.now() + ttl,
            )
        )

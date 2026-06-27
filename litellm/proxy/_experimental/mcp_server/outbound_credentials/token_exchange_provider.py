"""Composition root for the v2-native token_exchange (OBO) exchanger.

Wires the pure ``Rfc8693TokenExchanger`` to its runtime edges: the real httpx POST against the IdP and
the configured cache sizing/TTL constants. The exchanger (and its in-process cache) must be built once
and reused so the cache survives across requests, so ``LazyTokenExchanger`` builds it on first use and
holds it. The httpx client is acquired per call (its globals are not import-time ready), mirroring v1.
"""

from __future__ import annotations

from litellm._logging import verbose_logger
from litellm.constants import (
    MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
    MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
    MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
    MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    InMemoryTokenCacheBackend,
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchanger import (
    Rfc8693TokenExchanger,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    CredError,
    ServerSpec,
    TokenExchangeConfig,
)


async def _post_exchange_endpoint(url: str, form: dict[str, str]) -> dict[str, object] | None:
    from litellm.llms.custom_httpx.http_handler import (  # noqa: PLC0415
        get_async_httpx_client,  # pyright: ignore
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider  # noqa: PLC0415

    # litellm's httpx handler and httpx.Response are only partially typed; the IdP returns a JSON
    # object and the exchanger validates each field, so the untyped boundary is contained here.
    # A failed exchange is a miss, not a 500 (matches v1), so any error becomes None.
    headers = {"Accept": "application/json"}
    try:
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)  # pyright: ignore
        response = await client.post(url, headers=headers, data=form)  # pyright: ignore
        response.raise_for_status()  # pyright: ignore
        body: dict[str, object] = response.json()  # pyright: ignore
    except Exception as exc:  # noqa: BLE001
        verbose_logger.warning("MCP token exchange request failed: %s", exc)
        return None
    else:
        return body  # pyright: ignore


def build_token_exchanger() -> Rfc8693TokenExchanger:
    return Rfc8693TokenExchanger(
        _post_exchange_endpoint,
        cache=InMemoryTokenCacheBackend(max_size=MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE),
        default_ttl_seconds=MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
        min_ttl_seconds=MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
        expiry_buffer_seconds=MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
    )


class LazyTokenExchanger:
    """``TokenExchanger`` that builds the exchanger (and its process-lifetime cache) on first use.

    Built once, then reused, so the exchanged-token cache persists across requests rather than being
    discarded each call.
    """

    def __init__(self) -> None:
        self._exchanger: Rfc8693TokenExchanger | None = None

    async def exchange(
        self, subject_token: str, server: ServerSpec, config: TokenExchangeConfig
    ) -> Result[OAuthToken, CredError]:
        if self._exchanger is None:
            self._exchanger = build_token_exchanger()
        return await self._exchanger.exchange(subject_token, server, config)

"""Tests for the token_exchange composition root: build-once laziness and the HTTP edge contract.

`LazyTokenExchanger` must build its exchanger (and process-lifetime cache) exactly once;
`_post_exchange_endpoint` is the I/O edge that maps any transport/HTTP failure to None and parses a
JSON body on success.
"""

from unittest.mock import patch

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    Error,
    ServerSpec,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchange_provider import (
    LazyTokenExchanger,
    _post_exchange_endpoint,
    build_token_exchanger,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchanger import (
    Rfc8693TokenExchanger,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    TokenExchangeConfig,
)

_HTTP_CLIENT = "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"
_BUILD = "litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchange_provider.build_token_exchanger"


def test_build_token_exchanger_returns_an_exchanger():
    assert isinstance(build_token_exchanger(), Rfc8693TokenExchanger)


@pytest.mark.asyncio
async def test_lazy_builds_once_across_calls():
    incomplete = TokenExchangeConfig()  # misconfigured before any HTTP, so no network needed
    server = ServerSpec(server_id="s", resource="r", config=incomplete)
    with patch(_BUILD, wraps=build_token_exchanger) as spy:
        lazy = LazyTokenExchanger()
        first = await lazy.exchange("jwt", server, incomplete)
        second = await lazy.exchange("jwt", server, incomplete)
        assert spy.call_count == 1
    assert isinstance(first, Error) and isinstance(second, Error)
    assert first.error.tag == "misconfigured"


@pytest.mark.asyncio
async def test_post_returns_none_on_transport_error():
    with patch(_HTTP_CLIENT, side_effect=RuntimeError("boom")):
        result = await _post_exchange_endpoint("https://idp/token", {"grant_type": "x"})
    assert result is None


@pytest.mark.asyncio
async def test_post_parses_json_body_on_success():
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"access_token": "x", "expires_in": 60}

    class _Client:
        async def post(self, url, headers, data):
            return _Resp()

    with patch(_HTTP_CLIENT, return_value=_Client()):
        result = await _post_exchange_endpoint("https://idp/token", {"grant_type": "x"})
    assert result == {"access_token": "x", "expires_in": 60}

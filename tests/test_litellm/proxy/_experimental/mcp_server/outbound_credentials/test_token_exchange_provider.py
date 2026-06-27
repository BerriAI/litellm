"""Tests for the token_exchange composition root: the built exchanger and the HTTP edge contract.

`build_token_exchanger` wires the pure exchanger to its runtime edges; `_post_exchange_endpoint` is
the I/O edge that maps any transport/HTTP failure to None and parses a JSON body on success.
"""

from unittest.mock import patch

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchange_provider import (
    _post_exchange_endpoint,
    build_token_exchanger,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchanger import (
    Rfc8693TokenExchanger,
)

_HTTP_CLIENT = "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"


def test_build_token_exchanger_returns_an_exchanger():
    assert isinstance(build_token_exchanger(), Rfc8693TokenExchanger)


def test_build_gives_each_caller_an_independent_cache():
    # Separate builds must not share a cache, so one egress instance cannot serve another's tokens.
    assert build_token_exchanger() is not build_token_exchanger()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [["a", "b"], "a-string", 42], ids=["list", "str", "int"])
async def test_post_returns_none_on_non_object_json(payload):
    # A valid-but-non-object JSON body must become a miss, not crash field parsing downstream.
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return payload

    class _Client:
        async def post(self, url, headers, data):
            return _Resp()

    with patch(_HTTP_CLIENT, return_value=_Client()):
        result = await _post_exchange_endpoint("https://idp/token", {"grant_type": "x"})
    assert result is None

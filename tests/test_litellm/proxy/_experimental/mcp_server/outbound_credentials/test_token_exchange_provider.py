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
    SubjectTokenRejected,
    TokenExchangeClientError,
)

_HTTP_CLIENT = "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"


def _client_raising_4xx(body: object):
    """An httpx client whose POST returns a 4xx whose ``raise_for_status`` raises an HTTPStatusError
    carrying ``body`` as its JSON, so the RFC 6749 error-code classification can be driven."""
    import httpx

    request = httpx.Request("POST", "https://idp/token")
    response = httpx.Response(400, json=body, request=request)

    class _Resp:
        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError("bad request", request=request, response=response)

    class _Client:
        async def post(self, url, headers, data):
            return _Resp()

    return _Client()


def test_build_token_exchanger_returns_an_exchanger():
    assert isinstance(build_token_exchanger(), Rfc8693TokenExchanger)


def test_build_gives_each_caller_an_independent_cache():
    # Separate builds must not share a cache, so one egress instance cannot serve another's tokens.
    assert build_token_exchanger() is not build_token_exchanger()


@pytest.mark.asyncio
async def test_post_returns_none_on_transport_error():
    with patch(_HTTP_CLIENT, side_effect=RuntimeError("boom")):
        result = await _post_exchange_endpoint("https://idp/token", {"grant_type": "x"}, {})
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
        result = await _post_exchange_endpoint("https://idp/token", {"grant_type": "x"}, {})
    assert result == {"access_token": "x", "expires_in": 60}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", ["invalid_client", "unauthorized_client", "unsupported_grant_type", "invalid_target", "invalid_scope"]
)
async def test_post_maps_gateway_fault_4xx_to_client_error(code):
    # RFC 6749 5.2 gateway-fault codes must raise TokenExchangeClientError (-> 500), not the caller 401.
    with patch(_HTTP_CLIENT, return_value=_client_raising_4xx({"error": code})):
        with pytest.raises(TokenExchangeClientError):
            await _post_exchange_endpoint("https://idp/token", {"grant_type": "x"}, {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [{"error": "invalid_grant"}, {"error": "invalid_request"}, {}, {"error": 123}, "not-json-object"],
    ids=["invalid_grant", "invalid_request", "no_error", "non_str_error", "non_dict"],
)
async def test_post_maps_subject_fault_4xx_to_subject_rejected(body):
    # A subject-fault code (or an unparseable/absent error) is the caller's problem -> SubjectTokenRejected (401).
    with patch(_HTTP_CLIENT, return_value=_client_raising_4xx(body)):
        with pytest.raises(SubjectTokenRejected):
            await _post_exchange_endpoint("https://idp/token", {"grant_type": "x"}, {})


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
        result = await _post_exchange_endpoint("https://idp/token", {"grant_type": "x"}, {})
    assert result is None

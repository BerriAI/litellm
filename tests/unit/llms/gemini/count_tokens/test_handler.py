import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter


@pytest.mark.asyncio
async def test_acount_tokens_non_json_body_raises_api_error_with_502():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>proxy error page</html>")

    with pytest.raises(litellm.APIError) as exc_info:
        await GoogleAIStudioTokenCounter().acount_tokens(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
            api_key="test-key",
            client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )

    assert exc_info.value.status_code == 502
    assert "non-JSON" in exc_info.value.message


@pytest.mark.asyncio
async def test_acount_tokens_lets_internal_errors_propagate():
    def _handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("transport exploded")

    with pytest.raises(RuntimeError, match="transport exploded"):
        await GoogleAIStudioTokenCounter().acount_tokens(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            api_key="test-key",
            client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )


def _timing_out(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _litellm_handler_timing_out() -> AsyncHTTPHandler:
    handler = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(_timing_out))
    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client",
    (
        pytest.param(httpx.AsyncClient(transport=httpx.MockTransport(_timing_out)), id="httpx-client"),
        pytest.param(_litellm_handler_timing_out(), id="litellm-http-handler"),
    ),
)
async def test_acount_tokens_raises_connection_error_on_timeout(client):
    with pytest.raises(litellm.APIConnectionError):
        await GoogleAIStudioTokenCounter().acount_tokens(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            api_key="test-key",
            client=client,
        )

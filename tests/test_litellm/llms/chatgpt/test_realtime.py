import json

import httpx
import pytest

import litellm
from litellm.llms.chatgpt.realtime import ChatGPTRealtime
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.router import GenericLiteLLMParams


@pytest.mark.asyncio
async def test_chatgpt_call_keeps_oauth_and_frameless_session(chatgpt_tokens):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(201, text="v=0\r\n", headers={"location": "/v1/realtime/calls/rtc_test"})

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    response = await litellm.arealtime_calls(
        model="chatgpt/gpt-live-1-codex",
        openai_ephemeral_key="",
        sdp_body=b"v=0\r\n",
        session={"model": "chatgpt/gpt-live-1-codex", "audio": {"output": {"voice": "sol"}}},
        extra_query={"intent": "quicksilver", "architecture": "avas"},
        extra_headers={"openai-alpha": "quicksilver=v2"},
        client=client,
    )
    assert response.status_code == 201
    assert requests[0].url.path == "/backend-api/codex/realtime/calls"
    assert requests[0].url.params["architecture"] == "avas"
    assert requests[0].headers["authorization"] == "Bearer test-token-" + "default"
    assert json.loads(requests[0].content) == {
        "sdp": "v=0\r\n",
        "session": {"model": "gpt-live-1-codex", "audio": {"output": {"voice": "sol"}}},
    }
    await client.client.aclose()


@pytest.mark.parametrize("model,endpoint", [("gpt-realtime-1.5", "realtime"), ("gpt-live-1-codex", "live")])
def test_realtime_uses_platform_endpoint_with_oauth_headers(model, endpoint, chatgpt_tokens):
    handler = ChatGPTRealtime(
        GenericLiteLLMParams(),
        {
            "authorization": "Bearer proxy-key",
            "openai-alpha": "quicksilver=v2",
        },
    )
    assert handler._construct_url("https://api.openai.com/v1", {"model": model}) == (
        f"wss://api.openai.com/v1/{endpoint}?model={model}"
    )
    headers = handler._get_additional_headers("unused")
    assert headers["Authorization"] == "Bearer test-token-default"
    assert "authorization" not in headers
    assert headers["openai-alpha"] == "quicksilver=v2"

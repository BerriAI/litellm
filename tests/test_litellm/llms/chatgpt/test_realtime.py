import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from websockets.asyncio.server import serve

import litellm
from litellm.llms.chatgpt.realtime import ChatGPTRealtime
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.router import GenericLiteLLMParams


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-realtime-1.5", "gpt-live-1-codex"])
@pytest.mark.parametrize("call_id", [None, "rtc_existing"])
async def test_websocket_forwards_configured_headers_without_client_identity(model, call_id, chatgpt_tokens):
    captured = asyncio.get_running_loop().create_future()

    async def receive_connection(connection):
        captured.set_result(connection.request.headers)
        await connection.wait_closed()

    websocket = SimpleNamespace(
        headers={"authorization": "Bearer client", "cookie": "private-cookie", "openai-alpha": "client-value"},
        scope={},
        receive_text=AsyncMock(side_effect=RuntimeError("client disconnected")),
        send_text=AsyncMock(),
        close=AsyncMock(),
    )
    async with serve(receive_connection, "127.0.0.1", 0) as gateway:
        port = gateway.sockets[0].getsockname()[1]
        await asyncio.wait_for(litellm._arealtime(
            model=f"chatgpt/{model}", websocket=websocket, api_base=f"http://127.0.0.1:{port}",
            chatgpt_realtime_call_id=call_id,
            headers={"x-deployment-header": "configured"},
            extra_headers={"X-Gateway-Route": "voice", "OpenAI-Alpha": "configured-value",
                           "aUtHoRiZaTiOn": "Bearer wrong", "CHATGPT-ACCOUNT-ID": "wrong"},
        ), timeout=10)
        headers = await asyncio.wait_for(captured, timeout=5)
    assert headers["x-deployment-header"] == "configured"
    assert headers["x-gateway-route"] == "voice"
    assert headers["openai-alpha"] == "configured-value"
    assert headers["authorization"] == "Bearer test-token-default"
    assert headers["chatgpt-account-id"] == "test-account-default"
    assert "cookie" not in headers


@pytest.mark.asyncio
@pytest.mark.parametrize("api_base", [None, "https://voice.example/backend-api/codex"])
async def test_chatgpt_call_keeps_oauth_and_frameless_session(chatgpt_tokens, api_base):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(201, text="v=0\r\n", headers={"location": "/v1/realtime/calls/rtc_test"})

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    response = await litellm.arealtime_calls(
        model="chatgpt/gpt-live-1-codex",
        api_base=api_base,
        openai_ephemeral_key="",
        sdp_body=b"v=0\r\n",
        session={"model": "chatgpt/gpt-live-1-codex", "audio": {"output": {"voice": "sol"}}},
        extra_query={"intent": "quicksilver", "architecture": "avas"},
        extra_headers={
            "openai-alpha": "quicksilver=v2",
            "x-gateway-route": "voice",
            "aUtHoRiZaTiOn": "Bearer wrong",
            "CHATGPT-ACCOUNT-ID": "wrong",
        },
        client=client,
    )
    assert response.extensions["chatgpt_realtime"]["api_base"] == (api_base or "https://api.openai.com/v1")
    assert response.extensions["chatgpt_realtime"]["extra_headers"] == {"openai-alpha": "quicksilver=v2", "x-gateway-route": "voice"}
    assert requests[0].url.host == ("voice.example" if api_base else "chatgpt.com")
    assert response.status_code == 201
    assert requests[0].url.path == "/backend-api/codex/realtime/calls"
    assert requests[0].url.params["architecture"] == "avas"
    assert requests[0].headers["authorization"] == "Bearer test-token-" + "default"
    assert requests[0].headers["chatgpt-account-id"] == "test-account-default"
    assert requests[0].headers["openai-alpha"] == "quicksilver=v2"
    assert requests[0].headers["x-gateway-route"] == "voice"
    assert json.loads(requests[0].content) == {
        "sdp": "v=0\r\n",
        "session": {"model": "gpt-live-1-codex", "audio": {"output": {"voice": "sol"}}},
    }
    await client.client.aclose()


@pytest.mark.asyncio
async def test_openai_call_preserves_explicit_identity_headers():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(201, text="v=0\r\n")

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        response = await litellm.arealtime_calls(
            model="openai/gpt-realtime-1.5",
            openai_ephemeral_key="original-key",
            sdp_body=b"v=0\r\n",
            extra_headers={"Authorization": "Bearer explicit-key", "chatgpt-account-id": "custom-account"},
            client=client,
        )
        assert response.status_code == 201
        assert requests[0].headers["authorization"] == "Bearer explicit-key"
        assert requests[0].headers["chatgpt-account-id"] == "custom-account"
        assert requests[0].headers["content-type"].startswith("multipart/form-data")
    finally:
        await client.client.aclose()


@pytest.mark.parametrize("model,endpoint", [("gpt-realtime-1.5", "realtime"), ("gpt-live-1-codex", "live")])
def test_realtime_uses_platform_endpoint_with_oauth_headers(model, endpoint, chatgpt_tokens, local_model_cost_map):
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


@pytest.mark.parametrize("endpoint", ["live", "realtime"])
@pytest.mark.parametrize("call_id", [None, "rtc_metadata"])
def test_realtime_routes_new_models_using_registered_metadata(endpoint, call_id, chatgpt_tokens, local_model_cost_map):
    model = "metadata-voice-model"
    litellm.register_model({f"chatgpt/{model}": {
        "litellm_provider": "chatgpt", "mode": "realtime", "supported_endpoints": [f"/v1/{endpoint}"]
    }})
    handler = ChatGPTRealtime(GenericLiteLLMParams(chatgpt_realtime_call_id=call_id), {})
    expected = (
        f"wss://api.openai.com/v1/{endpoint}?model={model}" if call_id is None
        else f"wss://api.openai.com/v1/live/{call_id}" if endpoint == "live"
        else f"wss://api.openai.com/v1/realtime?call_id={call_id}"
    )
    assert handler._construct_url("https://api.openai.com/v1", {"model": model}) == expected


def test_realtime_unknown_model_keeps_standard_endpoint(chatgpt_tokens, local_model_cost_map):
    handler = ChatGPTRealtime(GenericLiteLLMParams(), {})
    assert handler._construct_url("https://api.openai.com/v1", {"model": "unknown-voice-model"}) == (
        "wss://api.openai.com/v1/realtime?model=unknown-voice-model"
    )


@pytest.mark.parametrize("env_name", ["CHATGPT_API_BASE", "OPENAI_CHATGPT_API_BASE"])
@pytest.mark.parametrize("api_base", [None, "https://deployment.example/codex"])
def test_realtime_routes_use_configured_gateway(monkeypatch, env_name, api_base, chatgpt_tokens):
    from litellm.llms.chatgpt.realtime import ChatGPTRealtimeHTTPConfig

    monkeypatch.delenv("CHATGPT_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_CHATGPT_API_BASE", raising=False)
    monkeypatch.setenv(env_name, "https://gateway.example/codex/")
    expected = api_base or "https://gateway.example/codex"
    config = ChatGPTRealtimeHTTPConfig(GenericLiteLLMParams())
    assert config.get_realtime_calls_url(api_base, "gpt-live-1-codex") == expected + "/realtime/calls"
    handler = ChatGPTRealtime(GenericLiteLLMParams(), {})
    assert handler._construct_url(handler.get_api_base(api_base), {"model": "gpt-realtime-1.5"}) == (
        expected.replace("https://", "wss://") + "/realtime?model=gpt-realtime-1.5"
    )

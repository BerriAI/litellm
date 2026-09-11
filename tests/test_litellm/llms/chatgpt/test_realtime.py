import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm.llms.chatgpt.realtime import ChatGPTRealtime
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.router import GenericLiteLLMParams

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["closed", "network"])
@pytest.mark.parametrize("hangup_status", [200, 503])
async def test_live_closed_observer_uses_independent_hangup(failure, hangup_status, chatgpt_tokens, monkeypatch):
    from websockets.exceptions import ConnectionClosedOK
    from websockets.frames import Close

    from litellm.caching.llm_caching_handler import LLMClientCache

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    handler = ChatGPTRealtime(
        GenericLiteLLMParams(
            chatgpt_realtime_call_id="rtc_live_closed",
            chatgpt_token_dir=chatgpt_tokens,
            extra_query={"gateway": "tenant", "tag": ["alpha +/&", "beta"]},
        ),
        {},
        {"x-gateway-token": "test-only"},
    )
    connection = SimpleNamespace(
        send=AsyncMock(
            side_effect=(
                ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)
                if failure == "closed"
                else OSError("socket unavailable")
            )
        )
    )
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(hangup_status)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        with patch("httpx.AsyncClient", return_value=client) as create_client:
            for _ in range(2):
                with pytest.raises(httpx.HTTPStatusError) if hangup_status == 503 else nullcontext():
                    await handler.close_call(connection, "gpt-live-1-codex", "https://gateway.example/v1")
                assert not client.is_closed
            create_client.assert_called_once()
    finally:
        await client.aclose()
    assert len(requests) == 2
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/v1/realtime/calls/rtc_live_closed/hangup"
    assert requests[0].url.params.get_list("tag") == ["alpha +/&", "beta"]
    assert requests[0].url.params["gateway"] == "tenant"
    assert requests[0].headers["x-gateway-token"] == "test-only"
    assert requests[0].headers["Authorization"] == "Bearer test-token-default"
    assert requests[0].extensions["timeout"]["read"] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["client_secrets", "transcription_sessions"])
@pytest.mark.parametrize("source", ["default", "explicit", "CHATGPT_API_BASE", "OPENAI_CHATGPT_API_BASE"])
async def test_realtime_session_urls_honor_gateway(endpoint, source, chatgpt_tokens, monkeypatch):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", chatgpt_tokens)
    monkeypatch.delenv("CHATGPT_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_CHATGPT_API_BASE", raising=False)
    gateway = "https://voice.example/custom/v1/"
    if source in ("CHATGPT_API_BASE", "OPENAI_CHATGPT_API_BASE"):
        monkeypatch.setenv(source, gateway)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"client_secret": {"value": "test-secret"}})

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    kwargs = {"model": "chatgpt/gpt-realtime-1.5", "client": client}
    if source == "explicit":
        kwargs["api_base"] = gateway
    try:
        if endpoint == "client_secrets":
            await litellm.acreate_realtime_client_secret(**kwargs)
        else:
            await litellm.acreate_realtime_transcription_session(**kwargs)
    finally:
        await client.client.aclose()
    base = "https://api.openai.com/v1" if source == "default" else gateway.rstrip("/")
    assert len(requests) == 1
    assert str(requests[0].url) == f"{base}/realtime/{endpoint}"
    assert requests[0].headers["authorization"] == "Bearer test-token-default"


@pytest.mark.asyncio
@pytest.mark.parametrize("inbound_headers", [{}, {"openai-alpha": "quicksilver=v2"}])
@pytest.mark.parametrize("model, endpoint", [("gpt-live-1-codex", "live"), ("gpt-realtime-1.5", "realtime")])
async def test_routed_call_preserves_deployment_gateway_headers(
    inbound_headers, model, endpoint, chatgpt_tokens, monkeypatch
):
    from litellm.llms.chatgpt.codex import (
        CodexRealtimeCall,
        CodexRealtimeOffer,
        build_call_request,
        build_sideband_request,
        parse_call_response,
    )

    monkeypatch.setenv("CHATGPT_TOKEN_DIR", chatgpt_tokens)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(201, text="v=0\r\n", headers={"location": "/v1/realtime/calls/rtc_test"})

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    router = litellm.Router(
        model_list=[
            {
                "model_name": "voice-gateway",
                "litellm_params": {
                    "model": f"chatgpt/{model}",
                    "api_base": "https://voice.example/backend-api/codex",
                    "extra_headers": {"x-gateway-route": "configured"},
                    "extra_query": {
                        "gateway_token": "configured",
                        "intent": "pinned-intent",
                        "count": 7,
                        "fraction": 1.5,
                        "enabled": True,
                        "disabled": False,
                        "blank": None,
                        "tag": ["alpha +/&", "beta"],
                        "empty": [],
                        "model": "other-model",
                        "call_id": "rtc_wrong",
                    },
                },
                "model_info": {"id": "selected-gateway-deployment"},
            }
        ],
        num_retries=0,
    )
    offer = CodexRealtimeOffer(sdp="v=0\r\n", session={"model": "voice-gateway"})
    try:
        response = await router.arealtime_calls(
            **build_call_request(offer, {"intent": "quicksilver", "architecture": "avas"}, inbound_headers),
            client=client,
        )
        assert requests[0].headers.get("x-gateway-route") == "configured"
        assert dict(requests[0].url.params) == {
            "gateway_token": "configured",
            "intent": "pinned-intent",
            "architecture": "avas",
            "count": "7",
            "fraction": "1.5",
            "enabled": "true",
            "disabled": "false",
            "blank": "",
            "tag": "alpha +/&",
            "model": "other-model",
            "call_id": "rtc_wrong",
        }
        assert requests[0].url.params.get_list("tag") == ["alpha +/&", "beta"]
        assert response.extensions["chatgpt_realtime"]["extra_query"] == {
            **dict(requests[0].url.params),
            "tag": ("alpha +/&", "beta"),
            "empty": (),
        }
        assert response.extensions["chatgpt_realtime"]["extra_headers"]["x-gateway-route"] == "configured"
        for name, value in inbound_headers.items():
            assert requests[0].headers[name] == value
        call = parse_call_response(response, alias="voice-gateway", owner="test-owner", expires_at=1)
        restored = CodexRealtimeCall.model_validate_json(call.model_dump_json())
        assert restored.model_id == "selected-gateway-deployment"
        assert restored.model == model
        handler = ChatGPTRealtime(GenericLiteLLMParams.model_validate(build_sideband_request(restored)), {})
        sideband_url = httpx.URL(handler._construct_url(restored.api_base, {"model": restored.model}))
        assert {key: value for key, value in sideband_url.params.items() if key != "call_id"} == {
            key: value for key, value in requests[0].url.params.items() if key not in ("model", "call_id")
        }
        assert sideband_url.params.get("call_id") == ("rtc_test" if endpoint == "realtime" else None)
        assert sideband_url.params.get_list("tag") == ["alpha +/&", "beta"]
        assert sideband_url.path.endswith("/realtime" if endpoint == "realtime" else "/live/rtc_test")
    finally:
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-realtime-1.5", "gpt-live-1-codex"])
@pytest.mark.parametrize("call_id", [None, "rtc_existing"])
async def test_websocket_forwards_configured_headers_without_client_identity(model, call_id, chatgpt_tokens):
    websocket = SimpleNamespace(
        headers={"authorization": "Bearer client", "cookie": "private-cookie", "openai-alpha": "client-value"},
        scope={},
        receive_text=AsyncMock(side_effect=RuntimeError("client disconnected")),
        send_text=AsyncMock(),
        close=AsyncMock(),
    )
    with patch("websockets.connect") as connect:
        connect.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("stop before streaming"))
        await litellm._arealtime(
            model=f"chatgpt/{model}",
            websocket=websocket,
            api_base="https://voice.example/codex",
            chatgpt_realtime_call_id=call_id,
            query_params={"model": model, "intent": "client-intent"},
            extra_query={"intent": "configured-intent", "tag": ["alpha +/&", "beta"]},
            headers={"x-deployment-header": "configured"},
            extra_headers={
                "X-Gateway-Route": "voice",
                "OpenAI-Alpha": "configured-value",
                "aUtHoRiZaTiOn": "Bearer wrong",
                "CHATGPT-ACCOUNT-ID": "wrong",
            },
        )
        connect.assert_called_once()
        headers = httpx.Headers(connect.call_args.kwargs["additional_headers"])
        upstream_url = httpx.URL(connect.call_args.args[0])
        assert upstream_url.params.get_list("intent") == ["configured-intent"]
        assert upstream_url.params.get_list("tag") == ["alpha +/&", "beta"]
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
        chatgpt_realtime_client_query={"intent": "untrusted-override", "architecture": "avas", "untrusted": "bad"},
        extra_headers={
            "openai-alpha": "quicksilver=v2",
            "x-gateway-route": "voice",
            "aUtHoRiZaTiOn": "Bearer wrong",
            "CHATGPT-ACCOUNT-ID": "wrong",
        },
        client=client,
    )
    assert response.extensions["chatgpt_realtime"]["api_base"] == (api_base or "https://api.openai.com/v1")
    assert response.extensions["chatgpt_realtime"]["extra_headers"] == {
        "openai-alpha": "quicksilver=v2",
        "x-gateway-route": "voice",
    }
    assert requests[0].url.host == ("voice.example" if api_base else "chatgpt.com")
    assert response.status_code == 201
    assert response.extensions["chatgpt_realtime"]["extra_query"] == {"intent": "quicksilver", "architecture": "avas"}
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
def test_new_realtime_session_preserves_gateway_query(endpoint, chatgpt_tokens, local_model_cost_map):
    model = "gpt-live-1-codex" if endpoint == "live" else "gpt-realtime-1.5"
    handler = ChatGPTRealtime(
        GenericLiteLLMParams(
            chatgpt_token_dir=chatgpt_tokens,
            chatgpt_realtime_client_query={"intent": "conversation", "architecture": "client-architecture"},
            extra_query={
                "gateway_token": "opaque +/& value",
                "intent": "gateway-intent",
                "architecture": "gateway-architecture",
                "model": "other-model",
                "call_id": "rtc_other",
            },
        ),
        {},
    )
    url = httpx.URL(handler._construct_url("https://gateway.example/v1", {"model": model, "intent": "query-intent"}))
    assert url.path == f"/v1/{endpoint}"
    assert dict(url.params) == {
        "model": model,
        "gateway_token": "opaque +/& value",
        "intent": "gateway-intent",
        "architecture": "gateway-architecture",
    }


@pytest.mark.asyncio
async def test_openai_http_call_does_not_require_websockets(monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm.llms.chatgpt.realtime", raising=False)
    for name in tuple(sys.modules):
        if name == "websockets" or name.startswith("websockets."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "websockets", None)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(201, text="v=0\r\n")

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        response = await litellm.arealtime_calls(
            model="openai/gpt-realtime-1.5",
            openai_ephemeral_key="test-only",
            sdp_body=b"v=0\r\n",
            api_key="test-only",
            client=client,
        )
        assert response.status_code == 201
        assert len(requests) == 1
        assert requests[0].url.path == "/v1/realtime/calls"
    finally:
        await client.client.aclose()


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


@pytest.mark.parametrize("model,endpoint", [("gpt-live-1-codex", "live"), ("gpt-realtime-1.5", "realtime")])
def test_sideband_restores_gateway_query_without_overriding_call(model, endpoint, chatgpt_tokens):
    handler = ChatGPTRealtime(
        GenericLiteLLMParams(
            chatgpt_realtime_call_id="rtc_selected",
            extra_query={"gateway_token": "opaque +/& value", "model": "other", "call_id": "rtc_other"},
        ),
        {},
    )
    url = httpx.URL(handler._construct_url("https://gateway.example/v1", {"model": model}))
    assert url.params["gateway_token"] == "opaque +/& value"
    assert "model" not in url.params
    if endpoint == "live":
        assert url.path == "/v1/live/rtc_selected"
        assert "call_id" not in url.params
    else:
        assert url.path == "/v1/realtime"
        assert url.params["call_id"] == "rtc_selected"


def test_client_cannot_forge_supervised_call_accounting(chatgpt_tokens):
    from litellm.llms.chatgpt.realtime import CallAccounting, accounts_for_call_usage

    assert accounts_for_call_usage(GenericLiteLLMParams(chatgpt_call_accounting={"supervised": True}))
    assert accounts_for_call_usage(GenericLiteLLMParams(chatgpt_call_accounting="supervised"))
    assert not accounts_for_call_usage(GenericLiteLLMParams(chatgpt_call_accounting=CallAccounting.SUPERVISED))


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-live-1-codex", "gpt-realtime-1.5"])
async def test_supervisor_connection_preserves_call_routing(model, chatgpt_tokens):
    handler = ChatGPTRealtime(
        GenericLiteLLMParams(
            chatgpt_token_dir=chatgpt_tokens,
            chatgpt_realtime_call_id="rtc_owner",
            extra_query={"gateway_token": "a+b&c"},
        ),
        {"openai-alpha": "quicksilver=v2"},
        {"x-gateway-token": "configured"},
    )
    connection = AsyncMock()
    with patch("websockets.connect", AsyncMock(return_value=connection)) as connect:
        assert await handler.open_call_connection(model, "https://gateway.example/v1") is connection
    url = httpx.URL(connect.call_args.args[0])
    assert url.params["gateway_token"] == "a+b&c"
    assert connect.call_args.kwargs["additional_headers"]["x-gateway-token"] == "configured"
    assert url.path.endswith("/rtc_owner") if model == "gpt-live-1-codex" else url.params["call_id"] == "rtc_owner"

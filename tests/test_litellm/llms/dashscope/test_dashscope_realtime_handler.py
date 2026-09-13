"""Unit tests for the DashScope realtime handler: URL construction, api base resolution, event forwarding."""

import json

import pytest
import websockets

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.llms.dashscope.realtime.handler import (
    DASHSCOPE_REALTIME_API_BASE,
    REALTIME_WEBSOCKET_PATH,
    DashScopeRealtime,
    resolve_dashscope_realtime_api_base,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    Usage,
)
from tests.llm_translation.realtime.base_realtime_tests import RealTimeWebSocketClient

dashscope_realtime = DashScopeRealtime()

WORKSPACE_API_BASE = "wss://ws-abc123.cn-beijing.maas.aliyuncs.com"


def test_construct_url_uses_dashscope_realtime_path():
    url = dashscope_realtime._construct_url(DASHSCOPE_REALTIME_API_BASE, {"model": "qwen3.5-omni-plus-realtime"})
    assert url == "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3.5-omni-plus-realtime"


def test_construct_url_upgrades_http_and_keeps_workspace_host():
    url = dashscope_realtime._construct_url("https://ws-abc123.cn-beijing.maas.aliyuncs.com", {"model": "m"})
    assert url == f"{WORKSPACE_API_BASE}/api-ws/v1/realtime?model=m"


def test_construct_url_omits_empty_query_params():
    url = dashscope_realtime._construct_url(WORKSPACE_API_BASE, {})
    assert url == f"{WORKSPACE_API_BASE}/api-ws/v1/realtime"


def test_construct_url_replaces_openai_style_path():
    url = dashscope_realtime._construct_url("wss://dashscope.aliyuncs.com/v1/realtime", {"model": "m"})
    assert url == "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=m"


def test_get_auth_headers_sends_bearer_token():
    assert dashscope_realtime.get_auth_headers("sk-test") == {"Authorization": "Bearer sk-test"}


def test_additional_headers_never_send_openai_beta():
    """DashScope needs no OpenAI-Beta opt-in header; its session shape is always the flat one."""
    assert dashscope_realtime._get_additional_headers("sk-test", openai_beta_realtime=True) == {
        "Authorization": "Bearer sk-test"
    }


def test_resolve_api_base_defaults_to_region_host():
    assert resolve_dashscope_realtime_api_base(None, None) == DASHSCOPE_REALTIME_API_BASE


def test_resolve_api_base_maps_chat_base_to_realtime_host():
    chat_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert resolve_dashscope_realtime_api_base(chat_base) == DASHSCOPE_REALTIME_API_BASE


def test_resolve_api_base_prefers_configured_workspace_host():
    chat_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert resolve_dashscope_realtime_api_base(WORKSPACE_API_BASE, chat_base) == WORKSPACE_API_BASE


def test_resolve_api_base_falls_back_to_dynamic_api_base():
    workspace = "wss://ws-abc123.ap-southeast-1.maas.aliyuncs.com"
    assert resolve_dashscope_realtime_api_base(None, workspace) == workspace


@pytest.mark.asyncio
async def test_proxy_forwards_events_to_and_from_a_dashscope_shaped_backend():
    """Drive the whole `_arealtime` path against a local stand-in for DashScope's backend.

    Locks the wire contract a real DashScope session depends on: the endpoint path, the
    bearer-only auth, and that the client's flat beta-style ``session.update`` reaches the
    backend unchanged. litellm remaps that payload into GA's nested shape unless the
    handler declares its backend a beta-protocol one, and DashScope drops the remapped
    modality and audio-format fields.
    """
    handshake: dict = {}
    client_events: list = []
    session_update: dict = {
        "type": "session.update",
        "session": {"modalities": ["text", "audio"], "input_audio_format": "pcm", "voice": "Ethan"},
    }

    async def backend_handler(backend_ws):
        handshake["path"] = backend_ws.request.path
        handshake["headers"] = backend_ws.request.headers
        await backend_ws.send(json.dumps({"type": "session.created", "session": {"id": "sess_1"}}))
        async for raw in backend_ws:
            client_events.append(json.loads(raw))

    async with websockets.serve(backend_handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client_ws = RealTimeWebSocketClient()
        client_ws.queue_client_message(json.dumps(session_update))
        await litellm._arealtime(
            model="dashscope/qwen3.5-omni-plus-realtime",
            websocket=client_ws,
            api_key="sk-dashscope-test",
            api_base=f"http://127.0.0.1:{port}",
            timeout=10,
        )

    assert handshake["path"] == f"{REALTIME_WEBSOCKET_PATH}?model=qwen3.5-omni-plus-realtime"
    assert handshake["headers"]["Authorization"] == "Bearer sk-dashscope-test"
    assert "OpenAI-Beta" not in handshake["headers"]
    assert client_events == [session_update]
    assert [message["type"] for message in client_ws.messages_received] == ["session.created"]


@pytest.mark.parametrize(
    "model",
    [
        "dashscope/qwen3.5-omni-plus-realtime",
        "dashscope/qwen3.5-omni-flash-realtime",
        "dashscope/qwen-audio-3.0-realtime-plus",
        "dashscope/qwen-audio-3.0-realtime-flash",
        "dashscope/qwen3.5-livetranslate-flash-realtime",
    ],
)
def test_cost_map_bills_realtime_audio_tokens_at_the_audio_rate(model, local_model_cost_map):
    """The cost map entries must price audio tokens separately, otherwise a realtime
    session silently bills audio at the far cheaper text rate."""
    model_info = litellm.get_model_info(model=model)
    assert model_info["mode"] == "realtime"
    assert "/v1/realtime" in model_info["supported_endpoints"]

    audio_input_rate = model_info["input_cost_per_audio_token"]
    audio_output_rate = model_info["output_cost_per_audio_token"]
    # A text rate that differs from the audio rate proves the audio rate was really applied.
    # The livetranslate family takes no text input, so it has no text rate at all.
    assert audio_input_rate != model_info.get("input_cost_per_token")
    assert audio_output_rate != model_info.get("output_cost_per_token")

    usage = Usage(
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
        prompt_tokens_details=PromptTokensDetailsWrapper(audio_tokens=1_000, text_tokens=0),
        completion_tokens_details=CompletionTokensDetailsWrapper(audio_tokens=500, text_tokens=0),
    )
    input_cost, output_cost = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="dashscope")

    assert input_cost == pytest.approx(1_000 * audio_input_rate)
    assert output_cost == pytest.approx(500 * audio_output_rate)


_RESERVED_WS_CLOSE_CODES = frozenset({1004, 1005, 1006, 1015})
_VALID_WS_CLOSE_CODES = (frozenset(range(1000, 1016)) - _RESERVED_WS_CLOSE_CODES) | frozenset(range(3000, 5000))


@pytest.mark.asyncio
async def test_rejected_handshake_closes_the_client_with_a_valid_websocket_code():
    """A rejected DashScope handshake must not reuse the upstream HTTP status as the code.

    WebSocket close codes are limited to RFC 6455's ranges. Passing 401 makes the server
    drop the code and the client observes a plain 1000 instead, so the upstream status
    belongs in the close reason rather than the code.
    """

    async def backend_handler(backend_ws):
        raise AssertionError("the upgrade should have been rejected before the handler runs")

    async def reject_upgrade(connection, request):
        return websockets.Response(401, "Unauthorized", websockets.Headers())

    async with websockets.serve(backend_handler, "127.0.0.1", 0, process_request=reject_upgrade) as server:
        port = server.sockets[0].getsockname()[1]
        client_ws = RealTimeWebSocketClient()
        await litellm._arealtime(
            model="dashscope/qwen3.5-omni-plus-realtime",
            websocket=client_ws,
            api_key="sk-rejected",
            api_base=f"http://127.0.0.1:{port}",
            timeout=10,
        )

    assert client_ws.close_code in _VALID_WS_CLOSE_CODES
    assert "401" in (client_ws.close_reason or "")

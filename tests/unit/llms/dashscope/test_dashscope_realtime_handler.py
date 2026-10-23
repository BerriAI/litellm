"""Unit tests for the DashScope realtime handler: URL construction, api base resolution, event forwarding."""

import asyncio
import json
from typing import Final
from unittest.mock import patch

import pytest
import websockets
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

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

dashscope_realtime = DashScopeRealtime()

WORKSPACE_API_BASE = "wss://ws-abc123.cn-beijing.maas.aliyuncs.com"

MODEL: Final = "dashscope/qwen3.5-omni-plus-realtime"


class _ScriptedClientWebSocket:
    """Client side of the proxy socket: replays scripted frames, records what the proxy sent."""

    def __init__(self, frames: list[str], headers: list[tuple[bytes, bytes]] | None = None) -> None:
        self.scope = {"headers": headers or []}
        self._frames = list(frames)
        self._closed = asyncio.Event()
        self.received: list[dict] = []
        self.close_code: int | None = None
        self.close_reason: str | None = None

    async def receive_text(self) -> str:
        if self._frames:
            return self._frames.pop(0)
        await self._closed.wait()
        raise RuntimeError("client disconnected")

    async def send_text(self, data: str) -> None:
        self.received.append(json.loads(data))

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.close_code = code
        self.close_reason = reason
        self._closed.set()


class _FakeBackendSocket:
    """DashScope's end of the socket: emits its opening frames, then records what the proxy forwards."""

    def __init__(self, opening_frames: list[str]) -> None:
        self._opening_frames = list(opening_frames)
        self._forwarded = asyncio.Event()
        self.forwarded: list[dict] = []

    async def recv(self, decode: bool = True) -> str:
        if self._opening_frames:
            return self._opening_frames.pop(0)
        await self._forwarded.wait()
        raise ConnectionClosedOK(Close(1000, ""), None)

    async def send(self, data: str) -> None:
        self.forwarded.append(json.loads(data))
        self._forwarded.set()


class _FakeBackendConnection:
    def __init__(self, socket: _FakeBackendSocket) -> None:
        self._socket = socket

    async def __aenter__(self) -> _FakeBackendSocket:
        return self._socket

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _connect_recorder(socket: _FakeBackendSocket, calls: list[dict]):
    def connect(url: str, **kwargs: object) -> _FakeBackendConnection:
        calls.append({"url": url, **kwargs})
        return _FakeBackendConnection(socket)

    return connect


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
    """Drive the whole `_arealtime` path against a stand-in for DashScope's backend.

    Locks the wire contract a real DashScope session depends on: the endpoint path, the
    bearer-only auth, and that the client's flat beta-style ``session.update`` reaches the
    backend unchanged. litellm remaps that payload into GA's nested shape unless the
    handler declares its backend a beta-protocol one, and DashScope drops the remapped
    modality and audio-format fields.
    """
    session_update: Final = {
        "type": "session.update",
        "session": {"modalities": ["text", "audio"], "input_audio_format": "pcm", "voice": "Ethan"},
    }
    client_ws: Final = _ScriptedClientWebSocket([json.dumps(session_update)])
    backend_ws: Final = _FakeBackendSocket([json.dumps({"type": "session.created", "session": {"id": "sess_1"}})])
    connect_calls: Final[list[dict]] = []

    with patch("websockets.connect", _connect_recorder(backend_ws, connect_calls)):
        await litellm._arealtime(
            model=MODEL,
            websocket=client_ws,
            api_key="sk-dashscope-test",
            api_base="https://dashscope.aliyuncs.com",
            timeout=10,
        )

    assert (
        connect_calls[0]["url"]
        == f"wss://dashscope.aliyuncs.com{REALTIME_WEBSOCKET_PATH}?model=qwen3.5-omni-plus-realtime"
    )
    assert connect_calls[0]["additional_headers"] == {"Authorization": "Bearer sk-dashscope-test"}
    assert backend_ws.forwarded == [session_update]
    assert [message["type"] for message in client_ws.received] == ["session.created"]


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


@pytest.mark.asyncio
async def test_client_beta_header_is_not_forwarded_upstream():
    """DashScope authenticates with the bearer token alone, so the client's ``OpenAI-Beta`` header stops at the proxy.

    The clients that speak the flat beta dialect are the ones that send this header, so without
    the override the header would reach DashScope on every working session.
    """
    session_update: Final = {"type": "session.update", "session": {"modalities": ["text"]}}
    client_ws: Final = _ScriptedClientWebSocket(
        [json.dumps(session_update)],
        headers=[(b"openai-beta", b"realtime=v1")],
    )
    backend_ws: Final = _FakeBackendSocket([json.dumps({"type": "session.created", "session": {"id": "sess_1"}})])
    connect_calls: Final[list[dict]] = []

    with patch("websockets.connect", _connect_recorder(backend_ws, connect_calls)):
        await litellm._arealtime(
            model=MODEL,
            websocket=client_ws,
            api_key="sk-dashscope-test",
            api_base="https://dashscope.aliyuncs.com",
            timeout=10,
        )

    assert connect_calls[0]["additional_headers"] == {"Authorization": "Bearer sk-dashscope-test"}
    assert backend_ws.forwarded == [session_update]


_RESERVED_WS_CLOSE_CODES = frozenset({1004, 1005, 1006, 1015})
_VALID_WS_CLOSE_CODES = (frozenset(range(1000, 1016)) - _RESERVED_WS_CLOSE_CODES) | frozenset(range(3000, 5000))


@pytest.mark.asyncio
async def test_rejected_handshake_closes_the_client_with_a_valid_websocket_code():
    """A rejected DashScope handshake must not reuse the upstream HTTP status as the code.

    WebSocket close codes are limited to RFC 6455's ranges. Passing 401 makes the server
    drop the code and the client observes a plain 1000 instead, so the upstream status
    belongs in the close reason rather than the code.
    """

    client_ws: Final = _ScriptedClientWebSocket([])
    refused: Final = websockets.exceptions.InvalidStatus(
        websockets.http11.Response(401, "Unauthorized", websockets.datastructures.Headers())
    )

    with patch("websockets.connect", side_effect=refused):
        await litellm._arealtime(
            model=MODEL,
            websocket=client_ws,
            api_key="sk-rejected",
            api_base="https://dashscope.aliyuncs.com",
            timeout=10,
        )

    assert client_ws.close_code in _VALID_WS_CLOSE_CODES
    assert "401" in (client_ws.close_reason or "")

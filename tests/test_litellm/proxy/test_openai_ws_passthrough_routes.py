"""OpenAI passthrough WebSocket route: registration, opt-in gating, refusals, and per-frame model checks."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import patch

import pytest
from starlette.routing import WebSocketRoute

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _OPENAI_WS_DISABLED_REFUSAL,
    _openai_websocket_refusal,
    _proxy_frame_model_gate,
    openai_websocket_proxy_route,
    router,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import WebsocketFrameModelGate

ENABLED: Final = MappingProxyType({"enable_openai_websocket_passthrough": True})
DISABLED_SETTINGS: Final = (
    MappingProxyType({}),
    MappingProxyType({"enable_openai_websocket_passthrough": False}),
    MappingProxyType({"enable_openai_websocket_passthrough": "false"}),
    MappingProxyType({"enable_openai_websocket_passthrough": None}),
)
GET_CREDENTIALS: Final = (
    "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
)


def test_openai_websocket_passthrough_routes_registered():
    ws_paths = {route.path for route in router.routes if isinstance(route, WebSocketRoute)}
    assert "/openai/{endpoint:path}" in ws_paths
    assert "/openai_passthrough/{endpoint:path}" in ws_paths


class _FakeWebSocket:
    def __init__(self, path: str, query: str, subprotocols: str | None = None) -> None:
        self.url = SimpleNamespace(path=path, query=query)
        self.headers = {"sec-websocket-protocol": subprotocols} if subprotocols else {}
        self.accepts: list[str | None] = []
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None

    async def accept(self, subprotocol: str | None = None) -> None:
        self.accepts.append(subprotocol)

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)

    def error_message(self) -> str:
        assert len(self.sent) == 1
        frame = json.loads(self.sent[0])
        assert frame["type"] == "error"
        return frame["error"]["message"]


@dataclass(frozen=True, slots=True)
class _RelayCall:
    target: str
    custom_headers: Mapping[str, str]
    forward_headers: bool
    endpoint: str
    accept_websocket: bool
    client_frame_model_gate: WebsocketFrameModelGate


class _FakeRelay:
    def __init__(self) -> None:
        self.calls: list[_RelayCall] = []

    async def __call__(
        self,
        *,
        websocket: _FakeWebSocket,
        target: str,
        custom_headers: dict[str, str],
        user_api_key_dict: UserAPIKeyAuth,
        forward_headers: bool,
        endpoint: str,
        accept_websocket: bool,
        client_frame_model_gate: WebsocketFrameModelGate,
    ) -> None:
        self.calls.append(
            _RelayCall(
                target=target,
                custom_headers=MappingProxyType(dict(custom_headers)),
                forward_headers=forward_headers,
                endpoint=endpoint,
                accept_websocket=accept_websocket,
                client_frame_model_gate=client_frame_model_gate,
            )
        )


class _FakeFrameModelGate:
    def __init__(self, refusal: str | None = None) -> None:
        self.refusal = refusal
        self.checked: list[str] = []

    async def __call__(self, model: str, valid_token: UserAPIKeyAuth, /) -> str | None:
        self.checked.append(model)
        return self.refusal


@dataclass(frozen=True, slots=True)
class _Served:
    relay: _FakeRelay
    frame_model_gate: _FakeFrameModelGate


async def _serve(
    websocket: _FakeWebSocket,
    endpoint: str,
    user_api_key_dict: UserAPIKeyAuth,
    general_settings: Mapping[str, object],
) -> _Served:
    served = _Served(relay=_FakeRelay(), frame_model_gate=_FakeFrameModelGate())
    await openai_websocket_proxy_route(
        websocket=websocket,
        endpoint=endpoint,
        user_api_key_dict=user_api_key_dict,
        general_settings=general_settings,
        relay=served.relay,
        frame_model_gate=served.frame_model_gate,
    )
    return served


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["openai", "openai_passthrough"])
async def test_openai_websocket_forwards_query_and_keeps_provider_auth(prefix, monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    websocket = _FakeWebSocket(f"/{prefix}/v1/realtime", "model=gpt-realtime-2.1")

    with patch(GET_CREDENTIALS, return_value="sk-provider"):
        served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), ENABLED)

    assert served.relay.calls == [
        _RelayCall(
            target="wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1",
            custom_headers=MappingProxyType({"Authorization": "Bearer sk-provider"}),
            forward_headers=False,
            endpoint=f"/{prefix}/v1/realtime",
            accept_websocket=False,
            client_frame_model_gate=served.frame_model_gate,
        )
    ]
    assert websocket.accepts == [None]
    assert websocket.sent == []
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_openai_websocket_accepts_first_client_subprotocol():
    websocket = _FakeWebSocket(
        "/openai/v1/realtime",
        "model=gpt-realtime-2.1",
        subprotocols="realtime, openai-insecure-api-key.sk-abc, openai-beta.realtime-v1",
    )

    with patch(GET_CREDENTIALS, return_value="sk-provider"):
        served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), ENABLED)

    assert websocket.accepts == ["realtime"]
    assert [call.accept_websocket for call in served.relay.calls] == [False]
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_openai_websocket_closes_cleanly_when_provider_credentials_missing():
    websocket = _FakeWebSocket("/openai/v1/realtime", "model=gpt-realtime-2.1")

    with patch(GET_CREDENTIALS, return_value=None):
        served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), ENABLED)

    assert websocket.closed is not None
    assert websocket.closed[0] == 1011
    assert "OPENAI_API_KEY" in websocket.closed[1]
    assert websocket.accepts == []
    assert served.relay.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["openai", "openai_passthrough"])
@pytest.mark.parametrize("general_settings", DISABLED_SETTINGS)
async def test_openai_websocket_refused_unless_explicitly_enabled(prefix, general_settings):
    websocket = _FakeWebSocket(f"/{prefix}/v1/realtime", "model=gpt-realtime-2.1")

    served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), general_settings)

    assert "enable_openai_websocket_passthrough" in websocket.error_message()
    assert websocket.accepts == [None]
    assert websocket.closed == (1008, _OPENAI_WS_DISABLED_REFUSAL.close_reason)
    assert served.relay.calls == []


@pytest.mark.parametrize("general_settings", DISABLED_SETTINGS)
def test_openai_websocket_refusal_is_disabled_for_falsy_settings(general_settings):
    assert _openai_websocket_refusal(general_settings) is _OPENAI_WS_DISABLED_REFUSAL


@pytest.mark.parametrize("value", [True, "true", "True"])
def test_openai_websocket_refusal_is_none_for_truthy_settings(value):
    assert _openai_websocket_refusal(MappingProxyType({"enable_openai_websocket_passthrough": value})) is None


@pytest.mark.asyncio
async def test_openai_websocket_refusal_echoes_requested_subprotocol():
    websocket = _FakeWebSocket(
        "/openai_passthrough/v1/realtime",
        "model=gpt-realtime-2.1",
        subprotocols="realtime, openai-beta.realtime-v1",
    )

    served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), MappingProxyType({}))

    assert websocket.accepts == ["realtime"]
    assert websocket.closed == (1008, _OPENAI_WS_DISABLED_REFUSAL.close_reason)
    assert served.relay.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_models",
    [["gpt-5.4-mini"], ["gpt-realtime-2.1", "gpt-5.4"], []],
)
async def test_openai_websocket_connects_model_restricted_keys(key_models):
    websocket = _FakeWebSocket("/openai/v1/responses", "")
    user_api_key_dict = UserAPIKeyAuth(models=key_models, token="hashed-fake", user_id="user-fake")

    with patch(GET_CREDENTIALS, return_value="sk-provider"):
        served = await _serve(websocket, "v1/responses", user_api_key_dict, ENABLED)

    assert len(served.relay.calls) == 1
    assert served.relay.calls[0].client_frame_model_gate is served.frame_model_gate
    assert websocket.sent == []
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_allows_a_model_the_key_owns():
    gate: Final = _proxy_frame_model_gate()
    token: Final = UserAPIKeyAuth(models=["gpt-realtime-2.1", "gpt-5.4-mini"])

    with patch("litellm.proxy.proxy_server.prisma_client", None):  # test-quality-ok: the gate needs no database
        assert await gate("gpt-realtime-2.1", token) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_model", ["gpt-5.4", "gpt-realtime-2.1-mini"])
async def test_proxy_frame_model_gate_refuses_a_model_the_key_lacks(requested_model):
    gate: Final = _proxy_frame_model_gate()
    token: Final = UserAPIKeyAuth(models=["gpt-realtime-2.1", "gpt-5.4-mini"])

    with patch("litellm.proxy.proxy_server.prisma_client", None):  # test-quality-ok: the gate needs no database
        refusal = await gate(requested_model, token)

    assert refusal is not None
    assert requested_model in refusal


@pytest.mark.asyncio
@pytest.mark.parametrize("unrestricted_models", [[], ["all-proxy-models"], ["*"]])
async def test_proxy_frame_model_gate_allows_every_model_for_unrestricted_keys(unrestricted_models):
    gate: Final = _proxy_frame_model_gate()
    token: Final = UserAPIKeyAuth(models=unrestricted_models)

    with patch("litellm.proxy.proxy_server.prisma_client", None):  # test-quality-ok: the gate needs no database
        assert await gate("gpt-realtime-2.1", token) is None

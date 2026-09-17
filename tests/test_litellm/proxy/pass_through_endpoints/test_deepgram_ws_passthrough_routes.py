"""Deepgram ``/v1/listen`` passthrough WebSocket route: registration, auth, credential injection, target URL."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocketDisconnect

from litellm.proxy._lazy_features import LAZY_FEATURES
from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _websocket_relay,
    deepgram_listen_websocket_route,
    router,
)

GET_CREDENTIALS: Final = (
    "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
)
USER_API_KEY_AUTH: Final = "litellm.proxy.auth.user_api_key_auth.user_api_key_auth"
LISTEN_PATHS: Final = ("/deepgram/v1/listen", "/deepgram/listen")


class _FakeWebSocket:
    def __init__(self, path: str, query: str) -> None:
        self.url = SimpleNamespace(path=path, query=query)
        self.headers = {"authorization": "Bearer sk-litellm-virtual", "x-api-key": "sk-caller-secret"}
        self.accepts: list[str | None] = []
        self.closed: tuple[int, str] | None = None

    async def accept(self, subprotocol: str | None = None) -> None:
        self.accepts.append(subprotocol)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


@dataclass(frozen=True, slots=True)
class _RelayCall:
    target: str
    custom_headers: Mapping[str, str]
    user_api_key_dict: UserAPIKeyAuth
    forward_headers: bool
    endpoint: str
    accept_websocket: bool


class _FakeRelay:
    def __init__(self) -> None:
        self.calls: list[_RelayCall] = []

    async def __call__(
        self,
        *,
        websocket: object,
        target: str,
        custom_headers: dict[str, str],
        user_api_key_dict: UserAPIKeyAuth,
        forward_headers: bool,
        endpoint: str,
        accept_websocket: bool,
    ) -> None:
        self.calls.append(
            _RelayCall(
                target=target,
                custom_headers=MappingProxyType(dict(custom_headers)),
                user_api_key_dict=user_api_key_dict,
                forward_headers=forward_headers,
                endpoint=endpoint,
                accept_websocket=accept_websocket,
            )
        )


async def _serve(websocket: _FakeWebSocket, user_api_key_dict: UserAPIKeyAuth | None = None) -> _FakeRelay:
    relay = _FakeRelay()
    await deepgram_listen_websocket_route(
        websocket=websocket,
        user_api_key_dict=user_api_key_dict or UserAPIKeyAuth(),
        relay=relay,
    )
    return relay


def test_deepgram_listen_websocket_routes_registered():
    ws_paths = {route.path for route in router.routes if isinstance(route, WebSocketRoute)}
    assert set(LISTEN_PATHS) <= ws_paths


@pytest.mark.parametrize("path", LISTEN_PATHS)
def test_deepgram_listen_is_a_lazily_loaded_mapped_pass_through_route(path):
    """The route must be reachable before the passthrough module is imported and must be authed and
    billed as a mapped pass-through route like the other provider prefixes."""
    feature = next(feature for feature in LAZY_FEATURES if feature.name == "llm_passthrough")
    assert feature.matches(path)
    assert any(path.startswith(prefix) for prefix in LiteLLMRoutes.mapped_pass_through_routes.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", LISTEN_PATHS)
async def test_deepgram_listen_forwards_query_and_injects_only_provider_auth(path, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    websocket = _FakeWebSocket(path, "encoding=linear16&sample_rate=16000&keywords=hi%3A2&keywords=there")
    caller = UserAPIKeyAuth(api_key="sk-litellm-virtual", team_id="team-stt")

    with patch(GET_CREDENTIALS, return_value="dg-provider-key") as get_credentials:
        relay = await _serve(websocket, caller)

    assert get_credentials.call_args.kwargs == {"custom_llm_provider": "deepgram", "region_name": None}
    assert relay.calls == [
        _RelayCall(
            target=(
                "wss://api.deepgram.com/v1/listen"
                "?encoding=linear16&sample_rate=16000&keywords=hi%3A2&keywords=there&model=nova-3"
            ),
            custom_headers=MappingProxyType({"Authorization": "Token dg-provider-key"}),
            user_api_key_dict=caller,
            forward_headers=False,
            endpoint=path,
            accept_websocket=False,
        )
    ]
    assert websocket.accepts == [None]
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_deepgram_listen_keeps_caller_chosen_model(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    websocket = _FakeWebSocket("/deepgram/v1/listen", "model=nova-2&language=en")

    with patch(GET_CREDENTIALS, return_value="dg-provider-key"):
        relay = await _serve(websocket)

    assert [call.target for call in relay.calls] == ["wss://api.deepgram.com/v1/listen?model=nova-2&language=en"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_target"),
    [
        ("", "wss://api.deepgram.com/v1/listen?model=nova-3"),
        ("model=", "wss://api.deepgram.com/v1/listen?model=nova-3"),
        ("model=&language=en", "wss://api.deepgram.com/v1/listen?language=en&model=nova-3"),
    ],
)
async def test_deepgram_listen_defaults_to_nova_3_when_no_model_is_named(query, expected_target, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    websocket = _FakeWebSocket("/deepgram/listen", query)

    with patch(GET_CREDENTIALS, return_value="dg-provider-key"):
        relay = await _serve(websocket)

    assert [call.target for call in relay.calls] == [expected_target]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_base", "expected_target"),
    [
        ("https://api.eu.deepgram.com/v1/", "wss://api.eu.deepgram.com/v1/listen?model=nova-3"),
        ("http://localhost:8080/v1", "ws://localhost:8080/v1/listen?model=nova-3"),
        ("wss://deepgram.internal.example/v1", "wss://deepgram.internal.example/v1/listen?model=nova-3"),
    ],
)
async def test_deepgram_listen_honours_server_configured_api_base(api_base, expected_target, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_BASE", api_base)
    websocket = _FakeWebSocket("/deepgram/v1/listen", "")

    with patch(GET_CREDENTIALS, return_value="dg-provider-key"):
        relay = await _serve(websocket)

    assert [call.target for call in relay.calls] == [expected_target]


@pytest.mark.asyncio
async def test_deepgram_listen_ignores_caller_supplied_api_base(monkeypatch):
    """V1: the server-configured Deepgram key must only ever go to the server-configured host."""
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    websocket = _FakeWebSocket("/deepgram/v1/listen", "api_base=wss%3A%2F%2Fattacker.example%2Fv1&model=nova-3")

    with patch(GET_CREDENTIALS, return_value="dg-provider-key"):
        relay = await _serve(websocket)

    assert [call.target for call in relay.calls] == [
        "wss://api.deepgram.com/v1/listen?api_base=wss%3A%2F%2Fattacker.example%2Fv1&model=nova-3"
    ]


@pytest.mark.asyncio
async def test_deepgram_listen_closes_cleanly_when_provider_credentials_missing():
    websocket = _FakeWebSocket("/deepgram/v1/listen", "model=nova-3")

    with patch(GET_CREDENTIALS, return_value=None):
        relay = await _serve(websocket)

    assert websocket.closed is not None
    assert websocket.closed[0] == 1011
    assert "DEEPGRAM_API_KEY" in websocket.closed[1]
    assert websocket.accepts == []
    assert relay.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        pytest.param("model=nova-3&callback=https%3A%2F%2Fsink.example%2Fdg", id="http callback"),
        pytest.param("callback=wss%3A%2F%2Fsink.example&callback_method=put&model=nova-3", id="ws callback"),
    ],
)
async def test_deepgram_listen_rejects_callback_delivery_that_would_go_unbilled(query, monkeypatch):
    """With ``callback`` set, Deepgram sends every Results and Metadata frame to the caller's URL and only a
    request id down this socket, so the proxy would meter zero seconds of audio; refuse before contacting Deepgram."""
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    websocket = _FakeWebSocket("/deepgram/v1/listen", query)

    with patch(GET_CREDENTIALS, return_value="dg-provider-key"):
        relay = await _serve(websocket)

    assert relay.calls == []
    assert websocket.closed is not None
    assert websocket.closed[0] == 1008
    assert "callback" in websocket.closed[1]
    assert "dg-provider-key" not in websocket.closed[1]


def _app_with_relay(relay: _FakeRelay) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_websocket_relay] = lambda: relay
    return app


def test_deepgram_listen_rejects_connections_without_a_litellm_key():
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))

    with patch(GET_CREDENTIALS, return_value="dg-provider-key") as get_credentials:
        with pytest.raises(WebSocketDisconnect) as disconnect:
            with client.websocket_connect("/deepgram/v1/listen?model=nova-3"):
                pass

    assert disconnect.value.code == 1008
    assert relay.calls == []
    get_credentials.assert_not_called()


def test_deepgram_listen_callback_rejection_reaches_the_client_as_a_policy_close(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))

    with (
        patch(GET_CREDENTIALS, return_value="dg-provider-key"),
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed"))),
    ):
        with pytest.raises(WebSocketDisconnect) as disconnect:
            with client.websocket_connect(
                "/deepgram/v1/listen?model=nova-3&callback=https%3A%2F%2Fsink.example%2Fdg",
                headers={"Authorization": "Bearer sk-litellm-virtual"},
            ) as connection:
                connection.receive_text()

    assert disconnect.value.code == 1008
    assert "callback" in disconnect.value.reason
    assert relay.calls == []


def test_deepgram_listen_authenticates_the_litellm_key_and_relays_to_deepgram(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))
    caller = UserAPIKeyAuth(api_key="hashed-sk-litellm", team_id="team-stt")

    with (
        patch(GET_CREDENTIALS, return_value="dg-provider-key"),
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=caller)) as auth,
    ):
        with client.websocket_connect(
            "/deepgram/v1/listen?model=nova-3&punctuate=true",
            headers={"Authorization": "Bearer sk-litellm-virtual"},
        ):
            pass

    assert auth.await_args.kwargs["api_key"] == "Bearer sk-litellm-virtual"
    assert relay.calls == [
        _RelayCall(
            target="wss://api.deepgram.com/v1/listen?model=nova-3&punctuate=true",
            custom_headers=MappingProxyType({"Authorization": "Token dg-provider-key"}),
            user_api_key_dict=caller,
            forward_headers=False,
            endpoint="/deepgram/v1/listen",
            accept_websocket=False,
        )
    ]


def test_deepgram_listen_echoes_the_browser_subprotocol_that_carries_the_litellm_key(monkeypatch):
    """Browsers cannot set headers, so they send the key as a subprotocol and abort the handshake unless the
    server echoes that subprotocol back; the key itself must still stay off the upstream connection."""
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))

    with (
        patch(GET_CREDENTIALS, return_value="dg-provider-key"),
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed"))),
    ):
        with client.websocket_connect(
            "/deepgram/v1/listen?model=nova-3",
            subprotocols=["openai-insecure-api-key.sk-litellm-virtual"],
        ) as connection:
            assert connection.accepted_subprotocol == "openai-insecure-api-key.sk-litellm-virtual"

    assert [call.custom_headers for call in relay.calls] == [
        MappingProxyType({"Authorization": "Token dg-provider-key"})
    ]
    assert [call.forward_headers for call in relay.calls] == [False]

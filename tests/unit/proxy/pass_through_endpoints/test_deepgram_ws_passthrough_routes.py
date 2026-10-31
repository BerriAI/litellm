"""Deepgram ``/v1/listen`` and ``/v1/speak`` passthrough WebSocket routes: registration, auth, credential injection,
target URL."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute
from starlette.types import Message
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import WEBSOCKET_CLOSE_REASON_MAX_BYTES
from litellm.litellm_core_utils.get_model_cost_map import adopt_model_cost_map
from litellm.llms.deepgram.common_utils import DeepgramSpeakMeter
from litellm.proxy._lazy_features import LAZY_FEATURES
from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import cache_key_object
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _websocket_relay,
    deepgram_listen_websocket_route,
    deepgram_speak_websocket_route,
    router,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import websocket_passthrough_request
from litellm.proxy.utils import InternalUsageCache, ProxyLogging, hash_token
from litellm.types.passthrough_endpoints.pass_through_endpoints import WebSocketSessionMeter

GET_CREDENTIALS: Final = (
    "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
)
USER_API_KEY_AUTH: Final = "litellm.proxy.auth.user_api_key_auth.user_api_key_auth"
LISTEN_PATHS: Final = ("/deepgram/v1/listen", "/deepgram/listen")
SPEAK_PATHS: Final = ("/deepgram/v1/speak", "/deepgram/speak")
NOVA_2_STREAMING_KEY: Final = "deepgram/streaming/nova-2"

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")


def _price_nova_2_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator-supplied streaming row: the bundled map prices only nova-3 for streaming."""
    monkeypatch.setitem(litellm.model_cost, NOVA_2_STREAMING_KEY, dict(litellm.model_cost["deepgram/streaming/nova-3"]))


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
    meter: WebSocketSessionMeter | None = None


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
        meter: WebSocketSessionMeter | None = None,
        model: str | None = None,
    ) -> None:
        self.calls.append(
            _RelayCall(
                target=target,
                custom_headers=MappingProxyType(dict(custom_headers)),
                user_api_key_dict=user_api_key_dict,
                forward_headers=forward_headers,
                endpoint=endpoint,
                accept_websocket=accept_websocket,
                meter=meter,
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
    _price_nova_2_streaming(monkeypatch)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "missing_key"),
    [
        pytest.param("model=nova-2", "deepgram/streaming/nova-2", id="model with only a pre-recorded price"),
        pytest.param("model=nova-99", "deepgram/streaming/nova-99", id="model unknown to the registry"),
        pytest.param(
            "model=nova-3&language=multi",
            "deepgram/streaming/nova-3-multilingual",
            id="multilingual session without its own price",
        ),
    ],
)
async def test_deepgram_listen_refuses_sessions_it_cannot_price(query, missing_key, monkeypatch):
    """A session with no streaming price would be logged at zero (or at the pre-recorded rate), letting a caller run
    up unmetered spend, so the proxy closes it before Deepgram is contacted and names the registry row to add."""
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.delitem(litellm.model_cost, missing_key, raising=False)
    assert "deepgram/nova-2" in litellm.model_cost
    websocket = _FakeWebSocket("/deepgram/v1/listen", query)

    with patch(GET_CREDENTIALS, return_value="dg-provider-key"):
        relay = await _serve(websocket)

    assert relay.calls == []
    assert websocket.closed is not None
    assert websocket.closed[0] == 1008
    assert missing_key in websocket.closed[1]
    assert "dg-provider-key" not in websocket.closed[1]


@pytest.mark.asyncio
async def test_deepgram_listen_refusal_for_a_long_unpriced_model_fits_a_close_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-provider-key")
    websocket: Final = _FakeWebSocket("/deepgram/v1/listen", "model=" + "%C3%A9" * 100)

    relay: Final = await _serve(websocket)

    assert relay.calls == []
    assert websocket.closed == (1008, "No streaming price for 'deepgram/streaming/" + "é" * 40)


@pytest.mark.asyncio
async def test_deepgram_listen_relays_once_the_operator_prices_the_model(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    websocket = _FakeWebSocket("/deepgram/v1/listen", "model=nova-2")
    with patch(GET_CREDENTIALS, return_value="dg-provider-key"):
        assert (await _serve(websocket)).calls == []

    _price_nova_2_streaming(monkeypatch)
    priced_websocket = _FakeWebSocket("/deepgram/v1/listen", "model=nova-2")
    with patch(GET_CREDENTIALS, return_value="dg-provider-key"):
        relay = await _serve(priced_websocket)

    assert [call.target for call in relay.calls] == ["wss://api.deepgram.com/v1/listen?model=nova-2"]
    assert priced_websocket.closed is None


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


async def _cache_restricted_key(virtual_key: str, models: list[str]) -> DualCache:
    cache = DualCache()
    await cache_key_object(
        hashed_token=hash_token(virtual_key),
        user_api_key_obj=UserAPIKeyAuth(token=hash_token(virtual_key), models=models),
        user_api_key_cache=cache,
        proxy_logging_obj=None,
    )
    return cache


@pytest.mark.parametrize(
    ("query", "expect_relay"),
    [
        pytest.param("model=nova-2", True, id="allowed model named"),
        pytest.param("model=nova-3", False, id="denied model named"),
        pytest.param("", False, id="model omitted, default denied"),
        pytest.param("model=&language=en", False, id="model blank, default denied"),
    ],
)
def test_deepgram_listen_authorizes_the_model_it_will_actually_send_upstream(query, expect_relay, monkeypatch):
    """A key allowed only ``nova-2`` must not reach ``nova-3`` by leaving ``model`` out and letting the proxy fill
    in its default: the real key auth path must see the same model the upstream target will carry."""
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "max_budget", 0.0)
    _price_nova_2_streaming(monkeypatch)
    cache = asyncio.run(_cache_restricted_key("sk-only-nova-2", ["nova-2"]))
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))

    with (
        patch(GET_CREDENTIALS, return_value="dg-provider-key"),
        patch.multiple(  # test-quality-ok: the real key auth path reads these proxy_server globals and has no injection seam
            "litellm.proxy.proxy_server",
            master_key="sk-master",
            prisma_client=MagicMock(),
            user_api_key_cache=cache,
            llm_model_list=None,
            llm_router=None,
        ),
    ):
        if expect_relay:
            with client.websocket_connect(
                f"/deepgram/v1/listen?{query}", headers={"Authorization": "Bearer sk-only-nova-2"}
            ):
                pass
            assert [call.target for call in relay.calls] == [f"wss://api.deepgram.com/v1/listen?{query}"]
            return
        with pytest.raises(WebSocketDisconnect) as disconnect:
            with client.websocket_connect(
                f"/deepgram/v1/listen?{query}", headers={"Authorization": "Bearer sk-only-nova-2"}
            ):
                pass

    assert disconnect.value.code == 1008
    assert relay.calls == []


def test_deepgram_listen_strips_a_second_model_that_would_outrank_the_authorized_one(monkeypatch):
    """Deepgram honours the last repeated ``model``; auth and pricing read the first. A key allowed only ``nova-2``
    must not smuggle ``nova-3`` past authorization behind an authorized first value."""
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "max_budget", 0.0)
    _price_nova_2_streaming(monkeypatch)
    cache = asyncio.run(_cache_restricted_key("sk-only-nova-2", ["nova-2"]))
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))

    with (
        patch(GET_CREDENTIALS, return_value="dg-provider-key"),
        patch.multiple(  # test-quality-ok: the real key auth path reads these proxy_server globals and has no injection seam
            "litellm.proxy.proxy_server",
            master_key="sk-master",
            prisma_client=MagicMock(),
            user_api_key_cache=cache,
            llm_model_list=None,
            llm_router=None,
        ),
    ):
        with client.websocket_connect(
            "/deepgram/v1/listen?model=nova-2&language=en&model=nova-3&language=multi",
            headers={"Authorization": "Bearer sk-only-nova-2"},
        ):
            pass

    assert [call.target for call in relay.calls] == ["wss://api.deepgram.com/v1/listen?model=nova-2&language=en"]


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


async def _serve_speak(websocket: _FakeWebSocket, user_api_key_dict: UserAPIKeyAuth | None = None) -> _FakeRelay:
    relay: Final = _FakeRelay()
    await deepgram_speak_websocket_route(
        websocket=websocket,
        user_api_key_dict=user_api_key_dict or UserAPIKeyAuth(),
        relay=relay,
    )
    return relay


@pytest.mark.asyncio
@pytest.mark.parametrize("path", SPEAK_PATHS)
async def test_deepgram_speak_relays_with_provider_auth_and_meters_the_session(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-provider-key")
    websocket: Final = _FakeWebSocket(path, "encoding=linear16&sample_rate=24000")
    caller: Final = UserAPIKeyAuth(api_key="sk-litellm-virtual", team_id="team-tts")
    target: Final = "wss://api.deepgram.com/v1/speak?encoding=linear16&sample_rate=24000&model=aura-asteria-en"

    relay: Final = await _serve_speak(websocket, caller)

    assert relay.calls == [
        _RelayCall(
            target=target,
            custom_headers=MappingProxyType({"Authorization": "Token dg-provider-key"}),
            user_api_key_dict=caller,
            forward_headers=False,
            endpoint=path,
            accept_websocket=False,
            meter=DeepgramSpeakMeter(target, litellm.model_cost["deepgram/aura"]["input_cost_per_character"]),
        )
    ]
    assert websocket.accepts == [None]
    assert websocket.closed is None


def _client_that_hangs_up(path: str, query: str) -> WebSocket:
    handshake: Final = iter(({"type": "websocket.connect"},))

    async def receive() -> Message:
        return next(handshake, {"type": "websocket.disconnect", "code": 1000})

    async def send(message: Message) -> None:
        return None

    return WebSocket({"type": "websocket", "path": path, "query_string": query.encode(), "headers": []}, receive, send)


class _UpstreamThatClosesAtOnce:
    async def recv(self, decode: bool | None = None) -> str:
        raise ConnectionClosedOK(rcvd=Close(1000, ""), sent=Close(1000, ""), rcvd_then_sent=True)

    async def send(self, message: str | bytes) -> None:
        return None

    async def close(self, code: int = 1000, reason: str = "") -> None:
        return None


class _UpstreamConnect:
    async def __aenter__(self) -> _UpstreamThatClosesAtOnce:
        return _UpstreamThatClosesAtOnce()

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


def _upstream_dials_under_the_real_rate_limiter(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from litellm.proxy import proxy_server

    fixed_now: Final = datetime.fromisoformat("2026-10-31T12:00:00+00:00")
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(DualCache(in_memory_cache=InMemoryCache(clock=fixed_now.timestamp))),
        time_provider=lambda: fixed_now,
        model_group_resolver=lambda _model: None,
    )
    monkeypatch.setattr(litellm, "callbacks", [limiter])
    monkeypatch.setattr(
        proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=proxy_server.user_api_key_cache)
    )
    dialed: Final[list[str]] = []

    def dial(target: str, additional_headers: Mapping[str, str]) -> _UpstreamConnect:
        dialed.append(target)
        return _UpstreamConnect()

    monkeypatch.setattr("litellm.proxy.pass_through_endpoints.pass_through_endpoints.connect", dial)
    return dialed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "path", "limited_model", "first_query", "second_query", "first_target"),
    [
        pytest.param(
            deepgram_listen_websocket_route,
            "/deepgram/v1/listen",
            "nova-3",
            "model=nova-3",
            "model=nova-3&model=nova-2",
            "wss://api.deepgram.com/v1/listen?model=nova-3",
            id="listen repeated model",
        ),
        pytest.param(
            deepgram_listen_websocket_route,
            "/deepgram/v1/listen",
            "nova-3",
            "",
            "",
            "wss://api.deepgram.com/v1/listen?model=nova-3",
            id="listen default model",
        ),
        pytest.param(
            deepgram_speak_websocket_route,
            "/deepgram/v1/speak",
            "aura-2-thalia-en",
            "model=aura-2-thalia-en",
            "model=aura-2-thalia-en&model=aura-asteria-en",
            "wss://api.deepgram.com/v1/speak?model=aura-2-thalia-en",
            id="speak repeated model",
        ),
        pytest.param(
            deepgram_speak_websocket_route,
            "/deepgram/v1/speak",
            "aura-asteria-en",
            "",
            "",
            "wss://api.deepgram.com/v1/speak?model=aura-asteria-en",
            id="speak default model",
        ),
    ],
)
async def test_deepgram_routes_refuse_a_second_session_over_the_keys_model_rpm_limit(
    route: Callable[..., Awaitable[None]],
    path: str,
    limited_model: str,
    first_query: str,
    second_query: str,
    first_target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-provider-key")
    dialed: Final = _upstream_dials_under_the_real_rate_limiter(monkeypatch)
    caller: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-deepgram-model-rpm"), metadata={"model_rpm_limit": {limited_model: 1}}
    )

    async def session(query: str) -> None:
        await route(
            websocket=_client_that_hangs_up(path, query),
            user_api_key_dict=caller,
            relay=websocket_passthrough_request,
        )

    await asyncio.create_task(session(first_query))
    with pytest.raises(HTTPException) as refused:
        await asyncio.create_task(session(second_query))

    assert (refused.value.status_code, refused.value.detail, dialed) == (
        429,
        f"Rate limit exceeded for model_per_key: {caller.api_key}:{limited_model}. Limit type: requests. "
        "Current limit: 1, Remaining: 0. Limit resets at: 2026-10-31 12:01:00 UTC",
        [first_target],
    )


@pytest.mark.asyncio
async def test_deepgram_speak_bills_at_the_connect_time_rate_after_a_cost_map_reload_drops_the_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-provider-key")
    rate: Final = litellm.model_cost["deepgram/aura-2"]["input_cost_per_character"]

    relay: Final = await _serve_speak(_FakeWebSocket("/deepgram/v1/speak", "model=aura-2-thalia-en"))
    meters: Final = [call.meter for call in relay.calls if isinstance(call.meter, DeepgramSpeakMeter)]
    adopt_model_cost_map({key: row for key, row in litellm.model_cost.items() if key != "deepgram/aura-2"})

    assert [meter.price(47).response_cost for meter in meters] == [47 * rate]


@pytest.mark.asyncio
async def test_deepgram_speak_honours_server_configured_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_BASE", "http://localhost:8080/v1")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-provider-key")

    relay: Final = await _serve_speak(_FakeWebSocket("/deepgram/v1/speak", "model=aura-2-thalia-en"))

    assert [call.target for call in relay.calls] == ["ws://localhost:8080/v1/speak?model=aura-2-thalia-en"]


@pytest.mark.asyncio
async def test_deepgram_speak_closes_cleanly_when_provider_credentials_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    websocket: Final = _FakeWebSocket("/deepgram/v1/speak", "model=aura-2-thalia-en")

    relay: Final = await _serve_speak(websocket)

    assert relay.calls == []
    assert websocket.accepts == []
    assert websocket.closed == (
        1011,
        "Required 'DEEPGRAM_API_KEY' in environment to make pass-through calls to Deepgram.",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "missing_key"),
    [
        pytest.param("model=aura-99-x-en", "deepgram/aura-99", id="generation without a row"),
        pytest.param("model=nova-3", "deepgram/nova-3", id="listen model has no per-character rate"),
    ],
)
async def test_deepgram_speak_refuses_voices_it_cannot_price(
    query: str, missing_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-provider-key")
    websocket: Final = _FakeWebSocket("/deepgram/v1/speak", query)

    relay: Final = await _serve_speak(websocket)

    assert relay.calls == []
    assert websocket.closed == (
        1008,
        f"No streaming price for '{missing_key}': add it to the model cost map to enable it",
    )


@pytest.mark.asyncio
async def test_deepgram_speak_refusal_for_a_long_unpriced_model_fits_a_close_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-provider-key")
    websocket: Final = _FakeWebSocket("/deepgram/v1/speak", f"model={'x' * 200}")

    relay: Final = await _serve_speak(websocket)

    assert relay.calls == []
    assert websocket.closed is not None
    assert (websocket.closed[0], len(websocket.closed[1].encode())) == (1008, WEBSOCKET_CLOSE_REASON_MAX_BYTES)


@pytest.mark.parametrize("path", SPEAK_PATHS)
@pytest.mark.parametrize(
    "query",
    [
        pytest.param("model=aura-2-thalia-en", id="allowed voice"),
        pytest.param("model=aura-2-thalia-en&model=aura-asteria-en", id="second voice stripped"),
    ],
)
def test_deepgram_speak_relays_only_the_voice_the_key_was_checked_against(
    path: str, query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key allowed only one voice must not reach another by repeating ``model``: the real key auth path must see
    the voice the upstream target will carry"""
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-provider-key")
    monkeypatch.setattr(litellm, "max_budget", 0.0)
    cache: Final = asyncio.run(_cache_restricted_key("sk-only-thalia", ["aura-2-thalia-en"]))
    relay: Final = _FakeRelay()
    client: Final = TestClient(_app_with_relay(relay))

    with patch.multiple(  # test-quality-ok: the real key auth path reads these proxy_server globals and has no injection seam
        "litellm.proxy.proxy_server",
        master_key="sk-master",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
        llm_model_list=None,
        llm_router=None,
    ):
        with client.websocket_connect(f"{path}?{query}", headers={"Authorization": "Bearer sk-only-thalia"}):
            pass

    assert [(call.target, call.endpoint) for call in relay.calls] == [
        ("wss://api.deepgram.com/v1/speak?model=aura-2-thalia-en", path)
    ]


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("model=aura-asteria-en", id="denied voice named"),
        pytest.param("", id="voice omitted, default denied"),
    ],
)
def test_deepgram_speak_refuses_a_voice_the_key_is_not_allowed(query: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_BASE", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-provider-key")
    monkeypatch.setattr(litellm, "max_budget", 0.0)
    cache: Final = asyncio.run(_cache_restricted_key("sk-only-thalia", ["aura-2-thalia-en"]))
    relay: Final = _FakeRelay()
    client: Final = TestClient(_app_with_relay(relay))

    with patch.multiple(  # test-quality-ok: the real key auth path reads these proxy_server globals and has no injection seam
        "litellm.proxy.proxy_server",
        master_key="sk-master",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
        llm_model_list=None,
        llm_router=None,
    ):
        with pytest.raises(WebSocketDisconnect) as disconnect:
            with client.websocket_connect(
                f"/deepgram/v1/speak?{query}", headers={"Authorization": "Bearer sk-only-thalia"}
            ):
                pass

    assert disconnect.value.code == 1008
    assert relay.calls == []

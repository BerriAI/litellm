import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocketDisconnect

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._lazy_features import LAZY_FEATURES
from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import cache_key_object
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _websocket_relay,
    elevenlabs_tts_multi_stream_input_websocket_route,
    elevenlabs_tts_stream_input_websocket_route,
    router,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import WebsocketFramePolicy
from litellm.proxy.utils import hash_token
from litellm.types.passthrough_endpoints.pass_through_endpoints import ForwardedClientText, RefusedClientText

GET_CREDENTIALS: Final = (
    "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
)
USER_API_KEY_AUTH: Final = "litellm.proxy.auth.user_api_key_auth.user_api_key_auth"
STREAM_PATH: Final = "/elevenlabs/v1/text-to-speech/voice1/stream-input"
MULTI_STREAM_PATH: Final = "/elevenlabs/v1/text-to-speech/voice1/multi-stream-input"
UPSTREAM: Final = "wss://api.elevenlabs.io/v1/text-to-speech/voice1"
PROVIDER_HEADERS: Final = MappingProxyType({"xi-api-key": "el-provider-key"})

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")


class _FakeWebSocket:
    def __init__(self, path: str, query: str) -> None:
        self.url = SimpleNamespace(path=path, query=query)
        self.headers = {"xi-api-key": "sk-litellm-virtual"}
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
    frame_policy: WebsocketFramePolicy


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
        frame_policy: WebsocketFramePolicy,
    ) -> None:
        self.calls.append(
            _RelayCall(
                target=target,
                custom_headers=MappingProxyType(dict(custom_headers)),
                user_api_key_dict=user_api_key_dict,
                forward_headers=forward_headers,
                endpoint=endpoint,
                accept_websocket=accept_websocket,
                frame_policy=frame_policy,
            )
        )


async def _serve(
    websocket: _FakeWebSocket, voice_id: str = "voice1", user_api_key_dict: UserAPIKeyAuth | None = None
) -> _FakeRelay:
    relay = _FakeRelay()
    await elevenlabs_tts_stream_input_websocket_route(
        websocket=websocket,
        voice_id=voice_id,
        user_api_key_dict=user_api_key_dict or UserAPIKeyAuth(),
        relay=relay,
    )
    return relay


def test_elevenlabs_tts_websocket_routes_registered():
    ws_paths = {route.path for route in router.routes if isinstance(route, WebSocketRoute)}
    assert {
        "/elevenlabs/v1/text-to-speech/{voice_id}/stream-input",
        "/elevenlabs/v1/text-to-speech/{voice_id}/multi-stream-input",
    } <= ws_paths


@pytest.mark.parametrize("path", [STREAM_PATH, MULTI_STREAM_PATH])
def test_elevenlabs_tts_is_a_lazily_loaded_mapped_pass_through_route(path):
    feature = next(feature for feature in LAZY_FEATURES if feature.name == "llm_passthrough")
    assert feature.matches(path)
    assert any(path.startswith(prefix) for prefix in LiteLLMRoutes.mapped_pass_through_routes.value)


@pytest.mark.asyncio
async def test_stream_input_relays_to_elevenlabs_with_only_the_proxy_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    websocket = _FakeWebSocket(STREAM_PATH, "model_id=eleven_flash_v2_5&output_format=pcm_16000")
    caller = UserAPIKeyAuth(api_key="sk-litellm-virtual", team_id="team-tts")

    with patch(GET_CREDENTIALS, return_value="el-provider-key") as get_credentials:
        relay = await _serve(websocket, user_api_key_dict=caller)

    assert get_credentials.call_args.kwargs == {"custom_llm_provider": "elevenlabs", "region_name": None}
    assert len(relay.calls) == 1
    call = relay.calls[0]
    assert call.target == f"{UPSTREAM}/stream-input?model_id=eleven_flash_v2_5&output_format=pcm_16000"
    assert call.custom_headers == PROVIDER_HEADERS
    assert call.user_api_key_dict is caller
    assert call.forward_headers is False
    assert call.endpoint == STREAM_PATH
    assert call.accept_websocket is False
    decision = call.frame_policy.client_text('{"text": "Hi ", "xi_api_key": "sk-litellm-virtual", "flush": true}')
    assert isinstance(decision, ForwardedClientText)
    assert json.loads(decision.text) == {"text": "Hi ", "flush": True}
    assert decision.recorded == {"text": "Hi "}
    assert isinstance(call.frame_policy.client_text('{"text": "Hi ", "x": "\\ud800"}'), RefusedClientText)
    assert call.frame_policy.recorded_upstream_frame({"audio": "AAAA", "isFinal": None}) == {"isFinal": None}
    assert websocket.accepts == [None]
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_multi_stream_input_relays_to_the_multi_stream_endpoint(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    websocket = _FakeWebSocket(MULTI_STREAM_PATH, "model_id=eleven_flash_v2_5")
    relay = _FakeRelay()

    with patch(GET_CREDENTIALS, return_value="el-provider-key"):
        await elevenlabs_tts_multi_stream_input_websocket_route(
            websocket=websocket, voice_id="voice1", user_api_key_dict=UserAPIKeyAuth(), relay=relay
        )

    assert [(call.target, call.endpoint) for call in relay.calls] == [
        (f"{UPSTREAM}/multi-stream-input?model_id=eleven_flash_v2_5", MULTI_STREAM_PATH)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_query"),
    [
        pytest.param("", "model_id=eleven_multilingual_v2", id="model omitted"),
        pytest.param(
            "model_id=&output_format=mp3_44100_128",
            "output_format=mp3_44100_128&model_id=eleven_multilingual_v2",
            id="model blank",
        ),
        pytest.param(
            "model_id=eleven_flash_v2_5&model_id=eleven_multilingual_v2",
            "model_id=eleven_flash_v2_5",
            id="second model stripped",
        ),
        pytest.param(
            "authorization=Bearer%20el-caller-key&single_use_token=sut&model_id=eleven_flash_v2_5",
            "model_id=eleven_flash_v2_5",
            id="caller credentials stripped",
        ),
    ],
)
async def test_upstream_query_carries_exactly_one_authorized_model_and_no_caller_credentials(
    query, expected_query, monkeypatch
):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    websocket = _FakeWebSocket(STREAM_PATH, query)

    with patch(GET_CREDENTIALS, return_value="el-provider-key"):
        relay = await _serve(websocket)

    assert [call.target for call in relay.calls] == [f"{UPSTREAM}/stream-input?{expected_query}"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_base", "expected_target"),
    [
        (
            "https://api.eu.residency.elevenlabs.io/",
            "wss://api.eu.residency.elevenlabs.io/v1/text-to-speech/voice1/stream-input?model_id=eleven_flash_v2_5",
        ),
        (
            "http://localhost:8080",
            "ws://localhost:8080/v1/text-to-speech/voice1/stream-input?model_id=eleven_flash_v2_5",
        ),
    ],
)
async def test_server_configured_api_base_is_honoured(api_base, expected_target, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_BASE", api_base)
    websocket = _FakeWebSocket(STREAM_PATH, "model_id=eleven_flash_v2_5")

    with patch(GET_CREDENTIALS, return_value="el-provider-key"):
        relay = await _serve(websocket)

    assert [call.target for call in relay.calls] == [expected_target]


@pytest.mark.asyncio
async def test_missing_provider_key_closes_before_the_handshake_completes():
    websocket = _FakeWebSocket(STREAM_PATH, "model_id=eleven_flash_v2_5")

    with patch(GET_CREDENTIALS, return_value=None):
        relay = await _serve(websocket)

    assert websocket.closed is not None
    assert websocket.closed[0] == 1011
    assert "ELEVENLABS_API_KEY" in websocket.closed[1]
    assert websocket.accepts == []
    assert relay.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_id", ["..", "a?model_id=eleven_v3", "a#b", "a b", "voixé"])
async def test_voice_ids_that_could_rewrite_the_upstream_url_are_refused(voice_id, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    websocket = _FakeWebSocket(STREAM_PATH, "model_id=eleven_flash_v2_5")

    with patch(GET_CREDENTIALS, return_value="el-provider-key"):
        relay = await _serve(websocket, voice_id=voice_id)

    assert relay.calls == []
    assert websocket.closed is not None
    assert websocket.closed[0] == 1008
    assert "voice_id" in websocket.closed[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "registry_key"),
    [
        pytest.param("model_id=eleven_not_in_registry", "elevenlabs/eleven_not_in_registry", id="unknown model"),
        pytest.param("model_id=scribe_v1", "elevenlabs/scribe_v1", id="row without a per character price"),
    ],
)
async def test_sessions_the_proxy_cannot_price_are_refused_before_elevenlabs_is_contacted(
    query, registry_key, monkeypatch
):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    websocket = _FakeWebSocket(STREAM_PATH, query)

    with patch(GET_CREDENTIALS, return_value="el-provider-key"):
        relay = await _serve(websocket)

    assert relay.calls == []
    assert websocket.closed is not None
    assert websocket.closed[0] == 1008
    assert registry_key in websocket.closed[1]
    assert "el-provider-key" not in websocket.closed[1]


@pytest.mark.asyncio
async def test_a_session_relays_once_the_operator_prices_the_model(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    with patch(GET_CREDENTIALS, return_value="el-provider-key"):
        assert (await _serve(_FakeWebSocket(STREAM_PATH, "model_id=eleven_operator_custom"))).calls == []

    monkeypatch.setitem(
        litellm.model_cost,
        "elevenlabs/eleven_operator_custom",
        dict(litellm.model_cost["elevenlabs/eleven_flash_v2_5"]),
    )
    with patch(GET_CREDENTIALS, return_value="el-provider-key"):
        relay = await _serve(_FakeWebSocket(STREAM_PATH, "model_id=eleven_operator_custom"))

    assert [call.target for call in relay.calls] == [f"{UPSTREAM}/stream-input?model_id=eleven_operator_custom"]


def _app_with_relay(relay: _FakeRelay) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_websocket_relay] = lambda: relay
    return app


def test_connections_without_a_litellm_key_are_refused():
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))

    with patch(GET_CREDENTIALS, return_value="el-provider-key") as get_credentials:
        with pytest.raises(WebSocketDisconnect) as disconnect:
            with client.websocket_connect(f"{STREAM_PATH}?model_id=eleven_flash_v2_5"):
                pass

    assert disconnect.value.code == 1008
    assert relay.calls == []
    get_credentials.assert_not_called()


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"xi-api-key": "sk-litellm-virtual"}, id="elevenlabs sdk header"),
        pytest.param({"api-key": "sk-litellm-virtual"}, id="api-key header"),
        pytest.param({"Authorization": "Bearer sk-litellm-virtual"}, id="bearer header"),
    ],
)
def test_the_litellm_key_is_accepted_where_elevenlabs_clients_send_theirs(headers, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))
    caller = UserAPIKeyAuth(api_key="hashed-sk-litellm", team_id="team-tts")

    with (
        patch(GET_CREDENTIALS, return_value="el-provider-key"),
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=caller)) as auth,
    ):
        with client.websocket_connect(f"{STREAM_PATH}?model_id=eleven_flash_v2_5", headers=headers):
            pass

    assert auth.await_args.kwargs["api_key"] == "Bearer sk-litellm-virtual"
    assert [(call.custom_headers, call.forward_headers, call.user_api_key_dict) for call in relay.calls] == [
        (PROVIDER_HEADERS, False, caller)
    ]


def test_a_bad_voice_id_reaches_the_client_as_a_policy_close(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))

    with (
        patch(GET_CREDENTIALS, return_value="el-provider-key"),
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed"))),
    ):
        with pytest.raises(WebSocketDisconnect) as disconnect:
            with client.websocket_connect(
                "/elevenlabs/v1/text-to-speech/a%3Fb/stream-input?model_id=eleven_flash_v2_5",
                headers={"xi-api-key": "sk-litellm-virtual"},
            ) as connection:
                connection.receive_text()

    assert disconnect.value.code == 1008
    assert "voice_id" in disconnect.value.reason
    assert relay.calls == []


def test_the_browser_subprotocol_carrying_the_litellm_key_is_echoed(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))

    with (
        patch(GET_CREDENTIALS, return_value="el-provider-key"),
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed"))),
    ):
        with client.websocket_connect(
            f"{STREAM_PATH}?model_id=eleven_flash_v2_5",
            subprotocols=["openai-insecure-api-key.sk-litellm-virtual"],
        ) as connection:
            assert connection.accepted_subprotocol == "openai-insecure-api-key.sk-litellm-virtual"

    assert [call.custom_headers for call in relay.calls] == [PROVIDER_HEADERS]


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
    ("query", "expected_query"),
    [
        pytest.param("model_id=eleven_flash_v2_5", "model_id=eleven_flash_v2_5", id="allowed model named"),
        pytest.param(
            "model_id=eleven_flash_v2_5&model_id=eleven_multilingual_v2",
            "model_id=eleven_flash_v2_5",
            id="denied model behind an allowed one is stripped",
        ),
        pytest.param("model_id=eleven_multilingual_v2", None, id="denied model named"),
        pytest.param("", None, id="model omitted, default denied"),
        pytest.param("model_id=", None, id="model blank, default denied"),
    ],
)
def test_key_model_access_is_checked_against_the_model_sent_upstream(query, expected_query, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "max_budget", 0.0)
    cache = asyncio.run(_cache_restricted_key("sk-only-flash", ["eleven_flash_v2_5"]))
    relay = _FakeRelay()
    client = TestClient(_app_with_relay(relay))

    with (
        patch(GET_CREDENTIALS, return_value="el-provider-key"),
        patch.multiple(  # test-quality-ok: the real key auth path reads these proxy_server globals and has no injection seam
            "litellm.proxy.proxy_server",
            master_key="sk-master",
            prisma_client=MagicMock(),
            user_api_key_cache=cache,
            llm_model_list=None,
            llm_router=None,
        ),
    ):
        if expected_query is not None:
            with client.websocket_connect(f"{STREAM_PATH}?{query}", headers={"xi-api-key": "sk-only-flash"}):
                pass
            assert [call.target for call in relay.calls] == [f"{UPSTREAM}/stream-input?{expected_query}"]
            return
        with pytest.raises(WebSocketDisconnect) as disconnect:
            with client.websocket_connect(f"{STREAM_PATH}?{query}", headers={"xi-api-key": "sk-only-flash"}):
                pass

    assert disconnect.value.code == 1008
    assert relay.calls == []

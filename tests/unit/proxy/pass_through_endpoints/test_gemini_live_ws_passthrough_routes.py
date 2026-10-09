"""Google AI Studio Live API passthrough WebSocket route: SDK path, auth on the setup model, credential injection."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocketDisconnect

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._lazy_features import LAZY_FEATURES
from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import cache_key_object
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import _websocket_relay, router
from litellm.proxy.utils import hash_token

GET_CREDENTIALS: Final = (
    "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
)
USER_API_KEY_AUTH: Final = "litellm.proxy.auth.user_api_key_auth.user_api_key_auth"
ROUTE: Final = "/gemini/ws/google.ai.generativelanguage.{api_version}.GenerativeService.BidiGenerateContent"
SDK_PATH: Final = "/gemini/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
UPSTREAM: Final = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
SETUP_FRAME: Final = json.dumps(
    {"setup": {"model": "models/gemini-live-2.5-flash-preview", "generationConfig": {"responseModalities": ["AUDIO"]}}}
)


@dataclass(frozen=True, slots=True)
class _RelayCall:
    target: str
    custom_headers: Mapping[str, str]
    user_api_key_dict: UserAPIKeyAuth
    forward_headers: bool
    endpoint: str
    accept_websocket: bool
    initial_client_frame: str | bytes | None
    model: str | None
    custom_llm_provider: str | None


class _FakeRelay:
    def __init__(self) -> None:
        self.calls: list[_RelayCall] = []

    async def __call__(
        self,
        *,
        websocket: WebSocket,
        target: str,
        custom_headers: dict[str, str],
        user_api_key_dict: UserAPIKeyAuth,
        forward_headers: bool,
        endpoint: str,
        accept_websocket: bool,
        initial_client_frame: str | bytes | None = None,
        model: str | None = None,
        custom_llm_provider: str | None = None,
    ) -> None:
        self.calls.append(
            _RelayCall(
                target=target,
                custom_headers=MappingProxyType(dict(custom_headers)),
                user_api_key_dict=user_api_key_dict,
                forward_headers=forward_headers,
                endpoint=endpoint,
                accept_websocket=accept_websocket,
                initial_client_frame=initial_client_frame,
                model=model,
                custom_llm_provider=custom_llm_provider,
            )
        )
        await websocket.close()


def _client(relay: _FakeRelay) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_websocket_relay] = lambda: relay
    return TestClient(app)


def _open_session(
    relay: _FakeRelay, path: str, frame: str = SETUP_FRAME, headers: Mapping[str, str] | None = None
) -> WebSocketDisconnect:
    with _client(relay).websocket_connect(path, headers=dict(headers or {})) as connection:
        connection.send_text(frame)
        with pytest.raises(WebSocketDisconnect) as disconnect:
            connection.receive_text()
    return disconnect.value


def test_gemini_live_route_is_registered_on_the_sdk_path():
    ws_paths = {route.path for route in router.routes if isinstance(route, WebSocketRoute)}
    assert ROUTE in ws_paths


def test_gemini_live_is_a_lazily_loaded_mapped_pass_through_route():
    feature = next(feature for feature in LAZY_FEATURES if feature.name == "llm_passthrough")
    assert feature.matches(SDK_PATH)
    assert any(SDK_PATH.startswith(prefix) for prefix in LiteLLMRoutes.mapped_pass_through_routes.value)


@pytest.mark.parametrize(
    ("query", "headers"),
    [
        pytest.param("?key=sk-litellm-virtual", {}, id="query key, as the JS SDK sends it"),
        pytest.param("", {"x-goog-api-key": "sk-litellm-virtual"}, id="x-goog-api-key, as the Python SDK sends it"),
        pytest.param("", {"Authorization": "Bearer sk-litellm-virtual"}, id="bearer, like the other ws routes"),
    ],
)
def test_gemini_live_authorizes_the_setup_model_and_relays_with_the_server_key_only(query, headers, monkeypatch):
    monkeypatch.delenv("GEMINI_API_BASE", raising=False)
    relay = _FakeRelay()
    caller = UserAPIKeyAuth(api_key="hashed-sk-litellm", team_id="team-voice")

    with (
        patch(GET_CREDENTIALS, return_value="gemini-server-key") as get_credentials,
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=caller)) as auth,
    ):
        disconnect = _open_session(relay, f"{SDK_PATH}{query}", headers=headers)

    assert disconnect.code == 1000
    assert auth.await_args.kwargs["api_key"] == "Bearer sk-litellm-virtual"
    assert json.loads(asyncio.run(auth.await_args.kwargs["request"].body())) == {
        "model": "gemini-live-2.5-flash-preview"
    }
    assert get_credentials.call_args.kwargs == {"custom_llm_provider": "gemini", "region_name": None}
    assert relay.calls == [
        _RelayCall(
            target=UPSTREAM,
            custom_headers=MappingProxyType({"x-goog-api-key": "gemini-server-key"}),
            user_api_key_dict=caller,
            forward_headers=False,
            endpoint=SDK_PATH,
            accept_websocket=False,
            initial_client_frame=SETUP_FRAME,
            model="gemini-live-2.5-flash-preview",
            custom_llm_provider="gemini",
        )
    ]


@pytest.mark.parametrize(
    ("api_base", "api_version", "expected_target"),
    [
        pytest.param(
            None,
            "v1alpha",
            "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent",
            id="v1alpha for proactive audio",
        ),
        pytest.param(
            "http://localhost:8089",
            "v1beta",
            "ws://localhost:8089/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
            id="operator GEMINI_API_BASE",
        ),
    ],
)
def test_gemini_live_keeps_the_api_version_and_honours_the_server_api_base(
    api_base, api_version, expected_target, monkeypatch
):
    if api_base is None:
        monkeypatch.delenv("GEMINI_API_BASE", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_BASE", api_base)
    relay = _FakeRelay()

    with (
        patch(GET_CREDENTIALS, return_value="gemini-server-key"),
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed"))),
    ):
        _open_session(
            relay,
            f"/gemini/ws/google.ai.generativelanguage.{api_version}.GenerativeService.BidiGenerateContent?key=sk-1",
        )

    assert [call.target for call in relay.calls] == [expected_target]


@pytest.mark.parametrize("api_version", ["v1beta%3Fkey%3Dx", "v1beta%23", "anything"])
def test_gemini_live_rejects_api_versions_that_are_not_a_version(api_version):
    relay = _FakeRelay()

    with (
        patch(GET_CREDENTIALS, return_value="gemini-server-key"),
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed"))) as auth,
    ):
        with pytest.raises(WebSocketDisconnect) as disconnect:
            with _client(relay).websocket_connect(
                f"/gemini/ws/google.ai.generativelanguage.{api_version}.GenerativeService.BidiGenerateContent?key=sk-1"
            ):
                pass

    assert disconnect.value.code == 1008
    auth.assert_not_awaited()
    assert relay.calls == []


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(json.dumps({"realtimeInput": {"text": "hi"}}), id="no setup frame first"),
        pytest.param(
            json.dumps({"setup": {"model": 'models/x", "metadata": {"tags": ["free"]}'}}), id="smuggled model"
        ),
    ],
)
def test_gemini_live_refuses_a_session_whose_first_frame_names_no_model_it_can_authorize(frame):
    relay = _FakeRelay()

    with (
        patch(GET_CREDENTIALS, return_value="gemini-server-key") as get_credentials,
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed"))) as auth,
    ):
        disconnect = _open_session(relay, f"{SDK_PATH}?key=sk-1", frame=frame)

    assert disconnect.code == 1008
    assert "setup" in disconnect.reason
    auth.assert_not_awaited()
    get_credentials.assert_not_called()
    assert relay.calls == []


def test_gemini_live_rejects_a_session_without_a_litellm_key():
    relay = _FakeRelay()

    with patch(GET_CREDENTIALS, return_value="gemini-server-key") as get_credentials:
        disconnect = _open_session(relay, SDK_PATH)

    assert disconnect.code == 1008
    get_credentials.assert_not_called()
    assert relay.calls == []


def test_gemini_live_names_the_missing_server_key_only_to_an_authenticated_caller():
    relay = _FakeRelay()

    with (
        patch(GET_CREDENTIALS, return_value=None),
        patch(USER_API_KEY_AUTH, new=AsyncMock(return_value=UserAPIKeyAuth(api_key="hashed"))),
    ):
        disconnect = _open_session(relay, f"{SDK_PATH}?key=sk-1")

    assert disconnect.code == 1011
    assert "GEMINI_API_KEY" in disconnect.reason
    assert relay.calls == []


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
    ("setup_model", "expect_relay"),
    [
        pytest.param("models/gemini-live-2.5-flash-preview", True, id="allowed model, sdk form"),
        pytest.param("gemini-live-2.5-flash-preview", True, id="allowed model, bare id"),
        pytest.param("models/gemini-3.1-flash-live-preview", False, id="denied model"),
    ],
)
def test_gemini_live_enforces_the_key_model_allowlist_on_the_setup_model(setup_model, expect_relay, monkeypatch):
    monkeypatch.delenv("GEMINI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "max_budget", 0.0)
    cache = asyncio.run(_cache_restricted_key("sk-only-flash-live", ["gemini-live-2.5-flash-preview"]))
    relay = _FakeRelay()

    with (
        patch(GET_CREDENTIALS, return_value="gemini-server-key"),
        patch.multiple(  # test-quality-ok: the real key auth path reads these proxy_server globals and has no injection seam
            "litellm.proxy.proxy_server",
            master_key="sk-master",
            prisma_client=MagicMock(),
            user_api_key_cache=cache,
            llm_model_list=None,
            llm_router=None,
        ),
    ):
        disconnect = _open_session(
            relay, f"{SDK_PATH}?key=sk-only-flash-live", frame=json.dumps({"setup": {"model": setup_model}})
        )

    if expect_relay:
        assert disconnect.code == 1000
        assert [call.model for call in relay.calls] == ["gemini-live-2.5-flash-preview"]
        return
    assert disconnect.code == 1008
    assert relay.calls == []

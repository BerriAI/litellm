"""OpenAI passthrough WebSocket route: registration, opt-in gating, and refusals."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import patch

import pytest
from starlette.routing import WebSocketRoute

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _OPENAI_WS_DISABLED_REFUSAL,
    _OPENAI_WS_MODEL_RESTRICTED_REFUSAL,
    _has_model_restrictions,
    _openai_websocket_refusal,
    _proxy_model_allowlists,
    openai_websocket_proxy_route,
    router,
)

Scopes = tuple[Sequence[str], ...]

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
    ) -> None:
        self.calls.append(
            _RelayCall(
                target=target,
                custom_headers=MappingProxyType(dict(custom_headers)),
                forward_headers=forward_headers,
                endpoint=endpoint,
                accept_websocket=accept_websocket,
            )
        )


class _FakeModelAllowlists:
    def __init__(self, scopes: Scopes) -> None:
        self.scopes = scopes
        self.calls: list[UserAPIKeyAuth] = []

    async def __call__(self, valid_token: UserAPIKeyAuth, /) -> Scopes:
        self.calls.append(valid_token)
        return self.scopes


@dataclass(frozen=True, slots=True)
class _Served:
    relay: _FakeRelay
    allowlists: _FakeModelAllowlists


async def _serve(
    websocket: _FakeWebSocket,
    endpoint: str,
    user_api_key_dict: UserAPIKeyAuth,
    general_settings: Mapping[str, object],
    scopes: Scopes = (),
) -> _Served:
    served = _Served(relay=_FakeRelay(), allowlists=_FakeModelAllowlists(scopes))
    await openai_websocket_proxy_route(
        websocket=websocket,
        endpoint=endpoint,
        user_api_key_dict=user_api_key_dict,
        general_settings=general_settings,
        relay=served.relay,
        model_allowlists=served.allowlists,
    )
    return served


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["openai", "openai_passthrough"])
async def test_openai_websocket_forwards_query_and_keeps_provider_auth(prefix, monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    websocket = _FakeWebSocket(f"/{prefix}/v1/realtime", "model=gpt-4o-realtime-preview")

    with patch(GET_CREDENTIALS, return_value="sk-provider"):
        served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), ENABLED)

    assert served.relay.calls == [
        _RelayCall(
            target="wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview",
            custom_headers=MappingProxyType({"Authorization": "Bearer sk-provider"}),
            forward_headers=False,
            endpoint=f"/{prefix}/v1/realtime",
            accept_websocket=False,
        )
    ]
    assert websocket.accepts == [None]
    assert websocket.sent == []
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_openai_websocket_accepts_first_client_subprotocol():
    websocket = _FakeWebSocket(
        "/openai/v1/realtime",
        "model=gpt-4o-realtime-preview",
        subprotocols="realtime, openai-insecure-api-key.sk-abc, openai-beta.realtime-v1",
    )

    with patch(GET_CREDENTIALS, return_value="sk-provider"):
        served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), ENABLED)

    assert websocket.accepts == ["realtime"]
    assert [call.accept_websocket for call in served.relay.calls] == [False]
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_openai_websocket_closes_cleanly_when_provider_credentials_missing():
    websocket = _FakeWebSocket("/openai/v1/realtime", "model=gpt-4o-realtime-preview")

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
    websocket = _FakeWebSocket(f"/{prefix}/v1/realtime", "model=gpt-4o-realtime-preview")

    served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), general_settings)

    assert "enable_openai_websocket_passthrough" in websocket.error_message()
    assert websocket.accepts == [None]
    assert websocket.closed == (1008, _OPENAI_WS_DISABLED_REFUSAL.close_reason)
    assert served.relay.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("general_settings", DISABLED_SETTINGS)
async def test_openai_websocket_refusal_is_disabled_for_falsy_settings(general_settings):
    refusal = await _openai_websocket_refusal(UserAPIKeyAuth(), general_settings, _FakeModelAllowlists(()))
    assert refusal is _OPENAI_WS_DISABLED_REFUSAL


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, "true", "True"])
async def test_openai_websocket_refusal_is_none_for_truthy_settings(value):
    settings = MappingProxyType({"enable_openai_websocket_passthrough": value})
    assert await _openai_websocket_refusal(UserAPIKeyAuth(), settings, _FakeModelAllowlists(())) is None


@pytest.mark.asyncio
async def test_openai_websocket_refusal_echoes_requested_subprotocol():
    websocket = _FakeWebSocket(
        "/openai_passthrough/v1/realtime",
        "model=gpt-4o-realtime-preview",
        subprotocols="realtime, openai-beta.realtime-v1",
    )

    served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), MappingProxyType({}))

    assert websocket.accepts == ["realtime"]
    assert websocket.closed == (1008, _OPENAI_WS_DISABLED_REFUSAL.close_reason)
    assert served.relay.calls == []


RESTRICTED_SCOPES: Final[tuple[Scopes, ...]] = (
    (("gpt-4o",),),
    ((), ("gpt-4o-realtime-preview",)),
    (("all-team-models",), ("gpt-4o",)),
    ((), ("all-proxy-models",), ("gpt-4o",)),
    ((), (), (), ("gpt-4o",)),
    (("*",), (), (), (), ("gpt-4o",)),
)
UNRESTRICTED_SCOPES: Final[tuple[Scopes, ...]] = (
    (),
    ((),),
    (("all-proxy-models",),),
    (("*",),),
    (("all-team-models",), ("all-proxy-models",)),
    ((), (), (), (), ()),
    (("*",), ("all-proxy-models",), ("all-team-models",), (), ()),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("scopes", RESTRICTED_SCOPES)
async def test_openai_websocket_rejects_model_restricted_identities(scopes):
    websocket = _FakeWebSocket("/openai/v1/realtime", "model=gpt-4o-realtime-preview")
    user_api_key_dict = UserAPIKeyAuth(token="hashed-fake", user_id="user-fake", team_id="team-fake")

    served = await _serve(websocket, "v1/realtime", user_api_key_dict, ENABLED, scopes)

    assert "model restrictions" in websocket.error_message()
    assert websocket.closed == (1008, _OPENAI_WS_MODEL_RESTRICTED_REFUSAL.close_reason)
    assert served.relay.calls == []
    assert served.allowlists.calls == [user_api_key_dict]


@pytest.mark.asyncio
@pytest.mark.parametrize("scopes", RESTRICTED_SCOPES)
async def test_openai_websocket_disabled_refusal_skips_allowlist_lookups(scopes):
    allowlists = _FakeModelAllowlists(scopes)

    refusal = await _openai_websocket_refusal(UserAPIKeyAuth(), MappingProxyType({}), allowlists)

    assert refusal is _OPENAI_WS_DISABLED_REFUSAL
    assert allowlists.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("scopes", UNRESTRICTED_SCOPES)
async def test_openai_websocket_allows_unrestricted_identities(scopes):
    websocket = _FakeWebSocket("/openai/v1/responses", "")

    with patch(GET_CREDENTIALS, return_value="sk-provider"):
        served = await _serve(websocket, "v1/responses", UserAPIKeyAuth(), ENABLED, scopes)

    assert len(served.relay.calls) == 1
    assert websocket.sent == []
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_proxy_model_allowlists_reads_the_token_scopes_without_a_database():
    token: Final = UserAPIKeyAuth(models=[], team_id="team-fake", team_models=["gpt-4o"])
    with patch("litellm.proxy.proxy_server.prisma_client", None):
        scopes = await _proxy_model_allowlists()(token)

    assert tuple(tuple(scope) for scope in scopes) == ((), ("gpt-4o",))
    assert _has_model_restrictions(scopes)

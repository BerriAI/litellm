"""OpenAI passthrough WebSocket route: registration, opt-in gating, refusals, and per-frame model checks."""

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.routing import WebSocketRoute

import litellm
from litellm.proxy._types import Litellm_EntityType, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _OPENAI_WS_DISABLED_REFUSAL,
    _OPENAI_WS_UNAVAILABLE_AUTH_REFUSAL,
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


GET_USER_OBJECT: Final = "litellm.proxy.auth.auth_checks.get_user_object"


def _personal_user(models: list[str]) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(user_id="user-lit7014", models=models, max_budget=None, spend=0.0)


@contextmanager
def _personal_user_allowlist(models: list[str]) -> Iterator[AsyncMock]:
    """Stand up the database-backed owner lookup the allowlist check runs, and report every call it made."""
    lookup: Final = AsyncMock(return_value=_personal_user(models))
    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),  # test-quality-ok: read at call time
        patch(GET_USER_OBJECT, lookup),
    ):
        yield lookup


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_refuses_a_model_the_person_behind_the_key_lacks():
    """
    A key with no team of its own carries its owner's model allowlist, the same
    one every HTTP route applies, so a frame naming a model that person cannot
    use is refused even when the key itself names no models at all.
    """
    gate: Final = _proxy_frame_model_gate()
    token: Final = UserAPIKeyAuth(models=[], user_id="user-lit7014")

    with _personal_user_allowlist(["gpt-5.4-mini"]):
        refusal = await gate("gpt-5.4", token)

    assert refusal is not None
    assert "gpt-5.4" in refusal


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_allows_a_model_the_person_behind_the_key_owns():
    gate: Final = _proxy_frame_model_gate()
    token: Final = UserAPIKeyAuth(models=[], user_id="user-lit7014")

    with _personal_user_allowlist(["gpt-5.4-mini", "gpt-realtime-2.1"]):
        assert await gate("gpt-realtime-2.1", token) is None


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_leaves_a_team_key_to_the_teams_allowlist():
    """
    A key that belongs to a team is governed by the team's models, not by the
    personal allowlist of whoever created it, so the owner's own restrictions
    must not close a session the team is entitled to.
    """
    gate: Final = _proxy_frame_model_gate()
    token: Final = UserAPIKeyAuth(models=[], user_id="user-lit7014", team_id="team-lit7014")

    with _personal_user_allowlist(["gpt-5.4-mini"]) as owner_lookup:
        assert await gate("gpt-5.4", token) is None

    assert owner_lookup.await_args_list == []


@pytest.mark.asyncio
@pytest.mark.parametrize("unrestricted_models", [[], ["all-proxy-models"], ["*"]])
async def test_proxy_frame_model_gate_allows_every_model_for_unrestricted_keys(unrestricted_models):
    gate: Final = _proxy_frame_model_gate()
    token: Final = UserAPIKeyAuth(models=unrestricted_models)

    with patch("litellm.proxy.proxy_server.prisma_client", None):  # test-quality-ok: the gate needs no database
        assert await gate("gpt-realtime-2.1", token) is None


ENFORCE_GROUP_BUDGET: Final = (
    "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.enforce_model_access_group_budget"
)


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_refuses_a_model_whose_access_group_is_out_of_budget():
    """
    A model that sits in a budgeted access group is refused on every other route once
    that group has spent its budget, so a frame naming it closes the session instead
    of quietly running on an empty pool. The key is allowed the model outright here,
    so nothing but the group's budget can produce the refusal.
    """
    exhausted: Final = litellm.BudgetExceededError(
        current_cost=2.0,
        max_budget=1.0,
        message="Budget has been exceeded! Model access group=tier-a Current cost: 2.0, Max budget: 1.0",
        entity_type=Litellm_EntityType.MODEL_ACCESS_GROUP.value,
        entity_id="tier-a",
    )
    gate: Final = _proxy_frame_model_gate()
    token: Final = UserAPIKeyAuth(models=["gpt-5.4-mini"])

    with (
        patch("litellm.proxy.proxy_server.prisma_client", None),  # test-quality-ok: the model check needs no database
        patch(ENFORCE_GROUP_BUDGET, AsyncMock(side_effect=exhausted)),
    ):
        refusal = await gate("gpt-5.4-mini", token)

    assert refusal is not None
    assert "Budget has been exceeded" in refusal
    assert "Model access group=tier-a" in refusal


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_refuses_out_loud_when_it_cannot_reach_the_budgets():
    """
    A gate that cannot answer has to close the session with a reason the caller can read.
    Letting the failure escape drops the socket with no frame at all, which the caller
    cannot tell apart from the network going away.
    """
    gate: Final = _proxy_frame_model_gate()
    token: Final = UserAPIKeyAuth(models=["gpt-5.4-mini"])

    with (
        patch("litellm.proxy.proxy_server.prisma_client", None),  # test-quality-ok: the model check needs no database
        patch(ENFORCE_GROUP_BUDGET, AsyncMock(side_effect=ConnectionError("connection pool is exhausted"))),
    ):
        refusal = await gate("gpt-5.4-mini", token)

    assert refusal == _OPENAI_WS_UNAVAILABLE_AUTH_REFUSAL
    assert "connection pool" not in refusal

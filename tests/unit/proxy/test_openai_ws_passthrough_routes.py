"""OpenAI passthrough WebSocket route: registration, opt-in gating, refusals, and per-frame model checks."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi import HTTPException
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket

import litellm
from litellm.llms.openai.workload_identity import _workload_identity_auth
from litellm.proxy._types import Litellm_EntityType, ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _OPENAI_WS_ALIASED_MODEL_REFUSAL,
    _OPENAI_WS_DISABLED_REFUSAL,
    _OPENAI_WS_UNAVAILABLE_AUTH_REFUSAL,
    _openai_websocket_refusal,
    _proxy_frame_model_gate,
    openai_websocket_proxy_route,
    router,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_websocket_passthrough_logging_handler import (
    openai_websocket_frame_kind,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    WebsocketFrameModelAllowed,
    WebsocketFrameModelGate,
    WebsocketFrameModelRefused,
    WebsocketFrameModelVerdict,
    WebsocketUpstreamFrameClassifier,
)

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
    client_frame_model_gate: WebsocketFrameModelGate | None
    upstream_frame_kind: WebsocketUpstreamFrameClassifier | None


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
        client_frame_model_gate: WebsocketFrameModelGate | None = None,
        upstream_frame_kind: WebsocketUpstreamFrameClassifier | None = None,
    ) -> None:
        self.calls.append(
            _RelayCall(
                target=target,
                custom_headers=MappingProxyType(dict(custom_headers)),
                forward_headers=forward_headers,
                endpoint=endpoint,
                accept_websocket=accept_websocket,
                client_frame_model_gate=client_frame_model_gate,
                upstream_frame_kind=upstream_frame_kind,
            )
        )


class _FakeFrameModelGate:
    def __init__(self, refusals: Mapping[str, WebsocketFrameModelRefused] = MappingProxyType({})) -> None:
        self.refusals = refusals
        self.checked: list[str] = []

    async def __call__(self, websocket: object, model: str, /) -> WebsocketFrameModelVerdict:
        self.checked.append(model)
        return self.refusals.get(model) or WebsocketFrameModelAllowed(access_groups=())


@dataclass(frozen=True, slots=True)
class _Served:
    relay: _FakeRelay
    frame_model_gate: _FakeFrameModelGate


async def _serve(
    websocket: _FakeWebSocket,
    endpoint: str,
    user_api_key_dict: UserAPIKeyAuth,
    general_settings: Mapping[str, object],
    frame_model_gate: _FakeFrameModelGate | None = None,
) -> _Served:
    served: Final = _Served(relay=_FakeRelay(), frame_model_gate=frame_model_gate or _FakeFrameModelGate())
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
    websocket: Final = _FakeWebSocket(f"/{prefix}/v1/realtime", "model=gpt-realtime-2.1")

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
            upstream_frame_kind=openai_websocket_frame_kind,
        )
    ]
    assert websocket.accepts == [None]
    assert websocket.sent == []
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_openai_websocket_reports_its_route_without_the_proxy_root_path(monkeypatch):
    """Route allowlists and spend rows key on the route the proxy serves, which a root path prefix is not part of"""
    monkeypatch.setenv("SERVER_ROOT_PATH", "/litellm")
    websocket: Final = _FakeWebSocket("/litellm/openai/v1/responses", "")

    with patch(GET_CREDENTIALS, return_value="sk-provider"):
        served: Final = await _serve(websocket, "v1/responses", UserAPIKeyAuth(), ENABLED)

    assert [call.endpoint for call in served.relay.calls] == ["/openai/v1/responses"]


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


TOKEN_EXCHANGE_URL: Final = "https://auth.openai.com/oauth/token"


@pytest.fixture
def openai_wif_token_file(monkeypatch, tmp_path):
    token_file = tmp_path / "subject_token.jwt"
    token_file.write_text("subject-token-from-file")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setenv("OPENAI_IDENTITY_PROVIDER_ID", "idp_test123")
    monkeypatch.setenv("OPENAI_SERVICE_ACCOUNT_ID", "user-test456")
    monkeypatch.setenv("OPENAI_IDENTITY_TOKEN_FILE", str(token_file))
    _workload_identity_auth.cache_clear()
    return token_file


@pytest.mark.asyncio
async def test_openai_websocket_uses_workload_identity_token_without_static_key(openai_wif_token_file):
    websocket = _FakeWebSocket("/openai_passthrough/v1/realtime", "model=gpt-realtime")

    with patch(GET_CREDENTIALS, return_value=None), respx.mock(assert_all_called=True) as upstream:
        upstream.post(TOKEN_EXCHANGE_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "wif-bearer", "expires_in": 3600})
        )
        served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), ENABLED)

    assert [call.custom_headers for call in served.relay.calls] == [
        MappingProxyType({"Authorization": "Bearer wif-bearer"})
    ]
    assert websocket.closed is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subject_token_present, exchange_outcome",
    [
        (True, httpx.Response(401, json={"error": "invalid_grant"})),
        (True, httpx.ConnectError("auth.openai.com unreachable")),
        (False, httpx.Response(200, json={"access_token": "wif-bearer", "expires_in": 3600})),
    ],
    ids=["rejected", "unreachable", "missing_subject_token"],
)
async def test_openai_websocket_closes_cleanly_when_workload_identity_exchange_fails(
    openai_wif_token_file, subject_token_present, exchange_outcome
):
    if not subject_token_present:
        openai_wif_token_file.unlink()
    websocket = _FakeWebSocket("/openai_passthrough/v1/realtime", "model=gpt-realtime")

    with patch(GET_CREDENTIALS, return_value=None), respx.mock(assert_all_called=False) as upstream:
        upstream.post(TOKEN_EXCHANGE_URL).mock(side_effect=exchange_outcome)
        served = await _serve(websocket, "v1/realtime", UserAPIKeyAuth(), ENABLED)

    assert websocket.closed == (1011, "OpenAI workload identity token exchange failed")
    assert websocket.accepts == []
    assert served.relay.calls == []


@pytest.mark.asyncio
async def test_openai_websocket_closes_cleanly_when_provider_credentials_missing():
    websocket: Final = _FakeWebSocket("/openai/v1/realtime", "model=gpt-realtime-2.1")

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
    websocket: Final = _FakeWebSocket(f"/{prefix}/v1/realtime", "model=gpt-realtime-2.1")

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
    websocket: Final = _FakeWebSocket("/openai/v1/responses", "")
    user_api_key_dict: Final = UserAPIKeyAuth(models=key_models, token="hashed-fake", user_id="user-fake")

    with patch(GET_CREDENTIALS, return_value="sk-provider"):
        served: Final = await _serve(websocket, "v1/responses", user_api_key_dict, ENABLED)

    assert len(served.relay.calls) == 1
    assert served.relay.calls[0].client_frame_model_gate is served.frame_model_gate
    assert websocket.sent == []
    assert websocket.closed is None


ALIAS_REFUSAL: Final = WebsocketFrameModelRefused(
    message=_OPENAI_WS_ALIASED_MODEL_REFUSAL, error_type=ProxyErrorTypes.key_model_access_denied.value
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "model=gpt-realtime-2.1",
        "model=gpt-realtime-2.1&model=gpt-realtime-2.1-mini",
        "model=gpt-realtime-2.1-mini&model=gpt-realtime-2.1",
    ],
    ids=["one_model", "refused_model_first", "refused_model_last"],
)
async def test_openai_websocket_refuses_an_opening_model_the_frame_gate_refuses(query):
    """
    The relay sends the connection URL to OpenAI unchanged, so every model it names has to pass the
    same gate as a frame, or an alias the upgrade resolves to an allowed model opens the session on the
    alias name itself
    """
    websocket: Final = _FakeWebSocket("/openai/v1/realtime", query)
    gate: Final = _FakeFrameModelGate(MappingProxyType({"gpt-realtime-2.1": ALIAS_REFUSAL}))

    with patch(GET_CREDENTIALS, return_value="sk-provider"):
        served: Final = await _serve(
            websocket, "v1/realtime", UserAPIKeyAuth(models=["gpt-realtime-2.1-mini"]), ENABLED, gate
        )

    assert websocket.error_message() == _OPENAI_WS_ALIASED_MODEL_REFUSAL
    assert websocket.closed == (1008, "This key cannot use a model alias over OpenAI websocket passthrough.")
    assert served.relay.calls == []
    assert "gpt-realtime-2.1" in gate.checked


@pytest.mark.asyncio
async def test_openai_websocket_relays_a_session_once_every_opening_model_passes_the_frame_gate():
    websocket: Final = _FakeWebSocket("/openai/v1/realtime", "model=gpt-realtime-2.1-mini")

    with patch(GET_CREDENTIALS, return_value="sk-provider"):
        served: Final = await _serve(
            websocket, "v1/realtime", UserAPIKeyAuth(models=["gpt-realtime-2.1-mini"]), ENABLED
        )

    assert served.frame_model_gate.checked == ["gpt-realtime-2.1-mini"]
    assert len(served.relay.calls) == 1
    assert websocket.closed is None


class _FrameAuthorizer:
    """Stands in for the full key, team, user and budget auth the gate re-runs for every model a frame names."""

    def __init__(self, token: UserAPIKeyAuth | None = None, denial: Exception | None = None) -> None:
        self.token = token or UserAPIKeyAuth()
        self.denial = denial
        self.authorized: list[str] = []

    async def __call__(self, websocket: object, model: str) -> UserAPIKeyAuth:
        self.authorized.append(model)
        if self.denial is not None:
            raise self.denial
        return self.token


SOCKET: Final = WebSocket({"type": "websocket"}, receive=AsyncMock(), send=AsyncMock())


async def _gate_verdict(authorizer: _FrameAuthorizer, model: str) -> WebsocketFrameModelVerdict:
    with patch("litellm.proxy.proxy_server.prisma_client", None):  # test-quality-ok: the model checks need no database
        return await _proxy_frame_model_gate(authorizer)(SOCKET, model)


def _refusal(verdict: WebsocketFrameModelVerdict) -> WebsocketFrameModelRefused:
    assert isinstance(verdict, WebsocketFrameModelRefused)
    return verdict


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_runs_full_auth_for_the_model_the_frame_names():
    authorizer: Final = _FrameAuthorizer(UserAPIKeyAuth(models=["gpt-realtime-2.1", "gpt-5.4-mini"]))

    verdict: Final = await _gate_verdict(authorizer, "gpt-realtime-2.1")

    assert verdict == WebsocketFrameModelAllowed(access_groups=())
    assert authorizer.authorized == ["gpt-realtime-2.1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_model", ["gpt-5.4", "gpt-realtime-2.1-mini"])
async def test_proxy_frame_model_gate_refuses_a_model_the_key_lacks(requested_model):
    authorizer: Final = _FrameAuthorizer(UserAPIKeyAuth(models=["gpt-realtime-2.1", "gpt-5.4-mini"]))

    refusal: Final = _refusal(await _gate_verdict(authorizer, requested_model))

    assert requested_model in refusal.message
    assert refusal.error_type == ProxyErrorTypes.key_model_access_denied.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "granted_models, refusal_names",
    [(["gpt-5.4-mini"], "gpt-5.4"), ([], "no model grants")],
    ids=["model_outside_the_grant", "no_grants"],
)
async def test_proxy_frame_model_gate_refuses_a_model_the_agent_was_not_granted(granted_models, refusal_names):
    from litellm.types.agents import AgentResponse

    token: Final = UserAPIKeyAuth(models=[], agent_id="agent-fake")
    token.managed_agent_policy = AgentResponse(
        agent_id="agent-fake",
        agent_name="Agent",
        agent_card_params={},
        object_permission={"object_permission_id": "grant-fake", "models": granted_models},
    )

    refusal: Final = _refusal(await _gate_verdict(_FrameAuthorizer(token), "gpt-5.4"))

    assert refusal_names in refusal.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token",
    [
        UserAPIKeyAuth(models=["gpt-5.4-mini"], aliases={"gpt-5.4": "gpt-5.4-mini"}),
        UserAPIKeyAuth(models=["gpt-5.4-mini"], team_model_aliases={"gpt-5.4": "gpt-5.4-mini"}),
    ],
    ids=["key_alias", "team_alias"],
)
async def test_proxy_frame_model_gate_refuses_an_alias_that_would_reach_openai_unchanged(token):
    """
    The passthrough sends each frame to OpenAI as written, so OpenAI runs the
    alias name itself. An alias pointing at an allowed model must not let a key
    limited to that model run the model the alias is named after.
    """
    refusal: Final = _refusal(await _gate_verdict(_FrameAuthorizer(token), "gpt-5.4"))

    assert refusal == WebsocketFrameModelRefused(
        message=_OPENAI_WS_ALIASED_MODEL_REFUSAL, error_type=ProxyErrorTypes.key_model_access_denied.value
    )


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_refuses_a_proxy_wide_alias_for_a_key_limited_to_its_target(monkeypatch):
    monkeypatch.setattr(litellm, "model_alias_map", {"gpt-5.4": "gpt-5.4-mini"})

    refusal: Final = _refusal(
        await _gate_verdict(_FrameAuthorizer(UserAPIKeyAuth(models=["gpt-5.4-mini"])), "gpt-5.4")
    )

    assert refusal.message == _OPENAI_WS_ALIASED_MODEL_REFUSAL


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_lets_a_key_that_may_use_any_model_send_an_alias():
    token: Final = UserAPIKeyAuth(models=[], aliases={"fast": "gpt-5.4-mini"})

    verdict: Final = await _gate_verdict(_FrameAuthorizer(token), "fast")

    assert verdict == WebsocketFrameModelAllowed(access_groups=())


@pytest.mark.asyncio
@pytest.mark.parametrize("unrestricted_models", [[], ["all-proxy-models"], ["*"]])
async def test_proxy_frame_model_gate_allows_every_model_for_unrestricted_keys(unrestricted_models):
    verdict: Final = await _gate_verdict(_FrameAuthorizer(UserAPIKeyAuth(models=unrestricted_models)), "gpt-5.4")

    assert verdict == WebsocketFrameModelAllowed(access_groups=())


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_hands_back_the_access_groups_that_authorized_the_model():
    """The turn a frame starts is billed to the groups that let its model through, so the gate has to report them."""
    token: Final = UserAPIKeyAuth(models=["gpt-5.4-mini"], matched_model_access_groups=["tier-a"])

    verdict: Final = await _gate_verdict(_FrameAuthorizer(token), "gpt-5.4-mini")

    assert verdict == WebsocketFrameModelAllowed(access_groups=("tier-a",))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_model",
    ["gpt-5.4-mini", "gpt-5.4"],
    ids=["allowed", "refused_by_the_model_check"],
)
async def test_proxy_frame_model_gate_releases_the_budget_it_reserved_for_the_check(requested_model):
    """
    Full auth reserves budget for the request it authorizes, but a frame check is
    not a request, so a reservation the gate leaves open would hold the key's
    spend counter above its real spend and refuse later calls on every route.
    """
    reservation: Final[dict[str, object]] = {"reserved_cost": 0.5, "entries": []}
    token: Final = UserAPIKeyAuth(models=["gpt-5.4-mini"], budget_reservation=reservation)

    await _gate_verdict(_FrameAuthorizer(token), requested_model)

    assert token.budget_reservation == {**reservation, "finalized": True}


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_does_not_count_the_sessions_own_reservation_as_key_spend():
    """
    The upgrade reserves up to the key's remaining budget for the session, and the key budget check
    counts open reservations as spend, so a frame checked while that reservation is open is refused
    on a key with budget left, where an HTTP call with the same key goes through.
    """
    connect_reservation: Final[dict[str, object]] = {"reserved_cost": 0.0001, "entries": []}
    socket: Final = WebSocket(
        {"type": "websocket", "state": {"budget_reservation": connect_reservation}}, receive=AsyncMock(), send=AsyncMock()
    )

    async def key_budget_check(websocket: object, model: str) -> UserAPIKeyAuth:
        if connect_reservation.get("finalized") is not True:
            raise litellm.BudgetExceededError(current_cost=0.0001, max_budget=0.0001)
        return UserAPIKeyAuth(models=["gpt-5.4-mini"])

    with patch("litellm.proxy.proxy_server.prisma_client", None):  # test-quality-ok: the model checks need no database
        verdict: Final = await _proxy_frame_model_gate(key_budget_check)(socket, "gpt-5.4-mini")

    assert verdict == WebsocketFrameModelAllowed(access_groups=())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "denial, error_type, message",
    [
        (
            ProxyException(
                message="Budget has been exceeded! Key=hashed Spend=2.0, Budget=1.0",
                type=ProxyErrorTypes.budget_exceeded,
                param=None,
                code=400,
            ),
            "budget_exceeded",
            "Budget has been exceeded! Key=hashed",
        ),
        (
            litellm.BudgetExceededError(
                current_cost=2.0,
                max_budget=1.0,
                message="Budget has been exceeded! Model access group=tier-a Current cost: 2.0, Max budget: 1.0",
                entity_type=Litellm_EntityType.MODEL_ACCESS_GROUP.value,
                entity_id="tier-a",
            ),
            "budget_exceeded",
            "Model access group=tier-a",
        ),
        (
            ProxyException(
                message="team not allowed to access model. This team can only access models=['gpt-5.4-mini']",
                type=ProxyErrorTypes.team_model_access_denied,
                param="model",
                code=401,
            ),
            "team_model_access_denied",
            "team not allowed to access model",
        ),
        (HTTPException(status_code=403, detail="Invalid Authorization header format"), "invalid_request_error", ""),
    ],
    ids=["key_budget", "group_budget", "team_allowlist", "malformed_auth"],
)
async def test_proxy_frame_model_gate_refuses_what_full_auth_denies_with_its_reason(denial, error_type, message):
    """A frame is denied for the same reasons a request is, and the caller reads which reason in the error event."""
    refusal: Final = _refusal(await _gate_verdict(_FrameAuthorizer(denial=denial), "gpt-5.4-mini"))

    assert refusal.error_type == error_type
    assert message in refusal.message


@pytest.mark.asyncio
async def test_proxy_frame_model_gate_refuses_out_loud_when_auth_cannot_answer():
    """
    A gate that cannot answer has to close the session with a reason the caller can read.
    Letting the failure escape drops the socket with no frame at all, which the caller
    cannot tell apart from the network going away.
    """
    refusal: Final = _refusal(
        await _gate_verdict(_FrameAuthorizer(denial=ConnectionError("connection pool is exhausted")), "gpt-5.4-mini")
    )

    assert refusal == WebsocketFrameModelRefused(message=_OPENAI_WS_UNAVAILABLE_AUTH_REFUSAL, error_type="server_error")

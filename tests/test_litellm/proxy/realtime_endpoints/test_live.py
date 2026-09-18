import json
import time
from contextlib import asynccontextmanager
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from prisma.builder import QueryBuilder

from litellm.llms.chatgpt.live import LiveDeployment
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import ModelAccessDeniedProxyException, UserAPIKeyAuth
from litellm.proxy.realtime_endpoints import live


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-live-encryption-key")


def handle(owner="owner", model_id=None):
    deployment = {"model": "gpt-live", "provider": "openai"}
    if model_id is not None:
        deployment["model_id"] = model_id
    return live._new_handle(
        "sess_upstream",
        "voice",
        LiveDeployment(**deployment),
        UserAPIKeyAuth(api_key=owner),
        None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["chatgpt_auth_profile", "chatgpt_token_dir", "chatgpt_auth_file"])
async def test_live_rejects_unsupported_deployment_credentials(monkeypatch, field):
    from litellm.proxy import proxy_server

    router = SimpleNamespace(
        async_get_available_deployment=AsyncMock(
            return_value={
                "litellm_params": {"model": "chatgpt/gpt-live-1", field: "other-account"},
                "model_info": {"id": "voice"},
            }
        ),
        async_routing_strategy_pre_call_checks=AsyncMock(),
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    with pytest.raises(HTTPException) as rejected:
        await live._deployment("voice", {})
    assert rejected.value.status_code == 400
    assert "deployment auth overrides are unsupported" in rejected.value.detail


def test_session_tokens_hide_credentials_and_enforce_owner_expiry_and_integrity():
    original = handle()
    token = live.encode_session(original)
    assert "deployment-a" not in token and "sess_upstream" not in token
    assert live.decode_session(token, live._owner(UserAPIKeyAuth(api_key="owner"))) == original
    for candidate, owner in (
        (token, "other-owner"),
        ("sess_upstream", original.owner),
        (token[:-8] + "aaaaaaaa", original.owner),
        (live.encode_session(original.model_copy(update={"expires_at": time.time() - 1})), original.owner),
    ):
        with pytest.raises(HTTPException) as rejected:
            live.decode_session(candidate, owner)
        assert rejected.value.status_code == 403


def test_handle_serializes_mappingproxy_without_losing_pinned_deployment():
    deployment = LiveDeployment(
        model="gpt-live",
        model_id="deployment-a",
        api_base="https://upstream.test/v1",
        extra_headers=MappingProxyType({"openai-beta": "test"}),
        extra_query=MappingProxyType({"architecture": "test"}),
    )
    original = live._new_handle("sess_upstream", "voice", deployment, UserAPIKeyAuth(api_key="owner"), None)
    assert live._pinned(live.decode_session(live.encode_session(original), original.owner)) == deployment


def test_json_value_iteratively_serializes_nested_mappingproxy_tuple_and_shared_subtree():
    shared = MappingProxyType({"deep": (1, 2)})
    value = MappingProxyType({"left": shared, "right": (shared,)})

    assert live._json_value(value) == {"left": {"deep": [1, 2]}, "right": [{"deep": [1, 2]}]}


@pytest.mark.parametrize("value", [{1: "invalid"}, {"invalid": object()}])
def test_json_value_rejects_non_json_objects_and_keys(value):
    with pytest.raises(ValueError, match="validation error"):
        live._json_value(value)


def test_json_value_rejects_cycles_and_excessive_depth():
    cycle = {}
    cycle["self"] = cycle
    with pytest.raises(ValueError, match="depth"):
        live._json_value(cycle)

    nested = json.loads('{"value":' * 257 + "null" + "}" * 257)
    with pytest.raises(ValueError, match="depth"):
        live._json_value(nested)


def test_only_protocol_session_ids_are_rewritten_and_application_values_survive():
    event = {
        "type": "session.started",
        "session": {"id": "raw", "instructions": "raw"},
        "session_id": "raw",
        "delta": "raw",
        "event": {"type": "response.output_text.delta", "delta": "raw", "session_id": "raw"},
    }
    rewritten = live.rewrite_session_ids(event, "raw", "public")
    assert rewritten["session"]["id"] == "public"
    assert rewritten["session_id"] == "public"
    assert rewritten["session"]["instructions"] == "raw"
    assert rewritten["delta"] == "raw"
    assert rewritten["event"] == event["event"]
    assert event["session"]["id"] == "raw"


@pytest.fixture
def route_client(monkeypatch):
    from litellm.proxy import proxy_server

    auth = UserAPIKeyAuth(api_key="owner")
    deployment = LiveDeployment(model="gpt-live", provider="openai", api_key="upstream-key", model_id="deployment-a")
    transport = SimpleNamespace(
        request=AsyncMock(
            return_value=httpx.Response(
                201, json={"session": {"id": "sess_upstream"}, "transport": {"type": "webrtc", "sdp": "answer"}}
            )
        )
    )
    selected = AsyncMock(return_value=deployment)
    supervised = AsyncMock()
    authenticated_bodies = []

    async def authenticate(request):
        authenticated_bodies.append(await request.json())
        return auth

    @asynccontextmanager
    async def precall(request, auth, model, **kwargs):
        yield live._Prepared({"model": model}, Mock(), None, kwargs.get("ownership"))

    monkeypatch.setattr(live, "_auth", authenticate)
    monkeypatch.setattr(live, "_precall", precall)
    monkeypatch.setattr(live, "_deployment", selected)
    monkeypatch.setattr(live, "_supervise", supervised)
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        SimpleNamespace(
            get_deployment=lambda model_id: (
                {
                    "model_name": "voice",
                    "litellm_params": {"model": "openai/gpt-live"},
                    "model_info": {"id": model_id},
                }
                if model_id == "deployment-a"
                else None
            )
        ),
    )
    factory = Mock(return_value=transport)
    monkeypatch.setattr(live, "LiveTransport", factory)
    app = FastAPI()
    app.include_router(live.router)
    return SimpleNamespace(
        client=TestClient(app),
        transport=transport,
        selected=selected,
        supervised=supervised,
        auth=auth,
        factory=factory,
        bodies=authenticated_bodies,
    )


@pytest.mark.parametrize("prefix", ["/v1/live", "/live", "/openai/v1/live"])
def test_create_preserves_configuration_and_returns_owned_json_session(route_client, prefix):
    body = {
        "session": {
            "model": "voice",
            "instructions": "hello",
            "input": [{"role": "user", "content": "hi"}],
            "audio": {"output": {"voice": "marin"}},
            "delegation": {"type": "client"},
            "future_option": {"enabled": True},
        },
        "transport": {"type": "webrtc", "sdp": "offer"},
        "api_base": "https://untrusted.test",
    }
    result = route_client.client.post(prefix + "/sessions", json=body)
    assert result.status_code == 201
    output = result.json()
    assert output["transport"] == {"type": "webrtc", "sdp": "answer"}
    original = live.decode_session(output["session"]["id"], live._owner(route_client.auth))
    assert original.session_id == "sess_upstream"
    assert original.deployment["api_key"] == "upstream-key"
    assert original.initialization_seconds == 15
    forwarded = route_client.transport.request.await_args.kwargs["body"]
    assert forwarded["session"] == {**body["session"], "model": "gpt-live"}
    assert route_client.bodies[0] == {**body, "model": "voice"}
    route_client.supervised.assert_awaited_once()
    assert route_client.factory.call_args.args[0].api_base is None


def test_fork_preserves_empty_overrides_and_pins_source_deployment(route_client):
    source = handle(model_id="deployment-a")
    token = live.encode_session(source)
    body = {"session": {}, "transport": {"type": "webrtc", "sdp": "offer"}}
    result = route_client.client.post(f"/v1/live/sessions/{token}/fork", json=body)
    assert result.status_code == 201
    assert route_client.transport.request.await_args.args == ("POST", "live/sessions/sess_upstream/fork")
    assert route_client.transport.request.await_args.kwargs["body"] == body
    assert route_client.factory.call_args.args[0].model_id == "deployment-a"
    route_client.selected.assert_not_awaited()


@pytest.mark.parametrize(
    "configured",
    [
        None,
        {
            "model_name": "voice",
            "litellm_params": {"model": "openai/gpt-replaced"},
            "model_info": {"id": "deployment-a"},
        },
        {
            "model_name": "voice",
            "litellm_params": {"model": "openai/gpt-live"},
            "model_info": {"id": "deployment-a", "blocked": True},
        },
    ],
    ids=["removed", "replaced", "blocked"],
)
def test_fork_rejects_removed_or_replaced_source_deployment(route_client, monkeypatch, configured):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server.llm_router, "get_deployment", lambda model_id: configured)
    token = live.encode_session(handle(model_id="deployment-a"))

    result = route_client.client.post(
        f"/v1/live/sessions/{token}/fork",
        json={"session": {}, "transport": {"type": "webrtc", "sdp": "offer"}},
    )

    assert result.status_code == 410
    assert "no longer available" in result.json()["detail"]
    route_client.transport.request.assert_not_awaited()


def test_fork_cannot_change_model_even_to_same_alias(route_client):
    token = live.encode_session(handle())
    response = route_client.client.post(f"/v1/live/sessions/{token}/fork", json={"session": {"model": "voice"}})
    assert response.status_code == 400
    route_client.transport.request.assert_not_awaited()


def test_cross_key_followup_and_raw_incoming_ids_never_contact_upstream(route_client):
    token = live.encode_session(handle("different-key"))
    for path in (f"{token}/hangup", "sess_other/accept", "sess_other/reject"):
        response = route_client.client.post(f"/v1/live/sessions/{path}", json={})
        assert response.status_code == 403
    route_client.transport.request.assert_not_awaited()


def test_recording_preserves_binary_body_status_and_content_headers(route_client):
    token = live.encode_session(handle())
    route_client.transport.request.return_value = httpx.Response(
        206,
        content=b"\x00\xffrecording",
        headers={
            "content-type": "video/mp4",
            "content-disposition": "attachment; filename=recording.mp4",
            "content-range": "bytes 0-10/20",
        },
    )
    result = route_client.client.get(f"/v1/live/sessions/{token}/content")
    assert result.status_code == 206
    assert result.content == b"\x00\xffrecording"
    assert result.headers["content-type"] == "video/mp4"
    assert result.headers["content-range"] == "bytes 0-10/20"


@pytest.mark.asyncio
async def test_pre_call_preserves_safe_body_options_and_refunds_failed_signaling(monkeypatch):
    from litellm.proxy import proxy_server

    auth = UserAPIKeyAuth(api_key="owner")
    authorize = AsyncMock()
    process = AsyncMock(return_value=({"model": "voice"}, Mock()))
    release = AsyncMock()
    monkeypatch.setattr(live, "_authorize", authorize)
    monkeypatch.setattr(live, "process_codex_request", process)
    monkeypatch.setattr(live, "release_or_invalidate_budget_reservation", release)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", SimpleNamespace(get_proxy_hook=lambda _: None))
    request = Request({"type": "http", "headers": []})
    synthetic = live._request(
        request,
        {
            "session": {"model": "voice", "instructions": "safe"},
            "api_base": "https://untrusted.test",
            "extra_headers": {"x-admin": "true"},
        },
    )
    with pytest.raises(RuntimeError, match="signaling failed"):
        async with live._precall(synthetic, auth, "voice"):
            raise RuntimeError("signaling failed")
    authorize.assert_awaited_once_with("voice", auth)
    data = process.await_args.args[1]
    assert data["session"]["instructions"] == "safe"
    assert "api_base" not in data and "extra_headers" not in data
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_transferred_supervisor_retains_budget_on_client_disconnect(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(live, "_authorize", AsyncMock())
    monkeypatch.setattr(live, "process_codex_request", AsyncMock(return_value=({"model": "voice"}, Mock())))
    release = AsyncMock()
    monkeypatch.setattr(live, "release_or_invalidate_budget_reservation", release)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", SimpleNamespace(get_proxy_hook=lambda _: None))
    request = live._request(Request({"type": "http", "headers": []}), {"model": "voice"})

    async def disconnect_after_transfer():
        async with live._precall(request, UserAPIKeyAuth(api_key="owner"), "voice") as prepared:
            prepared.transferred = True
            raise RuntimeError("client disconnected")

    with pytest.raises(RuntimeError, match="client disconnected"):
        await disconnect_after_transfer()
    release.assert_not_awaited()


@pytest.mark.asyncio
async def test_info_before_started_is_forwarded_without_rejecting_session():
    info = {"type": "info", "code": "data_channel_permissions", "message": "ready"}
    started = {"type": "session.started", "session": {"id": "sess_upstream"}}
    socket = SimpleNamespace(recv=AsyncMock(side_effect=[json.dumps(info), json.dumps(started)]))
    client = SimpleNamespace(send_json=AsyncMock())
    assert await live._wait_started(socket, client) == started
    client.send_json.assert_awaited_once_with(info)


@pytest.mark.asyncio
async def test_upstream_startup_error_is_forwarded_without_creating_session():
    error = {"type": "error", "error": {"code": "forbidden", "message": "Voice access denied"}}
    socket = SimpleNamespace(recv=AsyncMock(return_value=json.dumps(error)))
    client = SimpleNamespace(send_json=AsyncMock())
    with pytest.raises(HTTPException) as rejected:
        await live._wait_started(socket, client)
    assert rejected.value.status_code == 502
    client.send_json.assert_awaited_once_with(error)


@pytest.mark.asyncio
async def test_session_events_keep_stable_public_id_and_feed_shared_usage_sink():
    original = handle()
    token = live.encode_session(original)
    client = SimpleNamespace(send_text=AsyncMock(), scope={}, headers={})
    observer = SimpleNamespace(store_message=Mock())
    frontend = live._PublicSocket(client, original, token, UserAPIKeyAuth(api_key="owner"), observer)
    event = {
        "type": "session.updated",
        "session": {"id": original.session_id},
        "event": {"type": "response.completed", "response": {"id": "resp_a"}},
    }
    for _ in range(2):
        await frontend.send_text(json.dumps(event))
        assert json.loads(client.send_text.await_args.args[0])["session"]["id"] == token
    assert observer.store_message.call_count == 2


@pytest.mark.asyncio
async def test_websocket_delegation_model_update_is_authorized_before_forwarding(monkeypatch):
    authorize = AsyncMock(side_effect=HTTPException(403, "Model forbidden"))
    monkeypatch.setattr(live, "_authorize", authorize)
    message = {"type": "session.update", "session": {"delegation": {"responses": {"model": "unauthorized"}}}}
    client = SimpleNamespace(receive_text=AsyncMock(return_value=json.dumps(message)), scope={}, headers={})
    auth = UserAPIKeyAuth(api_key="owner")
    frontend = live._PublicSocket(client, handle(), "public", auth)
    with pytest.raises(HTTPException) as rejected:
        await frontend.receive_text()
    assert rejected.value.status_code == 403
    authorize.assert_awaited_once_with("unauthorized", auth)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits",
    [
        {"rpm_limit": 1},
        {"rpm_limit": 0},
        {"model_max_budget": {"backend": 1}},
        {"rpm_limit_per_model": {"backend": 0}},
        {"tpm_limit_per_model": {"backend": 0}},
        {"team_tpm_limit": 10},
        {"organization_rpm_limit": 0},
        {"organization_tpm_limit": 0},
        {"team_member_rpm_limit": 0},
        {"team_member_tpm_limit": 0},
        {"end_user_rpm_limit": 0},
        {"end_user_tpm_limit": 0},
        {"team_metadata": {"model_rpm_limit": {"backend": 1}}},
        {"metadata": {"scopes": [{"nested": {"model_tpm_limit": {"backend": 0}}}]}},
    ],
)
async def test_managed_delegation_fails_closed_for_unenforceable_constraints(limits):
    body = {"session": {"delegation": {"type": "responses", "responses": {"model": "backend"}}}}
    with pytest.raises(HTTPException) as rejected:
        await live._authorize_delegation(body, UserAPIKeyAuth(api_key="owner", **limits))
    assert rejected.value.status_code == 400
    assert "client delegation" in rejected.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("delegation", [{"type": "responses"}, {"type": "responses", "responses": {}}])
async def test_restricted_session_update_can_retain_backend_delegation_model(
    delegation,
):
    body = {"type": "session.update", "session": {"delegation": delegation}}
    result = await live._authorize_delegation(body, UserAPIKeyAuth(api_key="owner", models=["voice", "backend"]))
    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("session", [{}, {"delegation": None}, {"delegation": {"type": "client"}}])
async def test_constrained_webrtc_client_delegation_allows_frontend_updates(
    session,
):
    result = await live._authorize_delegation(
        {"session": session, "transport": {"type": "webrtc"}},
        UserAPIKeyAuth(api_key="owner", models=["voice"], rpm_limit=10),
    )
    assert result is None


@pytest.mark.parametrize(
    "limits",
    [
        {"max_budget": 1},
        {"team_max_budget": 1},
        {"user_max_budget": 1},
        {"end_user_max_budget": 1},
        {"organization_max_budget": 1},
        {"budget_limits": [{"budget_duration": "1d", "max_budget": 1}]},
    ],
)
def test_managed_constraints_detect_scalar_and_window_budgets(limits):
    assert live._managed_constraints(UserAPIKeyAuth(api_key="owner", **limits)) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "member_limit, default_limit, blocked",
    [(0, None, True), (1, None, True), (None, 1, True), (None, 0, False), (None, None, False)],
)
async def test_managed_delegation_checks_authoritative_member_and_default_budget(
    monkeypatch, member_limit, default_limit, blocked
):
    auth = UserAPIKeyAuth(
        api_key="owner", team_id="team", user_id="user", team_metadata={"team_member_budget_id": "budget"}
    )
    from litellm.proxy import proxy_server

    membership = AsyncMock(
        return_value=SimpleNamespace(litellm_budget_table=LiteLLM_BudgetTable(max_budget=member_limit))
    )
    db = SimpleNamespace(
        litellm_teammembership=SimpleNamespace(find_unique=membership),
        litellm_teamtable=SimpleNamespace(find_unique=AsyncMock(return_value=None)),
        litellm_budgettable=SimpleNamespace(
            find_unique=AsyncMock(return_value=LiteLLM_BudgetTable(max_budget=default_limit))
        ),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setattr(
        proxy_server, "user_api_key_cache", SimpleNamespace(async_get_cache=AsyncMock(return_value=None))
    )
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(live, "_authorize", AsyncMock())
    body = {"session": {"delegation": {"type": "responses", "responses": {"model": "backend"}}}}

    if blocked:
        with pytest.raises(HTTPException) as rejected:
            await live._authorize_delegation(body, auth)
        assert rejected.value.status_code == 400
        assert "client delegation" in rejected.value.detail
    else:
        await live._authorize_delegation(body, auth)
    assert membership.await_args.kwargs["where"] == {"user_id_team_id": {"user_id": "user", "team_id": "team"}}


@pytest.mark.asyncio
async def test_managed_delegation_rejects_unverifiable_member_budget_but_allows_client(monkeypatch):
    from litellm.proxy import proxy_server

    db = SimpleNamespace(
        litellm_teammembership=SimpleNamespace(find_unique=AsyncMock(side_effect=RuntimeError("Database unavailable")))
    )
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setattr(
        proxy_server, "user_api_key_cache", SimpleNamespace(async_get_cache=AsyncMock(return_value=None))
    )
    auth = UserAPIKeyAuth(api_key="owner", team_id="team", user_id="user")
    body = {"session": {"delegation": {"type": "responses", "responses": {"model": "backend"}}}}
    with pytest.raises(HTTPException) as rejected:
        await live._authorize_delegation(body, auth)
    assert rejected.value.status_code == 503
    await live._authorize_delegation({"session": {"delegation": {"type": "client"}}}, auth)


def test_managed_constraints_fails_closed_after_metadata_node_limit():
    auth = UserAPIKeyAuth(api_key="owner")
    auth.metadata = {"items": [{} for _ in range(4097)]}

    assert live._managed_constraints(auth) is True


@pytest.mark.parametrize(
    "limits",
    [
        {"model_max_budget": {}},
        {"team_metadata": {"model_rpm_limit": {}}},
        {"metadata": {"nested": [{"model_max_budget": {}}]}},
    ],
)
def test_empty_model_limit_maps_do_not_mark_delegation_as_managed(limits):
    assert live._managed_constraints(UserAPIKeyAuth(api_key="owner", **limits)) is False


def test_managed_constraints_terminates_on_cyclic_metadata_without_a_limit():
    metadata = {}
    metadata["self"] = metadata
    auth = UserAPIKeyAuth(api_key="owner")
    auth.metadata = metadata

    assert live._managed_constraints(auth) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        {"access_group_ids": ["restricted-group"]},
        {"project_id": "restricted-project"},
        {"org_id": "restricted-org"},
        {"team_id": "restricted-team"},
        {"team_id": "restricted-team", "user_id": "restricted-member"},
    ],
)
async def test_restricted_scopes_cannot_delegate_without_an_explicit_backend_model(monkeypatch, scope):
    monkeypatch.setattr(live, "_managed_member_budget", AsyncMock(return_value=False))
    body = {"session": {"delegation": {"type": "responses", "responses": {}}}}

    with pytest.raises(HTTPException) as rejected:
        await live._authorize_delegation(body, UserAPIKeyAuth(api_key="owner", **scope))

    assert rejected.value.status_code == 400
    assert "explicit authorized delegation.responses.model" in rejected.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        {"models": ["voice", "backend"]},
        {"access_group_ids": ["restricted-group"]},
        {"matched_model_access_groups": ["restricted-group"]},
        {"project_id": "restricted-project"},
        {"org_id": "restricted-org"},
        {"team_id": "restricted-team"},
        {"team_id": "restricted-team", "user_id": "restricted-member"},
    ],
)
async def test_restricted_webrtc_cannot_change_managed_model_outside_proxy(monkeypatch, scope):
    monkeypatch.setattr(live, "_managed_member_budget", AsyncMock(return_value=False))
    monkeypatch.setattr(live, "_authorize", AsyncMock())
    body = {
        "session": {"delegation": {"type": "responses", "responses": {"model": "backend"}}},
        "transport": {"type": "webrtc", "sdp": "offer"},
    }
    auth = UserAPIKeyAuth(api_key="owner", **scope)
    with pytest.raises(HTTPException) as rejected:
        await live._authorize_delegation(body, auth)
    assert rejected.value.status_code == 400
    body["session"]["client"] = {"data_channel": {"allowed_client_events": ["session.close"]}}
    await live._authorize_delegation(body, auth)


@pytest.mark.asyncio
async def test_budget_scope_releases_on_ownership_decode_error(monkeypatch):
    release = AsyncMock()
    monkeypatch.setattr(live, "release_or_invalidate_budget_reservation", release)
    auth = UserAPIKeyAuth(api_key="owner")
    with pytest.raises(HTTPException):
        async with live._budget_scope(auth):
            live.decode_session("raw-session-id", live._owner(auth))
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_authorization_rejection_still_releases_budget(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(live, "_authorize", AsyncMock(side_effect=HTTPException(403, "Model forbidden")))
    release = AsyncMock()
    monkeypatch.setattr(live, "release_or_invalidate_budget_reservation", release)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", SimpleNamespace(get_proxy_hook=lambda _: None))
    request = live._request(Request({"type": "http", "headers": []}), {"model": "forbidden"})
    with pytest.raises(HTTPException):
        async with live._precall(request, UserAPIKeyAuth(api_key="owner"), "forbidden"):
            pytest.fail("Upstream must not be reached")
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_precall_guardrail_mutations_are_used_without_forwarding_routing_options(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(live, "_authorize", AsyncMock())
    monkeypatch.setattr(live, "release_or_invalidate_budget_reservation", AsyncMock())
    process = AsyncMock(
        return_value=(
            {
                "model": "voice",
                "session": {"model": "voice", "instructions": "redacted"},
                "api_base": "https://internal.test",
            },
            Mock(),
        )
    )
    monkeypatch.setattr(live, "process_codex_request", process)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", SimpleNamespace(get_proxy_hook=lambda _: None))
    body = {"type": "session.start", "session": {"model": "voice", "instructions": "sensitive"}}
    request = live._request(Request({"type": "http", "headers": []}), body)
    async with live._precall(request, UserAPIKeyAuth(api_key="owner"), "voice") as prepared:
        forwarded = live._processed_body(body, prepared.processed)
        assert forwarded["session"]["instructions"] == "redacted"
        assert forwarded["type"] == "session.start"
        assert "api_base" not in forwarded
    assert process.await_args.args[1]["session"]["instructions"] == "sensitive"


@pytest.mark.asyncio
async def test_failed_live_signaling_releases_real_parallel_limiter_and_can_retry(monkeypatch):
    import litellm
    from litellm.proxy import proxy_server as server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import ProxyLogging

    proxy = ProxyLogging(UserApiKeyCache())
    monkeypatch.setattr(litellm, "callbacks", [])
    proxy._add_proxy_hooks()
    model_list = [{"model_name": "voice", "litellm_params": {"model": "openai/gpt-live-1", "api_key": "upstream-key"}}]
    monkeypatch.setattr(server, "proxy_logging_obj", proxy)
    monkeypatch.setattr(server, "general_settings", {})
    monkeypatch.setattr(server, "llm_model_list", model_list)
    monkeypatch.setattr(server, "llm_router", litellm.Router(model_list=model_list))
    auth = UserAPIKeyAuth(api_key="live-limiter-owner", max_parallel_requests=1)
    authenticate = AsyncMock(return_value=auth)
    monkeypatch.setattr(live, "user_api_key_auth", authenticate)
    transport = SimpleNamespace(
        request=AsyncMock(return_value=httpx.Response(403, json={"error": {"code": "forbidden"}}))
    )
    monkeypatch.setattr(live, "LiveTransport", Mock(return_value=transport))
    supervisor = AsyncMock()
    monkeypatch.setattr(live, "_supervise", supervisor)
    for _ in range(2):
        request = live._request(
            Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/live/sessions",
                    "query_string": b"",
                    "headers": [(b"content-type", b"application/json")],
                }
            ),
            {"session": {"model": "voice"}, "transport": {"type": "webrtc", "sdp": "offer"}},
        )
        result = await live.create_live_session(request)
        assert result.status_code == 403
        current = await proxy.internal_usage_cache.async_get_cache(
            "{api_key:live-limiter-owner}:max_parallel_requests", litellm_parent_otel_span=None, local_only=True
        )
        limiter = proxy.get_proxy_hook("parallel_request_limiter")
        assert limiter._gauge_in_flight_from_cache_value(current) == 0
    assert transport.request.await_count == 2
    supervisor.assert_not_awaited()


@pytest.mark.asyncio
async def test_inherited_managed_fork_cannot_bypass_new_key_constraints():
    source = handle().model_copy(
        update={"policy": {"delegation": {"type": "responses", "responses": {"model": "backend"}}}}
    )
    payload = live._policy_body({"session": {}, "transport": {"type": "webrtc", "sdp": "offer"}}, source)
    assert payload["session"]["delegation"]["responses"]["model"] == "backend"
    with pytest.raises(HTTPException) as rejected:
        await live._authorize_delegation(payload, UserAPIKeyAuth(api_key="owner", rpm_limit_per_model={"backend": 1}))
    assert rejected.value.status_code == 400


@pytest.mark.parametrize("protocol", ["http", "websocket"])
@pytest.mark.parametrize("startup_policy", [{"delegation": {"type": "responses", "responses": {"model": "allowed"}}}])
@pytest.mark.parametrize("overrides", [{}, {"delegation": {"responses": {}}}])
def test_restricted_fork_never_trusts_startup_delegation(route_client, protocol, startup_policy, overrides):
    from starlette.websockets import WebSocketDisconnect

    # The source may now use a revoked backend, including after an unrestricted WebRTC update.
    route_client.auth.models = ["voice", "allowed"]
    token = live.encode_session(handle().model_copy(update={"policy": startup_policy}))
    path = f"/v1/live/sessions/{token}/fork"
    if protocol == "http":
        response = route_client.client.post(path, json={"session": overrides})
        assert response.status_code == 400
    else:
        with route_client.client.websocket_connect(path, headers={"Authorization": "Bearer owner"}) as ws:
            ws.send_json({"type": "session.start", "session": overrides})
            with pytest.raises(WebSocketDisconnect) as rejected:
                ws.receive_json()
            assert rejected.value.code == 1008
    route_client.factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_policy", [{}, {"delegation": {"type": "responses", "responses": {"model": "old"}}}])
async def test_explicit_fork_backend_is_authorized_even_when_startup_policy_differs(monkeypatch, startup_policy):
    authorize = AsyncMock()
    monkeypatch.setattr(live, "_authorize", authorize)
    source = handle().model_copy(update={"policy": startup_policy})
    auth = UserAPIKeyAuth(api_key="owner", models=["voice", "allowed"])
    body = {"session": {"delegation": {"type": "responses", "responses": {"model": "allowed"}}}}
    await live._authorize_fork_policy(body, source, auth)
    authorize.assert_awaited_once_with("allowed", auth)
    authorize.side_effect = HTTPException(403, "Model revoked")
    with pytest.raises(HTTPException) as rejected:
        await live._authorize_fork_policy(body, source, auth)
    assert rejected.value.status_code == 403


def test_restricted_client_fork_can_inherit_delegation(route_client):
    route_client.auth.models = ["voice"]
    body = {"session": {}}
    token = live.encode_session(handle(model_id="deployment-a"))
    route_client.transport.request.return_value = httpx.Response(
        200, json={"session": {"id": "sess_fork"}, "transport": {"type": "webrtc", "sdp": "answer"}}
    )
    response = route_client.client.post(f"/v1/live/sessions/{token}/fork", json=body)
    assert response.status_code == 200
    assert response.json()["transport"]["sdp"] == "answer"
    route_client.transport.request.assert_awaited_once_with("POST", "live/sessions/sess_upstream/fork", body=body)


@pytest.mark.asyncio
@pytest.mark.parametrize("limits", [{"rpm_limit": 10}, {"tpm_limit": 100}, {"model_max_budget": {"backend": 1}}])
async def test_client_fork_can_inherit_immutable_delegation_with_new_limits(
    limits,
):
    result = await live._authorize_fork_policy({"session": {}}, handle(), UserAPIKeyAuth(api_key="owner", **limits))
    assert result is None


@pytest.mark.asyncio
async def test_explicit_managed_fork_still_rejected_for_new_rate_limits():
    with pytest.raises(HTTPException) as rejected:
        await live._authorize_fork_policy(
            {"session": {"delegation": {"type": "responses", "responses": {"model": "backend"}}}},
            handle(),
            UserAPIKeyAuth(api_key="owner", rpm_limit=10),
        )
    assert rejected.value.status_code == 400
    assert "cannot enforce" in rejected.value.detail


@pytest.mark.asyncio
async def test_startup_usage_buffer_preserves_nested_events_until_supervisor_owns_accounting():
    usage = {
        "type": "response.event",
        "event": {
            "type": "response.completed",
            "response": {
                "id": "resp_before_start",
                "model": "backend",
                "usage": {"input_tokens": 7, "output_tokens": 2},
            },
        },
    }
    started = {"type": "session.started", "session": {"id": "sess_upstream"}}
    backend = SimpleNamespace(recv=AsyncMock(side_effect=[json.dumps(usage), json.dumps(started)]))
    client = SimpleNamespace(send_json=AsyncMock())
    startup = live._StartupEvents()
    assert await live._wait_started(backend, client, startup) == started
    assert [json.loads(message) for message in startup.messages] == [usage]


def test_primary_websocket_authenticates_model_and_keeps_public_event_shape(route_client):
    from websockets.exceptions import ConnectionClosedOK
    from websockets.frames import Close

    event = {"type": "future.live.event", "session_id": "sess_upstream", "payload": {"untouched": ["a", 2]}}
    backend = SimpleNamespace(
        send=AsyncMock(),
        close=AsyncMock(),
        recv=AsyncMock(
            side_effect=[
                json.dumps({"type": "info", "code": "ready", "message": "Preparing"}),
                json.dumps({"type": "session.started", "session": {"id": "sess_upstream"}}),
                json.dumps(event),
                ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True),
            ]
        ),
    )
    route_client.transport.connect = AsyncMock(return_value=backend)
    observer = SimpleNamespace(store_message=Mock())
    route_client.supervised.return_value = observer
    with route_client.client.websocket_connect("/v1/live/sessions", headers={"Authorization": "Bearer owner"}) as ws:
        ws.send_json({"type": "session.start", "session": {"model": "voice", "instructions": "hello", "unknown": True}})
        assert ws.receive_json()["type"] == "info"
        started = ws.receive_json()
        public_id = started["session"]["id"]
        assert live.decode_session(public_id, live._owner(route_client.auth)).session_id == "sess_upstream"
        forwarded = ws.receive_json()
        assert forwarded == {**event, "session_id": public_id}
    assert route_client.bodies[0] == {}
    assert route_client.bodies[1]["model"] == "voice"
    assert route_client.bodies[1]["session"]["instructions"] == "hello"
    initial = json.loads(backend.send.await_args.args[0])
    assert initial == {
        "type": "session.start",
        "session": {"model": "gpt-live", "instructions": "hello", "unknown": True},
    }
    assert any(json.loads(call.args[0]) == event for call in observer.store_message.call_args_list)


@pytest.mark.asyncio
async def test_successful_live_session_holds_real_parallel_slot_until_supervisor_releases(monkeypatch):
    import litellm
    from litellm.proxy import proxy_server as server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import ProxyLogging

    proxy = ProxyLogging(UserApiKeyCache())
    monkeypatch.setattr(litellm, "callbacks", [])
    proxy._add_proxy_hooks()
    models = [{"model_name": "voice", "litellm_params": {"model": "openai/gpt-live-1", "api_key": "upstream-key"}}]
    monkeypatch.setattr(server, "proxy_logging_obj", proxy)
    monkeypatch.setattr(server, "general_settings", {})
    monkeypatch.setattr(server, "llm_model_list", models)
    monkeypatch.setattr(server, "llm_router", litellm.Router(model_list=models))
    auth = UserAPIKeyAuth(api_key="live-held-slot", max_parallel_requests=1)
    monkeypatch.setattr(live, "user_api_key_auth", AsyncMock(return_value=auth))
    transport = SimpleNamespace(
        request=AsyncMock(
            return_value=httpx.Response(
                201, json={"session": {"id": "sess_upstream"}, "transport": {"type": "webrtc", "sdp": "answer"}}
            )
        )
    )
    monkeypatch.setattr(live, "LiveTransport", Mock(return_value=transport))
    leases = []

    async def supervise(request, handle, auth, logger, lease):
        leases.append(lease)

    monkeypatch.setattr(live, "_supervise", supervise)

    def request():
        return live._request(
            Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/live/sessions",
                    "query_string": b"",
                    "headers": [(b"content-type", b"application/json")],
                }
            ),
            {"session": {"model": "voice"}, "transport": {"type": "webrtc", "sdp": "offer"}},
        )

    try:
        result = await live.create_live_session(request())
        assert result.status_code == 201
        assert leases[0] is not None
        with pytest.raises(HTTPException) as blocked:
            await live.create_live_session(request())
        assert blocked.value.status_code == 429
        assert transport.request.await_count == 1
    finally:
        for lease in leases:
            if lease is not None:
                await lease.close()
    current = await proxy.internal_usage_cache.async_get_cache(
        "{api_key:live-held-slot}:max_parallel_requests", litellm_parent_otel_span=None, local_only=True
    )
    limiter = proxy.get_proxy_hook("parallel_request_limiter")
    assert limiter._gauge_in_flight_from_cache_value(current) == 0


@pytest.mark.asyncio
async def test_supervisor_logger_keeps_deployment_pricing(monkeypatch):
    import litellm

    original = handle().model_copy(
        update={"deployment": {**handle().deployment, "model_id": "deployment-priced"}, "initialization_seconds": 15}
    )
    logger = Mock(litellm_params={}, model_call_details={})
    monkeypatch.setattr(live, "process_codex_request", AsyncMock(return_value=({"model": "voice"}, logger)))
    monkeypatch.setattr(litellm, "get_model_info", Mock(return_value={"input_cost_per_second": 0.1}))
    connection = SimpleNamespace(close=AsyncMock())
    transport = SimpleNamespace(connect=AsyncMock(return_value=connection), request=AsyncMock())
    monkeypatch.setattr(live, "LiveTransport", Mock(return_value=transport))
    started = AsyncMock()
    monkeypatch.setattr(live, "CALL_SUPERVISORS", SimpleNamespace(start=started))
    request = Request({"type": "http", "method": "POST", "path": "/v1/live/sessions", "headers": []})
    stream = await live._start_supervisor(request, original, UserAPIKeyAuth(api_key="owner"), None)
    assert stream.messages[0]["usage"]["seconds"] == 15
    started.assert_awaited_once()
    metadata = logger.update_from_kwargs.call_args.kwargs["kwargs"]["litellm_metadata"]
    assert metadata["model_info"]["id"] == "deployment-priced"
    assert metadata["model_info"]["input_cost_per_second"] == 0.1
    connection.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_supervisor_startup_failure_closes_observer_and_invalidates_unconfirmed_hangup(monkeypatch):
    from litellm.proxy.spend_tracking import budget_reservation

    logger = Mock(litellm_params={}, model_call_details={})
    monkeypatch.setattr(live, "process_codex_request", AsyncMock(return_value=({"model": "voice"}, logger)))
    connection = SimpleNamespace(close=AsyncMock())
    transport = SimpleNamespace(
        connect=AsyncMock(return_value=connection),
        request=AsyncMock(side_effect=httpx.ConnectError("upstream unavailable")),
    )
    monkeypatch.setattr(live, "LiveTransport", Mock(return_value=transport))
    monkeypatch.setattr(
        live, "CALL_SUPERVISORS", SimpleNamespace(start=AsyncMock(side_effect=RuntimeError("cannot start")))
    )
    invalidate = AsyncMock()
    monkeypatch.setattr(budget_reservation, "invalidate_budget_reservation_counters", invalidate)
    request = Request({"type": "http", "method": "POST", "path": "/v1/live/sessions", "headers": []})
    with pytest.raises(RuntimeError, match="cannot start"):
        await live._start_supervisor(request, handle(), UserAPIKeyAuth(api_key="owner"), None)
    connection.close.assert_awaited_once()
    invalidate.assert_awaited_once()


def test_admin_sip_accept_requires_exact_deployment_and_returns_owned_handle(route_client, monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LitellmUserRoles

    route_client.auth.user_role = LitellmUserRoles.PROXY_ADMIN
    monkeypatch.setattr(proxy_server, "llm_model_list", [{"model_name": "voice"}])
    route_client.transport.request.return_value = httpx.Response(200, content=b"")
    body = {"session": {"model": "voice", "type": "live", "instructions": "incoming"}}
    missing = route_client.client.post("/v1/live/sessions/sess_incoming/accept", json=body)
    assert missing.status_code == 400
    route_client.transport.request.assert_not_awaited()
    accepted = route_client.client.post(
        "/v1/live/sessions/sess_incoming/accept", json=body, headers={"x-litellm-live-model": "voice"}
    )
    assert accepted.status_code == 200 and accepted.content == b""
    token = accepted.headers["x-litellm-live-session-id"]
    owned = live.decode_session(token, live._owner(route_client.auth))
    assert owned.session_id == "sess_incoming" and owned.alias == "voice"
    assert route_client.transport.request.await_args.kwargs["body"]["session"]["model"] == "gpt-live"
    route_client.supervised.assert_awaited_once()
    raw_hangup = route_client.client.post("/v1/live/sessions/sess_incoming/hangup")
    assert raw_hangup.status_code == 403
    assert route_client.client.post(f"/v1/live/sessions/{token}/hangup").status_code == 200


def test_admin_sip_cannot_enroll_through_alias_with_multiple_accounts(route_client, monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LitellmUserRoles

    route_client.auth.user_role = LitellmUserRoles.PROXY_ADMIN
    monkeypatch.setattr(proxy_server, "llm_model_list", [{"model_name": "voice"}, {"model_name": "voice"}])
    response = route_client.client.post(
        "/v1/live/sessions/sess_incoming/reject", json={"status_code": 603}, headers={"x-litellm-live-model": "voice"}
    )
    assert response.status_code == 400
    route_client.transport.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_primary_first_frame_timeout_releases_authenticated_reservation(monkeypatch):
    import asyncio

    from fastapi import WebSocket

    messages = iter([{"type": "websocket.connect"}])

    async def receive():
        try:
            return next(messages)
        except StopIteration:
            raise asyncio.TimeoutError("first message timeout")

    sent = []

    async def send(message):
        sent.append(message)

    websocket = WebSocket(
        {
            "type": "websocket",
            "path": "/v1/live/sessions",
            "query_string": b"",
            "headers": [(b"authorization", b"Bearer owner")],
            "scheme": "ws",
            "server": ("localhost", 4000),
        },
        receive,
        send,
    )
    monkeypatch.setattr(live, "_auth", AsyncMock(return_value=UserAPIKeyAuth(api_key="owner")))
    release = AsyncMock()
    transport = Mock()
    monkeypatch.setattr(live, "release_or_invalidate_budget_reservation", release)
    monkeypatch.setattr(live, "LiveTransport", transport)
    await live.websocket_live_session(websocket)
    release.assert_awaited_once()
    transport.assert_not_called()
    assert sent[-1]["type"] == "websocket.close" and sent[-1]["code"] == 1008


@pytest.mark.asyncio
@pytest.mark.parametrize("key_limit,global_limit,allowed", [(None, None, True), (1, None, False), (None, 2, False)])
async def test_legacy_limiter_only_blocks_sessions_requiring_parallel_leases(
    monkeypatch, key_limit, global_limit, allowed
):
    from litellm.proxy import proxy_server
    from litellm.proxy.hooks.parallel_request_limiter import _PROXY_MaxParallelRequestsHandler

    legacy = Mock(spec=_PROXY_MaxParallelRequestsHandler)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", SimpleNamespace(get_proxy_hook=lambda _: legacy))
    monkeypatch.setattr(proxy_server, "general_settings", {"global_max_parallel_requests": global_limit})
    monkeypatch.setattr(live, "_authorize", AsyncMock())
    processor = AsyncMock(return_value=({"model": "voice"}, Mock()))
    monkeypatch.setattr(live, "process_codex_request", processor)
    monkeypatch.setattr(live, "release_or_invalidate_budget_reservation", AsyncMock())
    request = live._request(Request({"type": "http", "headers": []}), {"model": "voice"})
    auth = UserAPIKeyAuth(api_key="owner", max_parallel_requests=key_limit)
    if allowed:
        async with live._precall(request, auth, "voice") as prepared:
            assert prepared.lease is None
        processor.assert_awaited_once()
        return
    with pytest.raises(HTTPException) as rejected:
        async with live._precall(request, auth, "voice"):
            pytest.fail("Parallel-limited sessions require renewable leases")
    assert rejected.value.status_code == 400
    processor.assert_not_awaited()


@pytest.mark.parametrize("delegation", ["invalid", {"type": "responses", "responses": "invalid"}])
def test_malformed_delegation_returns_client_error_before_provider(route_client, delegation):
    result = route_client.client.post(
        "/v1/live/sessions",
        json={"session": {"model": "voice", "delegation": delegation}, "transport": {"type": "webrtc", "sdp": "offer"}},
    )
    assert result.status_code == 400
    route_client.transport.request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel_reserved", [False, True])
async def test_attachment_skips_parallel_admission_only_with_existing_session_lease(monkeypatch, parallel_reserved):
    from litellm.proxy import proxy_server
    from litellm.proxy.hooks.realtime_call_lease import is_realtime_call_attachment

    attachment = object()
    observed = []

    async def process(request, data, auth, model, route_type):
        observed.append(is_realtime_call_attachment(attachment))
        return data, Mock()

    monkeypatch.setattr(proxy_server, "proxy_logging_obj", SimpleNamespace(get_proxy_hook=lambda _: None))
    monkeypatch.setattr(live, "_authorize", AsyncMock())
    monkeypatch.setattr(live, "process_codex_request", process)
    monkeypatch.setattr(live, "release_or_invalidate_budget_reservation", AsyncMock())
    request = live._request(Request({"type": "http", "headers": []}), {"model": "voice"})
    async with live._precall(
        request, UserAPIKeyAuth(api_key="owner"), "voice", attachment=attachment, parallel_reserved=parallel_reserved
    ):
        pass
    assert observed == [parallel_reserved]


@pytest.mark.asyncio
async def test_live_observer_becomes_ready_without_session_started_event(monkeypatch):
    import asyncio

    from litellm.proxy.realtime_endpoints.call_supervision import CallSupervisor

    class Observer:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self.queue.get()

        async def close(self):
            self.closed = True

    observer = Observer()
    logger = Mock(litellm_params={}, model_call_details={})
    monkeypatch.setattr(live, "process_codex_request", AsyncMock(return_value=({"model": "voice"}, logger)))
    sink = SimpleNamespace(store_message=Mock(), log_messages=AsyncMock())
    monkeypatch.setattr(live, "RealTimeStreaming", Mock(return_value=sink))

    async def hangup(*args, **kwargs):
        await observer.queue.put(json.dumps({"type": "session.closed", "usage": {"seconds": 0}}))
        return httpx.Response(200, request=httpx.Request("POST", "https://upstream.test/hangup"))

    monkeypatch.setattr(
        live,
        "LiveTransport",
        Mock(return_value=SimpleNamespace(connect=AsyncMock(return_value=observer), request=hangup)),
    )
    supervisors = []

    def build_supervisor(*args, **kwargs):
        return CallSupervisor(
            *args, **kwargs, ready_timeout=0.1, drain_timeout=0.1, termination_timeout=0.2, logging_timeout=0.2
        )

    async def start(supervisor):
        supervisors.append(supervisor)
        await supervisor.start()

    monkeypatch.setattr(live, "CallSupervisor", build_supervisor)
    monkeypatch.setattr(live, "CALL_SUPERVISORS", SimpleNamespace(start=start))
    request = Request({"type": "http", "method": "POST", "path": "/v1/live/sessions", "headers": []})
    try:
        result = await asyncio.wait_for(
            live._start_supervisor(request, handle(), UserAPIKeyAuth(api_key="owner"), None), 0.5
        )
        assert result is sink
        sink.store_message.assert_not_called()
    finally:
        for supervisor in supervisors:
            await supervisor.close()
    assert observer.closed


@pytest.fixture
def isolated_live_model_auth(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(live, "can_key_call_resolved_model", AsyncMock())
    monkeypatch.setattr(proxy_server, "llm_model_list", [])
    monkeypatch.setattr(proxy_server, "llm_router", None)


@pytest.mark.asyncio
async def test_authorize_enforces_authoritative_personal_user_models(monkeypatch, isolated_live_model_auth):
    user_loader = AsyncMock(return_value=LiteLLM_UserTable(user_id="user-only", models=["voice"]))
    monkeypatch.setattr(live, "get_user_object", user_loader)
    auth = UserAPIKeyAuth(api_key="owner", models=[], user_id="user-only")

    await live._authorize("voice", auth)
    with pytest.raises(ModelAccessDeniedProxyException) as rejected:
        await live._authorize("backend", auth)
    assert "user can only access" in rejected.value.internal_message

    assert user_loader.await_count == 2


@pytest.mark.asyncio
async def test_authorize_enforces_authoritative_organization_models(monkeypatch, isolated_live_model_auth):
    org_loader = AsyncMock(
        return_value=LiteLLM_OrganizationTable(
            organization_id="org-only",
            budget_id="budget",
            created_by="admin",
            updated_by="admin",
            models=["voice"],
        )
    )
    monkeypatch.setattr(live, "get_org_object", org_loader)
    auth = UserAPIKeyAuth(api_key="owner", models=[], org_id="org-only")

    await live._authorize("voice", auth)
    with pytest.raises(ModelAccessDeniedProxyException) as rejected:
        await live._authorize("backend", auth)
    assert "org can only access" in rejected.value.internal_message

    assert org_loader.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "auth_kwargs", "loader_name"),
    [
        ("user", {"user_id": "user-only"}, "get_user_object"),
        ("organization", {"org_id": "org-only"}, "get_org_object"),
    ],
)
async def test_authorize_fails_closed_when_principal_grant_lookup_fails(
    monkeypatch, isolated_live_model_auth, scope, auth_kwargs, loader_name
):
    loader = AsyncMock(side_effect=RuntimeError(f"{scope} lookup unavailable"))
    monkeypatch.setattr(live, loader_name, loader)

    with pytest.raises(HTTPException) as rejected:
        await live._authorize("backend", UserAPIKeyAuth(api_key="owner", **auth_kwargs))

    assert rejected.value.status_code == 503
    assert "verify Live" in str(rejected.value.detail)


def test_restricted_models_marks_user_scoped_identity_as_restricted():
    assert live._restricted_models(UserAPIKeyAuth(api_key="owner", user_id="user-only")) is True


@pytest.mark.asyncio
async def test_sparse_responses_update_without_model_remains_valid_for_user_scoped_identity():
    result = await live._authorize_delegation(
        {"type": "session.update", "session": {"delegation": {"type": "responses", "responses": {}}}},
        UserAPIKeyAuth(api_key="owner", user_id="user-only"),
    )
    assert result is None


def test_managed_constraints_uses_exact_metadata_keys():
    for key in ("max_budget_alert_emails", "model_max_budget_usage"):
        assert live._managed_constraints(UserAPIKeyAuth(api_key="owner", metadata={key: {"backend": 1}})) is False


@pytest.mark.asyncio
async def test_managed_budget_reads_authoritative_member_budget(monkeypatch):
    from litellm.proxy import proxy_server

    membership = LiteLLM_TeamMembership(
        user_id="member",
        team_id="team",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1),
    )

    def find_unique(*, where, include):
        QueryBuilder(method="find_unique", arguments={"where": where}).build_query()
        return membership

    db = SimpleNamespace(
        litellm_teammembership=SimpleNamespace(find_unique=AsyncMock(side_effect=find_unique)),
        litellm_teamtable=SimpleNamespace(find_unique=AsyncMock(return_value=None)),
    )
    cache = SimpleNamespace(async_get_cache=AsyncMock(return_value=None), async_set_cache=AsyncMock())
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)

    assert await live._managed_member_budget(UserAPIKeyAuth(api_key="owner", team_id="team", user_id="member")) is True


@pytest.mark.asyncio
async def test_managed_budget_fails_closed_when_membership_repository_is_unreadable(monkeypatch):
    from litellm.proxy import proxy_server

    failure = RuntimeError("database unavailable")
    db = SimpleNamespace(
        litellm_teammembership=SimpleNamespace(find_unique=AsyncMock(side_effect=failure)),
    )
    cache = SimpleNamespace(async_get_cache=AsyncMock(return_value=None), async_set_cache=AsyncMock())
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)

    with pytest.raises(HTTPException) as rejected:
        await live._managed_member_budget(UserAPIKeyAuth(api_key="owner", team_id="team", user_id="member"))

    assert rejected.value.status_code == 503


@pytest.mark.asyncio
async def test_managed_budget_checks_project_team_and_model_group_tables(monkeypatch):
    from litellm.proxy import proxy_server

    project = LiteLLM_ProjectTable(
        project_id="project",
        budget_id="project-budget",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1),
    )
    team = LiteLLM_TeamTable(team_id="team", budget_limits=[{"budget_duration": "1d", "max_budget": 1}])
    group = SimpleNamespace(
        access_group_name="group",
        litellm_budget_table=SimpleNamespace(max_budget=1),
    )

    def find_project(*, where, include):
        QueryBuilder(method="find_unique", arguments={"where": where}).build_query()
        return project

    def find_groups(*, where, include):
        QueryBuilder(method="find_many", arguments={"where": where}).build_query()
        return [group]

    db = SimpleNamespace(
        litellm_teamtable=SimpleNamespace(find_unique=AsyncMock(return_value=team)),
        litellm_projecttable=SimpleNamespace(find_unique=AsyncMock(side_effect=find_project)),
        litellm_modelaccessgroupbudgettable=SimpleNamespace(find_many=AsyncMock(side_effect=find_groups)),
    )
    cache = SimpleNamespace(async_get_cache=AsyncMock(return_value=None), async_set_cache=AsyncMock())
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(live, "collect_matched_model_access_groups", AsyncMock(return_value=("group",)))

    assert await live._managed_member_budget(UserAPIKeyAuth(api_key="owner", project_id="project")) is True
    assert await live._managed_member_budget(UserAPIKeyAuth(api_key="owner", team_id="team")) is True
    assert (
        await live._managed_member_budget(
            UserAPIKeyAuth(api_key="owner", matched_model_access_groups=["voice-group"]), model="backend"
        )
        is True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_budget, blocked", [(None, False), (0, False), (1, True)])
async def test_managed_budget_uses_the_delegated_model_group(monkeypatch, backend_budget, blocked):
    from litellm.proxy import proxy_server

    rows = [
        SimpleNamespace(access_group_name="voice-group", litellm_budget_table=SimpleNamespace(max_budget=1)),
        SimpleNamespace(
            access_group_name="backend-group", litellm_budget_table=SimpleNamespace(max_budget=backend_budget)
        ),
    ]
    db = SimpleNamespace(
        litellm_modelaccessgroupbudgettable=SimpleNamespace(find_many=AsyncMock(return_value=rows)),
    )
    cache = SimpleNamespace(async_get_cache=AsyncMock(return_value=None), async_set_cache=AsyncMock())
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "llm_router", SimpleNamespace())
    monkeypatch.setattr(live, "collect_matched_model_access_groups", AsyncMock(return_value=("backend-group",)))

    auth = UserAPIKeyAuth(api_key="owner", models=["voice", "backend-group"])
    assert await live._managed_member_budget(auth, model="backend") is blocked


@pytest.mark.asyncio
async def test_responses_delegation_fails_closed_when_inherited_org_lookup_fails(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import auth_checks

    team = LiteLLM_TeamTable(team_id="team", organization_id="org", models=["*"])
    group = SimpleNamespace(
        access_group_name="backend-group",
        litellm_budget_table=SimpleNamespace(max_budget=1),
    )
    db = SimpleNamespace(
        litellm_teamtable=SimpleNamespace(find_unique=AsyncMock(return_value=team)),
        litellm_modelaccessgroupbudgettable=SimpleNamespace(find_many=AsyncMock(return_value=[group])),
    )
    cache = SimpleNamespace(async_get_cache=AsyncMock(return_value=None), async_set_cache=AsyncMock())
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", None)
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        SimpleNamespace(get_model_access_groups=lambda model_name, team_id=None: {"backend-group"}),
    )
    monkeypatch.setattr(
        auth_checks,
        "get_org_object",
        AsyncMock(side_effect=RuntimeError("organization lookup unavailable")),
    )
    monkeypatch.setattr(live, "_authorize", AsyncMock())

    with pytest.raises(HTTPException) as rejected:
        await live._authorize_delegation(
            {"session": {"delegation": {"type": "responses", "responses": {"model": "backend"}}}},
            UserAPIKeyAuth(api_key="owner", models=["*"], team_id="team"),
        )

    assert rejected.value.status_code == 503


@pytest.mark.asyncio
async def test_authorize_checks_organization_inherited_from_team(monkeypatch):
    from litellm.proxy import proxy_server

    team = SimpleNamespace(organization_id="org")
    org = SimpleNamespace(models=["backend"])
    team_loader = AsyncMock(return_value=team)
    org_loader = AsyncMock(return_value=org)
    org_check = Mock()
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", object())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", None)
    monkeypatch.setattr(proxy_server, "llm_model_list", [])
    monkeypatch.setattr(proxy_server, "llm_router", object())
    monkeypatch.setattr(live, "can_key_call_resolved_model", AsyncMock())
    monkeypatch.setattr(live, "get_team_object", team_loader, raising=False)
    monkeypatch.setattr(live, "get_org_object", org_loader)
    monkeypatch.setattr(live, "can_org_access_model", org_check)

    await live._authorize("backend", UserAPIKeyAuth(api_key="owner", team_id="team"))

    team_loader.assert_awaited_once()
    org_loader.assert_awaited_once()
    org_check.assert_called_once_with(model="backend", org_object=org, llm_router=proxy_server.llm_router)


@pytest.mark.asyncio
async def test_managed_budget_fails_closed_when_member_scope_lookup_fails_after_snapshot(monkeypatch):
    from litellm.proxy import proxy_server

    team = LiteLLM_TeamTable(team_id="team", models=["*"])
    membership = LiteLLM_TeamMembership(
        user_id="member",
        team_id="team",
        litellm_budget_table=LiteLLM_BudgetTable(allowed_models=["backend-group"]),
    )
    group = SimpleNamespace(
        access_group_name="backend-group",
        litellm_budget_table=SimpleNamespace(max_budget=1),
    )
    db = SimpleNamespace(
        litellm_teammembership=SimpleNamespace(
            find_unique=AsyncMock(side_effect=[membership, RuntimeError("membership lookup unavailable")])
        ),
        litellm_teamtable=SimpleNamespace(find_unique=AsyncMock(return_value=team)),
        litellm_modelaccessgroupbudgettable=SimpleNamespace(find_many=AsyncMock(return_value=[group])),
    )
    cache = SimpleNamespace(async_get_cache=AsyncMock(return_value=None), async_set_cache=AsyncMock())
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", None)
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        SimpleNamespace(get_model_access_groups=lambda model_name, team_id=None: {"backend-group"}),
    )
    monkeypatch.setattr(live, "_authorize", AsyncMock())

    with pytest.raises(HTTPException) as rejected:
        await live._authorize_delegation(
            {"session": {"delegation": {"type": "responses", "responses": {"model": "backend"}}}},
            UserAPIKeyAuth(api_key="owner", models=["*"], team_id="team", user_id="member"),
        )

    assert rejected.value.status_code == 503


def test_json_conversion_rejects_unsupported_parent_container(monkeypatch):
    class _ForeignMapping:
        def validate_python(self, value):
            return {1: "coerced"}

    monkeypatch.setattr(live, "_MAPPING", _ForeignMapping())
    with pytest.raises(TypeError, match="Invalid Live JSON conversion target"):
        live._json_value(MappingProxyType({"nested": "value"}))


def test_rewrite_session_ids_serializes_non_object_events_without_touching_ids():
    assert live.rewrite_session_ids(["live", {"id": "raw"}], "raw", "public") == ["live", {"id": "raw"}]
    assert live.rewrite_session_ids("public", "raw", "public") == "public"


def test_owner_requires_authenticated_api_key():
    with pytest.raises(HTTPException) as rejected:
        live._owner(UserAPIKeyAuth())
    assert rejected.value.status_code == 403
    assert rejected.value.detail == "Live sessions require an authenticated API key"


def _streamed_request(chunks: list[bytes]) -> Request:
    pending = list(chunks)

    async def receive():
        return {"type": "http.request", "body": pending.pop(0), "more_body": bool(pending)}

    return Request({"type": "http", "method": "POST", "headers": [], "query_string": b""}, receive=receive)


@pytest.mark.asyncio
async def test_body_rejects_streams_larger_than_the_offer_limit():
    with pytest.raises(HTTPException) as rejected:
        await live._body(_streamed_request([b"a" * (8 * 1024 * 1024), b"b"]))
    assert rejected.value.status_code == 413
    assert rejected.value.detail == "Live request exceeds the 8 MiB limit"


@pytest.mark.asyncio
async def test_body_rejects_json_that_is_not_an_object():
    with pytest.raises(HTTPException) as rejected:
        await live._body(_streamed_request([b"[1,2]"]))
    assert rejected.value.status_code == 400
    assert rejected.value.detail == "Expected a JSON object"


def test_session_model_requires_object_and_model():
    with pytest.raises(HTTPException) as not_object:
        live._session_model({"session": "voice"})
    assert not_object.value.status_code == 400 and not_object.value.detail == "session must be a JSON object"
    with pytest.raises(HTTPException) as no_model:
        live._session_model({"session": {}})
    assert no_model.value.status_code == 400 and no_model.value.detail == "session.model is required"


@pytest.mark.asyncio
async def test_team_organization_lookup_maps_failures_to_service_unavailable(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", object())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", object())
    monkeypatch.setattr(live, "get_team_object", AsyncMock(side_effect=RuntimeError("database unavailable")))

    with pytest.raises(HTTPException) as rejected:
        await live._live_organization_id(UserAPIKeyAuth(api_key="owner", team_id="team"))
    assert rejected.value.status_code == 503
    assert rejected.value.detail == "Could not verify Live team organization model access"


@pytest.mark.asyncio
async def test_direct_user_authorization_fails_closed_when_user_lookup_fails(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "llm_model_list", [])
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", object())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", object())
    monkeypatch.setattr(live, "can_key_call_resolved_model", AsyncMock())
    monkeypatch.setattr(live, "get_user_object", AsyncMock(side_effect=RuntimeError("database unavailable")))

    with pytest.raises(HTTPException) as rejected:
        await live._authorize("voice", UserAPIKeyAuth(api_key="owner", user_id="user"))
    assert rejected.value.status_code == 503
    assert rejected.value.detail == "Could not verify Live user model access"


@pytest.mark.asyncio
async def test_direct_org_authorization_fails_closed_when_org_lookup_fails(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "llm_model_list", [])
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", object())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", object())
    monkeypatch.setattr(live, "can_key_call_resolved_model", AsyncMock())
    monkeypatch.setattr(live, "get_org_object", AsyncMock(side_effect=RuntimeError("database unavailable")))

    with pytest.raises(HTTPException) as rejected:
        await live._authorize("voice", UserAPIKeyAuth(api_key="owner", org_id="org-1"))
    assert rejected.value.status_code == 503
    assert rejected.value.detail == "Could not verify Live organization model access"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth,lookup,expected",
    [
        (UserAPIKeyAuth(api_key="owner", user_id="user"), "get_user_object", "Could not verify Live user model access"),
        (
            UserAPIKeyAuth(api_key="owner", org_id="org-1"),
            "get_org_object",
            "Could not verify Live organization model access",
        ),
    ],
    ids=["user", "organization"],
)
async def test_authorization_fails_closed_when_the_principal_row_is_missing(monkeypatch, auth, lookup, expected):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "llm_model_list", [])
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", object())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", object())
    monkeypatch.setattr(live, "can_key_call_resolved_model", AsyncMock())
    monkeypatch.setattr(live, "can_user_call_model", AsyncMock())
    monkeypatch.setattr(live, "can_org_access_model", Mock())
    monkeypatch.setattr(live, lookup, AsyncMock(return_value=None))

    with pytest.raises(HTTPException) as rejected:
        await live._authorize("voice", auth)
    assert rejected.value.status_code == 503
    assert rejected.value.detail == expected


@pytest.mark.asyncio
async def test_deployment_requires_router_and_supported_provider(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", None)
    with pytest.raises(HTTPException) as without_router:
        await live._deployment("voice", {})
    assert without_router.value.status_code == 503
    assert without_router.value.detail == "Live requires a configured model deployment"

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        SimpleNamespace(
            async_get_available_deployment=AsyncMock(
                return_value={
                    "litellm_params": {"model": "bedrock/voice"},
                    "model_info": {"id": "deployment-a"},
                }
            ),
            async_routing_strategy_pre_call_checks=AsyncMock(),
        ),
    )
    with pytest.raises(HTTPException) as wrong_provider:
        await live._deployment("voice", {})
    assert wrong_provider.value.status_code == 400
    assert "OpenAI or ChatGPT" in wrong_provider.value.detail


@pytest.mark.parametrize("payload", [{}, {"session": {"id": 5}}, {"session": None}])
def test_session_id_requires_upstream_string_id(payload):
    with pytest.raises(HTTPException) as rejected:
        live._session_id(payload)
    assert rejected.value.status_code == 502
    assert rejected.value.detail == "Upstream did not return a Live session ID"


@pytest.mark.asyncio
async def test_live_team_membership_prefers_reservation_cache_and_sentinel(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
    from litellm.proxy.common_utils.user_api_key_cache import NO_TEAM_MEMBERSHIP_SENTINEL

    membership = LiteLLM_TeamMembership(user_id="user", team_id="team")
    auth = UserAPIKeyAuth(api_key="owner", user_id="user", team_id="team")
    monkeypatch.setattr(
        proxy_server,
        "user_api_key_cache",
        SimpleNamespace(
            async_get_cache=AsyncMock(return_value=CacheCodec.serialize(membership, model_type=LiteLLM_TeamMembership))
        ),
    )
    restored = await live._live_team_membership(auth)
    assert restored is not None and restored.user_id == "user" and restored.team_id == "team"

    monkeypatch.setattr(
        proxy_server,
        "user_api_key_cache",
        SimpleNamespace(async_get_cache=AsyncMock(return_value=NO_TEAM_MEMBERSHIP_SENTINEL)),
    )
    assert await live._live_team_membership(auth) is None


@pytest.mark.asyncio
async def test_live_team_uses_team_cache_before_database(monkeypatch):
    from litellm.proxy import proxy_server

    team = SimpleNamespace(team_id="team", models=["*"])
    monkeypatch.setattr(
        proxy_server, "user_api_key_cache", SimpleNamespace(async_get_cache=AsyncMock(return_value=team))
    )
    assert await live._live_team(UserAPIKeyAuth(api_key="owner", team_id="team")) is team


@pytest.mark.parametrize(
    "team",
    [
        SimpleNamespace(budget_limits=3, rpm_limit=None, tpm_limit=None, max_budget=None, model_max_budget=None),
        SimpleNamespace(budget_limits=None, rpm_limit=5, tpm_limit=None, max_budget=None, model_max_budget=None),
        SimpleNamespace(
            budget_limits=None, rpm_limit=None, tpm_limit=None, max_budget=None, model_max_budget={"voice": 1}
        ),
    ],
    ids=["scalar-windows", "scalar-rpm", "model-max-budget"],
)
def test_team_budget_fields_short_circuit_before_metadata_scan(team):
    assert live._live_team_budget_configured(UserAPIKeyAuth(api_key="owner"), team) is True


@pytest.mark.parametrize(
    "value,zero_is_limit,expected",
    [
        (None, False, False),
        ({"max_budget": 0}, True, True),
        ({"max_budget": 0}, False, False),
        ({"max_budget": "unlimited"}, False, True),
        ({"rpm_limit": 2}, False, True),
    ],
    ids=["missing", "zero-as-limit", "zero-unlimited", "non-numeric-limit", "other-limit"],
)
def test_live_budget_configured_separates_zero_from_non_numeric_limits(value, zero_is_limit, expected):
    assert live._live_budget_configured(value, zero_is_limit=zero_is_limit) is expected


@pytest.mark.asyncio
async def test_live_default_budget_uses_cached_team_member_budget(monkeypatch):
    from litellm.proxy import proxy_server

    budget = LiteLLM_BudgetTable(max_budget=1)
    monkeypatch.setattr(
        proxy_server, "user_api_key_cache", SimpleNamespace(async_get_cache=AsyncMock(return_value=budget))
    )
    team = SimpleNamespace(metadata={"team_member_budget_id": "budget-1"})
    auth = UserAPIKeyAuth(api_key="owner", user_id="user", team_id="team")
    assert await live._live_default_budget(auth, team) is budget


@pytest.mark.asyncio
async def test_live_project_uses_cache_or_reports_missing_row(monkeypatch):
    from litellm.proxy import proxy_server

    project = SimpleNamespace(project_id="project-1")
    monkeypatch.setattr(
        proxy_server, "user_api_key_cache", SimpleNamespace(async_get_cache=AsyncMock(return_value=project))
    )
    auth = UserAPIKeyAuth(api_key="owner", project_id="project-1")
    assert await live._live_project(auth) is project

    monkeypatch.setattr(
        proxy_server, "user_api_key_cache", SimpleNamespace(async_get_cache=AsyncMock(return_value=None))
    )
    monkeypatch.setattr(
        live,
        "ProjectRepository",
        lambda client: SimpleNamespace(table=SimpleNamespace(find_unique=AsyncMock(return_value=None))),
    )
    assert await live._live_project(auth) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "project, managed",
    [
        (
            SimpleNamespace(
                litellm_budget_table=None,
                budget_id=None,
                model_rpm_limit={"voice": 5},
                model_tpm_limit=None,
                metadata=None,
            ),
            True,
        ),
        (
            SimpleNamespace(
                litellm_budget_table=None,
                budget_id=None,
                model_rpm_limit=None,
                model_tpm_limit=None,
                metadata={"rpm_limit": 5},
            ),
            True,
        ),
        (
            SimpleNamespace(
                litellm_budget_table=None, budget_id=None, model_rpm_limit=None, model_tpm_limit=None, metadata=None
            ),
            False,
        ),
    ],
    ids=["model-rate-limit", "metadata-limit", "nothing"],
)
async def test_project_budget_falls_back_to_rate_limits_and_metadata(monkeypatch, project, managed):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", object())
    assert await live._live_project_budget_configured(UserAPIKeyAuth(api_key="owner"), project) is managed


@pytest.mark.asyncio
async def test_model_group_budget_requires_model_name():
    assert await live._live_model_group_budget_configured(UserAPIKeyAuth(api_key="owner"), None, None, None) is False


@pytest.mark.asyncio
async def test_managed_member_budget_fails_closed_without_database(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "llm_router", object())
    with pytest.raises(HTTPException) as rejected:
        await live._managed_member_budget(UserAPIKeyAuth(api_key="owner", team_id="team"), "voice")
    assert rejected.value.status_code == 503
    assert rejected.value.detail == "Could not verify Live managed budgets"


@pytest.mark.asyncio
async def test_backend_delegation_without_responses_contract_requires_named_model(monkeypatch):
    monkeypatch.setattr(live, "_managed_member_budget", AsyncMock(return_value=False))

    await live._authorize_delegation(
        {"session": {"delegation": {"type": "backend"}}},
        UserAPIKeyAuth(api_key="owner"),
    )

    with pytest.raises(HTTPException) as rejected:
        await live._authorize_delegation(
            {"type": "session.start", "session": {"delegation": {"type": "responses"}}},
            UserAPIKeyAuth(api_key="owner", models=["voice"]),
        )
    assert rejected.value.status_code == 400
    assert "explicit authorized delegation.responses.model" in rejected.value.detail


@pytest.mark.asyncio
async def test_precall_aborts_when_transferred_quota_lease_cannot_renew(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.hooks.parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3

    lease = SimpleNamespace(start=Mock(), renew=AsyncMock(return_value=False), close=AsyncMock())
    limiter = Mock(spec=_PROXY_MaxParallelRequestsHandler_v3)
    limiter.transfer_realtime_call_slot = Mock(return_value=lease)
    limiter.async_post_call_failure_hook = AsyncMock()
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", SimpleNamespace(get_proxy_hook=lambda _: limiter))
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(live, "_authorize", AsyncMock())
    monkeypatch.setattr(live, "process_codex_request", AsyncMock(return_value=({"model": "voice"}, Mock())))
    request = live._request(Request({"type": "http", "headers": []}), {"session": {"model": "voice"}})

    with pytest.raises(HTTPException) as lost:
        async with live._precall(request, UserAPIKeyAuth(api_key="owner"), "voice"):
            pytest.fail("session must not start when the quota reservation is lost")

    assert lost.value.status_code == 503 and "quota reservation was lost" in lost.value.detail
    lease.start.assert_called_once()
    lease.close.assert_awaited_once()
    limiter.async_post_call_failure_hook.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervise_starts_observer_under_isolated_request_stash(monkeypatch):
    start = AsyncMock(return_value="stream")
    monkeypatch.setattr(live, "_start_supervisor", start)
    request = Request({"type": "http", "headers": []})
    auth = UserAPIKeyAuth(api_key="owner")
    source = handle()

    assert await live._supervise(request, source, auth, None, None) == "stream"
    assert (
        start.await_args.args[0] is request and start.await_args.args[1] is source and start.await_args.args[2] is auth
    )


@pytest.mark.asyncio
async def test_observer_frontend_swallows_traffic_and_hangup_checks_upstream_status(monkeypatch):
    from starlette.websockets import WebSocketState

    connection = SimpleNamespace(close=AsyncMock())
    transport = SimpleNamespace(
        connect=AsyncMock(return_value=connection),
        request=AsyncMock(
            return_value=httpx.Response(
                502, request=httpx.Request("POST", "http://upstream.test/live/sessions/sess_upstream/hangup")
            )
        ),
    )
    monkeypatch.setattr(live, "LiveTransport", Mock(return_value=transport))
    monkeypatch.setattr(
        live, "process_codex_request", AsyncMock(return_value=({"model": "voice"}, Mock(litellm_params={})))
    )
    monkeypatch.setattr(live, "CALL_SUPERVISORS", SimpleNamespace(start=AsyncMock()))
    supervisor_init = Mock()
    monkeypatch.setattr(live, "CallSupervisor", supervisor_init)
    stream_cls = Mock()
    monkeypatch.setattr(live, "RealTimeStreaming", stream_cls)
    request = Request({"type": "http", "headers": []})

    await live._start_supervisor(request, handle(), UserAPIKeyAuth(api_key="owner"), None)

    frontend = stream_cls.call_args.args[0]
    frontend.client_state = WebSocketState.CONNECTED
    frontend.application_state = WebSocketState.CONNECTED
    assert await frontend.receive() == {"type": "websocket.disconnect", "code": 1000}
    assert await frontend.send({"type": "websocket.send", "text": "tick"}) is None
    hangup = supervisor_init.call_args.args[4]
    with pytest.raises(httpx.HTTPStatusError):
        await hangup()
    transport.request.assert_awaited_once_with("POST", "live/sessions/sess_upstream/hangup")


@pytest.mark.asyncio
async def test_observer_startup_failure_still_hangs_up_and_keeps_the_original_error(monkeypatch):
    from litellm.proxy.spend_tracking import budget_reservation

    connection = SimpleNamespace(close=AsyncMock())
    transport = SimpleNamespace(
        connect=AsyncMock(return_value=connection),
        request=AsyncMock(
            return_value=httpx.Response(
                200, request=httpx.Request("POST", "http://upstream.test/live/sessions/sess_upstream/hangup")
            )
        ),
    )
    monkeypatch.setattr(live, "LiveTransport", Mock(return_value=transport))
    monkeypatch.setattr(
        live, "process_codex_request", AsyncMock(return_value=({"model": "voice"}, Mock(litellm_params={})))
    )
    monkeypatch.setattr(live, "CALL_SUPERVISORS", SimpleNamespace(start=AsyncMock()))
    monkeypatch.setattr(live, "RealTimeStreaming", Mock())
    monkeypatch.setattr(live, "CallSupervisor", Mock(side_effect=RuntimeError("supervisor refused the call")))
    invalidate = AsyncMock()
    monkeypatch.setattr(budget_reservation, "invalidate_budget_reservation_counters", invalidate)
    request = Request({"type": "http", "headers": []})

    with pytest.raises(RuntimeError, match="supervisor refused the call"):
        await live._start_supervisor(request, handle(), UserAPIKeyAuth(api_key="owner"), None)

    transport.request.assert_awaited_once_with("POST", "live/sessions/sess_upstream/hangup")
    invalidate.assert_not_awaited()
    connection.close.assert_awaited_once()


def test_admin_sip_accept_rejects_model_mismatch_and_passes_upstream_errors_through(route_client, monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LitellmUserRoles

    route_client.auth.user_role = LitellmUserRoles.PROXY_ADMIN
    monkeypatch.setattr(proxy_server, "llm_model_list", [{"model_name": "voice"}])
    mismatch = route_client.client.post(
        "/v1/live/sessions/sess_incoming/accept",
        json={"session": {"model": "other", "type": "live"}},
        headers={"x-litellm-live-model": "voice"},
    )
    assert mismatch.status_code == 400 and "must match" in mismatch.json()["detail"]
    route_client.transport.request.assert_not_awaited()

    route_client.transport.request.return_value = httpx.Response(503, json={"error": "gateway down"})
    failed = route_client.client.post(
        "/v1/live/sessions/sess_incoming/accept",
        json={"session": {"model": "voice", "type": "live"}},
        headers={"x-litellm-live-model": "voice"},
    )
    assert failed.status_code == 503 and "x-litellm-live-session-id" not in failed.headers
    route_client.supervised.assert_not_awaited()


@pytest.mark.asyncio
async def test_public_sideband_rejects_restart_and_model_change_then_rewrites_ids():
    websocket = SimpleNamespace(
        receive_text=AsyncMock(return_value=json.dumps({"type": "session.start"})),
        send_text=AsyncMock(),
        close=AsyncMock(),
        scope={},
        headers=Mock(),
    )
    public = live._PublicSocket(websocket, handle(), "public", UserAPIKeyAuth(api_key="owner"))

    with pytest.raises(HTTPException) as restart:
        await public.receive_text()
    assert restart.value.status_code == 400 and restart.value.detail == "Session has already started"

    websocket.receive_text.return_value = json.dumps({"type": "session.update", "session": {"model": "other"}})
    with pytest.raises(HTTPException) as model:
        await public.receive_text()
    assert model.value.status_code == 400 and model.value.detail == "Session model cannot change"

    websocket.receive_text.return_value = json.dumps({"type": "custom", "session_id": "public"})
    assert json.loads(await public.receive_text()) == {"type": "custom", "session_id": "sess_upstream"}


def test_startup_events_overflow_fails_closed():
    events = live._StartupEvents()
    for _ in range(128):
        events.store({"type": "info"})
    with pytest.raises(HTTPException) as overflowed:
        events.store({"type": "info"})
    assert overflowed.value.status_code == 502


def test_websocket_requires_api_key_then_session_start(route_client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as anonymous:
        with route_client.client.websocket_connect("/v1/live/sessions") as ws:
            ws.receive_json()
    assert anonymous.value.code == 1008

    with route_client.client.websocket_connect("/v1/live/sessions", headers={"Authorization": "Bearer owner"}) as ws:
        ws.send_json({"type": "ping"})
        with pytest.raises(WebSocketDisconnect) as wrong_first:
            ws.receive_json()
    assert wrong_first.value.code == 1008
    route_client.transport.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_attached_socket_authorizes_policy_and_reuses_source_without_start(route_client, monkeypatch):
    from starlette.websockets import WebSocket

    streams: list = []

    class AttachedStream:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.bidirectional_forward = AsyncMock()
            streams.append(self)

    backend = SimpleNamespace(send=AsyncMock(), close=AsyncMock(), recv=AsyncMock())
    route_client.transport.connect = AsyncMock(return_value=backend)
    monkeypatch.setattr(live, "RealTimeStreaming", AttachedStream)
    authorize = AsyncMock()
    monkeypatch.setattr(live, "_authorize_delegation", authorize)
    token = live.encode_session(handle(model_id="deployment-a"))
    inbound = iter([{"type": "websocket.connect"}, {"type": "websocket.disconnect", "code": 1000}])
    sent: list = []

    async def receive():
        return next(inbound)

    async def send(message):
        sent.append(message)

    websocket = WebSocket(
        {
            "type": "websocket",
            "path": f"/v1/live/sessions/{token}/attach",
            "query_string": b"",
            "headers": [(b"authorization", b"Bearer owner")],
            "scheme": "ws",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "subprotocols": [],
        },
        receive,
        send,
    )

    await live.websocket_live_session(websocket, token)

    assert sent == [{"type": "websocket.accept", "subprotocol": None, "headers": []}]
    authorize.assert_awaited_once()
    assert authorize.await_args.args[1] is route_client.auth
    route_client.transport.connect.assert_awaited_once_with("live/sessions/sess_upstream/attach")
    backend.send.assert_not_awaited()
    backend.close.assert_awaited_once()
    frontend = streams[0].args[0]
    assert frontend.public_id == token and frontend.handle.session_id == "sess_upstream" and frontend.observer is None
    route_client.supervised.assert_not_awaited()
    streams[0].bidirectional_forward.assert_awaited_once()


def test_websocket_connection_failure_closes_with_internal_error(route_client):
    from starlette.websockets import WebSocketDisconnect

    route_client.transport.connect = AsyncMock(return_value=None)
    with route_client.client.websocket_connect("/v1/live/sessions", headers={"Authorization": "Bearer owner"}) as ws:
        ws.send_json({"type": "session.start", "session": {"model": "voice"}})
        with pytest.raises(WebSocketDisconnect) as internal:
            ws.receive_json()
    assert internal.value.code == 1011
    route_client.transport.request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["rejected", "crashed"])
async def test_websocket_close_after_asgi_completion_is_swallowed(monkeypatch, stage):
    from fastapi import WebSocket

    class CompletedWebSocket(WebSocket):
        async def close(self, code=1000, reason=None):
            raise RuntimeError("ASGI send channel already completed")

    headers = [] if stage == "rejected" else [(b"authorization", b"Bearer owner")]
    sent = []

    async def receive():
        return {"type": "websocket.disconnect"}

    async def send(message):
        sent.append(message)

    websocket = CompletedWebSocket(
        {
            "type": "websocket",
            "path": "/v1/live/sessions",
            "query_string": b"",
            "headers": headers,
            "scheme": "ws",
            "server": ("localhost", 4000),
        },
        receive,
        send,
    )
    if stage == "crashed":
        monkeypatch.setattr(live, "_auth", AsyncMock(side_effect=ConnectionError("redis down")))

    await live.websocket_live_session(websocket)

    assert sent == []

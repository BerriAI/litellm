import json
import time
from contextlib import asynccontextmanager
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from litellm.llms.chatgpt.live import LiveDeployment
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.realtime_endpoints import live


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-live-encryption-key")


def handle(owner="owner"):
    return live._new_handle(
        "sess_upstream",
        "voice",
        LiveDeployment(model="gpt-live"),
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
    source = handle()
    source = source.model_copy(update={"deployment": {**source.deployment, "model_id": "deployment-a"}})
    token = live.encode_session(source)
    body = {"session": {}, "transport": {"type": "webrtc", "sdp": "offer"}}
    result = route_client.client.post(f"/v1/live/sessions/{token}/fork", json=body)
    assert result.status_code == 201
    assert route_client.transport.request.await_args.args == ("POST", "live/sessions/sess_upstream/fork")
    assert route_client.transport.request.await_args.kwargs["body"] == body
    assert route_client.factory.call_args.args[0].model_id == "deployment-a"
    route_client.selected.assert_not_awaited()


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
        {"model_max_budget": {"backend": 1}},
        {"team_tpm_limit": 10},
        {"team_metadata": {"model_rpm_limit": {"backend": 1}}},
    ],
)
async def test_managed_delegation_fails_closed_for_unenforceable_constraints(limits):
    body = {"session": {"delegation": {"type": "responses", "responses": {"model": "backend"}}}}
    with pytest.raises(HTTPException) as rejected:
        await live._authorize_delegation(body, UserAPIKeyAuth(api_key="owner", **limits))
    assert rejected.value.status_code == 400
    assert "client delegation" in rejected.value.detail


@pytest.mark.asyncio
async def test_restricted_webrtc_cannot_change_managed_model_outside_proxy(monkeypatch):
    monkeypatch.setattr(live, "_authorize", AsyncMock())
    body = {
        "session": {"delegation": {"type": "responses", "responses": {"model": "backend"}}},
        "transport": {"type": "webrtc", "sdp": "offer"},
    }
    auth = UserAPIKeyAuth(api_key="owner", models=["voice", "backend"])
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
@pytest.mark.parametrize(
    "startup_policy", [{}, {"delegation": {"type": "responses", "responses": {"model": "allowed"}}}]
)
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


def test_restricted_fork_can_explicitly_select_client_delegation(route_client):
    route_client.auth.models = ["voice"]
    body = {"session": {"delegation": {"type": "client"}}}
    token = live.encode_session(handle())
    route_client.transport.request.return_value = httpx.Response(
        200, json={"session": {"id": "sess_fork"}, "transport": {"type": "webrtc", "sdp": "answer"}}
    )
    response = route_client.client.post(f"/v1/live/sessions/{token}/fork", json=body)
    assert response.status_code == 200
    assert response.json()["transport"]["sdp"] == "answer"
    route_client.transport.request.assert_awaited_once_with("POST", "live/sessions/sess_upstream/fork", body=body)


@pytest.mark.parametrize("protocol", ["http", "websocket"])
@pytest.mark.parametrize("limits", [{"rpm_limit": 10}, {"tpm_limit": 100}, {"model_max_budget": {"backend": 1}}])
def test_fork_with_new_limits_cannot_trust_old_client_policy(route_client, protocol, limits):
    from starlette.websockets import WebSocketDisconnect

    # An unrestricted source could have switched to managed delegation after its handle was issued.
    for key, value in limits.items():
        setattr(route_client.auth, key, value)
    token = live.encode_session(handle())
    path = f"/v1/live/sessions/{token}/fork"
    if protocol == "http":
        response = route_client.client.post(path, json={"session": {}})
        assert response.status_code == 400
    else:
        with route_client.client.websocket_connect(path, headers={"Authorization": "Bearer owner"}) as ws:
            ws.send_json({"type": "session.start", "session": {}})
            with pytest.raises(WebSocketDisconnect) as rejected:
                ws.receive_json()
            assert rejected.value.code == 1008
    route_client.factory.assert_not_called()


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

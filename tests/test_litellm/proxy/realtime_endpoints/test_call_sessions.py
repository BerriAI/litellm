import hashlib
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, WebSocket

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.realtime_endpoints import call_sessions as codex
from litellm.llms.chatgpt.codex import CodexRealtimeCall
from litellm.proxy.realtime_endpoints.call_sessions import decode_call, encode_call


@pytest.mark.asyncio
@pytest.mark.parametrize("logged_success", [False, True])
@pytest.mark.parametrize("disconnect_error", [False, True])
async def test_sideband_preserves_pending_cost_reconciliation(monkeypatch, logged_success, disconnect_error):
    import litellm
    from unittest.mock import AsyncMock
    from litellm.litellm_core_utils.realtime_streaming import REALTIME_SESSION_SUCCESS_LOGGED_KEY

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-only-salt-for-codex-realtime")
    call = CodexRealtimeCall(call_id="rtc_test", model="gpt-live-1-codex", alias="voice",
        owner=hashlib.sha256(b"Bearer owner").hexdigest(), expires_at=time.time()+300)
    auth = UserAPIKeyAuth()
    auth.budget_reservation = {"reserved_cost": 0.55, "input_cost": 0.0, "finalized": False, "entries": []}
    logger = SimpleNamespace(model_call_details={})
    monkeypatch.setattr(codex, "can_key_call_resolved_model", AsyncMock())
    monkeypatch.setattr(codex, "process_codex_request", AsyncMock(return_value=({}, logger)))

    async def forward(**kwargs):
        if logged_success:
            logger.model_call_details[REALTIME_SESSION_SUCCESS_LOGGED_KEY] = True
        if disconnect_error:
            raise RuntimeError("Backend disconnected")

    monkeypatch.setattr(litellm, "_arealtime", forward)
    websocket = WebSocket({"type": "websocket", "path": "/v1/live/opaque", "query_string": b"",
        "headers": [(b"authorization", b"Bearer owner")]},
        AsyncMock(return_value={"type": "websocket.connect"}), AsyncMock())
    if disconnect_error:
        with pytest.raises(RuntimeError, match="Backend disconnected"):
            await codex.codex_realtime_sideband(websocket, encode_call(call), auth)
    else:
        await codex.codex_realtime_sideband(websocket, encode_call(call), auth)
    assert auth.budget_reservation["finalized"] is not logged_success


def test_sideband_token_binds_owner_and_model(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-only-salt-for-codex-realtime")
    call = CodexRealtimeCall(
        call_id="rtc_test",
        model="gpt-live-1-codex",
        alias="gpt-live-1-codex",
        owner=hashlib.sha256(b"Bearer test-owner").hexdigest(),
        expires_at=time.time() + 300,
    )
    token = encode_call(call)
    assert "/" not in token
    assert decode_call(token, "Bearer test-owner") == call
    with pytest.raises(HTTPException) as error:
        decode_call(token, "Bearer different-owner")
    assert error.value.status_code == 403
    with pytest.raises(HTTPException):
        decode_call(token[:30] + "tampered" + token[30:], "Bearer test-owner")


def test_sideband_rejects_expired_token(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-only-salt-for-codex-realtime")
    call = CodexRealtimeCall(
        call_id="rtc_test",
        model="gpt-realtime-1.5",
        alias="gpt-realtime-1.5",
        owner=hashlib.sha256(b"Bearer test-owner").hexdigest(),
        expires_at=time.time() - 1,
    )
    with pytest.raises(HTTPException):
        decode_call(encode_call(call), "Bearer test-owner")


@pytest.mark.parametrize("token", ["", "rtc_other", "rtc_litellm_%%%%", "rtc_litellm_a"])
def test_sideband_rejects_malformed_tokens(token):
    with pytest.raises(HTTPException):
        decode_call(token, "Bearer test-owner")


@pytest.mark.asyncio
async def test_sideband_rejects_revoked_model_access(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-only-salt-for-codex-realtime")
    call = CodexRealtimeCall(
        call_id="rtc_test",
        model="gpt-live-1-codex",
        alias="voice",
        owner=hashlib.sha256(b"Bearer test-owner").hexdigest(),
        expires_at=time.time() + 300,
    )
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    async def deny_model(**kwargs):
        raise ProxyException("Model access revoked", "auth_error", "model", 403)

    monkeypatch.setattr(codex, "can_key_call_resolved_model", deny_model)
    websocket = WebSocket(
        {"type": "websocket", "headers": [(b"authorization", b"Bearer test-owner")]}, receive, send
    )
    await codex.codex_realtime_sideband(websocket, encode_call(call), UserAPIKeyAuth())
    assert sent == [{"type": "websocket.close", "code": 1008, "reason": "Invalid realtime call"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("call_id", ["rtc_raw", "", "rtc_litellm_invalid"])
async def test_realtime_endpoint_rejects_untrusted_call_ids(monkeypatch, call_id):
    from unittest.mock import AsyncMock
    from fastapi import WebSocket
    from litellm.proxy import proxy_server as server
    from litellm.proxy._types import UserAPIKeyAuth

    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    route = AsyncMock()
    monkeypatch.setattr(server, "route_request", route)
    websocket = WebSocket({"type": "websocket", "headers": [], "query_string": b""}, receive, send)
    await server.realtime_websocket_endpoint(
        websocket, model="gpt-realtime-1.5", call_id=call_id,
        intent=None, guardrails=None, user_api_key_dict=UserAPIKeyAuth()
    )
    assert sent == [{"type": "websocket.close", "code": 1008, "reason": "Invalid realtime call"}]
    route.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("multipart", [False, True])
@pytest.mark.parametrize("credential", ["authorization", "api-key", "subprotocol"])
async def test_offer_exchange_wraps_call_and_filters_client_headers(monkeypatch, multipart, credential):
    import json
    from unittest.mock import AsyncMock

    import httpx
    from fastapi import Request, WebSocket
    from litellm.proxy import common_request_processing, proxy_server
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.realtime_endpoints import call_sessions as codex
    import litellm

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-only-salt-for-codex-realtime")
    session = {"model": "voice-alias", "audio": {"output": {"voice": "sol"}}}
    if multipart:
        body_request = httpx.Request("POST", "http://test/v1/realtime/calls", files={
            "sdp": (None, "v=0\r\n"), "session": (None, json.dumps(session))
        })
    else:
        body_request = httpx.Request("POST", "http://test/v1/realtime/calls", json={"sdp": "v=0\r\n", "session": session})
    body = body_request.read()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/v1/realtime/calls",
        "query_string": b"intent=quicksilver&architecture=avas&untrusted=bad",
        "headers": [(b"content-type", body_request.headers["content-type"].encode()),
                    (b"authorization", b"Bearer owner"), (b"openai-alpha", b"quicksilver=v2"),
                    (b"x-untrusted", b"bad")]}, receive)
    auth = UserAPIKeyAuth()
    authenticate = AsyncMock(return_value=auth)
    authorize = AsyncMock()
    monkeypatch.setattr(codex, "user_api_key_auth", authenticate)
    monkeypatch.setattr(codex, "can_key_call_resolved_model", authorize)

    class Processor:
        def __init__(self, data):
            self.data = data

        async def common_processing_pre_call_logic(self, **kwargs):
            assert kwargs["user_api_key_dict"] is auth
            if kwargs["route_type"] == "_arealtime":
                assert self.data["model"] == "voice-alias"
                assert self.data["guardrails"] == ["query-guardrail"]
                assert await kwargs["request"].json() == {"model": "voice-alias"}
                return {**self.data, "metadata": {"guardrails": ["policy-guardrail"], "user_api_key_team_id": "team"}}, None
            return self.data, None

    monkeypatch.setattr(common_request_processing, "ProxyBaseLLMRequestProcessing", Processor)

    async def route(**kwargs):
        data = kwargs["data"]
        assert data["sdp_body"] == b"v=0\r\n"
        assert data["session"] == session
        assert data["extra_headers"] == {"openai-alpha": "quicksilver=v2"}
        assert data["extra_query"] == {"intent": "quicksilver", "architecture": "avas"}

        async def respond():
            return httpx.Response(201, content=b"v=0\r\nanswer", headers={"Location": "/v1/realtime/calls/rtc_private"},
                extensions={"chatgpt_realtime": {"model": "gpt-live-1-codex", "api_base": "https://voice.example/codex"}})
        return respond()

    monkeypatch.setattr(proxy_server, "route_request", route)
    response = await codex.create_codex_realtime_call(request)
    assert response.status_code == 201
    assert response.body == b"v=0\r\nanswer"
    token = response.headers["location"].rsplit("/", 1)[-1]
    call = codex.decode_call(token, "Bearer owner")
    assert call.call_id == "rtc_private"
    assert call.alias == "voice-alias"
    assert call.model == "gpt-live-1-codex"
    assert "rtc_private" not in token
    assert time.time() < call.expires_at < time.time() + 3601
    authorize.assert_awaited_once()

    sent = []

    async def send(message):
        sent.append(message)

    async def receive_ws():
        return {"type": "websocket.connect"}

    credential_headers = {
        "authorization": [(b"authorization", b"Bearer owner")],
        "api-key": [(b"api-key", b"owner")],
        "subprotocol": [(b"sec-websocket-protocol", b"realtime, openai-insecure-api-key.owner")],
    }
    websocket = WebSocket({"type": "websocket", "path": "/v1/live/opaque",
        "query_string": b"guardrails=query-guardrail", "headers": credential_headers[credential]}, receive_ws, send)
    forward = AsyncMock()
    monkeypatch.setattr(litellm, "_arealtime", forward)
    await codex.codex_realtime_sideband(websocket, token, auth)
    assert sent[0]["type"] == "websocket.accept"
    if credential == "subprotocol":
        assert sent[0]["subprotocol"] == "realtime"
    assert forward.await_args.kwargs["metadata"] == {"guardrails": ["policy-guardrail"], "user_api_key_team_id": "team"}
    assert forward.await_args.kwargs["chatgpt_realtime_call_id"] == "rtc_private"
    assert forward.await_args.kwargs["model"] == "chatgpt/gpt-live-1-codex"
    assert forward.await_args.kwargs["api_base"] == "https://voice.example/codex"
    assert authorize.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not json", b'{}', b'{"sdp":"v=0","session":{}}'])
async def test_invalid_offers_fail_before_authentication(monkeypatch, body):
    from unittest.mock import AsyncMock
    from fastapi import Request
    from litellm.proxy.realtime_endpoints import call_sessions as codex

    async def receive():
        return {"type": "http.request", "body": body}

    request = Request({"type": "http", "headers": [(b"content-type", b"application/json")]}, receive)
    authenticate = AsyncMock()
    monkeypatch.setattr(codex, "user_api_key_auth", authenticate)
    with pytest.raises(HTTPException) as error:
        await codex.create_codex_realtime_call(request)
    assert error.value.status_code == 400
    authenticate.assert_not_called()


@pytest.mark.asyncio
async def test_sideband_pre_call_block_prevents_upstream_connection(monkeypatch):
    from unittest.mock import AsyncMock
    from fastapi import WebSocket
    import litellm
    from litellm.proxy import common_request_processing
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.realtime_endpoints import call_sessions as codex

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-only-salt-for-codex-realtime")
    call = CodexRealtimeCall(call_id="rtc_test", model="gpt-live-1-codex", alias="voice",
        owner=hashlib.sha256(b"Bearer owner").hexdigest(), expires_at=time.time()+300)
    token = encode_call(call)
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    class BlockingProcessor:
        def __init__(self, data):
            assert data["model"] == "voice"

        async def common_processing_pre_call_logic(self, **kwargs):
            assert kwargs["route_type"] == "_arealtime"
            raise HTTPException(403, "Policy blocked this call")

    forward = AsyncMock()
    monkeypatch.setattr(litellm, "_arealtime", forward)
    monkeypatch.setattr(codex, "can_key_call_resolved_model", AsyncMock())
    monkeypatch.setattr(common_request_processing, "ProxyBaseLLMRequestProcessing", BlockingProcessor)
    websocket = WebSocket({"type": "websocket", "path": "/v1/live/opaque", "query_string": b"",
        "headers": [(b"authorization", b"Bearer owner")]}, receive, send)
    await codex.codex_realtime_sideband(websocket, token, UserAPIKeyAuth())
    forward.assert_not_called()
    assert sent == [{"type": "websocket.close", "code": 1008, "reason": "Realtime pre-call rejected"}]

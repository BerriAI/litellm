import hashlib
import time

import pytest
from fastapi import HTTPException, WebSocket

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.realtime_endpoints import codex
from litellm.proxy.realtime_endpoints.codex import CodexRealtimeCall, decode_call, encode_call


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

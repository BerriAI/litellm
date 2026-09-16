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
@pytest.mark.parametrize("multipart", [False, True])
@pytest.mark.parametrize("content_length", [None, "1", "999999999"])
async def test_oversized_offer_stops_before_auth_or_multipart_files(monkeypatch, multipart, content_length):
    import json
    from unittest.mock import AsyncMock, Mock

    from fastapi import Request

    monkeypatch.setattr(codex, "MAX_REALTIME_OFFER_BYTES", 1024)
    if multipart:
        body = (
            b'--Boundary\r\nContent-Disposition: form-data; name="extra"; filename="large.bin"\r\n\r\n'
            + b"x" * 2048
            + b"\r\n--Boundary--\r\n"
        )
        media_type = b"Multipart/Form-Data; boundary=Boundary"
    else:
        body = json.dumps({"sdp": "x" * 2048, "session": {"model": "voice"}}).encode()
        media_type = b"application/json"
    chunks = [body[offset : offset + 256] for offset in range(0, len(body), 256)]
    received = []

    async def receive():
        chunk = chunks.pop(0)
        received.append(len(chunk))
        return {"type": "http.request", "body": chunk, "more_body": bool(chunks)}

    headers = [(b"content-type", media_type)]
    if content_length is not None:
        headers.append((b"content-length", content_length.encode()))
    request = Request({"type": "http", "headers": headers}, receive)
    if multipart:
        from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

        assert await _read_request_body(request) == {}
    authenticate = AsyncMock()
    create_file = Mock(side_effect=AssertionError("Oversized offers must not create temporary files"))
    monkeypatch.setattr(codex, "user_api_key_auth", authenticate)
    monkeypatch.setattr("starlette.formparsers.SpooledTemporaryFile", create_file)
    with pytest.raises(HTTPException) as rejected:
        await codex.create_codex_realtime_call(request)
    assert rejected.value.status_code == 413
    assert sum(received) <= 1280
    assert chunks
    authenticate.assert_not_awaited()
    create_file.assert_not_called()


@pytest.mark.asyncio
async def test_offer_at_size_limit_keeps_body_available_for_custom_auth(monkeypatch):
    import json

    from fastapi import Request

    monkeypatch.setattr(codex, "MAX_REALTIME_OFFER_BYTES", 1024)
    empty = {"sdp": "", "session": {"model": "voice"}}
    sdp = "x" * (1024 - len(json.dumps(empty).encode()))
    body = json.dumps({"sdp": sdp, "session": {"model": "voice"}}).encode()
    chunks = [body[:512], body[512:]]

    async def receive():
        return {"type": "http.request", "body": chunks.pop(0), "more_body": bool(chunks)}

    request = Request({"type": "http", "headers": [(b"content-type", b"application/json")]}, receive)
    offer = await codex.read_codex_offer(request)
    assert offer.sdp == sdp
    assert offer.session.model == "voice"
    assert await request.body() == body
    assert not chunks


@pytest.mark.asyncio
async def test_empty_pre_read_multipart_offer_returns_invalid_offer():
    from fastapi import Request

    async def receive():
        return {"type": "http.request", "body": b"--Boundary--\r\n", "more_body": False}

    request = Request(
        {"type": "http", "headers": [(b"content-type", b"multipart/form-data; boundary=Boundary")]}, receive
    )
    assert not await request.form()
    with pytest.raises(HTTPException) as rejected:
        await codex.create_codex_realtime_call(request)
    assert rejected.value.status_code == 400


@pytest.mark.asyncio
async def test_oversized_pre_read_offer_is_rejected_before_decoding(monkeypatch):
    from fastapi import Request

    monkeypatch.setattr(codex, "MAX_REALTIME_OFFER_BYTES", 1024)

    async def receive():
        return {"type": "http.request", "body": b"x" * 2048, "more_body": False}

    request = Request({"type": "http", "headers": [(b"content-type", b"application/json")]}, receive)
    await request.body()
    with pytest.raises(HTTPException) as rejected:
        await codex.read_codex_offer(request)
    assert rejected.value.status_code == 413


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_mixed_case_offer_preserves_boundary_metadata_and_closes_extra_files(monkeypatch, malformed):
    from fastapi import Request
    from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

    boundary = "AbCdEf123"
    fields = {
        "sdp": "v=0",
        "session": "invalid" if malformed else '{"model":"voice"}',
        "metadata": '{"policy":"keep"}',
        "extra_policy": "keep",
    }
    body = (
        "".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            for name, value in fields.items()
        )
        + f'--{boundary}\r\nContent-Disposition: form-data; name="extra_file"; filename="test.txt"\r\n\r\nextra\r\n--{boundary}--\r\n'
    ).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {"type": "http", "headers": [(b"content-type", f'Multipart/Form-Data; boundary="{boundary}"'.encode())]},
        receive,
    )
    assert not await request.form()
    if malformed:
        with pytest.raises(HTTPException) as error:
            await codex.create_codex_realtime_call(request)
        assert error.value.status_code == 400
        assert (await request.form())["extra_file"].file.closed
        return
    first = await codex.read_codex_offer(request)
    second = await codex.read_codex_offer(request)
    assert first == second
    parsed = await _read_request_body(request)
    assert parsed["metadata"] == {"policy": "keep"}
    assert parsed["extra_policy"] == "keep"
    assert not parsed["extra_file"].file.closed

    async def deny_auth(**kwargs):
        assert kwargs["request"] is request
        auth_form = await request.form()
        assert auth_form["extra_policy"] == "keep"
        assert await auth_form["extra_file"].read() == b"extra"
        assert not auth_form["extra_file"].file.closed
        raise HTTPException(403, "policy denied")

    monkeypatch.setattr(codex, "user_api_key_auth", deny_auth)
    with pytest.raises(HTTPException, match="policy denied"):
        await codex.create_codex_realtime_call(request)
    assert parsed["extra_file"].file.closed
    assert request.headers["content-type"] == f'Multipart/Form-Data; boundary="{boundary}"'


@pytest.mark.asyncio
@pytest.mark.parametrize("multipart", [False, True])
@pytest.mark.parametrize("policy", ["budget", "personal_models"])
@pytest.mark.parametrize("mixed_case", [False, True])
@pytest.mark.parametrize("pre_read", [False, True])
async def test_offer_auth_enforces_session_model_policy_before_upstream(monkeypatch, multipart, policy, mixed_case, pre_read):
    import json
    from unittest.mock import AsyncMock, MagicMock

    import httpx
    from fastapi import Request, Response

    import litellm
    from litellm.exceptions import BudgetExceededError
    from litellm.proxy import proxy_server as server
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.auth.auth_checks import common_checks
    from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
    from litellm.proxy.realtime_endpoints.endpoints import proxy_realtime_calls

    session = {"model": "forbidden-voice"}
    payload = (
        {"files": {"sdp": (None, "v=0"), "session": (None, json.dumps(session)), "model": (None, "body-decoy")}}
        if multipart
        else {"json": {"sdp": "v=0", "session": session, "model": "body-decoy"}}
    )
    outbound = httpx.Request("POST", "http://localhost/v1/realtime/calls", **payload)
    body = outbound.read()
    content_type = outbound.headers["content-type"]
    if mixed_case:
        content_type = content_type.replace("multipart/form-data", "Multipart/Form-Data").replace(
            "application/json", "Application/JSON"
        )
    receives = []

    async def receive():
        receives.append(True)
        assert len(receives) == 1
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/realtime/calls",
            "query_string": b"model=query-decoy&policy=keep",
            "client": ("127.0.0.7", 1234),
            "headers": [
                (b"content-type", content_type.encode()),
                (b"x-policy-key", b"Bearer test-key"),
                (b"x-custom-policy", b"preserved"),
                (b"x-litellm-model", b"header-decoy"),
            ],
        },
        receive,
    )
    token = UserAPIKeyAuth(token="test-key", user_id="personal-user", model_max_budget={"forbidden-voice": 0})
    budget = AsyncMock(side_effect=BudgetExceededError(current_cost=1, max_budget=0))
    upstream = AsyncMock()
    original_request = request

    async def custom_auth(request: Request, api_key: str):
        assert request is original_request
        assert request.headers["content-type"] == content_type
        assert api_key == "test-key"
        assert request.headers["x-custom-policy"] == "preserved"
        assert request.query_params["policy"] == "keep"
        assert request.client.host == "127.0.0.7"
        if multipart:
            assert (await request.form())["model"] == "body-decoy"
        parsed = await _read_request_body(request)
        assert parsed["model"] == "body-decoy"
        assert isinstance(parsed["session"], str) is multipart
        if policy == "personal_models":
            await common_checks(
                request_body=parsed,
                team_object=None,
                user_object=LiteLLM_UserTable(
                    user_id="personal-user", models=["allowed-voice", "body-decoy", "query-decoy", "header-decoy"]
                ),
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route="/v1/realtime/calls",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=request,
                skip_budget_checks=True,
            )
        return token

    custom = AsyncMock(side_effect=custom_auth)
    monkeypatch.setattr(server, "general_settings", {"litellm_key_header_name": "x-policy-key"})
    monkeypatch.setattr(server, "user_custom_auth", custom)
    monkeypatch.setattr(server, "llm_router", None)
    monkeypatch.setattr(server, "llm_model_list", [])
    monkeypatch.setattr(server, "model_max_budget_limiter", SimpleNamespace(is_key_within_model_budget=budget))
    monkeypatch.setattr(server, "route_request", upstream)
    monkeypatch.setattr(litellm, "enable_post_custom_auth_checks", True, raising=False)
    if pre_read:
        await _read_request_body(request)
    with pytest.raises(ProxyException) as denied:
        await proxy_realtime_calls(request, Response())
    if policy == "personal_models":
        assert "user not allowed to access model" in str(denied.value)
        assert "forbidden-voice" in str(denied.value)
    custom.assert_awaited_once()
    upstream.assert_not_awaited()
    if policy == "budget":
        budget.assert_awaited_once()
        assert budget.await_args.kwargs["model"] == "forbidden-voice"


@pytest.mark.asyncio
@pytest.mark.parametrize("route_type", ["arealtime_calls", "_arealtime"])
@pytest.mark.parametrize("observer", [False, True])
async def test_codex_processing_merges_model_guardrails(monkeypatch, route_type, observer):
    from fastapi import Request
    from litellm import Router
    from litellm.proxy import proxy_server as server
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.realtime_endpoints.call_sessions import process_codex_request

    class PolicyHook:
        async def pre_call_hook(self, user_api_key_dict, data, call_type, *, internal_realtime_observer=False):
            assert internal_realtime_observer is observer
            if "model-policy" in data.get("metadata", {}).get("guardrails", []):
                raise HTTPException(403, "Model policy rejected request")
            return data

    router = Router(model_list=[{
        "model_name": "voice-policy",
        "litellm_params": {"model": "openai/gpt-realtime-1.5", "api_key": "test", "guardrails": ["model-policy"]},
    }])
    monkeypatch.setattr(server, "llm_router", router)
    monkeypatch.setattr(server, "proxy_logging_obj", PolicyHook())
    request = Request({"type": "http", "method": "POST", "path": "/v1/realtime/calls", "headers": [], "query_string": b"", "scheme": "http", "server": ("localhost", 80)})
    with pytest.raises(HTTPException) as error:
        await process_codex_request(
            request,
            {"model": "voice-policy"},
            UserAPIKeyAuth(),
            "voice-policy",
            route_type,
            internal_realtime_observer=observer,
        )
    assert error.value.status_code == 403
    assert error.value.detail == "Model policy rejected request"


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


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["normal", "disconnect", "pre_call", "admission"])
async def test_supervised_attachments_release_real_limiter_before_reconnect(monkeypatch, ending):
    import asyncio
    from unittest.mock import AsyncMock

    import litellm
    from litellm.caching.caching import DualCache
    from litellm.proxy import proxy_server as server
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _PROXY_MaxParallelRequestsHandler_v3,
        _request_stash,
        get_request_stash,
    )
    from litellm.proxy.utils import InternalUsageCache

    cache = DualCache()
    limiter = _PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(cache))
    auth = UserAPIKeyAuth(api_key="attachment-owner", max_parallel_requests=1, tpm_limit=10000)
    token_key = limiter.create_rate_limit_keys(key="api_key", value=auth.api_key, rate_limit_type="tokens")
    parallel_key = f"{{api_key:{auth.api_key}}}:max_parallel_requests"
    call = CodexRealtimeCall(
        call_id="rtc_test",
        model="gpt-live-1-codex",
        alias="voice",
        usage_supervised=True,
        owner=hashlib.sha256(b"Bearer owner").hexdigest(),
        expires_at=time.time() + 300,
    )
    monkeypatch.setenv("LITELLM_SALT_KEY", "attachment-cleanup-test")
    monkeypatch.setattr(codex, "can_key_call_resolved_model", AsyncMock())
    monkeypatch.setattr(server, "proxy_logging_obj", SimpleNamespace(get_proxy_hook=lambda name: limiter))

    async def process(request, data, selected_auth, model, call_type):
        await limiter.async_pre_call_hook(
            user_api_key_dict=selected_auth,
            cache=cache,
            data={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 50},
            call_type="completion",
        )
        assert get_request_stash().reserved_tokens > 0
        if ending == "pre_call":
            raise RuntimeError("Later policy rejected attachment")
        return data, SimpleNamespace(model_call_details={})

    async def forward(**kwargs):
        if ending == "disconnect":
            raise asyncio.CancelledError()

    monkeypatch.setattr(codex, "process_codex_request", process)
    monkeypatch.setattr(litellm, "_arealtime", forward)
    blocker_stash = None
    blocker_reserved = 0
    if ending == "admission":
        setup_token = _request_stash.set(None)
        try:
            await limiter.async_pre_call_hook(
                user_api_key_dict=auth,
                cache=cache,
                data={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 50},
                call_type="completion",
            )
            blocker_stash = get_request_stash()
            blocker_reserved = blocker_stash.reserved_tokens
        finally:
            _request_stash.reset(setup_token)
    for _ in range(3):
        stash_token = _request_stash.set(None)
        try:
            websocket = WebSocket(
                {
                    "type": "websocket",
                    "path": "/v1/live/opaque",
                    "query_string": b"",
                    "headers": [(b"authorization", b"Bearer owner")],
                },
                AsyncMock(return_value={"type": "websocket.connect"}),
                AsyncMock(),
            )
            if ending == "disconnect":
                with pytest.raises(asyncio.CancelledError):
                    await codex.codex_realtime_sideband(websocket, encode_call(call), auth)
            else:
                await codex.codex_realtime_sideband(websocket, encode_call(call), auth)
            assert limiter._gauge_in_flight_from_cache_value(await cache.async_get_cache(parallel_key)) == int(
                ending == "admission"
            )
            assert int(await cache.async_get_cache(token_key) or 0) == blocker_reserved
        finally:
            _request_stash.reset(stash_token)
    if blocker_stash is not None:
        cleanup_token = _request_stash.set(blocker_stash)
        try:
            await limiter.async_release_realtime_attachment({}, auth)
        finally:
            _request_stash.reset(cleanup_token)


def test_sideband_token_binds_owner_and_model(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-only-salt-for-codex-realtime")
    call = CodexRealtimeCall(
        call_id="rtc_test",
        model="gpt-live-1-codex",
        alias="gpt-live-1-codex",
        extra_headers={"x-gateway-secret": "configured-secret"},
        owner=hashlib.sha256(b"Bearer test-owner").hexdigest(),
        expires_at=time.time() + 300,
    )
    token = encode_call(call)
    assert "/" not in token
    assert "configured-secret" not in token
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
@pytest.mark.parametrize("credential", ["authorization", "api-key", "subprotocol", "x-litellm-api-key", "custom"])
@pytest.mark.parametrize("signaling_credential", ["authorization", "api-key", "x-litellm-api-key", "mixed"])
async def test_offer_exchange_wraps_call_and_filters_client_headers(
    monkeypatch, multipart, credential, signaling_credential
):
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
        body_request = httpx.Request(
            "POST",
            "http://test/v1/realtime/calls",
            files={"sdp": (None, "v=0\r\n"), "session": (None, json.dumps(session))},
        )
    else:
        body_request = httpx.Request(
            "POST", "http://test/v1/realtime/calls", json={"sdp": "v=0\r\n", "session": session}
        )
    body = body_request.read()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    signaling_headers = (
        [(b"authorization", b"Bearer other-owner"), (b"x-litellm-api-key", b"owner")]
        if signaling_credential == "mixed"
        else [(signaling_credential.encode(), b"Bearer owner" if signaling_credential == "authorization" else b"owner")]
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/realtime/calls",
            "scheme": "http",
            "server": ("localhost", 80),
            "query_string": b"intent=quicksilver&architecture=avas&untrusted=bad",
            "headers": [
                (b"content-type", body_request.headers["content-type"].encode()),
                *signaling_headers,
                *([(b"x-proxy-key", b"Bearer owner")] if credential == "custom" else []),
                (b"openai-alpha", b"quicksilver=v2"),
                (b"x-untrusted", b"bad"),
            ],
        },
        receive,
    )
    auth = UserAPIKeyAuth()
    authorize = AsyncMock()
    monkeypatch.setattr(proxy_server, "master_key", "owner")
    monkeypatch.setattr(
        proxy_server, "general_settings", {"litellm_key_header_name": "x-proxy-key"} if credential == "custom" else {}
    )
    monkeypatch.setattr(codex, "can_key_call_resolved_model", authorize)

    class Processor:
        def __init__(self, data):
            self.data = data

        async def common_processing_pre_call_logic(self, **kwargs):
            assert isinstance(kwargs["user_api_key_dict"], UserAPIKeyAuth)
            if kwargs["route_type"] == "_arealtime":
                assert self.data["model"] == "voice-alias"
                assert self.data["guardrails"] == ["query-guardrail"]
                assert await kwargs["request"].json() == {"model": "voice-alias"}
                return {
                    **self.data,
                    "extra_headers": {
                        "X-Hook-Required": "policy-value",
                        "x-gateway-token": "untrusted-override",
                        "Authorization": "Bearer untrusted",
                    },
                    "extra_query": {"gateway_token": "untrusted-override"},
                    "metadata": {"guardrails": ["policy-guardrail"], "user_api_key_team_id": "team"},
                }, None
            return self.data, None

    monkeypatch.setattr(common_request_processing, "ProxyBaseLLMRequestProcessing", Processor)

    async def route(**kwargs):
        data = kwargs["data"]
        assert data["sdp_body"] == b"v=0\r\n"
        assert data["session"] == session
        assert data["chatgpt_realtime_client_headers"] == {"openai-alpha": "quicksilver=v2"}
        assert "extra_headers" not in data
        assert data["chatgpt_realtime_client_query"] == {"intent": "quicksilver", "architecture": "avas"}

        async def respond():
            return httpx.Response(
                201,
                content=b"v=0\r\nanswer",
                headers={"Location": "/v1/realtime/calls/rtc_private"},
                extensions={
                    "chatgpt_realtime": {
                        "model": "gpt-live-1-codex",
                        "api_base": "https://voice.example/codex",
                        "extra_headers": {"X-Gateway-Token": "pinned-value"},
                        "extra_query": {"gateway_token": "pinned-query-value"},
                    }
                },
            )

        return respond()

    monkeypatch.setattr(proxy_server, "route_request", route)
    supervise = AsyncMock()
    monkeypatch.setattr(codex, "supervise_codex_call", supervise)
    response = await codex.create_codex_realtime_call(request)
    assert response.status_code == 201
    assert response.body == b"v=0\r\nanswer"
    token = response.headers["location"].rsplit("/", 1)[-1]
    call = codex.decode_call(token, "Bearer owner")
    assert call.call_id == "rtc_private"
    assert call.alias == "voice-alias"
    assert call.model == "gpt-live-1-codex"
    assert call.usage_supervised
    supervise.assert_awaited_once()
    assert "rtc_private" not in token
    assert "pinned-query-value" not in token
    assert call.extra_query == {"gateway_token": "pinned-query-value"}
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
        "x-litellm-api-key": [(b"x-litellm-api-key", b"owner")],
        "custom": [(b"x-proxy-key", b"Bearer owner")],
        "subprotocol": [(b"sec-websocket-protocol", b"realtime, openai-insecure-api-key.owner")],
    }
    websocket = WebSocket(
        {
            "type": "websocket",
            "path": "/v1/live/opaque",
            "query_string": b"guardrails=query-guardrail",
            "headers": credential_headers[credential],
        },
        receive_ws,
        send,
    )
    forward = AsyncMock()
    monkeypatch.setattr(litellm, "_arealtime", forward)
    await codex.codex_realtime_sideband(websocket, token, auth)
    assert sent[0]["type"] == "websocket.accept"
    if credential == "subprotocol":
        assert sent[0]["subprotocol"] == "realtime"
    assert forward.await_args.kwargs["extra_headers"] == {
        "x-hook-required": "policy-value",
        "x-gateway-token": "pinned-value",
    }
    assert forward.await_args.kwargs["metadata"] == {"guardrails": ["policy-guardrail"], "user_api_key_team_id": "team"}
    assert forward.await_args.kwargs["extra_query"] == {"gateway_token": "pinned-query-value"}
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


@pytest.mark.asyncio
@pytest.mark.parametrize("observer_fails", [False, True])
async def test_signaling_transfers_reservation_only_to_ready_observer(monkeypatch, observer_fails):
    import json
    from unittest.mock import AsyncMock

    import httpx
    from fastapi import Request

    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-only-reservation-transfer")
    reservation = {"reserved_cost": 0.55, "input_cost": 0.0, "finalized": False, "entries": []}
    auth = UserAPIKeyAuth(budget_reservation=reservation)
    monkeypatch.setattr(codex, "user_api_key_auth", AsyncMock(return_value=auth))
    monkeypatch.setattr(codex, "can_key_call_resolved_model", AsyncMock())
    monkeypatch.setattr(proxy_server, "general_settings", {})
    process = AsyncMock(return_value=({}, None))
    monkeypatch.setattr(codex, "process_codex_request", process)

    async def response():
        return httpx.Response(
            201,
            text="v=0\r\n",
            headers={"Location": "/v1/realtime/calls/rtc_ready"},
            extensions={"chatgpt_realtime": {"model": "gpt-live-1-codex"}},
        )

    async def route(**kwargs):
        return response()

    monkeypatch.setattr(proxy_server, "route_request", route)

    async def supervise(request, call, owner):
        assert owner is auth
        assert not owner.budget_reservation["finalized"]
        assert call.usage_supervised
        if observer_fails:
            await codex.release_or_invalidate_budget_reservation(budget_reservation=owner.budget_reservation)
            raise RuntimeError("Observer unavailable")

    monkeypatch.setattr(codex, "supervise_codex_call", supervise)

    async def receive():
        return {"type": "http.request", "body": json.dumps({"sdp": "v=0", "session": {"model": "voice"}}).encode()}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/realtime/calls",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"authorization", b"Bearer owner")],
        },
        receive,
    )
    if observer_fails:
        with pytest.raises(RuntimeError, match="Observer unavailable"):
            await codex.create_codex_realtime_call(request)
    else:
        assert (await codex.create_codex_realtime_call(request)).status_code == 201
    assert process.await_args.args[2].budget_reservation is None
    assert auth.budget_reservation["finalized"] is observer_fails


@pytest.mark.asyncio
async def test_supervisor_policy_failure_hangs_up_before_releasing(monkeypatch):
    from unittest.mock import AsyncMock

    from fastapi import Request

    call = CodexRealtimeCall(
        call_id="rtc_open", model="gpt-live-1-codex", alias="voice", owner="owner", expires_at=time.time() + 60
    )
    auth = UserAPIKeyAuth(budget_reservation={"reserved_cost": 0.5, "finalized": False, "entries": []})
    monkeypatch.setattr(codex, "process_codex_request", AsyncMock(side_effect=HTTPException(403, "Policy rejected")))
    closed = []

    class Handler:
        def __init__(self, *args):
            pass

        @staticmethod
        def get_api_base(base):
            return "https://gateway.test/v1"

        async def hangup_call(self, base):
            assert not auth.budget_reservation["finalized"]
            closed.append(base)

    monkeypatch.setattr(codex, "ChatGPTRealtime", Handler)
    request = Request({"type": "http", "headers": [], "method": "POST", "path": "/v1/realtime/calls"})
    with pytest.raises(HTTPException) as error:
        await codex.supervise_codex_call(request, call, auth)
    assert error.value.status_code == 403
    assert closed == ["https://gateway.test/v1"]
    assert auth.budget_reservation["finalized"]


@pytest.mark.asyncio
@pytest.mark.parametrize("hangup_fails", [False, True])
async def test_supervisor_constructor_failure_closes_effective_connection(monkeypatch, hangup_fails, caplog):
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import Request

    call = CodexRealtimeCall(
        call_id="rtc_open", model="gpt-live-1-codex", alias="voice", owner="owner", expires_at=time.time() + 60
    )
    auth = UserAPIKeyAuth(budget_reservation={"reserved_cost": 0.5, "finalized": False, "entries": []})
    logger = MagicMock()
    logger.litellm_params = {}
    connection = AsyncMock()
    handlers = []
    invalidate = AsyncMock()
    release = AsyncMock()
    monkeypatch.setattr(codex, "invalidate_budget_reservation_counters", invalidate, raising=False)
    monkeypatch.setattr(codex, "release_or_invalidate_budget_reservation", release)
    monkeypatch.setattr(
        codex, "process_codex_request", AsyncMock(return_value=({"extra_headers": {"x-hook": "effective"}}, logger))
    )

    class Handler:
        def __init__(self, params, headers, extra_headers):
            self.headers = extra_headers
            handlers.append(self)

        @staticmethod
        def get_api_base(base):
            return "https://gateway.test/v1"

        async def open_call_connection(self, model, base):
            return connection

        async def hangup_call(self, base):
            assert self.headers["x-hook"] == "effective"
            if hangup_fails:
                raise RuntimeError("private-cleanup-credential")

    monkeypatch.setattr(codex, "ChatGPTRealtime", Handler)
    monkeypatch.setattr(codex, "RealTimeStreaming", MagicMock(side_effect=ValueError("original constructor failure")))
    request = Request({"type": "http", "headers": [], "method": "POST", "path": "/v1/realtime/calls"})
    with pytest.raises(ValueError, match="original constructor failure"):
        await codex.supervise_codex_call(request, call, auth)
    connection.close.assert_awaited_once()
    assert len(handlers) == 1
    if hangup_fails:
        invalidate.assert_awaited_once_with(budget_reservation=auth.budget_reservation)
        release.assert_not_awaited()
    else:
        release.assert_awaited_once_with(budget_reservation=auth.budget_reservation)
        invalidate.assert_not_awaited()
    assert "private-cleanup-credential" not in caplog.text


@pytest.mark.asyncio
async def test_attachment_releases_quota_before_upstream_close_handshake(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    import websockets

    import litellm
    from litellm.proxy import proxy_server as server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.hooks.parallel_request_limiter_v3 import _request_stash
    from litellm.proxy.utils import ProxyLogging

    proxy = ProxyLogging(UserApiKeyCache())
    monkeypatch.setattr(litellm, "callbacks", [])
    proxy._add_proxy_hooks()
    monkeypatch.setattr(server, "proxy_logging_obj", proxy)
    monkeypatch.setattr(
        server,
        "llm_router",
        litellm.Router(
            model_list=[
                {"model_name": "voice", "litellm_params": {"model": "openai/gpt-realtime-1.5", "api_key": "test"}}
            ]
        ),
    )
    monkeypatch.setattr(codex, "can_key_call_resolved_model", AsyncMock())
    monkeypatch.setenv("LITELLM_SALT_KEY", "attachment-close-order-test")
    from litellm.llms.chatgpt.authenticator import Authenticator

    monkeypatch.setattr(Authenticator, "get_access_token", lambda self: "test-token")
    monkeypatch.setattr(Authenticator, "get_account_id", lambda self: "test-account")
    closing = asyncio.Event()
    finish_close = asyncio.Event()

    class Backend:
        async def recv(self, **kwargs):
            await asyncio.Event().wait()

        async def send(self, value):
            return None

    class Connection:
        async def __aenter__(self):
            return Backend()

        async def __aexit__(self, *args):
            closing.set()
            await finish_close.wait()

    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: Connection())
    auth = UserAPIKeyAuth(api_key="close-order-owner", max_parallel_requests=1)
    call = CodexRealtimeCall(
        call_id="rtc_test",
        model="gpt-live-1-codex",
        alias="voice",
        usage_supervised=True,
        owner=hashlib.sha256(b"Bearer owner").hexdigest(),
        expires_at=time.time() + 300,
    )
    ws = WebSocket(
        {
            "type": "websocket",
            "path": "/v1/live/test",
            "query_string": b"",
            "headers": [(b"authorization", b"Bearer owner")],
            "scheme": "ws",
            "server": ("localhost", 80),
        },
        AsyncMock(side_effect=[{"type": "websocket.connect"}, {"type": "websocket.disconnect", "code": 1000}]),
        AsyncMock(),
    )
    token = _request_stash.set(None)
    request = asyncio.create_task(codex.codex_realtime_sideband(ws, encode_call(call), auth))
    try:
        await asyncio.wait_for(closing.wait(), timeout=5)
        limiter = proxy.get_proxy_hook("parallel_request_limiter")
        value = await proxy.internal_usage_cache.async_get_cache(
            "{api_key:close-order-owner}:max_parallel_requests", litellm_parent_otel_span=None, local_only=True
        )
        assert limiter._gauge_in_flight_from_cache_value(value) == 0
    finally:
        finish_close.set()
        await asyncio.wait_for(request, timeout=5)
        _request_stash.reset(token)
    value = await proxy.internal_usage_cache.async_get_cache(
        "{api_key:close-order-owner}:max_parallel_requests", litellm_parent_otel_span=None, local_only=True
    )
    assert limiter._gauge_in_flight_from_cache_value(value) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "provider", "observer", "renewal", "legacy_key", "legacy_global"])
async def test_signaling_keeps_or_releases_owned_call_lease(monkeypatch, failure):
    import json
    from unittest.mock import AsyncMock, MagicMock

    import httpx
    from fastapi import Request
    from litellm.proxy.hooks.realtime_call_lease import RealtimeCallLease

    from litellm.proxy import proxy_server as server
    from litellm.proxy.hooks.parallel_request_limiter import _PROXY_MaxParallelRequestsHandler
    from litellm.proxy.hooks.parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3

    auth = UserAPIKeyAuth(max_parallel_requests=None if failure == "legacy_global" else 1)
    lease = MagicMock(spec=RealtimeCallLease)
    lease.renew = AsyncMock(return_value=failure != "renewal")
    lease.close = AsyncMock()
    legacy = failure in ("legacy_key", "legacy_global")
    limiter = MagicMock(spec=_PROXY_MaxParallelRequestsHandler if legacy else _PROXY_MaxParallelRequestsHandler_v3)
    if not legacy:
        limiter.transfer_realtime_call_slot.return_value = lease
    proxy = MagicMock()
    proxy.get_proxy_hook.return_value = limiter
    monkeypatch.setattr(server, "proxy_logging_obj", proxy)
    monkeypatch.setattr(
        server, "general_settings", {"global_max_parallel_requests": 1} if failure == "legacy_global" else {}
    )
    monkeypatch.setattr(codex, "user_api_key_auth", AsyncMock(return_value=auth))
    monkeypatch.setattr(codex, "can_key_call_resolved_model", AsyncMock())
    process = AsyncMock(return_value=({}, None))
    monkeypatch.setattr(codex, "process_codex_request", process)
    monkeypatch.setenv("LITELLM_SALT_KEY", "lease-transfer-test")

    async def route(**kwargs):
        lease.start.assert_called_once()
        if failure == "provider":
            raise RuntimeError("Provider unavailable")

        async def respond():
            return httpx.Response(
                201,
                text="v=0\r\n",
                headers={"Location": "/v1/realtime/calls/rtc_lease"},
                extensions={"chatgpt_realtime": {"model": "gpt-live-1-codex"}},
            )

        return respond()

    async def supervise(request, call, owner, selected_lease):
        assert owner is auth
        assert selected_lease is lease
        assert call.parallel_reserved
        if failure == "observer":
            raise RuntimeError("Observer unavailable")

    monkeypatch.setattr(server, "route_request", route)
    monkeypatch.setattr(codex, "supervise_codex_call", supervise)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/realtime/calls",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"authorization", b"Bearer owner")],
        },
        AsyncMock(
            return_value={
                "type": "http.request",
                "body": json.dumps({"sdp": "v=0", "session": {"model": "voice"}}).encode(),
            }
        ),
    )
    if failure is None:
        response = await codex.create_codex_realtime_call(request)
        token = response.headers["location"].rsplit("/", 1)[-1]
        assert codex.decode_call(token, "Bearer owner").parallel_reserved
        lease.close.assert_not_awaited()
    else:
        with pytest.raises((RuntimeError, HTTPException)) as raised:
            await codex.create_codex_realtime_call(request)
        if legacy:
            assert raised.value.status_code == 400
            assert "V3 rate limiter" in raised.value.detail
            process.assert_not_awaited()
            lease.close.assert_not_awaited()
        else:
            if failure == "renewal":
                assert raised.value.status_code == 503
            lease.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_signaling_rejection_after_admission_refunds_parallel_slot(monkeypatch):
    import json
    from unittest.mock import AsyncMock

    from fastapi import Request

    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy import proxy_server as server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import ProxyLogging

    proxy = ProxyLogging(UserApiKeyCache())
    monkeypatch.setattr(litellm, "callbacks", [])
    proxy._add_proxy_hooks()
    limiter = proxy.get_proxy_hook("parallel_request_limiter")
    key = "{api_key:rejected-signaling-owner}:max_parallel_requests"

    class Reject(CustomLogger):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            current = await proxy.internal_usage_cache.async_get_cache(key, litellm_parent_otel_span=None, local_only=True)
            assert limiter._gauge_in_flight_from_cache_value(current) == 1
            raise RuntimeError("Policy rejected after admission")

    litellm.callbacks.append(Reject())
    monkeypatch.setattr(server, "proxy_logging_obj", proxy)
    monkeypatch.setattr(server, "general_settings", {})
    monkeypatch.setattr(
        server,
        "llm_router",
        litellm.Router(
            model_list=[
                {"model_name": "voice", "litellm_params": {"model": "openai/gpt-realtime-1.5", "api_key": "test"}}
            ]
        ),
    )
    auth = UserAPIKeyAuth(api_key="rejected-signaling-owner", max_parallel_requests=1)
    monkeypatch.setattr(codex, "user_api_key_auth", AsyncMock(return_value=auth))
    monkeypatch.setattr(codex, "can_key_call_resolved_model", AsyncMock())
    route = AsyncMock()
    monkeypatch.setattr(server, "route_request", route)
    for _ in range(2):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/realtime/calls",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
            },
            AsyncMock(
                return_value={
                    "type": "http.request",
                    "body": json.dumps({"sdp": "v=0", "session": {"model": "voice"}}).encode(),
                }
            ),
        )
        with pytest.raises(RuntimeError, match="Policy rejected after admission"):
            await codex.create_codex_realtime_call(request)
        current = await proxy.internal_usage_cache.async_get_cache(key, litellm_parent_otel_span=None, local_only=True)
        assert limiter._gauge_in_flight_from_cache_value(current) == 0
    route.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_order", ["before", "after", "cancel"])
async def test_signaling_settles_tokens_once_with_isolated_sdk_callbacks(monkeypatch, callback_order):
    import asyncio
    import json
    from datetime import datetime
    from unittest.mock import AsyncMock

    import httpx
    import litellm
    from fastapi import Request
    from litellm.proxy import proxy_server as server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.hooks.parallel_request_limiter_v3 import get_request_stash, isolated_request_stash
    from litellm.proxy.utils import ProxyLogging

    proxy = ProxyLogging(UserApiKeyCache())
    monkeypatch.setattr(litellm, "callbacks", [])
    proxy._add_proxy_hooks()
    limiter = proxy.get_proxy_hook("parallel_request_limiter")
    monkeypatch.setattr(server, "proxy_logging_obj", proxy)
    monkeypatch.setattr(server, "general_settings", {})
    monkeypatch.setattr(
        server,
        "llm_router",
        litellm.Router(
            model_list=[
                {"model_name": "voice", "litellm_params": {"model": "openai/gpt-realtime-1.5", "api_key": "test"}}
            ]
        ),
    )
    auth = UserAPIKeyAuth(api_key="signaling-settlement-owner", max_parallel_requests=1, tpm_limit=10000)
    monkeypatch.setattr(codex, "user_api_key_auth", AsyncMock(return_value=auth))
    monkeypatch.setattr(codex, "can_key_call_resolved_model", AsyncMock())
    supervisor = AsyncMock()
    monkeypatch.setattr(codex, "supervise_codex_call", supervisor)
    monkeypatch.setenv("LITELLM_SALT_KEY", "signaling-settlement-test")
    ready, release_callback = asyncio.Event(), asyncio.Event()
    callbacks = []

    async def counter(kind):
        return await proxy.internal_usage_cache.async_get_cache(
            f"{{api_key:{auth.api_key}}}:{kind}", litellm_parent_otel_span=None, local_only=True
        )

    async def route(**kwargs):
        assert get_request_stash() is None
        assert await counter("tokens") > 0

        async def callback():
            await release_callback.wait()
            assert get_request_stash() is None
            await limiter.async_log_success_event(
                kwargs={
                    "litellm_call_id": kwargs["data"]["litellm_call_id"],
                    "standard_logging_object": {"metadata": {"user_api_key_hash": auth.api_key}},
                },
                response_obj=litellm.ModelResponse(usage=litellm.Usage()),
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        async def respond():
            assert get_request_stash() is None
            ready.set()
            if callback_order == "cancel":
                await asyncio.Event().wait()
            callbacks.append(asyncio.create_task(callback()))
            if callback_order == "before":
                release_callback.set()
                await callbacks[0]
                assert await counter("tokens") > 0
            return httpx.Response(
                201,
                text="v=0\r\n",
                headers={"Location": "/v1/realtime/calls/rtc_settlement"},
                extensions={"chatgpt_realtime": {"model": "gpt-live-1-codex"}},
            )

        return respond()

    monkeypatch.setattr(server, "route_request", route)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/realtime/calls",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"authorization", b"Bearer owner")],
        },
        AsyncMock(
            return_value={
                "type": "http.request",
                "body": json.dumps({"sdp": "v=0", "session": {"model": "voice"}}).encode(),
            }
        ),
    )
    with isolated_request_stash():
        signaling = asyncio.create_task(codex.create_codex_realtime_call(request))
    await asyncio.wait_for(ready.wait(), timeout=2)
    if callback_order == "cancel":
        signaling.cancel()
        with pytest.raises(asyncio.CancelledError):
            await signaling
        supervisor.assert_not_awaited()
    else:
        assert (await signaling).status_code == 201
        assert limiter._gauge_in_flight_from_cache_value(await counter("max_parallel_requests")) == 1
        await supervisor.call_args.args[3].close()
    assert await counter("tokens") == 0
    release_callback.set()
    await asyncio.gather(*callbacks)
    assert await counter("tokens") == 0
    assert limiter._gauge_in_flight_from_cache_value(await counter("max_parallel_requests")) == 0

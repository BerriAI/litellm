import asyncio
from collections.abc import Generator, Mapping
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request
from pydantic import TypeAdapter
from starlette.types import ASGIApp

import litellm
from litellm import ModelResponse
from litellm.caching.caching import DualCache
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.context_compaction import (
    _SummaryRequest,
    _summary_request,
    proxy_summary_executor_scope,
    summary_request_updates,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
    _request_stash,
    get_request_stash,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.router_strategy.complexity_router.context_compaction import (
    NativeRequest,
    current_summary_executor,
)

_BODY: Final = TypeAdapter(dict[str, object])
_KEY: Final = "sk-test-context-compaction"
_KEY_HASH: Final = hash_token(_KEY)
_MESSAGES: Final = [{"role": "user", "content": "Private older conversation"}]


def parent_request(app: ASGIApp) -> Request:
    return Request({
        "type": "http", "method": "POST", "scheme": "https", "path": "/gateway/v1/messages",
        "root_path": "/gateway", "query_string": b"", "server": ("gateway.test", 443),
        "client": ("192.0.2.10", 2345), "app": app,
        "headers": [(b"authorization", f"Bearer {_KEY}".encode()), (b"content-length", b"99999"),
                    (b"x-litellm-call-id", b"parent"), (b"litellm-disable-message-redaction", b"true"),
                    (b"x-app", b"cli"), (b"x-claude-code-agent-id", b"agent"),
                    (b"x-claude-code-session-id", b"session"), (b"x-litellm-model-id", b"target-id"),
                    (b"x-litellm-model", b"target"), (b"anthropic-beta", b"existing-beta")],
    })


@pytest.fixture
def summary_proxy(
    protocol: Literal["responses", "messages"], monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[_PROXY_MaxParallelRequestsHandler_v3, AsyncMock], None, None]:
    limiter: Final = _PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(DualCache()))
    inference: Final = AsyncMock()
    monkeypatch.setattr(litellm, "acompact_responses" if protocol == "responses" else "anthropic_messages", inference)
    model: Final = "openai/gpt-5.6" if protocol == "responses" else "anthropic/claude-sonnet-5"
    router: Final = litellm.Router(model_list=[{
        "model_name": "summary", "litellm_params": {"model": model, "api_key": "test-only-key"},
    }])
    monkeypatch.setattr(proxy_server, "master_key", "sk-test-master")
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "max_parallel_request_limiter", limiter)
    monkeypatch.setattr(litellm, "callbacks", [limiter])
    token: Final = _request_stash.set(None)
    try:
        yield limiter, inference
    finally:
        _request_stash.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "denied", "route-denied", "cancel", "timeout", "failed"])
@pytest.mark.parametrize("protocol", ["responses", "messages"])
async def test_real_proxy_summary_auth_identity_and_lifecycle(
    outcome: Literal["success", "denied", "route-denied", "cancel", "timeout", "failed"], protocol: Literal["responses", "messages"],
    summary_proxy: tuple[_PROXY_MaxParallelRequestsHandler_v3, AsyncMock],
) -> None:
    limiter, inference = summary_proxy
    native_payload: Final = (
        {"input": _MESSAGES, "instructions": "Preserve the constraints"} if protocol == "responses" else
        {"messages": _MESSAGES, "max_tokens": 128, "system": "Preserve the constraints", "tools": [],
         "compaction": {"type": "summarize"}}
    )
    native: Final = NativeRequest(protocol, {
        **native_payload, "model": "target", "api_key": "payload-credential", "extra_body": {"model": "target"},
        "stream": True,
        "extra_headers": {
            "Anthropic-Beta": "existing-beta, compact-2026-09-04",
            "authorization": "Bearer other", "x-api-key": "other",
        },
    }, frozenset({"native-test"}))
    result: Final = {
        "id": "native-result", "usage": {"input_tokens": 10, "output_tokens": 2},
        **({"object": "response.compaction", "output": [{"type": "compaction", "encrypted_content": "opaque"}]}
           if protocol == "responses" else {
               "type": "message", "role": "assistant", "stop_reason": "compaction",
               "content": [{"type": "compaction", "content": "opaque", "signature": "signed"}],
           }),
    }
    auth: Final = UserAPIKeyAuth(
        api_key=_KEY_HASH, models=["target"] if outcome == "denied" else ["target", "summary"],
        max_parallel_requests=1, rpm_limit=10,
        allowed_routes=["/v1/chat/completions"] if outcome == "route-denied" else ["llm_api_routes"],
    )
    await proxy_server.user_api_key_cache.async_set_cache(_KEY_HASH, auth, model_type=UserAPIKeyAuth)
    parent_data: Final = {
        "model": "target", "litellm_call_id": "parent", "turn_off_message_logging": True,
        "allowed_model_region": "eu", "user": "end-user", "litellm_session_id": "session",
        "metadata": {"tags": ["cost-center"]},
    }
    await limiter.async_pre_call_hook(auth, DualCache(), parent_data, "acompletion")
    parent: Final = get_request_stash()
    assert parent is not None and parent.parallel_slot is not None
    acquisition: Final = parent.parallel_slot
    entered: Final = asyncio.Event()
    stopped: Final = asyncio.Event()
    application: Final = AsyncMock(wraps=proxy_server.app.__call__)

    async def receive(**provider_data: object) -> Mapping[str, object]:
        data: Final = _BODY.validate_python(_BODY.validate_python(provider_data["proxy_server_request"])["body"])
        assert get_request_stash() is not parent
        assert parent.parallel_slot is acquisition
        child: Final = Request(application.call_args.args[0])
        assert child.url.path == "/gateway/v1/" + ("responses/compact" if protocol == "responses" else "messages")
        assert child.headers["authorization"] == f"Bearer {_KEY}"
        assert "x-api-key" not in child.headers
        assert child.headers["anthropic-beta"] == "existing-beta,compact-2026-09-04"
        if protocol == "responses":
            assert "stream" not in provider_data
        else:
            assert provider_data["stream"] is False
        assert child.client.host == "192.0.2.10"
        assert not {"litellm-disable-message-redaction", "x-app", "x-claude-code-agent-id",
                    "x-litellm-model-id", "x-litellm-model"}.intersection(child.headers)
        assert child.headers["x-claude-code-session-id"] == "session"
        assert data["litellm_call_id"] != "parent"
        for name, expected in (
            ("turn_off_message_logging", True), ("num_retries", 0), ("max_retries", 0),
            ("disable_fallbacks", True), ("allowed_model_region", "eu"), ("user", "end-user"),
            ("litellm_session_id", "session"), ("model", "summary"), *native_payload.items(),
        ):
            assert data[name] == expected
        assert provider_data.get("api_key") != "payload-credential"
        assert "extra_body" not in data
        metadata: Final = _BODY.validate_python(data["litellm_metadata"])
        assert metadata[INTERNAL_CALL_ORIGIN_METADATA_KEY] == "context_compaction"
        assert metadata["user_api_key"] == _KEY_HASH
        assert "cost-center" in metadata["tags"]
        with pytest.raises(ValueError, match="nested"):
            await executor("summary", native, 10)
        entered.set()
        try:
            if outcome in ("cancel", "timeout"):
                await asyncio.Event().wait()
            if outcome == "failed":
                raise litellm.APIError(503, "Private older conversation", llm_provider="test", model="summary")
            return result
        finally:
            stopped.set()

    inference.side_effect = receive
    assert current_summary_executor() is None
    with proxy_summary_executor_scope(parent_request(application), parent_data, limiter):
        executor: Final = current_summary_executor()
        assert executor is not None
        task: Final = asyncio.create_task(executor("summary", native, 0.2 if outcome == "timeout" else 10))
        if outcome in ("denied", "route-denied", "failed"):
            with pytest.raises(litellm.APIError) as denied:
                await task
            assert denied.value.status_code in ((503,) if outcome == "failed" else (401, 403))
            assert "Private older conversation" not in str(denied.value)
            assert inference.await_count == int(outcome == "failed")
            cached: Final = await proxy_server.user_api_key_cache.async_get_cache(_KEY_HASH, model_type=UserAPIKeyAuth)
            assert cached.models == auth.models and cached.spend == 0
        else:
            if outcome == "success":
                assert await task == result
            else:
                await asyncio.wait_for(entered.wait(), 5)
                if outcome == "cancel":
                    task.cancel()
                with pytest.raises(asyncio.CancelledError if outcome == "cancel" else TimeoutError):
                    await task
            inference.assert_awaited_once()
            assert stopped.is_set()
    assert current_summary_executor() is None
    assert get_request_stash() is parent and parent.parallel_slot is acquisition
    assert not parent.reservation_released
    with pytest.raises(ValueError, match="closed"):
        await executor("summary", native, 10)
    await limiter.async_post_call_success_hook(parent_data, auth, ModelResponse())
    assert parent.parallel_slot is None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app)) as client:
        rejected: Final = await client.post(
            "https://gateway.test/v1/" + ("responses/compact" if protocol == "responses" else "messages"),
            headers={"authorization": f"Bearer {_KEY}"},
            json={**native_payload, "model": "summary", "turn_off_message_logging": True},
        )
    assert rejected.is_client_error and "turn_off_message_logging" in rejected.text
    assert proxy_server.general_settings == {}


@pytest.mark.parametrize("parent_redact", [False, True, None])
@pytest.mark.parametrize("recipient_redact", [False, True, None])
@pytest.mark.parametrize("global_redact", [False, True])
def test_summary_privacy_never_lowers_recipient_policy(
    parent_redact: bool | None, recipient_redact: bool | None,
    global_redact: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "turn_off_message_logging", global_redact)
    summary: Final = _SummaryRequest(call_id="parent", redact=parent_redact)
    token: Final = _summary_request.set(summary)
    recipient_data: Final = {"turn_off_message_logging": recipient_redact, "metadata": {}}
    try:
        updates: Final = summary_request_updates(
            parent_request(FastAPI()), UserAPIKeyAuth(), recipient_data, "metadata"
        )
        if parent_redact is True or recipient_redact is True or global_redact:
            assert updates["turn_off_message_logging"] is True
            assert summary.redact is True
        else:
            assert "turn_off_message_logging" not in updates
    finally:
        _summary_request.reset(token)

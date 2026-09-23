import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

import litellm
from litellm._logging import trace_id_var
from litellm.caching.caching import Cache, CacheMode, LiteLLMCacheType
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.experimental_pass_through.messages import handler as python_messages_handler
from litellm.rust_bridge.catalog import CacheRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.response_cache import resolve_response_cache
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    MESSAGES,
    MESSAGES_EVENTS,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
    request_body,
)

pytestmark = pytest.mark.requires_rust_extension

STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS)


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": MESSAGES_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


def assert_served_natively(server: RecordingServer) -> None:
    assert len(server.requests) == 1
    assert not server.requests[0].headers.get("user-agent", "").startswith("python-httpx")


@pytest.mark.asyncio
async def test_native_messages_callbacks_see_the_provider_request_and_the_public_response(
    messages_server: RecordingServer,
) -> None:
    await drain_logging()
    recorder: Final = RecordingLogger()

    response: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, callbacks=[recorder], litellm_call_id="messages-success")
    )

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]
    sent: Final = messages_server.requests[0]
    assert sent.path == "/v1/messages"
    assert sent.body == {"model": "claude-sonnet-5", "messages": list(MESSAGES), "max_tokens": 64, "stream": False}
    pre_call: Final = recorder.wait_for("log_pre_api_call")
    assert request_body(pre_call[0].kwargs) == sent.body
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].call_type == "anthropic_messages"
    assert success[0].kwargs["litellm_call_id"] == "messages-success"
    assert success[0].response.choices[0].message.content == "Hello from native Messages"


@pytest.mark.asyncio
async def test_native_messages_pre_call_body_edit_reaches_the_provider(messages_server: RecordingServer) -> None:
    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["temperature"] = 0.25

    await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[Edit()]))

    assert messages_server.requests[0].body["temperature"] == 0.25


@pytest.mark.parametrize(
    "context_management",
    (
        {"edits": [{"type": "compact_20260112"}]},
        [{"type": "compaction", "compact_threshold": 2048}],
    ),
)
@pytest.mark.asyncio
async def test_native_messages_matches_python_request_cleanup_and_beta_headers(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, context_management: object
) -> None:
    messages_server.expected_requests = 2
    messages: Final = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "  "},
                {"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}},
                {"type": "text", "text": "hello", "provider_specific_fields": {"source": "other"}},
            ],
        }
    ]
    kwargs: Final = arguments(
        messages_server,
        messages=messages,
        metadata={"user_id": "user", "internal": "private"},
        system="Keep it short",
        stop_sequences=["END"],
        temperature=1,
        tools=[{"name": "Bash", "input_schema": {"type": "object"}}],
        tool_choice={"type": "auto"},
        context_management=context_management,
        extra_headers={"x-request-id": "parity"},
        provider_specific_header=[
            {"custom_llm_provider": "azure_ai", "extra_headers": {"x-ignored": "other"}},
            {"custom_llm_provider": "anthropic, bedrock", "extra_headers": {"x-scope": "selected"}},
        ],
    )

    monkeypatch.setenv("LITELLM_RUST", "0")
    await litellm.anthropic.messages.acreate(**kwargs)
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(python_messages_handler, "base_llm_http_handler", None)
    await litellm.anthropic.messages.acreate(**kwargs)

    python_request, rust_request = messages_server.requests
    assert python_request.body == rust_request.body
    assert python_request.headers["anthropic-beta"] == rust_request.headers["anthropic-beta"]
    assert rust_request.headers["x-request-id"] == "parity"
    assert rust_request.headers["x-scope"] == "selected"
    assert "x-ignored" not in rust_request.headers
    assert not rust_request.headers.get("user-agent", "").startswith("python-httpx")


@pytest.mark.parametrize(
    ("model", "options"),
    (
        ("claude-opus-4-7", {"reasoning_effort": "low", "max_tokens": 8192}),
        (
            "claude-sonnet-4-5",
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}, "max_tokens": 8192},
        ),
        ("claude-opus-4-7", {"thinking": {"type": "disabled"}, "speed": "fast"}),
    ),
)
@pytest.mark.asyncio
async def test_native_messages_matches_python_thinking_and_reasoning_rules(
    messages_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    options: dict[str, object],
) -> None:
    messages_server.expected_requests = 2
    kwargs: Final = arguments(messages_server, model=model, **options)

    monkeypatch.setenv("LITELLM_RUST", "0")
    await litellm.anthropic.messages.acreate(**kwargs)
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(python_messages_handler, "base_llm_http_handler", None)
    await litellm.anthropic.messages.acreate(**kwargs)

    python_request, rust_request = messages_server.requests
    assert rust_request.body == python_request.body
    assert rust_request.headers.get("anthropic-beta") == python_request.headers.get("anthropic-beta")
    assert not rust_request.headers.get("user-agent", "").startswith("python-httpx")


@pytest.mark.asyncio
async def test_native_invalid_reasoning_effort_fails_without_upstream_request(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages_server.expected_requests = 0
    monkeypatch.setattr(python_messages_handler, "base_llm_http_handler", None)

    with pytest.raises(litellm.BadRequestError):
        await litellm.anthropic.messages.acreate(
            **arguments(messages_server, model="claude-opus-4-7", reasoning_effort="invalid")
        )

    assert not messages_server.requests


@pytest.mark.asyncio
async def test_native_messages_matches_python_prompt_cache_injection(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages_server.expected_requests = 2
    kwargs: Final = arguments(
        messages_server,
        messages=[{"role": "user", "content": [{"type": "text", "text": "Keep this context"}]}],
        cache_control_injection_points=[{"location": "message", "index": -1}],
    )

    monkeypatch.setenv("LITELLM_RUST", "0")
    await litellm.anthropic.messages.acreate(**kwargs)
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(python_messages_handler, "base_llm_http_handler", None)
    await litellm.anthropic.messages.acreate(**kwargs)

    python_request, rust_request = messages_server.requests
    assert rust_request.body == python_request.body
    assert not rust_request.headers.get("user-agent", "").startswith("python-httpx")


@pytest.mark.parametrize("enabled_by_request", (False, True))
@pytest.mark.asyncio
async def test_native_messages_matches_python_default_prompt_cache_injection(
    messages_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    enabled_by_request: bool,
) -> None:
    messages_server.expected_requests = 2
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", not enabled_by_request)
    kwargs: Final = arguments(
        messages_server,
        system="Stable instructions",
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
            {"role": "user", "content": [{"type": "text", "text": "Latest"}]},
        ],
        **({"enable_prompt_caching": True} if enabled_by_request else {}),
    )

    monkeypatch.setenv("LITELLM_RUST", "0")
    await litellm.anthropic.messages.acreate(**kwargs)
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(python_messages_handler, "base_llm_http_handler", None)
    await litellm.anthropic.messages.acreate(**kwargs)

    python_request, rust_request = messages_server.requests
    assert rust_request.body == python_request.body
    assert rust_request.body["system"][0]["cache_control"]["type"] == "ephemeral"
    assert rust_request.body["messages"][-1]["content"][-1]["cache_control"]["type"] == "ephemeral"


@pytest.mark.asyncio
async def test_same_provider_request_hook_runs_once_and_uses_native_transport(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Edit(CustomLogger):
        calls = 0

        async def async_pre_request_hook(self, model, messages, kwargs):
            self.calls += 1
            return {**kwargs, "temperature": 0.25}

    hook: Final = Edit()
    monkeypatch.setattr(litellm, "callbacks", [hook])
    monkeypatch.setattr(python_messages_handler, "base_llm_http_handler", None)

    await litellm.anthropic.messages.acreate(**arguments(messages_server, model="claude-sonnet-4-5"))

    assert hook.calls == 1
    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["temperature"] == 0.25


@pytest.mark.asyncio
async def test_provider_changing_request_hook_continues_on_python_once(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Redirect(CustomLogger):
        calls = 0

        async def async_pre_request_hook(self, model, messages, kwargs):
            self.calls += 1
            return {**kwargs, "litellm_params": {"custom_llm_provider": "bedrock"}}

    hook: Final = Redirect()
    calls: Final = []

    async def python_handler(**kwargs: object) -> object:
        calls.append(kwargs)
        return MESSAGES_RESPONSE

    messages_server.expected_requests = 0
    monkeypatch.setattr(litellm, "callbacks", [hook])
    monkeypatch.setattr(python_messages_handler, "anthropic_messages_handler", python_handler)

    result: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert hook.calls == 1
    assert len(calls) == 1
    assert calls[0]["custom_llm_provider"] == "bedrock"
    assert result["content"] == MESSAGES_RESPONSE["content"]
    assert not messages_server.requests


@pytest.mark.asyncio
async def test_native_request_hooks_keep_deployment_and_pre_call_order(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: Final = []

    class Ordered(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            events.append("deployment")
            return kwargs

        async def async_pre_request_hook(self, model, messages, kwargs):
            events.append("request")
            return {**kwargs, "temperature": 0.25}

        def log_pre_api_call(self, model, messages, kwargs):
            events.append("pre_call")

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append("success")

    monkeypatch.setattr(litellm, "callbacks", [Ordered()])
    monkeypatch.setattr(python_messages_handler, "base_llm_http_handler", None)

    await litellm.anthropic.messages.acreate(**arguments(messages_server, model="claude-sonnet-4-5"))
    await drain_logging()

    assert events == ["deployment", "request", "pre_call", "success"]
    assert messages_server.requests[0].body["temperature"] == 0.25


@pytest.mark.asyncio
async def test_native_request_hook_can_enable_streaming(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Stream(CustomLogger):
        async def async_pre_request_hook(self, model, messages, kwargs):
            return {**kwargs, "stream": True}

    messages_server.enqueue(STREAM)
    monkeypatch.setattr(litellm, "callbacks", [Stream()])
    monkeypatch.setattr(python_messages_handler, "base_llm_http_handler", None)

    stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))
    chunks: Final = b"".join([chunk async for chunk in stream])

    assert chunks == sse_payload()
    assert messages_server.requests[0].body["stream"] is True


@pytest.mark.asyncio
async def test_native_messages_stream_preserves_python_sse_event_order(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages_server.expected_requests = 2
    messages_server.enqueue(STREAM)
    messages_server.enqueue(STREAM)

    monkeypatch.setenv("LITELLM_RUST", "0")
    python_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    python_events: Final = b"".join([chunk async for chunk in python_stream])

    monkeypatch.setenv("LITELLM_RUST", "1")
    rust_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    rust_events: Final = b"".join([chunk async for chunk in rust_stream])

    assert python_events == rust_events == sse_payload()
    assert messages_server.requests[0].body == messages_server.requests[1].body


@pytest.mark.asyncio
async def test_native_messages_stream_preserves_incomplete_provider_response(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages_server.expected_requests = 2
    incomplete: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS[:-1])
    messages_server.enqueue(incomplete)
    messages_server.enqueue(incomplete)

    monkeypatch.setenv("LITELLM_RUST", "0")
    python_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    python_events: Final = b"".join([chunk async for chunk in python_stream])

    monkeypatch.setenv("LITELLM_RUST", "1")
    rust_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    rust_events: Final = b"".join([chunk async for chunk in rust_stream])

    assert python_events == rust_events == b"".join(incomplete.payloads())
    assert b"event: message_stop" not in rust_events


@pytest.mark.asyncio
async def test_native_messages_provider_error_reaches_caller_and_failure_callbacks_as_one_public_error(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(
        ResponseSpec(body={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}, status=400)
    )
    observed: Final = []

    class Observe(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("sync", kwargs["exception"]))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("async", kwargs["exception"]))

    with pytest.raises(litellm.BadRequestError) as raised:
        await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[Observe()]))

    assert_served_natively(messages_server)
    assert [phase for phase, _ in observed] == ["sync", "async"]
    assert all(error is raised.value for _, error in observed)


@pytest.mark.asyncio
async def test_native_messages_retries_invalid_thinking_signature_with_clean_history(
    messages_server: RecordingServer,
) -> None:
    messages_server.expected_requests = 2
    messages_server.enqueue(
        ResponseSpec(
            body={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "Invalid signature in thinking block"},
            },
            status=400,
        )
    )
    recorder: Final = RecordingLogger()
    messages: Final = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "previous", "signature": "expired"},
                {"type": "text", "text": "answer"},
            ],
        },
        {"role": "user", "content": "continue"},
    ]

    response: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, messages=messages, callbacks=[recorder])
    )

    assert response["content"] == MESSAGES_RESPONSE["content"]
    assert [block["type"] for block in messages_server.requests[0].body["messages"][0]["content"]] == [
        "thinking",
        "text",
    ]
    assert messages_server.requests[1].body["messages"][0]["content"] == [{"type": "text", "text": "answer"}]
    assert len(recorder.wait_for("log_pre_api_call")) == 1
    assert len(await recorder.wait_for_async("async_log_success_event")) == 1


@pytest.mark.asyncio
async def test_provider_changing_hook_runs_once_before_native_request(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages_server.expected_requests = 0

    class Reroute(CustomLogger):
        calls = 0

        async def async_pre_request_hook(self, model, messages, kwargs):
            self.calls += 1
            return {**kwargs, "litellm_params": {"custom_llm_provider": "openai"}, "mock_response": "rerouted"}

    hook: Final = Reroute()
    monkeypatch.setattr(litellm, "callbacks", [hook])

    response: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert response["content"][0]["text"] == "rerouted"
    assert hook.calls == 1
    assert messages_server.requests == []


@pytest.mark.parametrize("native_cache", (False, True))
@pytest.mark.asyncio
async def test_native_messages_share_the_configured_response_cache(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, native_cache: bool
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    if native_cache:
        cache._native_cache = resolve_response_cache(  # pyright: ignore[reportPrivateUsage]  # select the native store for this cache integration test
            cache,
            rules=(CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.LOCAL})),),
        )
        assert cache._native_cache is not None  # pyright: ignore[reportPrivateUsage]  # confirm the selected backend

        async def unexpected_python_cache(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("native inference accessed the Python cache facade")

        monkeypatch.setattr(cache, "async_get_cache", unexpected_python_cache)
        monkeypatch.setattr(cache, "async_add_cache", unexpected_python_cache)
    monkeypatch.setattr(litellm, "cache", cache)

    first: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))
    await asyncio.sleep(0.05)
    second: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert first == second
    assert_served_natively(messages_server)


@pytest.mark.asyncio
async def test_native_cache_hit_runs_success_callbacks_without_replaying_inference(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    await drain_logging()
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    cache._native_cache = resolve_response_cache(  # pyright: ignore[reportPrivateUsage]  # configure native storage for inference
        cache,
        rules=(CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.LOCAL})),),
    )
    assert cache._native_cache is not None  # pyright: ignore[reportPrivateUsage]  # require the configured backend
    monkeypatch.setattr(litellm, "cache", cache)
    recorder: Final = RecordingLogger()
    kwargs: Final = arguments(messages_server, callbacks=[recorder], litellm_trace_id="cache-child")
    token: Final = trace_id_var.set("cache-hit-parent")

    try:
        first: Final = await litellm.anthropic.messages.acreate(**kwargs)
        await asyncio.sleep(0.05)
        second: Final = await litellm.anthropic.messages.acreate(**kwargs)
        successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)

        assert first == second
        assert_served_natively(messages_server)
        assert len(successes) == 2
        assert successes[1].kwargs["cache_hit"] is True
        assert trace_id_var.get() == "cache-hit-parent"
    finally:
        trace_id_var.reset(token)


@pytest.mark.asyncio
async def test_native_cache_hit_runs_deployment_hook_once_per_call(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    cache._native_cache = resolve_response_cache(  # pyright: ignore[reportPrivateUsage]  # configure native storage for the cache hit
        cache,
        rules=(CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.LOCAL})),),
    )
    assert cache._native_cache is not None  # pyright: ignore[reportPrivateUsage]  # require native storage
    events: Final = []

    class Deployment(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            events.append("deployment")
            return kwargs

    monkeypatch.setattr(litellm, "cache", cache)
    monkeypatch.setattr(litellm, "callbacks", [Deployment()])

    first: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))
    await asyncio.sleep(0.05)
    second: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert first == second
    assert_served_natively(messages_server)
    assert events == ["deployment", "deployment"]


@pytest.mark.parametrize("native_cache", (False, True))
@pytest.mark.asyncio
async def test_native_messages_no_cache_replaces_stored_response(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, native_cache: bool
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    if native_cache:
        cache._native_cache = resolve_response_cache(  # pyright: ignore[reportPrivateUsage]  # exercise native cache directives
            cache,
            rules=(CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.LOCAL})),),
        )
        assert cache._native_cache is not None  # pyright: ignore[reportPrivateUsage]  # require the configured backend
    monkeypatch.setattr(litellm, "cache", cache)
    messages_server.expected_requests = 2
    replacement: Final = {
        **MESSAGES_RESPONSE,
        "id": "msg_replacement",
        "content": [{"type": "text", "text": "replacement"}],
    }
    messages_server.enqueue(ResponseSpec(body=MESSAGES_RESPONSE))
    messages_server.enqueue(ResponseSpec(body=replacement))

    first: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))
    await asyncio.sleep(0.05)
    refreshed: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, cache={"no-cache": True}))
    await asyncio.sleep(0.05)
    cached: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert first["id"] != refreshed["id"]
    assert cached == refreshed
    assert len(messages_server.requests) == 2


@pytest.mark.parametrize(
    ("mode", "control", "requests"),
    (
        (CacheMode.default_on, {"no-store": True}, 2),
        (CacheMode.default_off, {}, 2),
        (CacheMode.default_off, {"use-cache": True}, 1),
    ),
)
@pytest.mark.asyncio
async def test_native_cache_honors_store_and_opt_in_policy(
    messages_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    mode: CacheMode,
    control: dict[str, bool],
    requests: int,
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, mode=mode)
    cache._native_cache = resolve_response_cache(  # pyright: ignore[reportPrivateUsage]  # configure native cache policy
        cache,
        rules=(CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.LOCAL})),),
    )
    assert cache._native_cache is not None  # pyright: ignore[reportPrivateUsage]  # require native storage
    monkeypatch.setattr(litellm, "cache", cache)
    messages_server.expected_requests = requests
    kwargs: Final = arguments(messages_server, **({"cache": control} if control else {}))

    await litellm.anthropic.messages.acreate(**kwargs)
    await asyncio.sleep(0.05)
    await litellm.anthropic.messages.acreate(**kwargs)

    assert len(messages_server.requests) == requests


@pytest.mark.parametrize("native_cache", (False, True))
@pytest.mark.asyncio
async def test_native_messages_stream_is_replayed_from_the_configured_cache(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, native_cache: bool
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    if native_cache:
        cache._native_cache = resolve_response_cache(  # pyright: ignore[reportPrivateUsage]  # select the native store for this cache integration test
            cache,
            rules=(CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.LOCAL})),),
        )
        assert cache._native_cache is not None  # pyright: ignore[reportPrivateUsage]  # confirm the selected backend

        async def unexpected_python_cache(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("native inference accessed the Python cache facade")

        monkeypatch.setattr(cache, "async_get_cache", unexpected_python_cache)
        monkeypatch.setattr(cache, "async_add_cache", unexpected_python_cache)
    monkeypatch.setattr(litellm, "cache", cache)
    messages_server.enqueue(STREAM)

    first_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    first: Final = b"".join([chunk async for chunk in first_stream])
    second_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    second: Final = b"".join([chunk async for chunk in second_stream])

    assert first == second == sse_payload()
    assert second_stream._hidden_params["cache_hit"] is True
    assert_served_natively(messages_server)


@pytest.mark.asyncio
async def test_abandoned_native_cached_stream_does_not_store_partial_events(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    cache._native_cache = resolve_response_cache(  # pyright: ignore[reportPrivateUsage]  # configure the native stream cache
        cache,
        rules=(CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.LOCAL})),),
    )
    assert cache._native_cache is not None  # pyright: ignore[reportPrivateUsage]  # require native storage
    monkeypatch.setattr(litellm, "cache", cache)
    messages_server.expected_requests = 2
    messages_server.enqueue(STREAM)
    messages_server.enqueue(STREAM)

    first: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    await first.__anext__()
    await first.aclose()
    second: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))

    assert b"".join([chunk async for chunk in second]) == sse_payload()
    assert len(messages_server.requests) == 2


@pytest.mark.asyncio
async def test_native_cached_stream_replay_logs_success_once(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    await drain_logging()
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    cache._native_cache = resolve_response_cache(  # pyright: ignore[reportPrivateUsage]  # configure native stream caching
        cache,
        rules=(CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.LOCAL})),),
    )
    assert cache._native_cache is not None  # pyright: ignore[reportPrivateUsage]  # require native storage
    monkeypatch.setattr(litellm, "cache", cache)
    recorder: Final = RecordingLogger()
    kwargs: Final = arguments(messages_server, stream=True, callbacks=[recorder])
    messages_server.enqueue(STREAM)

    first: Final = await litellm.anthropic.messages.acreate(**kwargs)
    assert b"".join([chunk async for chunk in first]) == sse_payload()
    second: Final = await litellm.anthropic.messages.acreate(**kwargs)
    assert b"".join([chunk async for chunk in second]) == sse_payload()
    successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)

    assert_served_natively(messages_server)
    assert len(successes) == 2
    assert successes[1].kwargs["cache_hit"] is True


def sse_payload() -> bytes:
    return b"".join(STREAM.payloads())


@pytest.mark.asyncio
async def test_native_messages_stream_relays_provider_events_and_logs_success_once_after_the_last_chunk(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    first: Final = await anext(stream)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names
    rest: Final = [chunk async for chunk in stream]

    assert first + b"".join(rest) == sse_payload()
    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["stream"] is True
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].kwargs["stream"] is True
    assert success[0].kwargs["completion_start_time"] is not None
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_native_messages_stream_closed_early_logs_success_once_for_what_was_delivered(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    await anext(stream)
    await stream.aclose()

    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


def test_native_sync_messages_stream_relays_provider_events_and_logs_success_once(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = litellm.anthropic.messages.create(**arguments(messages_server, stream=True, callbacks=[recorder]))
    assert isinstance(stream, Iterator)

    assert b"".join(stream) == sse_payload()
    assert_served_natively(messages_server)
    assert len(recorder.wait_for("async_log_success_event")) == 1


def test_native_sync_messages_returns_the_provider_message(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    response: Final = litellm.anthropic.messages.create(**arguments(messages_server, callbacks=[recorder]))

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]
    assert len(recorder.wait_for("log_success_event")) == 1

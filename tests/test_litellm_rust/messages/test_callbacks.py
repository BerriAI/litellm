from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import rebound
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


@pytest.fixture(autouse=True)
def opt_messages_into_rust() -> Iterator[None]:
    with rebound(catalog, "RULES", (RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN), *catalog.RULES)):
        yield


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
    assert get_hidden_params_dict(stream)["additional_headers"]["x-litellm-rust"] == "true"
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
    assert get_hidden_params_dict(stream)["additional_headers"]["x-litellm-rust"] == "true"

    assert b"".join(stream) == sse_payload()
    assert_served_natively(messages_server)
    assert len(recorder.wait_for("async_log_success_event")) == 1


def test_native_sync_messages_returns_the_provider_message(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    response: Final = litellm.anthropic.messages.create(**arguments(messages_server, callbacks=[recorder]))

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]
    assert len(recorder.wait_for("log_success_event")) == 1


@pytest.mark.asyncio
async def test_native_messages_pre_call_sees_the_shaped_optional_params(
    messages_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    await litellm.anthropic.messages.acreate(
        **arguments(messages_server, callbacks=[recorder], temperature=0.2, top_k=3, drop_params=True)
    )

    sent: Final = messages_server.requests[0].body
    assert not {"temperature", "top_k"} & sent.keys()
    pre_call: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(pre_call, dict)
    optional_params: Final = pre_call["optional_params"]
    assert isinstance(optional_params, dict)
    assert not {"model", "messages", "temperature", "top_k"} & optional_params.keys()
    assert optional_params["max_tokens"] == sent["max_tokens"]


@pytest.mark.asyncio
async def test_native_messages_failing_pre_call_logger_does_not_fail_the_call(messages_server: RecordingServer) -> None:
    class Broken(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            raise RuntimeError("logger exploded")

    response: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[Broken()]))

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]


@pytest.mark.asyncio
async def test_native_messages_stream_success_log_carries_usage_rebuilt_from_the_relayed_events(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    async for _ in stream:
        pass

    success: Final = await recorder.wait_for_async("async_log_success_event")
    usage: Final = success[0].response.usage
    assert usage.completion_tokens == MESSAGES_EVENTS[4][1]["usage"]["output_tokens"]
    assert usage.prompt_tokens == MESSAGES_RESPONSE["usage"]["input_tokens"]
    assert success[0].response.choices[0].message.content == "Hello from native Messages"


def test_native_messages_dispatches_each_callback_phase_once_when_logger_is_registered_multiple_times(
    messages_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    litellm.anthropic.messages.create(
        **arguments(
            messages_server,
            callbacks=[recorder, recorder],
            success_callback=[recorder],
            failure_callback=[recorder],
        )
    )
    recorder.wait_for("log_success_event")

    assert recorder.names.count("log_pre_api_call") == 1
    assert recorder.names.count("logging_hook") == 1
    assert recorder.names.count("log_success_event") == 1
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_native_pre_request_hooks_share_the_callers_objects_and_chain_edits(
    messages_server: RecordingServer,
) -> None:
    import asyncio

    messages: Final = [{"role": "user", "content": "hello"}]
    tools: Final = [{"name": "original", "input_schema": {"type": "object"}}]
    renamed: Final = [{"name": "renamed", "input_schema": {"type": "object"}}]
    task: Final = asyncio.current_task()

    class Rewrite(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            return {**kwargs, "messages": caller_messages}

        async def async_pre_request_hook(
            self, model: str, messages: object, kwargs: dict[str, object]
        ) -> dict[str, object]:
            assert asyncio.current_task() is task
            assert kwargs["tools"] is tools
            assert kwargs["litellm_params"] == {"custom_llm_provider": "anthropic"}
            return {**kwargs, "tools": renamed, "max_tokens": 128}

    class Observe(CustomLogger):
        async def async_pre_request_hook(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            assert asyncio.current_task() is task
            assert kwargs["tools"] is renamed
            assert kwargs["max_tokens"] == 128
            assert messages is caller_messages

    caller_messages: Final = messages
    litellm.callbacks.extend((Rewrite(), Observe()))
    recorder: Final = RecordingLogger()
    await litellm.anthropic.messages.acreate(
        **arguments(messages_server, messages=messages, tools=tools, callbacks=[recorder])
    )
    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["tools"] == renamed
    assert messages_server.requests[0].body["max_tokens"] == 128
    assert request_body(recorder.wait_for("log_pre_api_call")[0].kwargs) == messages_server.requests[0].body


@pytest.mark.asyncio
async def test_native_pre_request_rejection_preserves_the_error_without_sending(
    messages_server: RecordingServer,
) -> None:
    refused: Final = ValueError("pre-request refused")

    class Reject(CustomLogger):
        async def async_pre_request_hook(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            raise refused

    messages_server.expected_requests = 0
    litellm.callbacks.append(Reject())
    recorder: Final = RecordingLogger()
    with pytest.raises(ValueError, match="pre-request refused") as raised:
        await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[recorder]))
    assert raised.value is refused
    assert messages_server.requests == []
    failure: Final = await recorder.wait_for_async("async_log_failure_event")
    assert len(failure) == 1
    assert failure[0].kwargs["exception"] is refused
    assert "async_log_success_event" not in recorder.names


@pytest.mark.asyncio
async def test_native_pre_request_stream_conversion_preserves_content_and_billed_usage(
    messages_server: RecordingServer,
) -> None:
    class Convert(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: object, kwargs: dict[str, object]
        ) -> dict[str, object]:
            return {**kwargs, "stream": False}

    litellm.callbacks.append(Convert())
    recorder: Final = RecordingLogger()
    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    chunks: Final = tuple([chunk async for chunk in stream])
    assert chunks and b"event: message_stop" in chunks[-1]
    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["stream"] is False
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].response.usage.completion_tokens == MESSAGES_RESPONSE["usage"]["output_tokens"]
    assert success[0].response.usage.prompt_tokens == MESSAGES_RESPONSE["usage"]["input_tokens"]
    assert success[0].response.choices[0].message.content == MESSAGES_RESPONSE["content"][0]["text"]


@pytest.mark.asyncio
async def test_abandoned_stream_collects_a_cycle_through_retained_callback_headers(
    messages_server: RecordingServer,
) -> None:
    import gc
    import weakref

    from tests.test_litellm_rust.support.requests import request_headers

    class Capture(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.headers: dict[str, object] | None = None

        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            self.headers = request_headers(kwargs)

    async def create_cycle() -> weakref.ReferenceType[object]:
        observer: Final = Capture()
        stream: Final = await litellm.anthropic.messages.acreate(
            **arguments(messages_server, stream=True, callbacks=[observer])
        )
        assert observer.headers is not None
        observer.headers["cycle"] = stream
        return weakref.ref(stream)

    from tests.test_litellm_rust.support.isolation import isolated_callback_registries

    messages_server.enqueue(STREAM)
    with isolated_callback_registries():
        reference: Final = await create_cycle()
        await drain_logging()
    gc.collect()
    assert reference() is None

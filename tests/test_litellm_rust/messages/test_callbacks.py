from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.pass_through.messages.handler import anthropic_messages
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES
from litellm.rust_bridge.public_call import native_call, signature
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
@pytest.mark.parametrize("blocked", (False, True), ids=("accepted", "blocked"))
async def test_native_messages_deferred_success_waits_for_proxy_acceptance(
    messages_server: RecordingServer, blocked: bool
) -> None:
    recorder: Final = RecordingLogger()
    logger: Final = Logging(
        model=MESSAGES_MODEL,
        messages=list(MESSAGES),
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.now(),
        litellm_call_id="deferred-messages",
        function_id="deferred-messages",
        dynamic_async_success_callbacks=[recorder],
    )
    logger.defer_async_logging = True
    binding: Final = NATIVE_AMESSAGES.load()
    assert binding is not None
    response: Final = await binding(
        native_call(signature(anthropic_messages), (), arguments(messages_server, litellm_logging_obj=logger))
    )
    assert isinstance(response, dict)
    assert_served_natively(messages_server)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names

    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logger, exception_raised=blocked)
    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logger, exception_raised=False)
    await drain_logging()

    assert getattr(logger, "_native_pending_logging", None) is None
    if blocked:
        assert "async_log_success_event" not in recorder.names
    else:
        success: Final = await recorder.wait_for_async("async_log_success_event")
        assert len(success) == 1
        assert success[0].response.choices[0].message.content == response["content"][0]["text"]


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


@pytest.mark.asyncio
async def test_native_messages_does_not_restore_keywords_deleted_by_a_pre_call_hook(
    messages_server: RecordingServer,
) -> None:
    class DropKeywords(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            return {name: value for name, value in kwargs.items() if name not in ("system", "extra_headers")}

    call_arguments: Final = arguments(
        messages_server,
        system="remove this instruction",
        extra_headers={"x-deleted": "remove this header"},
        headers={"x-kept": "keep this header"},
        stream=False,
    )
    with rebound(litellm, "callbacks", [DropKeywords()]):
        await litellm.anthropic.messages.acreate(**call_arguments)

    assert_served_natively(messages_server)
    sent: Final = messages_server.requests[0]
    assert sent.body == {
        "model": MESSAGES_MODEL.removeprefix("anthropic/"),
        "messages": call_arguments["messages"],
        "max_tokens": call_arguments["max_tokens"],
        "stream": call_arguments["stream"],
    }
    assert sent.headers.get("x-deleted") is None
    assert sent.headers.get("x-kept") == "keep this header"

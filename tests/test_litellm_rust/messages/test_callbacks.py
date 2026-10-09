from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.pass_through.messages.handler import anthropic_messages
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import catalog
from litellm.rust_bridge.bindings import native_exception_types
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES
from litellm.rust_bridge.public_call import native_call, signature
from litellm.types.utils import Choices, ModelResponse
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
async def test_native_messages_pre_request_edits_reach_provider_in_the_caller_task(
    messages_server: RecordingServer,
) -> None:
    import asyncio
    from contextvars import ContextVar

    caller: Final = asyncio.current_task()
    marker: Final = ContextVar("native-messages-pre-request", default="caller")
    tools: Final = [{"name": "lookup", "input_schema": {"type": "object"}}]
    original_choice: Final = {"type": "auto"}
    prepared_choice: Final = {"type": "tool", "name": "lookup"}

    class Edit(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            return {**kwargs, "hook_order": "deployment"}

        async def async_pre_request_hook(
            self, model: str, messages: list[object], kwargs: dict[str, object]
        ) -> dict[str, object]:
            await asyncio.sleep(0)
            assert asyncio.current_task() is caller
            assert marker.get() == "caller"
            assert kwargs["hook_order"] == "deployment"
            assert kwargs["tool_choice"] is original_choice
            marker.set("hook")
            return {**kwargs, "tools": tools, "tool_choice": prepared_choice, "stop_sequences": ["callback-stop"]}

    binding: Final = NATIVE_AMESSAGES.load()
    assert binding is not None
    with rebound(litellm, "callbacks", [Edit()]):
        response: Final = await binding(
            native_call(signature(anthropic_messages), (), arguments(messages_server, tool_choice=original_choice))
        )

    assert isinstance(response, dict)
    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["tools"] == tools
    assert messages_server.requests[0].body["stop_sequences"] == ["callback-stop"]
    assert messages_server.requests[0].body["tool_choice"] == prepared_choice
    assert response["content"] == MESSAGES_RESPONSE["content"]
    assert marker.get() == "hook"


@pytest.mark.asyncio
async def test_native_messages_agentic_loop_runs_inside_the_wrapper_and_its_answer_is_what_gets_logged(
    messages_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()
    replacement: Final[Mapping[str, object]] = {
        **MESSAGES_RESPONSE,
        "content": [{"type": "text", "text": "from the agentic loop"}],
    }
    seen: Final[list[object]] = []

    class Loop(CustomLogger):
        async def async_should_run_agentic_loop(
            self,
            response: object,
            model: str,
            messages: object,
            tools: object,
            stream: bool,
            custom_llm_provider: str,
            kwargs: dict[str, object],
        ) -> tuple[bool, dict[str, object]]:
            seen.append(response)
            assert kwargs["api_key"] == "test-key"
            return True, {}

        async def async_run_agentic_loop(
            self,
            tools: object,
            model: str,
            messages: object,
            response: object,
            anthropic_messages_provider_config: object,
            anthropic_messages_optional_request_params: object,
            logging_obj: object,
            stream: bool,
            kwargs: dict[str, object],
        ) -> object:
            return replacement

        async def async_post_call_success_deployment_hook(
            self, request_data: object, response: object, call_type: object
        ) -> None:
            seen.append(response)

    binding: Final = NATIVE_AMESSAGES.load()
    assert binding is not None
    with rebound(litellm, "callbacks", [Loop()]):
        response: Final = await binding(
            native_call(signature(anthropic_messages), (), arguments(messages_server, callbacks=[recorder]))
        )

    assert_served_natively(messages_server)
    assert isinstance(response, dict)
    assert dict(response)["content"] == replacement["content"]
    assert len(seen) == 2
    assert isinstance(seen[0], dict) and seen[0]["content"] == MESSAGES_RESPONSE["content"]
    assert seen[1] is response
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    logged: Final = success[0].response
    assert isinstance(logged, ModelResponse)
    choice: Final = logged.choices[0]
    assert isinstance(choice, Choices)
    assert choice.message.content == "from the agentic loop"


@pytest.mark.asyncio
async def test_streaming_with_an_agentic_loop_hook_falls_back_to_python_which_runs_the_loop_at_end_of_stream(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    seen: Final[list[bool]] = []

    class Loop(CustomLogger):
        async def async_should_run_agentic_loop(
            self,
            response: object,
            model: str,
            messages: object,
            tools: object,
            stream: bool,
            custom_llm_provider: str,
            kwargs: dict[str, object],
        ) -> tuple[bool, dict[str, object]]:
            seen.append(stream)
            return False, {}

    binding: Final = NATIVE_AMESSAGES.load()
    assert binding is not None
    exceptions: Final = native_exception_types()
    assert exceptions is not None
    declined, _ = exceptions
    with rebound(litellm, "callbacks", [Loop()]):
        with pytest.raises(declined):
            await binding(native_call(signature(anthropic_messages), (), arguments(messages_server, stream=True)))
        assert not messages_server.requests
        stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
        assert isinstance(stream, AsyncIterator)
        assert not seen
        chunks: Final = [chunk async for chunk in stream]

    assert chunks
    assert len(messages_server.requests) == 1
    assert seen == [True]


@pytest.mark.asyncio
async def test_native_messages_pre_request_failure_never_sends_or_replays_provider_work(
    messages_server: RecordingServer,
) -> None:
    messages_server.expected_requests = 0
    failure: Final = RuntimeError("callback rejected")
    recorder: Final = RecordingLogger()

    class Reject(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.failure: Exception | None = None

        async def async_pre_request_hook(self, model: str, messages: list[object], kwargs: dict[str, object]) -> None:
            raise failure

        async def async_post_call_failure_deployment_hook(
            self, request_data: object, exception: Exception, call_type: object
        ) -> None:
            self.failure = exception

    binding: Final = NATIVE_AMESSAGES.load()
    assert binding is not None
    reject: Final = Reject()
    with rebound(litellm, "callbacks", [reject]), pytest.raises(RuntimeError) as raised:
        await binding(native_call(signature(anthropic_messages), (), arguments(messages_server, callbacks=[recorder])))

    assert raised.value is failure
    assert isinstance(reject.failure, RuntimeError)
    assert str(reject.failure) == str(failure)
    assert messages_server.requests == []
    failures: Final = recorder.wait_for("async_log_failure_event")
    assert len(failures) == 1
    assert failures[0].kwargs["exception"] is failure


@pytest.mark.asyncio
async def test_cancellation_during_request_preparation_stops_before_provider_and_terminal_callbacks(
    messages_server: RecordingServer,
) -> None:
    messages_server.expected_requests = 0
    import asyncio

    started: Final = asyncio.Event()
    hold: Final = asyncio.Event()
    recorder: Final = RecordingLogger()

    class Wait(CustomLogger):
        async def async_pre_request_hook(self, model: str, messages: list[object], kwargs: dict[str, object]) -> None:
            started.set()
            await hold.wait()

    binding: Final = NATIVE_AMESSAGES.load()
    assert binding is not None
    with rebound(litellm, "callbacks", [Wait()]):
        task: Final = asyncio.ensure_future(
            binding(native_call(signature(anthropic_messages), (), arguments(messages_server, callbacks=[recorder])))
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    await drain_logging()
    assert messages_server.requests == []
    assert "async_log_failure_event" not in recorder.names
    assert "async_log_success_event" not in recorder.names


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

    class AsyncOnly(CustomLogger):
        async def async_pre_request_hook(self, model: str, messages: list[object], kwargs: dict[str, object]) -> None:
            raise AssertionError("asynchronous request hooks cannot run on synchronous Messages")

    with rebound(litellm, "callbacks", [AsyncOnly()]):
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

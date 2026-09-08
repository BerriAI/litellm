import asyncio
import copy
import json
import threading
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.utils import CallTypes
from tests.test_litellm_rust.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.contracts import (
    MESSAGES,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
    request_body,
    request_headers,
)
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


async def call_messages(server: RecordingServer, callbacks: list[CustomLogger], **kwargs: object):
    return await litellm.anthropic.messages.acreate(
        model=MESSAGES_MODEL,
        messages=MESSAGES,
        max_tokens=64,
        api_key="test-key",
        api_base=server.base_url,
        callbacks=callbacks,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_messages_pre_call_receives_expected_provider_request(messages_server: RecordingServer) -> None:
    observations: Final = []

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append((model, messages, copy.deepcopy(kwargs["additional_args"])))

    await call_messages(messages_server, [Observe()])

    assert len(observations) == 1
    model, messages, additional_args = observations[0]
    assert model == "claude-sonnet-4-5-20250929"
    assert messages == MESSAGES
    assert additional_args["api_base"] == f"{messages_server.base_url}/v1/messages"
    assert additional_args["complete_input_dict"] == {
        "model": "claude-sonnet-4-5-20250929",
        "messages": MESSAGES,
        "max_tokens": 64,
    }
    assert additional_args["headers"]["x-api-key"] == "test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_after_edit", [False, True])
async def test_messages_pre_call_edits_reach_later_callbacks_and_provider(
    messages_server: RecordingServer, raise_after_edit: bool
) -> None:
    observed: Final = []

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["temperature"] = 0.25
            request_headers(kwargs)["x-audit-tag"] = "reviewed"
            if raise_after_edit:
                raise RuntimeError("audit exporter unavailable")

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append((copy.deepcopy(request_body(kwargs)), dict(request_headers(kwargs))))

    await call_messages(messages_server, [Edit(), Observe()])

    assert observed[0][0]["temperature"] == 0.25
    assert observed[0][1]["x-audit-tag"] == "reviewed"
    assert messages_server.requests[0].body["temperature"] == 0.25
    assert messages_server.requests[0].headers["x-audit-tag"] == "reviewed"


@pytest.mark.asyncio
async def test_messages_pre_call_rebinding_does_not_replace_inflight_request(
    messages_server: RecordingServer,
) -> None:
    observed: Final = []

    class Rebind(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["additional_args"]["complete_input_dict"] = {"replacement": True}

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append(request_body(kwargs))

    await call_messages(messages_server, [Rebind(), Observe()])

    assert observed == [{"replacement": True}]
    assert messages_server.requests[0].body["messages"] == MESSAGES


@pytest.mark.asyncio
async def test_messages_pre_call_state_reaches_terminal_callbacks(messages_server: RecordingServer) -> None:
    token: Final = object()
    observed: Final = []
    finished: Final = asyncio.Event()

    class Stash(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["test-token"] = token

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(kwargs["test-token"])
            finished.set()

    await call_messages(messages_server, [Stash()])
    await asyncio.wait_for(finished.wait(), timeout=10)

    assert observed == [token]


@pytest.mark.asyncio
async def test_messages_callbacks_run_once(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    await call_messages(messages_server, [recorder])
    await recorder.wait_for_async("async_log_success_event")

    assert recorder.names.count("log_pre_api_call") == 1
    assert recorder.names.count("async_logging_hook") == 1
    assert recorder.names.count("async_log_success_event") == 1
    assert "log_failure_event" not in recorder.names
    assert "async_log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_messages_logging_drain_waits_for_suspended_callback(messages_server: RecordingServer) -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    finished: Final = asyncio.Event()

    class SuspendedLogger(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            started.set()
            await release.wait()
            finished.set()

    await call_messages(messages_server, [SuspendedLogger()])
    draining: Final = asyncio.create_task(drain_logging())
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        await asyncio.sleep(0)
        assert not draining.done()
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.wait_for(draining, timeout=10)

    assert finished.is_set()


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="accepted native Messages errors are replayed through the Python transport")
async def test_messages_failure_callbacks_receive_original_provider_error(messages_server: RecordingServer) -> None:
    messages_server.default_response = ResponseSpec(body={"error": {"message": "provider unavailable"}}, status=500)
    messages_server.expected_requests = None
    recorder: Final = RecordingLogger()

    with pytest.raises(litellm.InternalServerError) as caught:
        await call_messages(messages_server, [recorder])
    events: Final = await recorder.wait_for_async("async_log_failure_event")

    assert len(events) == 1
    assert events[0].call_type == "anthropic_messages"
    assert events[0].kwargs["exception"] is caught.value
    assert len(messages_server.requests) == 1
    assert "async_log_success_event" not in recorder.names


@pytest.mark.asyncio
async def test_messages_success_callbacks_receive_expected_context(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    await call_messages(
        messages_server,
        [recorder],
        litellm_call_id="messages-success",
        metadata={"source": "callback-test"},
    )
    async_events: Final = await recorder.wait_for_async("async_log_success_event")

    assert len(async_events) == 1
    event: Final = async_events[0]
    assert event.call_type == "anthropic_messages"
    assert event.kwargs["litellm_call_id"] == "messages-success"
    assert event.kwargs["litellm_params"]["metadata"]["source"] == "callback-test"
    assert event.response.choices[0].message.content == "Hello from native Messages"


@pytest.mark.asyncio
async def test_messages_pre_call_runs_in_callers_execution_context(messages_server: RecordingServer) -> None:
    caller_thread: Final = threading.current_thread()
    observations: Final = []

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observations.append((asyncio.get_running_loop(), threading.current_thread()))

    caller_loop: Final = asyncio.get_running_loop()
    await call_messages(messages_server, [Observe()])

    assert observations == [(caller_loop, caller_thread)]


@pytest.mark.asyncio
async def test_messages_stream_logs_success_after_exhaustion(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()
    stream: Final = await call_messages(messages_server, [recorder], stream=True)

    assert "async_log_success_event" not in recorder.names
    chunks: Final = [chunk async for chunk in stream]
    events: Final = await recorder.wait_for_async("async_log_success_event")

    assert chunks
    assert len(events) == 1
    assert "async_log_stream_event" not in recorder.names
    assert events[0].kwargs["complete_streaming_response"] is not None


@pytest.mark.asyncio
async def test_messages_compression_hook_replaces_messages_sent_to_provider(messages_server: RecordingServer) -> None:
    compressed_messages: Final = [{"role": "user", "content": "Compressed context"}]
    call_types: Final = []

    class CompressMessages(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            call_types.append(call_type)
            kwargs["messages"] = compressed_messages
            return kwargs

    litellm.callbacks.append(CompressMessages())

    await call_messages(messages_server, [])

    assert call_types == [CallTypes.anthropic_messages]
    assert messages_server.requests[0].body["messages"] == compressed_messages


@pytest.mark.asyncio
async def test_messages_post_call_guardrail_replacement_reaches_caller_and_logging(
    messages_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()
    call_types: Final = []

    class ReviewResponse(CustomLogger):
        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            call_types.append(call_type)
            response["content"][0]["text"] = "Reviewed response"
            return response

    litellm.callbacks.append(ReviewResponse())

    response: Final = await call_messages(messages_server, [recorder])
    events: Final = await recorder.wait_for_async("async_log_success_event")

    assert call_types == [CallTypes.anthropic_messages]
    assert response["content"][0]["text"] == "Reviewed response"
    assert events[0].response.choices[0].message.content == "Reviewed response"


@pytest.mark.asyncio
async def test_messages_logging_hook_replacement_reaches_later_loggers_only(messages_server: RecordingServer) -> None:
    observations: Final = []
    exported: Final = asyncio.Event()

    class RecordGuardrailVerdict(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            observations.append("guardrail")
            return {**kwargs, "guardrail-verdict": "allowed"}, result

    class ExportLog(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            return kwargs, result

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            observations.append(("export", kwargs["guardrail-verdict"]))
            exported.set()

    response: Final = await call_messages(messages_server, [RecordGuardrailVerdict(), ExportLog()])
    await asyncio.wait_for(exported.wait(), timeout=10)

    assert observations == ["guardrail", ("export", "allowed")]
    assert response["content"][0]["text"] == "Hello from native Messages"


@pytest.mark.asyncio
async def test_messages_success_callback_failure_does_not_skip_later_loggers(
    messages_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    class UnavailableExporter(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            raise RuntimeError("exporter unavailable")

    response: Final = await call_messages(messages_server, [UnavailableExporter(), recorder])
    events: Final = await recorder.wait_for_async("async_log_success_event")

    assert response["content"][0]["text"] == "Hello from native Messages"
    assert len(events) == 1
    assert "async_log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_messages_concurrent_calls_keep_callback_state_isolated(messages_server: RecordingServer) -> None:
    messages_server.expected_requests = 4
    tokens: Final = {f"messages-{index}": object() for index in range(4)}
    terminal_state: Final = []

    class CorrelateCallState(RecordingLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            kwargs["correlation-token"] = tokens[kwargs["litellm_call_id"]]
            super().log_pre_api_call(model, messages, kwargs)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            terminal_state.append((kwargs["litellm_call_id"], kwargs["correlation-token"]))
            await super().async_log_success_event(kwargs, response_obj, start_time, end_time)

    correlate: Final = CorrelateCallState()

    await asyncio.gather(*(call_messages(messages_server, [correlate], litellm_call_id=call_id) for call_id in tokens))
    await correlate.wait_for_async("async_log_success_event", count=4)

    assert len(terminal_state) == 4
    assert all(token is tokens[call_id] for call_id, token in terminal_state)


@pytest.mark.asyncio
async def test_messages_cancelled_call_runs_no_terminal_callbacks(messages_server: RecordingServer) -> None:
    messages_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE, delay=0.5)
    recorder: Final = RecordingLogger()
    task: Final = asyncio.create_task(call_messages(messages_server, [recorder]))

    async with asyncio.timeout(10):
        while not messages_server.requests:
            await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)

    assert recorder.names.count("log_pre_api_call") == 1
    assert "log_success_event" not in recorder.names
    assert "async_log_success_event" not in recorder.names
    assert "log_failure_event" not in recorder.names
    assert "async_log_failure_event" not in recorder.names

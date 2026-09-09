import asyncio
import copy
import threading
from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.rust_bridge.provenance import has_rust_response_marker
from litellm.types.utils import CallTypes
from tests.test_litellm_rust.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.contracts import (
    MESSAGES,
    MESSAGES_EVENTS,
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
        "stream": False,
    }
    assert additional_args["headers"]["x-api-key"] == "test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("raise_after_edit", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_messages_pre_call_edits_reach_later_callbacks_and_provider(
    messages_server: RecordingServer, raise_after_edit: bool, native: bool, stream: bool
) -> None:
    litellm.rust(native)
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

    if stream:
        messages_server.enqueue(ResponseSpec(body=None, events=MESSAGES_EVENTS))
    response: Final = await call_messages(messages_server, [Edit(), Observe()], stream=stream)
    if stream:
        async for _ in response:
            pass

    assert observed[0][0]["temperature"] == 0.25
    assert observed[0][1]["x-audit-tag"] == "reviewed"
    assert "temperature" not in messages_server.requests[0].body
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
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("provider", ["anthropic", "azure_ai"])
@pytest.mark.parametrize("raise_after_edit", [False, True])
async def test_messages_retained_aliases_preserve_identity_and_snapshot_timing(
    messages_server: RecordingServer, native: bool, provider: str, raise_after_edit: bool
) -> None:
    litellm.rust(native)
    retained: Final = []
    observed: Final = []

    class Retain(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            body = request_body(kwargs)
            message = kwargs["messages"][0]
            assert body["messages"][0] is message
            assert body["messages"][0]["content"] is message["content"]
            assert body["messages"][0]["content"][0] is message["content"][0]
            retained.append(message["content"][0])

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            retained[0]["text"] = "changed through retained reference"
            if raise_after_edit:
                raise RuntimeError("export failed after mutation")

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            block = request_body(kwargs)["messages"][0]["content"][0]
            observed.append((block is retained[0], block["text"]))

    response: Final = await litellm.anthropic.messages.acreate(
        model=MESSAGES_MODEL.replace("anthropic/", f"{provider}/"),
        messages=[{"role": "user", "content": [{"type": "text", "text": "original"}]}],
        max_tokens=64,
        api_key="test-key",
        api_base=messages_server.base_url,
        callbacks=[Retain(), Edit(), Observe()],
    )
    assert observed == [(True, "changed through retained reference")]
    assert retained[0]["text"] == "changed through retained reference"
    assert messages_server.requests[0].body["messages"][0]["content"][0]["text"] == "original"
    if native:
        assert response["_hidden_params"]["additional_headers"]["x-litellm-rust"] == "true"


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
async def test_messages_unconsumed_stream_close_logs_failure_once(messages_server: RecordingServer) -> None:
    messages_server.enqueue(ResponseSpec(body=None, events=MESSAGES_EVENTS))
    recorder: Final = RecordingLogger()
    stream: Final = await call_messages(messages_server, [recorder], stream=True)
    assert "async_log_success_event" not in recorder.names
    await stream.aclose()
    await stream.aclose()
    await recorder.wait_for_async("async_log_failure_event")
    assert recorder.names.count("async_log_failure_event") == 1
    assert "async_log_success_event" not in recorder.names


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
    messages_server.enqueue(ResponseSpec(body=None, events=MESSAGES_EVENTS))
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
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_messages_compression_hook_replaces_messages_sent_to_provider(
    messages_server: RecordingServer, native: bool, stream: bool
) -> None:
    litellm.rust(native)
    compressed_messages: Final = [{"role": "user", "content": "Compressed context"}]
    call_types: Final = []
    recorder: Final = RecordingLogger()

    class CompressMessages(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            assert not messages_server.requests
            call_types.append(call_type)
            return {**kwargs, "messages": compressed_messages}

    litellm.callbacks.append(CompressMessages())

    if stream:
        messages_server.enqueue(ResponseSpec(body=None, events=MESSAGES_EVENTS))
    response: Final = await call_messages(messages_server, [recorder], stream=stream)
    assert has_rust_response_marker(response) is native
    if stream:
        assert [chunk async for chunk in response]
    await recorder.wait_for_async("async_log_success_event")

    assert call_types == [CallTypes.anthropic_messages]
    assert messages_server.requests[0].body["messages"] == compressed_messages
    pre_calls: Final = tuple(event for event in recorder.events if event.name == "log_pre_api_call")
    assert len(pre_calls) == 1
    assert request_body(pre_calls[0].kwargs)["messages"] == compressed_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_messages_deployment_rejection_prevents_provider_call(
    messages_server: RecordingServer, native: bool, stream: bool
) -> None:
    litellm.rust(native)
    messages_server.expected_requests = 0
    calls: Final[list[CallTypes | None]] = []
    recorder: Final = RecordingLogger()

    class Reject(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: CallTypes | None
        ) -> dict[str, object]:
            calls.append(call_type)
            raise litellm.BadRequestError(
                message="deployment policy rejected request", model=MESSAGES_MODEL, llm_provider="anthropic"
            )

    litellm.callbacks.append(Reject())

    with pytest.raises(litellm.BadRequestError, match="deployment policy rejected request"):
        await call_messages(messages_server, [recorder], stream=stream)

    assert calls == [CallTypes.anthropic_messages]
    assert not messages_server.requests
    assert "log_pre_api_call" not in recorder.names


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_messages_provider_open_failure_notifies_deployment_once_and_preserves_error(
    messages_server: RecordingServer, native: bool, stream: bool
) -> None:
    litellm.rust(native)
    messages_server.enqueue(ResponseSpec(body={"error": {"message": "provider unavailable"}}, status=500))
    calls: Final[list[tuple[CallTypes | None, Exception, int]]] = []

    class FailingObserver(CustomLogger):
        async def async_post_call_failure_deployment_hook(
            self,
            request_data: Mapping[str, object],
            exception: Exception,
            call_type: CallTypes | None,
            fallback_depth: int | None = None,
        ) -> None:
            calls.append((call_type, exception, len(messages_server.requests)))
            raise RuntimeError("deployment observer unavailable")

    litellm.callbacks.append(FailingObserver())
    recorder: Final = RecordingLogger()

    with pytest.raises(litellm.InternalServerError) as raised:
        await call_messages(messages_server, [recorder], stream=stream)
    await recorder.wait_for_async("async_log_failure_event")

    assert len(calls) == 1
    call_type, exception, request_count = calls[0]
    assert call_type == CallTypes.anthropic_messages
    assert isinstance(exception, litellm.InternalServerError)
    assert exception.status_code == raised.value.status_code == 500
    assert request_count == len(messages_server.requests) == 1
    assert recorder.names.count("async_log_failure_event") == 1
    assert "async_log_success_event" not in recorder.names


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
@pytest.mark.parametrize("stream", [False, True])
async def test_messages_cancelled_call_runs_no_terminal_callbacks(
    messages_server: RecordingServer, stream: bool
) -> None:
    messages_server.default_response = ResponseSpec(
        body=MESSAGES_RESPONSE,
        delay=0.5,
        events=MESSAGES_EVENTS if stream else (),
    )
    recorder: Final = RecordingLogger()
    task: Final = asyncio.create_task(call_messages(messages_server, [recorder], stream=stream))

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


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 500])
async def test_whole_call_with_supplied_logger_still_owns_terminal_callbacks(
    messages_server: RecordingServer, status: int
) -> None:
    from litellm.rust_bridge import messages as bridge

    events: Final = []

    class SuppliedLogger:
        def pre_call(self, **kwargs):
            events.append(("pre", kwargs))

        async def async_success_handler(self, response, start, end):
            events.append(("success", response))

        def _should_run_sync_callbacks_for_async_calls(self):
            return False

        def failure_handler(self, error, trace, start, end):
            events.append(("sync_failure", error))

        async def async_failure_handler(self, error, trace, start, end):
            events.append(("async_failure", error))

    messages_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE, status=status)
    arguments: Final = {
        "model": "claude-opus-5",
        "body": {"model": "claude-opus-5", "messages": MESSAGES, "max_tokens": 64},
        "api_key": "test",
        "api_base": messages_server.base_url,
        "custom_llm_provider": "anthropic",
        "extra_headers": {},
        "timeout": 5.0,
        "logging_obj": SuppliedLogger(),
    }
    if status == 200:
        response: Final = await bridge.amessages(**arguments)
        await drain_logging()
        assert [name for name, value in events] == ["pre", "success"]
        assert events[1][1] is response
        return

    with pytest.raises(litellm.InternalServerError) as raised:
        await bridge.amessages(**arguments)
    assert [name for name, value in events] == ["pre", "sync_failure", "async_failure"]
    assert events[1][1] is raised.value
    assert events[2][1] is raised.value


@pytest.mark.asyncio
async def test_direct_bridge_stream_owns_callbacks(messages_server: RecordingServer) -> None:
    from litellm.rust_bridge import messages as bridge

    recorder: Final = RecordingLogger()
    messages_server.enqueue(ResponseSpec(body=None, events=MESSAGES_EVENTS))
    stream: Final = await bridge.amessages(
        model=MESSAGES_MODEL.split("/", 1)[-1],
        body={"model": MESSAGES_MODEL.split("/", 1)[-1], "messages": MESSAGES, "max_tokens": 64, "stream": True},
        api_key="test-key",
        api_base=messages_server.base_url,
        custom_llm_provider="anthropic",
        extra_headers=None,
        timeout=5.0,
        request_arguments={"callbacks": [recorder]},
    )
    assert "async_log_success_event" not in recorder.names
    assert b"message_stop" in b"".join([chunk async for chunk in stream])
    events: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(events) == 1
    assert events[0].stream is True
    assert events[0].response.choices[0].message.content == "Hello from native Messages"
    assert recorder.names.count("log_pre_api_call") == 1
    await stream.aclose()

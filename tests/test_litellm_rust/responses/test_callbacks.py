from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import Final

import pytest
from pydantic import BaseModel

import litellm
from litellm.integrations.argilla import ArgillaLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.integrations.argilla import ArgillaItem
from litellm.types.utils import StandardLoggingPayload
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.callbacks import (
    CALL_ID,
    RESPONSES_SURFACE,
    HookRecorder,
    LegacyAgenticLoop,
    PlannedAgenticLoop,
    agentic_hooks,
    assert_transport,
    gap,
    consume,
    response_id,
    with_id,
)
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import RESPONSES_EVENTS, RESPONSES_RESPONSE, request_body

pytestmark = pytest.mark.requires_rust_extension

SURFACE: Final = RESPONSES_SURFACE
STREAM: Final = ResponseSpec(body=None, events=RESPONSES_EVENTS)
SYNC_HANDLERS_ON_ASYNC_CALLS: Final = gap("native async calls also run the sync success and failure handlers")
SYNC_DEPLOYMENT_HOOK: Final = gap("native sync calls skip the async pre-call deployment hook")
STREAMING: Final = gap("native Responses streaming declines to Python", raises=RuntimeError)
BODY_HOOKS: Final = gap("native Responses does not run the handler body hooks")


def success_signal(asynchronous: bool) -> str:
    return "async_log_success_event" if asynchronous else "log_success_event"


@pytest.fixture
def server(recording_server: RecordingServer) -> Iterator[RecordingServer]:
    recording_server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
    recording_server.expected_requests = None
    yield recording_server
    recording_server.responses.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook,asynchronous,expected",
    (
        pytest.param("log_pre_api_call", True, 1, id="async-log_pre_api_call"),
        pytest.param("log_post_api_call", True, 1, id="async-log_post_api_call"),
        pytest.param("async_log_success_event", True, 1, id="async-async_log_success_event"),
        pytest.param("async_logging_hook", True, 1, id="async-async_logging_hook"),
        pytest.param("redact_standard_logging_payload_from_model_call_details", True, 1, id="async-redact"),
        pytest.param("async_pre_call_deployment_hook", True, 1, id="async-async_pre_call_deployment_hook"),
        pytest.param(
            "async_post_call_success_deployment_hook", True, 1, id="async-async_post_call_success_deployment_hook"
        ),
        pytest.param("log_success_event", True, 0, id="async-log_success_event"),
        pytest.param("logging_hook", True, 0, id="async-logging_hook"),
        pytest.param("log_pre_api_call", False, 1, id="sync-log_pre_api_call"),
        pytest.param("log_post_api_call", False, 1, id="sync-log_post_api_call"),
        pytest.param("log_success_event", False, 1, id="sync-log_success_event"),
        pytest.param("logging_hook", False, 1, id="sync-logging_hook"),
        pytest.param(
            "async_pre_call_deployment_hook",
            False,
            1,
            id="sync-async_pre_call_deployment_hook",
            marks=SYNC_DEPLOYMENT_HOOK,
        ),
        pytest.param(
            "async_post_call_success_deployment_hook", False, 0, id="sync-async_post_call_success_deployment_hook"
        ),
    ),
)
async def test_success_hook_fires_the_expected_number_of_times(
    server: RecordingServer, hook: str, asynchronous: bool, expected: int
) -> None:
    recorder: Final = HookRecorder()

    with rebound(litellm, "callbacks", [recorder]):
        await SURFACE.call(server, asynchronous=asynchronous)
        await recorder.wait_for_async(success_signal(asynchronous))

    assert_transport(server)
    assert recorder.names.count(hook) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook,asynchronous,expected",
    (
        pytest.param("async_log_failure_event", True, 1, id="async-async_log_failure_event"),
        pytest.param(
            "async_post_call_failure_deployment_hook", True, 1, id="async-async_post_call_failure_deployment_hook"
        ),
        pytest.param("log_model_group_rate_limit_error", True, 1, id="async-log_model_group_rate_limit_error"),
        pytest.param("log_failure_event", True, 0, id="async-log_failure_event", marks=SYNC_HANDLERS_ON_ASYNC_CALLS),
        pytest.param("log_failure_event", False, 1, id="sync-log_failure_event"),
    ),
)
async def test_failure_hook_fires_the_expected_number_of_times(
    server: RecordingServer, hook: str, asynchronous: bool, expected: int
) -> None:
    server.default_response = ResponseSpec(body=SURFACE.rate_limited, status=429)
    recorder: Final = HookRecorder()

    with rebound(litellm, "callbacks", [recorder]):
        with pytest.raises(litellm.RateLimitError):
            await SURFACE.call(server, asynchronous=asynchronous, metadata={"model_group_size": 1})
        await recorder.wait_for_async("async_log_failure_event" if asynchronous else "log_failure_event")

    assert_transport(server)
    assert recorder.names.count(hook) == expected


@pytest.mark.asyncio
async def test_failure_events_carry_the_error_raised_to_the_caller(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body=SURFACE.bad_request, status=400)
    recorder: Final = RecordingLogger()

    with rebound(litellm, "callbacks", [recorder]):
        with pytest.raises(litellm.BadRequestError) as raised:
            await SURFACE.call(server)
        events: Final = await recorder.wait_for_async("async_log_failure_event")

    assert_transport(server)
    assert isinstance(events[0].kwargs, dict)
    assert events[0].kwargs["exception"] is raised.value


@pytest.mark.asyncio
async def test_pre_api_call_sees_the_wire_body(server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    await SURFACE.call(server, callbacks=[recorder])

    assert_transport(server)
    pre_call: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(pre_call, dict)
    assert request_body(pre_call) == server.requests[0].body


@pytest.mark.asyncio
async def test_pre_api_call_edits_reach_the_provider(server: RecordingServer) -> None:
    class Edit(CustomLogger):
        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            request_body(kwargs)["temperature"] = 0.25

    await SURFACE.call(server, callbacks=[Edit()])

    assert_transport(server)
    assert isinstance(server.requests[0].body, dict)
    assert server.requests[0].body["temperature"] == 0.25


@pytest.mark.asyncio
async def test_logging_hook_and_redaction_edits_reach_the_async_success_event(server: RecordingServer) -> None:
    class Rewrite(HookRecorder):
        async def async_logging_hook(
            self, kwargs: dict[str, object], result: object, call_type: str
        ) -> tuple[dict[str, object], object]:
            return {**kwargs, "logging_hook_ran": True}, result

        def redact_standard_logging_payload_from_model_call_details(
            self, model_call_details: dict[str, object]
        ) -> dict[str, object]:
            return {**model_call_details, "redacted": True}

    recorder: Final = Rewrite()

    with rebound(litellm, "callbacks", [recorder]):
        await SURFACE.call(server)
        success: Final = await recorder.wait_for_async("async_log_success_event")

    assert isinstance(success[0].kwargs, dict)
    assert success[0].kwargs["logging_hook_ran"] is True
    assert success[0].kwargs["redacted"] is True


@pytest.mark.asyncio
async def test_pre_call_deployment_hook_rewrites_the_wire_request(server: RecordingServer) -> None:
    class Rewrite(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            return {**kwargs, "extra_headers": {"x-deployment-hook": "rewritten"}}

    with rebound(litellm, "callbacks", [Rewrite()]):
        await SURFACE.call(server)

    assert_transport(server)
    assert server.requests[0].headers["x-deployment-hook"] == "rewritten"


@pytest.mark.asyncio
async def test_pre_call_deployment_hook_can_delete_keywords(server: RecordingServer) -> None:
    class DropKeywords(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            return {name: value for name, value in kwargs.items() if name not in ("instructions", "extra_headers")}

    with rebound(litellm, "callbacks", [DropKeywords()]):
        await SURFACE.call(
            server,
            instructions="remove this instruction",
            extra_headers={"x-deleted": "remove this header"},
        )

    assert_transport(server)
    sent: Final = server.requests[0]
    assert isinstance(sent.body, dict)
    assert "instructions" not in sent.body
    assert "x-deleted" not in sent.headers


@pytest.mark.asyncio
async def test_post_call_success_deployment_hook_replaces_the_response(server: RecordingServer) -> None:
    class Replace(CustomLogger):
        async def async_post_call_success_deployment_hook(
            self, request_data: dict[str, object], response: object, call_type: object
        ) -> object:
            return with_id(response, "replaced")

    with rebound(litellm, "callbacks", [Replace()]):
        response: Final = await SURFACE.call(server)

    assert_transport(server)
    assert response_id(response) == "replaced"


@pytest.mark.asyncio
async def test_a_failing_logger_does_not_fail_the_call(server: RecordingServer) -> None:
    class Broken(CustomLogger):
        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            raise RuntimeError("logger exploded")

    response: Final = await SURFACE.call(server, callbacks=[Broken()])

    assert_transport(server)
    assert response_id(response) == RESPONSES_RESPONSE["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event,asynchronous,dynamic,expected",
    (
        pytest.param("input", False, False, 1, id="global-input"),
        pytest.param("success", True, False, 1, id="global-async-success"),
        pytest.param("success", True, True, 1, id="request-async-success"),
        pytest.param("success", False, False, 0, id="global-sync-success", marks=SYNC_HANDLERS_ON_ASYNC_CALLS),
        pytest.param("success", False, True, 0, id="request-sync-success", marks=SYNC_HANDLERS_ON_ASYNC_CALLS),
        pytest.param("failure", True, False, 1, id="global-async-failure"),
        pytest.param("failure", False, False, 1, id="global-sync-failure"),
        pytest.param("failure", False, True, 1, id="request-sync-failure"),
    ),
)
async def test_function_callback_fires_the_expected_number_of_times(
    server: RecordingServer, event: str, asynchronous: bool, dynamic: bool, expected: int
) -> None:
    recorder: Final = RecordingLogger()

    def callback(kwargs: dict[str, object], *_: object) -> None:
        recorder._record(event, kwargs)

    async def async_callback(kwargs: dict[str, object], *_: object) -> None:
        recorder._record(event, kwargs)

    registered: Final = async_callback if asynchronous else callback
    global_list: Final = f"_async_{event}_callback" if asynchronous else f"{event}_callback"
    request_callbacks: Final = {f"{event}_callback": [registered]} if dynamic else {}
    if event == "failure":
        server.default_response = ResponseSpec(body=SURFACE.bad_request, status=400)

    with rebound(litellm, "callbacks", [recorder]), rebound(litellm, global_list, [] if dynamic else [registered]):
        try:
            await SURFACE.call(server, **request_callbacks)
        except litellm.BadRequestError:
            assert event == "failure"
        await recorder.wait_for_async(f"async_log_{'failure' if event == 'failure' else 'success'}_event")

    assert_transport(server)
    assert recorder.names.count(event) == expected


@pytest.mark.asyncio
async def test_dataset_hook_returning_none_drops_the_record_from_the_success_sink(server: RecordingServer) -> None:
    datasets: Final = RecordingLogger()

    class Drop(CustomLogger):
        async def async_dataset_hook(
            self, logged_data: ArgillaItem, standard_logging_payload: StandardLoggingPayload | None
        ) -> None:
            datasets._record("async_dataset_hook", response=logged_data)

    server.enqueue(ResponseSpec(body={"items": [{"id": "callback-dataset"}]}))
    with rebound(litellm, "argilla_transformation_object", {"prompt": "messages", "answer": "response"}):
        sink: Final = ArgillaLogger(
            argilla_api_key="test-key",
            argilla_base_url=server.base_url,
            argilla_dataset_name="callback-dataset",
            flush_interval=3600,
        )
        with rebound(litellm, "callbacks", [sink, Drop()]):
            await SURFACE.call(server)
            await datasets.wait_for_async("async_dataset_hook")
            await sink.flush_queue()

    assert [request.method for request in server.requests] == ["GET", "POST"]


@pytest.mark.asyncio
@STREAMING
async def test_stream_relays_provider_events_and_logs_success_once_after_the_last_chunk(
    server: RecordingServer,
) -> None:
    server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await SURFACE.call(server, stream=True, callbacks=[recorder])
    assert isinstance(stream, AsyncIterator)
    first: Final = await anext(stream)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names
    rest: Final = [chunk async for chunk in stream]

    assert_transport(server)
    assert [getattr(chunk, "type", None) for chunk in (first, *rest)] == [event for event, _ in RESPONSES_EVENTS]
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert isinstance(success[0].kwargs, dict)
    assert success[0].kwargs["completion_start_time"] is not None
    assert response_id(success[0].response) == RESPONSES_RESPONSE["id"]
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
@STREAMING
async def test_sync_stream_relays_provider_events_and_logs_success_once(server: RecordingServer) -> None:
    server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await SURFACE.call(server, asynchronous=False, stream=True, callbacks=[recorder])
    chunks: Final = await consume(stream)

    assert_transport(server)
    assert [getattr(chunk, "type", None) for chunk in chunks] == [event for event, _ in RESPONSES_EVENTS]
    assert len(await recorder.wait_for_async("log_success_event")) == 1


@pytest.mark.asyncio
@STREAMING
async def test_body_streaming_deployment_hook_rewrites_chunks_before_delivery(server: RecordingServer) -> None:
    seen: Final[list[str]] = []

    class Rewrite(CustomLogger):
        async def async_post_call_streaming_deployment_hook(
            self, request_data: dict[str, object], response_chunk: BaseModel, call_type: object
        ) -> BaseModel:
            seen.append(getattr(response_chunk, "type", ""))
            if getattr(response_chunk, "type", None) != "response.output_text.delta":
                return response_chunk
            return response_chunk.model_copy(update={"delta": "rewritten"})

    server.enqueue(STREAM)
    with rebound(litellm, "callbacks", [Rewrite()]):
        stream: Final = await SURFACE.call(server, stream=True)
        chunks: Final = await consume(stream)

    assert_transport(server)
    assert seen == [event for event, _ in RESPONSES_EVENTS]
    assert [getattr(chunk, "delta", None) for chunk in chunks] == [None, "rewritten", None]


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_planned_agentic_loop_sends_the_followup_and_replaces_the_response(server: RecordingServer) -> None:
    loop: Final = PlannedAgenticLoop()
    server.enqueue(ResponseSpec(body=RESPONSES_RESPONSE))
    server.default_response = ResponseSpec(body={**RESPONSES_RESPONSE, "id": "followup"})

    with rebound(litellm, "callbacks", [loop]):
        response: Final = await SURFACE.call(server)
        await drain_logging()

    assert_transport(server, count=2)
    assert isinstance(server.requests[1].body, dict)
    assert server.requests[1].body["input"] == [{"role": "user", "content": "followup"}]
    assert response_id(response) == "agentic-replaced"
    assert agentic_hooks(loop) == (
        "async_should_run_agentic_loop",
        "async_build_agentic_loop_plan",
        "async_should_run_agentic_loop",
        "async_post_agentic_loop_response_hook",
        "async_agentic_loop_cleanup_hook",
    )


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_legacy_agentic_loop_replaces_the_response(server: RecordingServer) -> None:
    loop: Final = LegacyAgenticLoop()

    with rebound(litellm, "callbacks", [loop]):
        response: Final = await SURFACE.call(server)

    assert_transport(server)
    assert response_id(response) == "agentic-replaced"
    assert agentic_hooks(loop) == ("async_should_run_agentic_loop", "async_run_agentic_loop")


@pytest.mark.asyncio
async def test_body_logging_setup_drops_a_logger_fn_from_a_supplied_logging_object(server: RecordingServer) -> None:
    observed: Final[list[object]] = []
    logging_obj: Final = Logging(
        model=RESPONSES_SURFACE.request["model"],
        messages=[],
        stream=False,
        start_time=datetime.now(),
        call_type="aresponses",
        litellm_call_id=CALL_ID,
        function_id=CALL_ID,
    )
    logging_obj.update_environment_variables(litellm_params={"logger_fn": observed.append}, optional_params={})

    await SURFACE.call(server, litellm_logging_obj=logging_obj)

    assert_transport(server)
    assert observed == []

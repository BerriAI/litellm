import asyncio
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import datetime
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.integrations.argilla import ArgillaLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.integrations.argilla import ArgillaItem
from litellm.types.utils import StandardLoggingPayload
from tests.test_litellm_rust.responses.data import (
    BAD_REQUEST,
    RATE_LIMITED,
    RESPONSES_RESPONSE,
    SYNC_DEPLOYMENT_HOOK,
    call,
)
from tests.test_litellm_rust.support.callback_recorder import HookRecorder, RecordingLogger, request_body
from tests.test_litellm_rust.support.gaps import gap
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, relaxed
from tests.test_litellm_rust.support.results import response_id, with_id
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

SYNC_HANDLERS_ON_ASYNC_CALLS: Final = gap("native async calls also run the sync success and failure handlers")
_OBJECT: Final = TypeAdapter(dict[str, object])
_HEADERS: Final = TypeAdapter(dict[str, str])


def success_signal(asynchronous: bool) -> str:
    return "async_log_success_event" if asynchronous else "log_success_event"


@pytest.fixture
def server(server: RecordingServer) -> Iterator[RecordingServer]:
    yield from relaxed(server)


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
        await call(server, asynchronous=asynchronous)
        await recorder.wait_for_async(success_signal(asynchronous))

    assert_served_natively(server)
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
        pytest.param("log_failure_event", True, 0, id="async-log_failure_event"),
        pytest.param("log_failure_event", False, 1, id="sync-log_failure_event"),
    ),
)
async def test_failure_hook_fires_the_expected_number_of_times(
    server: RecordingServer, hook: str, asynchronous: bool, expected: int
) -> None:
    server.default_response = ResponseSpec(body=RATE_LIMITED, status=429)
    recorder: Final = HookRecorder()

    with rebound(litellm, "callbacks", [recorder]):
        with pytest.raises(litellm.RateLimitError):
            await call(server, asynchronous=asynchronous, metadata={"model_group_size": 1})
        await recorder.wait_for_async("async_log_failure_event" if asynchronous else "log_failure_event")

    assert_served_natively(server)
    assert recorder.names.count(hook) == expected


@pytest.mark.asyncio
async def test_failure_events_carry_the_error_raised_to_the_caller(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body=BAD_REQUEST, status=400)
    recorder: Final = RecordingLogger()

    with rebound(litellm, "callbacks", [recorder]):
        with pytest.raises(litellm.BadRequestError) as raised:
            await call(server)
        events: Final = await recorder.wait_for_async("async_log_failure_event")

    assert_served_natively(server)
    assert isinstance(events[0].kwargs, dict)
    assert events[0].kwargs["exception"] is raised.value


@pytest.mark.asyncio
async def test_pre_api_call_sees_the_wire_body(server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    await call(server, callbacks=[recorder])

    assert_served_natively(server)
    pre_call: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(pre_call, dict)
    assert request_body(pre_call) == server.requests[0].body


@pytest.mark.asyncio
async def test_pre_api_call_edits_reach_the_provider(server: RecordingServer) -> None:
    class Edit(CustomLogger):
        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            request_body(kwargs)["temperature"] = 0.25

    await call(server, callbacks=[Edit()])

    assert_served_natively(server)
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
        await call(server)
        success: Final = await recorder.wait_for_async("async_log_success_event")

    assert isinstance(success[0].kwargs, dict)
    assert success[0].kwargs["logging_hook_ran"] is True
    assert success[0].kwargs["redacted"] is True


@pytest.mark.asyncio
async def test_pre_call_deployment_hooks_compose_the_wire_request_in_registration_order(
    server: RecordingServer,
) -> None:
    class Rewrite(CustomLogger):
        def __init__(self, marker: str) -> None:
            super().__init__()
            self.marker = marker

        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            headers: Final = _HEADERS.validate_python(kwargs["extra_headers"])
            return {
                **kwargs,
                "extra_headers": {**headers, "x-deployment-hook": headers["x-deployment-hook"] + self.marker},
            }

    with rebound(litellm, "callbacks", [Rewrite("first"), Rewrite("-second")]):
        await call(server, extra_headers={"x-deployment-hook": "caller-", "x-kept": "unchanged"})

    assert_served_natively(server)
    assert server.requests[0].headers["x-deployment-hook"] == "caller-first-second"
    assert server.requests[0].headers["x-kept"] == "unchanged"


@pytest.mark.asyncio
async def test_pre_call_deployment_rejection_logs_failure_without_provider_io(server: RecordingServer) -> None:
    rejection: Final = litellm.BadRequestError(message="deployment rejected", model="test-model", llm_provider="openai")
    recorder: Final = RecordingLogger()

    class Reject(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            raise rejection

    with rebound(litellm, "callbacks", [Reject(), recorder]):
        with pytest.raises(litellm.BadRequestError, match="deployment rejected") as raised:
            await call(server)
        failures: Final = await recorder.wait_for_async("async_log_failure_event")

    assert raised.value is rejection
    assert server.requests == []
    assert len(failures) == 1
    assert _OBJECT.validate_python(failures[0].kwargs)["exception"] is raised.value
    assert not {"log_pre_api_call", "log_success_event", "async_log_success_event"} & set(recorder.names)


@pytest.mark.asyncio
async def test_pre_call_deployment_hook_can_delete_keywords(server: RecordingServer) -> None:
    class DropKeywords(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            return {name: value for name, value in kwargs.items() if name not in ("instructions", "extra_headers")}

    with rebound(litellm, "callbacks", [DropKeywords()]):
        await call(
            server,
            instructions="remove this instruction",
            extra_headers={"x-deleted": "remove this header"},
        )

    assert_served_natively(server)
    sent: Final = server.requests[0]
    assert isinstance(sent.body, dict)
    assert "instructions" not in sent.body
    assert "x-deleted" not in sent.headers


@pytest.mark.asyncio
async def test_post_call_success_deployment_hook_replaces_the_response(server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    class Replace(CustomLogger):
        async def async_post_call_success_deployment_hook(
            self, request_data: dict[str, object], response: object, call_type: object
        ) -> object:
            return with_id(response, "replaced")

    with rebound(litellm, "callbacks", [Replace(), recorder]):
        response: Final = await call(server)
        success: Final = await recorder.wait_for_async("async_log_success_event")

    assert_served_natively(server)
    assert response_id(response) == "replaced"
    assert len(success) == 1
    assert response_id(success[0].response) == response_id(response)


@pytest.mark.asyncio
async def test_concurrent_calls_keep_callback_context_metadata_and_responses_isolated(server: RecordingServer) -> None:
    context: Final = ContextVar("responses-callback-context", default="outside")
    first_entered: Final = asyncio.Event()
    second_entered: Final = asyncio.Event()
    server.enqueue(ResponseSpec(body={**RESPONSES_RESPONSE, "id": "resp_first"}))
    server.enqueue(ResponseSpec(body={**RESPONSES_RESPONSE, "id": "resp_second"}))

    class Observe(RecordingLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            arrived, other = (
                (first_entered, second_entered)
                if kwargs["litellm_call_id"] == "first"
                else (second_entered, first_entered)
            )
            arrived.set()
            await asyncio.wait_for(other.wait(), timeout=5)
            return kwargs

        def log_pre_api_call(self, model: object, messages: object, kwargs: dict[str, object]) -> None:
            self._record("log_pre_api_call", {**kwargs, "caller_context": context.get()})

        async def async_log_success_event(
            self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            self._record("async_log_success_event", {**kwargs, "caller_context": context.get()}, response_obj)

    recorder: Final = Observe()

    async def invoke(marker: str) -> object:
        token: Final = context.set(marker)
        try:
            return await call(server, litellm_call_id=marker, metadata={"marker": marker})
        finally:
            context.reset(token)

    with rebound(litellm, "callbacks", [recorder]):
        responses: Final = await asyncio.gather(invoke("first"), invoke("second"))
        success: Final = await recorder.wait_for_async("async_log_success_event", count=2)

    assert_served_natively(server, count=2)
    expected: Final = {"first": responses[0], "second": responses[1]}
    pre_call: Final = recorder.wait_for("log_pre_api_call", count=2)
    assert len(pre_call) == len(success) == 2
    assert {_OBJECT.validate_python(event.kwargs)["litellm_call_id"] for event in success} == expected.keys()
    for event in (*pre_call, *success):
        details: Final = _OBJECT.validate_python(event.kwargs)
        call_id: Final = str(details["litellm_call_id"])
        params: Final = _OBJECT.validate_python(details["litellm_params"])
        assert details["caller_context"] == call_id
        assert _OBJECT.validate_python(params["metadata"])["marker"] == call_id
    for event in success:
        logged_call_id: Final = str(_OBJECT.validate_python(event.kwargs)["litellm_call_id"])
        assert response_id(event.response) == response_id(expected[logged_call_id])
    assert context.get() == "outside"


@pytest.mark.asyncio
async def test_a_failing_logger_does_not_fail_the_call(server: RecordingServer) -> None:
    class Broken(CustomLogger):
        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            raise RuntimeError("logger exploded")

    response: Final = await call(server, callbacks=[Broken()])

    assert_served_natively(server)
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
        server.default_response = ResponseSpec(body=BAD_REQUEST, status=400)

    with rebound(litellm, "callbacks", [recorder]), rebound(litellm, global_list, [] if dynamic else [registered]):
        try:
            await call(server, **request_callbacks)
        except litellm.BadRequestError:
            assert event == "failure"
        await recorder.wait_for_async(f"async_log_{'failure' if event == 'failure' else 'success'}_event")

    assert_served_natively(server)
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
            await call(server)
            await datasets.wait_for_async("async_dataset_hook")
            await sink.flush_queue()

    assert [request.method for request in server.requests] == ["GET", "POST"]

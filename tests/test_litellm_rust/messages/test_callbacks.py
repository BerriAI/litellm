import asyncio
import json
from contextvars import ContextVar
from datetime import datetime
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.integrations.argilla import ArgillaLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.integrations.argilla import ArgillaItem
from litellm.types.utils import ModelResponse, StandardLoggingPayload
from tests.test_litellm_rust.messages.data import (
    BAD_REQUEST,
    MESSAGES_RESPONSE,
    RATE_LIMITED,
    call,
    messages_reply,
)
from tests.test_litellm_rust.support.callback_recorder import HookRecorder, RecordingLogger, request_body
from tests.test_litellm_rust.support.gaps import gap
from tests.test_litellm_rust.support.isolation import rebound
from tests._support.provider import SharedProvider
from tests._support.wire import Reply
from tests.test_litellm_rust.support.results import response_id, with_id
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

SYNC_HANDLERS_ON_ASYNC_CALLS: Final = gap("native async calls also run the sync success and failure handlers")
_OBJECT: Final = TypeAdapter(dict[str, object])
_HEADERS: Final = TypeAdapter(dict[str, str])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook,expected",
    (
        pytest.param("log_pre_api_call", 1),
        pytest.param("log_post_api_call", 1),
        pytest.param("async_log_success_event", 1),
        pytest.param("async_logging_hook", 1),
        pytest.param("redact_standard_logging_payload_from_model_call_details", 1, id="redact"),
        pytest.param("async_pre_call_deployment_hook", 1),
        pytest.param("async_post_call_success_deployment_hook", 1),
        pytest.param("log_success_event", 0),
        pytest.param("logging_hook", 0),
    ),
)
async def test_success_hook_fires_the_expected_number_of_times(
    server: SharedProvider, hook: str, expected: int
) -> None:
    recorder: Final = HookRecorder()
    server.expect(messages_reply())

    with rebound(litellm, "callbacks", [recorder]):
        response: Final = await call(server)
        await recorder.wait_for_async("async_log_success_event")

    assert_served_natively(server, response)
    assert recorder.names.count(hook) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook,expected",
    (
        pytest.param("async_log_failure_event", 1),
        pytest.param("async_post_call_failure_deployment_hook", 1),
        pytest.param("log_model_group_rate_limit_error", 1),
        pytest.param("log_failure_event", 0, marks=SYNC_HANDLERS_ON_ASYNC_CALLS),
    ),
)
async def test_failure_hook_fires_the_expected_number_of_times(
    server: SharedProvider, hook: str, expected: int
) -> None:
    server.expect(messages_reply(RATE_LIMITED, status=429))
    recorder: Final = HookRecorder()

    with rebound(litellm, "callbacks", [recorder]):
        with pytest.raises(litellm.RateLimitError):
            await call(server, metadata={"model_group_size": 1})
        await recorder.wait_for_async("async_log_failure_event")

    assert_served_natively(server)
    assert recorder.names.count(hook) == expected


@pytest.mark.asyncio
async def test_failure_events_carry_the_error_raised_to_the_caller(server: SharedProvider) -> None:
    server.expect(messages_reply(BAD_REQUEST, status=400))
    recorder: Final = RecordingLogger()

    with rebound(litellm, "callbacks", [recorder]):
        with pytest.raises(litellm.BadRequestError) as raised:
            await call(server)
        events: Final = await recorder.wait_for_async("async_log_failure_event")

    assert_served_natively(server)
    assert isinstance(events[0].kwargs, dict)
    assert events[0].kwargs["exception"] is raised.value


@pytest.mark.asyncio
async def test_pre_api_call_sees_the_wire_body(server: SharedProvider) -> None:
    recorder: Final = RecordingLogger()
    server.expect(messages_reply())

    response: Final = await call(server, callbacks=[recorder])

    requests: Final = assert_served_natively(server, response)
    pre_call: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(pre_call, dict)
    assert request_body(pre_call) == json.loads(requests[0].body)


@pytest.mark.asyncio
async def test_pre_api_call_edits_reach_the_provider(server: SharedProvider) -> None:
    class Edit(CustomLogger):
        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            request_body(kwargs)["temperature"] = 0.25

    server.expect(messages_reply())
    response: Final = await call(server, callbacks=[Edit()])

    requests: Final = assert_served_natively(server, response)
    body: Final = json.loads(requests[0].body)
    assert isinstance(body, dict)
    assert body["temperature"] == 0.25


@pytest.mark.asyncio
async def test_pre_api_call_sees_the_shaped_optional_params(server: SharedProvider) -> None:
    recorder: Final = RecordingLogger()
    server.expect(messages_reply())

    await call(server, callbacks=[recorder], temperature=0.2, top_k=3, drop_params=True)

    requests: Final = assert_served_natively(server)
    sent: Final = json.loads(requests[0].body)
    assert isinstance(sent, dict)
    assert not {"temperature", "top_k"} & sent.keys()
    pre_call: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(pre_call, dict)
    optional_params: Final = pre_call["optional_params"]
    assert isinstance(optional_params, dict)
    assert not {"model", "messages", "temperature", "top_k"} & optional_params.keys()
    assert optional_params["max_tokens"] == sent["max_tokens"]


@pytest.mark.asyncio
async def test_logging_hook_and_redaction_edits_reach_the_async_success_event(server: SharedProvider) -> None:
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
    server.expect(messages_reply())

    with rebound(litellm, "callbacks", [recorder]):
        await call(server)
        success: Final = await recorder.wait_for_async("async_log_success_event")

    assert_served_natively(server)
    assert isinstance(success[0].kwargs, dict)
    assert success[0].kwargs["logging_hook_ran"] is True
    assert success[0].kwargs["redacted"] is True


@pytest.mark.asyncio
async def test_pre_call_deployment_hooks_compose_the_wire_request_in_registration_order(
    server: SharedProvider,
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

    server.expect(messages_reply())
    with rebound(litellm, "callbacks", [Rewrite("first"), Rewrite("-second")]):
        response: Final = await call(server, extra_headers={"x-deployment-hook": "caller-", "x-kept": "unchanged"})

    requests: Final = assert_served_natively(server, response)
    assert requests[0].headers["x-deployment-hook"] == "caller-first-second"
    assert requests[0].headers["x-kept"] == "unchanged"


@pytest.mark.asyncio
async def test_pre_call_deployment_rejection_logs_failure_without_provider_io(server: SharedProvider) -> None:
    rejection: Final = litellm.BadRequestError(
        message="deployment rejected", model="test-model", llm_provider="anthropic"
    )
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
    requests: Final = server.received()
    assert requests == ()
    assert len(failures) == 1
    assert _OBJECT.validate_python(failures[0].kwargs)["exception"] is raised.value
    assert not {"log_pre_api_call", "log_success_event", "async_log_success_event"} & set(recorder.names)


@pytest.mark.asyncio
async def test_pre_call_deployment_hook_can_delete_keywords(server: SharedProvider) -> None:
    class DropKeywords(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            return {name: value for name, value in kwargs.items() if name not in ("system", "extra_headers")}

    server.expect(messages_reply())
    with rebound(litellm, "callbacks", [DropKeywords()]):
        response: Final = await call(
            server,
            system="remove this instruction",
            extra_headers={"x-deleted": "remove this header"},
            headers={"x-kept": "keep this header"},
        )

    requests: Final = assert_served_natively(server, response)
    sent: Final = requests[0]
    body: Final = json.loads(sent.body)
    assert isinstance(body, dict)
    assert "system" not in body
    assert "x-deleted" not in sent.headers
    assert sent.headers["x-kept"] == "keep this header"


@pytest.mark.asyncio
async def test_post_call_success_deployment_hook_replaces_the_response(server: SharedProvider) -> None:
    recorder: Final = RecordingLogger()
    server.expect(messages_reply())

    class Replace(CustomLogger):
        async def async_post_call_success_deployment_hook(
            self, request_data: dict[str, object], response: object, call_type: object
        ) -> object:
            return with_id(response, "replaced")

    with rebound(litellm, "callbacks", [Replace(), recorder]):
        response: Final = await call(server)
        success: Final = await recorder.wait_for_async("async_log_success_event")

    assert_served_natively(server, response)
    assert response_id(response) == "replaced"
    assert len(success) == 1
    assert isinstance(success[0].response, ModelResponse)
    assert success[0].response.id == response_id(response)


@pytest.mark.asyncio
async def test_concurrent_calls_keep_callback_context_metadata_and_responses_isolated(server: SharedProvider) -> None:
    context: Final = ContextVar("messages-callback-context", default="outside")
    first_entered: Final = asyncio.Event()
    second_entered: Final = asyncio.Event()
    server.expect(
        messages_reply({**MESSAGES_RESPONSE, "id": "msg_first"}),
        messages_reply({**MESSAGES_RESPONSE, "id": "msg_second"}),
    )

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

    assert_served_natively(server, *responses)
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
        assert isinstance(event.response, ModelResponse)
        assert event.response.id == response_id(expected[logged_call_id])
    assert context.get() == "outside"


@pytest.mark.asyncio
async def test_a_failing_logger_does_not_fail_the_call(server: SharedProvider) -> None:
    class Broken(CustomLogger):
        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            raise RuntimeError("logger exploded")

    server.expect(messages_reply())
    response: Final = await call(server, callbacks=[Broken()])

    assert_served_natively(server, response)
    assert response_id(response) == MESSAGES_RESPONSE["id"]


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
    server: SharedProvider, event: str, asynchronous: bool, dynamic: bool, expected: int
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
        server.expect(messages_reply(BAD_REQUEST, status=400))
    else:
        server.expect(messages_reply())

    with rebound(litellm, "callbacks", [recorder]), rebound(litellm, global_list, [] if dynamic else [registered]):
        try:
            await call(server, **request_callbacks)
        except litellm.BadRequestError:
            assert event == "failure"
        await recorder.wait_for_async(f"async_log_{'failure' if event == 'failure' else 'success'}_event")

    assert_served_natively(server)
    assert recorder.names.count(event) == expected


@pytest.mark.asyncio
async def test_pre_call_rule_rejects_before_provider_io(server: SharedProvider) -> None:
    checked: Final[list[str]] = []

    def reject(text: str) -> bool:
        checked.append(text)
        return False

    with rebound(litellm, "pre_call_rules", [reject]):
        with pytest.raises(litellm.APIResponseValidationError):
            await call(server)

    assert checked == ["Hello"]
    requests: Final = server.received()
    assert requests == ()


@pytest.mark.asyncio
async def test_dataset_hook_returning_none_drops_the_record_from_the_success_sink(server: SharedProvider) -> None:
    datasets: Final = RecordingLogger()

    class Drop(CustomLogger):
        async def async_dataset_hook(
            self, logged_data: ArgillaItem, standard_logging_payload: StandardLoggingPayload | None
        ) -> None:
            datasets._record("async_dataset_hook", response=logged_data)

    server.expect(
        Reply(body=json.dumps({"items": [{"id": "callback-dataset"}]}).encode()),
        messages_reply(),
    )
    with rebound(litellm, "argilla_transformation_object", {"prompt": "messages", "answer": "response"}):
        sink: Final = ArgillaLogger(
            argilla_api_key="test-key",
            argilla_base_url=server.wire.url,
            argilla_dataset_name="callback-dataset",
            flush_interval=3600,
        )
        with rebound(litellm, "callbacks", [sink, Drop()]):
            await call(server)
            await datasets.wait_for_async("async_dataset_hook")
            await sink.flush_queue()

    requests: Final = server.received()
    assert [request.method for request in requests] == ["GET", "POST"]


@pytest.mark.asyncio
async def test_sync_call_returns_the_provider_message(server: SharedProvider) -> None:
    recorder: Final = RecordingLogger()
    server.expect(messages_reply())

    response: Final = await call(server, asynchronous=False, callbacks=[recorder])

    assert_served_natively(server, response)
    assert response_id(response) == MESSAGES_RESPONSE["id"]
    assert len(await recorder.wait_for_async("log_success_event")) == 1

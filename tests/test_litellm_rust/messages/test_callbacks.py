from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.integrations.argilla import ArgillaLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.pass_through.messages import interceptors
from litellm.llms.anthropic.pass_through.messages.interceptors.base import MessagesInterceptor
from litellm.types.integrations.argilla import ArgillaItem
from litellm.types.utils import StandardLoggingPayload
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.callbacks import (
    CALL_ID,
    MESSAGES_SURFACE,
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
from tests.test_litellm_rust.support.requests import MESSAGES_EVENTS, MESSAGES_RESPONSE, request_body

pytestmark = pytest.mark.requires_rust_extension

SURFACE: Final = MESSAGES_SURFACE
SYNC_HANDLERS_ON_ASYNC_CALLS: Final = gap("native async calls also run the sync success and failure handlers")
BODY_HOOKS: Final = gap("native Messages does not run the handler body hooks")
STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS)


@pytest.fixture
def server(recording_server: RecordingServer) -> Iterator[RecordingServer]:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    recording_server.expected_requests = None
    yield recording_server
    recording_server.responses.clear()


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
    server: RecordingServer, hook: str, expected: int
) -> None:
    recorder: Final = HookRecorder()

    with rebound(litellm, "callbacks", [recorder]):
        await SURFACE.call(server)
        await recorder.wait_for_async("async_log_success_event")

    assert_transport(server)
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
    server: RecordingServer, hook: str, expected: int
) -> None:
    server.default_response = ResponseSpec(body=SURFACE.rate_limited, status=429)
    recorder: Final = HookRecorder()

    with rebound(litellm, "callbacks", [recorder]):
        with pytest.raises(litellm.RateLimitError):
            await SURFACE.call(server, metadata={"model_group_size": 1})
        await recorder.wait_for_async("async_log_failure_event")

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
async def test_pre_api_call_sees_the_shaped_optional_params(server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    await SURFACE.call(server, callbacks=[recorder], temperature=0.2, top_k=3, drop_params=True)

    sent: Final = server.requests[0].body
    assert isinstance(sent, dict)
    assert not {"temperature", "top_k"} & sent.keys()
    pre_call: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(pre_call, dict)
    optional_params: Final = pre_call["optional_params"]
    assert isinstance(optional_params, dict)
    assert not {"model", "messages", "temperature", "top_k"} & optional_params.keys()
    assert optional_params["max_tokens"] == sent["max_tokens"]


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
            return {name: value for name, value in kwargs.items() if name not in ("system", "extra_headers")}

    with rebound(litellm, "callbacks", [DropKeywords()]):
        await SURFACE.call(
            server,
            system="remove this instruction",
            extra_headers={"x-deleted": "remove this header"},
            headers={"x-kept": "keep this header"},
        )

    assert_transport(server)
    sent: Final = server.requests[0]
    assert isinstance(sent.body, dict)
    assert "system" not in sent.body
    assert "x-deleted" not in sent.headers
    assert sent.headers["x-kept"] == "keep this header"


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
async def test_pre_call_rule_rejects_before_provider_io(server: RecordingServer) -> None:
    checked: Final[list[str]] = []

    def reject(text: str) -> bool:
        checked.append(text)
        return False

    with rebound(litellm, "pre_call_rules", [reject]):
        with pytest.raises(litellm.APIResponseValidationError):
            await SURFACE.call(server)

    assert checked == ["Hello"]
    assert server.requests == []


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
    assert first + b"".join(rest) == b"".join(STREAM.payloads())
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert isinstance(success[0].kwargs, dict)
    assert success[0].kwargs["completion_start_time"] is not None
    usage: Final = success[0].response.usage
    assert usage.prompt_tokens == MESSAGES_RESPONSE["usage"]["input_tokens"]
    assert usage.completion_tokens == MESSAGES_RESPONSE["usage"]["output_tokens"]
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_sync_call_returns_the_provider_message(server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    response: Final = await SURFACE.call(server, asynchronous=False, callbacks=[recorder])

    assert_transport(server)
    assert response_id(response) == MESSAGES_RESPONSE["id"]
    assert len(await recorder.wait_for_async("log_success_event")) == 1


@pytest.mark.asyncio
async def test_sync_stream_relays_provider_events_and_logs_success_once(server: RecordingServer) -> None:
    server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await SURFACE.call(server, asynchronous=False, stream=True, callbacks=[recorder])
    chunks: Final = await consume(stream)

    assert_transport(server)
    assert b"".join(chunks) == b"".join(STREAM.payloads())
    assert len(await recorder.wait_for_async("async_log_success_event")) == 1


@pytest.mark.asyncio
async def test_stream_closed_early_logs_success_once(server: RecordingServer) -> None:
    server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await SURFACE.call(server, stream=True, callbacks=[recorder])
    assert isinstance(stream, AsyncIterator)
    await anext(stream)
    await stream.aclose()

    assert len(await recorder.wait_for_async("async_log_success_event")) == 1
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_pre_request_hook_rewrites_the_provider_request(server: RecordingServer) -> None:
    class Rewrite(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
        ) -> dict[str, object]:
            return {**kwargs, "system": "body hook instruction"}

    with rebound(litellm, "callbacks", [Rewrite()]):
        await SURFACE.call(server)

    assert_transport(server)
    assert isinstance(server.requests[0].body, dict)
    assert server.requests[0].body.get("system") == "body hook instruction"


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_websearch_short_circuit_answers_without_the_provider(server: RecordingServer) -> None:
    class ShortCircuit(WebSearchInterceptionLogger):
        async def try_short_circuit_search(
            self,
            model: str,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None,
            custom_llm_provider: str | None,
            kwargs: dict[str, object] | None = None,
        ) -> object:
            return {**MESSAGES_RESPONSE, "id": "short-circuited"}

    with rebound(litellm, "callbacks", [ShortCircuit()]):
        response: Final = await SURFACE.call(server)

    assert response_id(response) == "short-circuited"
    assert server.requests == []


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_interceptor_replaces_the_provider_call(server: RecordingServer) -> None:
    class Replace(MessagesInterceptor):
        def can_handle(self, tools: list[dict[str, object]] | None, custom_llm_provider: str | None) -> bool:
            return True

        async def handle(self, **kwargs: object) -> object:
            return {**MESSAGES_RESPONSE, "id": "intercepted"}

    with rebound(interceptors, "_interceptors", [Replace()]):
        response: Final = await SURFACE.call(server)

    assert response_id(response) == "intercepted"
    assert server.requests == []


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_planned_agentic_loop_sends_the_followup_and_replaces_the_response(server: RecordingServer) -> None:
    loop: Final = PlannedAgenticLoop()
    server.enqueue(ResponseSpec(body=MESSAGES_RESPONSE))
    server.default_response = ResponseSpec(body={**MESSAGES_RESPONSE, "id": "followup"})

    with rebound(litellm, "callbacks", [loop]):
        response: Final = await SURFACE.call(server)
        await drain_logging()

    assert_transport(server, count=2)
    assert isinstance(server.requests[1].body, dict)
    assert server.requests[1].body["messages"] == [{"role": "user", "content": "followup"}]
    assert response_id(response) == "agentic-replaced"
    assert agentic_hooks(loop) == (
        "async_should_run_agentic_loop",
        "async_build_agentic_loop_plan",
        "async_should_run_agentic_loop",
        "async_post_agentic_loop_response_hook",
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
        model=MESSAGES_SURFACE.request["model"],
        messages=[],
        stream=False,
        start_time=datetime.now(),
        call_type="anthropic_messages",
        litellm_call_id=CALL_ID,
        function_id=CALL_ID,
    )
    logging_obj.update_environment_variables(litellm_params={"logger_fn": observed.append}, optional_params={})

    await SURFACE.call(server, litellm_logging_obj=logging_obj)

    assert_transport(server)
    assert observed == []

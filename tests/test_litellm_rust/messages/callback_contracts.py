from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import datetime
from typing import Final, Literal, TypeAlias

import pytest

import litellm
from litellm.integrations.argilla import ArgillaLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.integrations.argilla import ArgillaItem
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import StandardLoggingPayload
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_MODEL, MESSAGES_RESPONSE

Surface: TypeAlias = Literal["messages", "responses"]
XFAIL: Final = pytest.mark.xfail(
    reason="Native inference callback parity with the Python wrapper and body is pending",
    strict=False,
    raises=(AssertionError, TimeoutError, pytest.fail.Exception),
)
BACKENDS: Final = (pytest.param(False, id="python"), pytest.param(True, id="rust", marks=XFAIL))
CALLABLE_REGISTRATIONS: Final = (
    pytest.param("input", False, False, id="global-input-function"),
    pytest.param("success", False, False, id="global-sync-success-function"),
    pytest.param("success", False, True, id="request-sync-success-function"),
    pytest.param("success", True, False, id="global-async-success-function"),
    pytest.param("success", True, True, id="request-async-success-function"),
    pytest.param("failure", False, False, id="global-sync-failure-function"),
    pytest.param("failure", False, True, id="request-sync-failure-function"),
    pytest.param("failure", True, False, id="global-async-failure-function"),
)
RESPONSES_MODEL: Final = "openai/gpt-6.1-sol"
RESPONSES_RESPONSE: Final = {
    "id": "resp_callbacks",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": RESPONSES_MODEL.removeprefix("openai/"),
    "output": [
        {
            "id": "msg_callbacks",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hello from callbacks", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
}
SUCCESS_HOOKS: Final = (
    "log_pre_api_call",
    "log_post_api_call",
    "log_success_event",
    "logging_hook",
    "async_log_success_event",
    "async_logging_hook",
    "redact_standard_logging_payload_from_model_call_details",
)
FAILURE_HOOKS: Final = (
    "log_failure_event",
    "async_log_failure_event",
    "async_post_call_failure_deployment_hook",
    "log_model_group_rate_limit_error",
)
AGENTIC_HOOKS: Final = (
    "async_should_run_agentic_loop",
    "async_run_agentic_loop",
    "async_build_agentic_loop_plan",
    "async_post_agentic_loop_response_hook",
    "async_agentic_loop_cleanup_hook",
)


def provider_response(surface: Surface) -> dict[str, object]:
    return dict(MESSAGES_RESPONSE if surface == "messages" else RESPONSES_RESPONSE)


def response_id(response: object) -> object:
    if isinstance(response, ResponsesAPIResponse):
        return ResponsesAPIRequestUtils.decode_responses_api_response_id(response.id).get("response_id") or response.id
    return response.get("id") if isinstance(response, dict) else None


async def call(
    server: RecordingServer, surface: Surface, native: bool, asynchronous: bool = True, **kwargs: object
) -> object:
    try:
        return await _call(server, surface, native, asynchronous, **kwargs)
    except RuntimeError as error:
        if native and str(error).startswith(f"Rust {surface} bridge declined"):
            pytest.fail(str(error))
        raise


async def _call(
    server: RecordingServer, surface: Surface, native: bool, asynchronous: bool, **kwargs: object
) -> object:
    route: Final = Route.MESSAGES if surface == "messages" else Route.RESPONSES
    rollout: Final = Rollout.RUST_REQUIRED if native else Rollout.PYTHON_ONLY
    arguments: Final = {
        "api_key": "test-key",
        "api_base": server.base_url,
        "litellm_call_id": "callback-contract",
        **kwargs,
    }
    with rebound(catalog, "RULES", (RouteRule(route, rollout), *catalog.RULES)):
        if surface == "messages":
            if asynchronous:
                return await litellm.anthropic.messages.acreate(
                    model=MESSAGES_MODEL, messages=list(MESSAGES), max_tokens=64, **arguments
                )
            return litellm.anthropic.messages.create(
                model=MESSAGES_MODEL, messages=list(MESSAGES), max_tokens=64, **arguments
            )
        if asynchronous:
            return await litellm.aresponses(model=RESPONSES_MODEL, input="Hello", **arguments)
        return litellm.responses(model=RESPONSES_MODEL, input="Hello", **arguments)


def assert_transport(server: RecordingServer, native: bool, count: int = 1) -> None:
    assert len(server.requests) == count
    if native:
        assert all(not request.headers.get("user-agent", "").startswith("python-httpx") for request in server.requests)


class ContractLogger(RecordingLogger):
    def log_post_api_call(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._record("log_post_api_call", kwargs, response_obj)

    async def async_logging_hook(
        self, kwargs: dict[str, object], result: object, call_type: str
    ) -> tuple[dict[str, object], object]:
        self._record("async_logging_hook", kwargs, result)
        return {**kwargs, "callback_logging_hook": True}, result

    def redact_standard_logging_payload_from_model_call_details(
        self, model_call_details: dict[str, object]
    ) -> dict[str, object]:
        self._record("redact_standard_logging_payload_from_model_call_details", model_call_details)
        return {**model_call_details, "callback_redacted": True}

    async def async_post_call_failure_deployment_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: Exception,
        call_type: object,
        *,
        fallback_depth: int | None = None,
    ) -> None:
        self._record("async_post_call_failure_deployment_hook", dict(request_data), original_exception)

    async def log_model_group_rate_limit_error(
        self, exception: Exception, original_model_group: str | None, kwargs: dict[str, object]
    ) -> None:
        self._record("log_model_group_rate_limit_error", kwargs, exception)


async def assert_success_contract(
    server: RecordingServer, surface: Surface, native: bool, hook: str, asynchronous: bool
) -> None:
    recorder: Final = ContractLogger()
    server.default_response = ResponseSpec(body=provider_response(surface))
    with rebound(litellm, "callbacks", [recorder]):
        response: Final = await call(server, surface, native, asynchronous)
        if asynchronous and hook in ("log_success_event", "logging_hook"):
            await recorder.wait_for_async("async_log_success_event", timeout=1)
            assert hook not in recorder.names
            assert_transport(server, native)
            return
        events: Final = await recorder.wait_for_async(hook, timeout=1)
    assert_transport(server, native)
    assert response_id(response) == provider_response(surface)["id"]
    assert len(events) == 1
    details: Final = events[0].kwargs
    assert isinstance(details, dict)
    assert details["litellm_call_id"] == "callback-contract"
    if hook in ("log_success_event", "async_log_success_event", "logging_hook", "async_logging_hook"):
        assert events[0].response is not None
    if hook == "async_log_success_event":
        assert details["callback_redacted"] is True
        assert details["callback_logging_hook"] is True
    if hook == "log_pre_api_call":
        additional: Final = details["additional_args"]
        assert isinstance(additional, dict)
        assert additional["complete_input_dict"] == server.requests[0].body


async def assert_failure_contract(server: RecordingServer, surface: Surface, native: bool, hook: str) -> None:
    recorder: Final = ContractLogger()
    server.default_response = ResponseSpec(
        status=429,
        body={"type": "error", "error": {"type": "rate_limit_error", "message": "rate limited"}},
    )
    with rebound(litellm, "callbacks", [recorder]):
        with pytest.raises(litellm.RateLimitError) as raised:
            await call(server, surface, native, num_retries=0, metadata={"model_group_size": 1})
        if hook == "log_failure_event":
            await recorder.wait_for_async("async_log_failure_event", timeout=1)
            assert hook not in recorder.names
            assert_transport(server, native)
            return
        events: Final = await recorder.wait_for_async(hook, timeout=1)
    assert_transport(server, native)
    assert len(events) == 1
    if hook == "async_post_call_failure_deployment_hook":
        assert type(events[0].response) is type(raised.value)
        assert events[0].response is not raised.value
    elif hook == "log_model_group_rate_limit_error":
        assert events[0].response is raised.value
    else:
        details: Final = events[0].kwargs
        assert isinstance(details, dict)
        assert details["exception"] is raised.value


class DeploymentLogger(RecordingLogger):
    async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: object) -> dict[str, object]:
        self._record("async_pre_call_deployment_hook", kwargs)
        return {**kwargs, "extra_headers": {"x-callback-contract": "rewritten"}}

    async def async_post_call_success_deployment_hook(
        self, request_data: dict[str, object], response: object, call_type: object
    ) -> object:
        self._record("async_post_call_success_deployment_hook", request_data, response)
        if isinstance(response, ResponsesAPIResponse):
            return response.model_copy(update={"id": "callback-replaced"})
        assert isinstance(response, dict)
        return {**response, "id": "callback-replaced"}


async def assert_deployment_contract(server: RecordingServer, surface: Surface, native: bool, hook: str) -> None:
    recorder: Final = DeploymentLogger()
    server.default_response = ResponseSpec(body=provider_response(surface))
    with rebound(litellm, "callbacks", [recorder]):
        response: Final = await call(server, surface, native)
    assert_transport(server, native)
    assert recorder.names.count(hook) == 1
    if hook == "async_pre_call_deployment_hook":
        assert server.requests[0].headers["x-callback-contract"] == "rewritten"
    else:
        assert response_id(response) == "callback-replaced"


async def assert_callable_contract(
    server: RecordingServer, surface: Surface, native: bool, event: str, asynchronous: bool, dynamic: bool
) -> None:
    recorder: Final = RecordingLogger()
    server.default_response = ResponseSpec(
        body=provider_response(surface)
        if event != "failure"
        else {"error": {"message": "bad", "type": "invalid_request_error"}},
        status=400 if event == "failure" else 200,
    )

    def input_callback(kwargs: dict[str, object]) -> None:
        recorder._record("input", kwargs)

    def terminal_callback(kwargs: dict[str, object], response: object, start: datetime, end: datetime) -> None:
        recorder._record(event, kwargs, response)

    async def async_terminal_callback(
        kwargs: dict[str, object], response: object, start: datetime, end: datetime
    ) -> None:
        recorder._record(event, kwargs, response)

    callback: Final = (
        input_callback if event == "input" else (async_terminal_callback if asynchronous else terminal_callback)
    )
    attribute: Final = f"{'_async_' if asynchronous and event != 'input' else ''}{event}_callback"
    with rebound(litellm, attribute, [] if dynamic else [callback]):
        arguments: Final = {f"{event}_callback": [callback]} if dynamic else {}
        if event == "failure":
            with pytest.raises(litellm.BadRequestError):
                await call(server, surface, native, num_retries=0, **arguments)
        else:
            await call(server, surface, native, **arguments)
        if (event == "success" and not asynchronous) or (event == "failure" and asynchronous and dynamic):
            await drain_logging()
            assert event not in recorder.names
            assert_transport(server, native)
            return
        events: Final = await recorder.wait_for_async(event, timeout=1)
    assert_transport(server, native)
    assert len(events) == 1
    details: Final = events[0].kwargs
    assert isinstance(details, dict)
    assert details["litellm_call_id"] == "callback-contract"
    assert details["log_event_type"] == ("pre_api_call" if event == "input" else "post_api_call")


class AgenticLogger(RecordingLogger):
    async def async_should_run_agentic_loop(
        self,
        response: object,
        model: str,
        messages: list[dict[str, object]],
        tools: object,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict[str, object],
    ) -> tuple[bool, dict[str, object]]:
        self._record("async_should_run_agentic_loop", kwargs, response)
        return response_id(response) != "followup", {"tool_calls": [{"id": "tool_contract", "name": "lookup"}]}

    async def async_build_agentic_loop_plan(
        self,
        tools: object,
        model: str,
        messages: list[dict[str, object]],
        response: object,
        anthropic_messages_provider_config: object,
        anthropic_messages_optional_request_params: dict[str, object],
        logging_obj: object,
        stream: bool,
        kwargs: dict[str, object],
    ) -> AgenticLoopPlan:
        self._record("async_build_agentic_loop_plan", kwargs, response)
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch.model_validate(
                {"messages": [{"role": "user", "content": "followup"}]}
            ),
        )

    async def async_post_agentic_loop_response_hook(
        self, response: object, plan: AgenticLoopPlan, kwargs: dict[str, object]
    ) -> object:
        self._record("async_post_agentic_loop_response_hook", kwargs, response)
        if isinstance(response, ResponsesAPIResponse):
            return response.model_copy(update={"id": "agentic-replaced"})
        assert isinstance(response, dict)
        return {**response, "id": "agentic-replaced"}

    async def async_agentic_loop_cleanup_hook(self, plan: AgenticLoopPlan, kwargs: dict[str, object]) -> None:
        self._record("async_agentic_loop_cleanup_hook", kwargs, plan)


class LegacyAgenticLogger(RecordingLogger):
    async def async_should_run_agentic_loop(
        self,
        response: object,
        model: str,
        messages: list[dict[str, object]],
        tools: object,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict[str, object],
    ) -> tuple[bool, dict[str, object]]:
        self._record("async_should_run_agentic_loop", kwargs, response)
        return True, {"tool_calls": [{"id": "tool_contract", "name": "lookup"}]}

    async def async_run_agentic_loop(
        self,
        tools: object,
        model: str,
        messages: list[dict[str, object]],
        response: object,
        anthropic_messages_provider_config: object,
        anthropic_messages_optional_request_params: dict[str, object],
        logging_obj: object,
        stream: bool,
        kwargs: dict[str, object],
    ) -> object:
        self._record("async_run_agentic_loop", kwargs, response)
        if isinstance(response, ResponsesAPIResponse):
            return response.model_copy(update={"id": "agentic-replaced"})
        assert isinstance(response, dict)
        return {**response, "id": "agentic-replaced"}


async def assert_agentic_contract(server: RecordingServer, surface: Surface, native: bool, hook: str) -> None:
    planned: Final = hook not in ("async_should_run_agentic_loop", "async_run_agentic_loop")
    recorder: Final = AgenticLogger() if planned else LegacyAgenticLogger()
    server.default_response = ResponseSpec(body=provider_response(surface))
    if planned:
        server.expected_requests = None
        server.enqueue(ResponseSpec(body=provider_response(surface)))
        server.default_response = ResponseSpec(body={**provider_response(surface), "id": "followup"})
    with rebound(litellm, "callbacks", [recorder]):
        response: Final = await call(server, surface, native)
        await drain_logging()
    assert_transport(server, native, count=2 if planned else 1)
    assert recorder.names.count(hook) >= 1
    assert response_id(response) == "agentic-replaced"


async def consume(stream: object) -> tuple[object, ...]:
    if isinstance(stream, AsyncIterator):
        return tuple([chunk async for chunk in stream])
    assert isinstance(stream, Iterator)
    return tuple(stream)


async def assert_logger_fn_contract(server: RecordingServer, surface: Surface, native: bool) -> None:
    recorder: Final = RecordingLogger()
    server.default_response = ResponseSpec(body=provider_response(surface))

    def logger_fn(details: dict[str, object]) -> None:
        recorder._record("logger_fn", details)

    logger: Final = Logging(
        model=MESSAGES_MODEL if surface == "messages" else RESPONSES_MODEL,
        messages=list(MESSAGES),
        stream=False,
        start_time=datetime.now(),
        call_type="anthropic_messages" if surface == "messages" else "aresponses",
        litellm_call_id="callback-contract",
        function_id="callback-contract",
    )
    logger.update_environment_variables(litellm_params={"logger_fn": logger_fn}, optional_params={})
    await call(server, surface, native, litellm_logging_obj=logger)

    assert_transport(server, native)
    assert recorder.events == ()


async def assert_dataset_contract(server: RecordingServer, surface: Surface, native: bool) -> None:
    recorder: Final = RecordingLogger()

    class FilterDataset(RecordingLogger):
        async def async_dataset_hook(
            self, logged_data: ArgillaItem, standard_logging_payload: StandardLoggingPayload | None
        ) -> None:
            recorder._record("async_dataset_hook", response=logged_data)

    server.expected_requests = None
    server.enqueue(ResponseSpec(body={"items": [{"id": "callback-dataset"}]}))
    server.default_response = ResponseSpec(body=provider_response(surface))
    with rebound(litellm, "argilla_transformation_object", {"prompt": "messages", "answer": "response"}):
        sink: Final = ArgillaLogger(
            argilla_api_key="test-key",
            argilla_base_url=server.base_url,
            argilla_dataset_name="callback-dataset",
            flush_interval=3600,
        )
        with rebound(litellm, "callbacks", [sink, FilterDataset()]):
            await call(server, surface, native)
            events: Final = await recorder.wait_for_async("async_dataset_hook", timeout=1)
            await sink.flush_queue()

    assert len(events) == 1
    data: Final = events[0].response
    assert isinstance(data, dict)
    fields: Final = data["fields"]
    assert isinstance(fields, dict)
    assert fields["prompt"]
    assert len(server.requests) == 2
    assert server.requests[0].method == "GET"
    assert server.requests[1].method == "POST"

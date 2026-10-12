import asyncio
import datetime
import inspect
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import pytest
from pydantic import TypeAdapter

import litellm
from litellm._internal_context import is_internal_call
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.rust_bridge.host import callbacks_legacy_python as legacy
from litellm.rust_bridge.host.callbacks_legacy_python import failure_handler, setup
from litellm.types.integrations.custom_logger import AgenticLoopPlan
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import ModelResponse

_OCR_KWARGS: Final = MappingProxyType(
    {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
    }
)


def _supplied_logger() -> Logging:
    return Logging(
        model="mistral/mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="supplied",
        function_id="supplied",
    )


def test_native_stream_headers_reach_spend_callbacks() -> None:
    logger: Final = _supplied_logger()
    legacy.stream_opened(logger, {"additional_headers": {"llm_provider-request-id": "req_native"}})

    assert logger.stream is True
    assert logger.model_call_details["response_headers"] == {"llm_provider-request-id": "req_native"}


def test_setup_reuses_a_supplied_logger() -> None:
    supplied: Final = _supplied_logger()
    result: Final = setup(
        "aocr", (), {**_OCR_KWARGS, "litellm_logging_obj": supplied}, datetime.datetime.now(), asynchronous=True
    )
    assert result.logger is supplied


@pytest.mark.parametrize("explicit_provider", (None, "openai"))
def test_cache_hit_finalization_preserves_execution_provider_attribution(explicit_provider: str | None) -> None:
    now: Final = datetime.datetime.now()
    kwargs: Final = {
        "model": "openai/cache-test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "custom_llm_provider": explicit_provider,
        "metadata": {"user_api_key": "key-hash"},
    }
    prepared: Final = setup("acompletion", (), kwargs, now, asynchronous=True)
    legacy.update_logging(
        prepared.logger,
        prepared.kwargs,
        "resolved-cache-model",
        {},
        {**prepared.logger.litellm_params, "custom_llm_provider": "azure"},
        "azure",
    )
    prepared.logger.model_call_details.update({"cache_hit": True, "cache_key": "cached-response"})
    response: Final = ModelResponse(model="cache-test-model")
    legacy.finalize(response, prepared.logger, prepared.kwargs, now, now)
    assert prepared.logger.model_call_details["custom_llm_provider"] == "azure"
    assert prepared.logger.model_call_details["model"] == "resolved-cache-model"
    assert prepared.logger.litellm_params["metadata"]["user_api_key"] == "key-hash"
    assert response._hidden_params["cache_key"] == "cached-response"
    assert response._hidden_params["response_cost"] == 0


@pytest.mark.parametrize(
    "call_type, kwargs",
    [
        ("aocr", _OCR_KWARGS),
        ("aembedding", MappingProxyType({"model": "text-embedding-3-large", "input": ["hi"]})),
    ],
    ids=["ocr", "embedding"],
)
def test_setup_builds_a_logger_when_none_is_supplied(call_type: str, kwargs: Mapping[str, object]) -> None:
    result: Final = setup(call_type, (), kwargs, datetime.datetime.now(), asynchronous=True)
    assert result.logger.litellm_call_id == result.kwargs["litellm_call_id"]


def _budget_reservation() -> dict:
    return {"reserved_cost": 0.5, "entries": [], "finalized": False, "callback_bound": False}


def _kwargs_with_a_budget_reservation(reservation: dict) -> dict[str, object]:
    return {**_OCR_KWARGS, "metadata": {"user_api_key_budget_reservation": reservation}}


def test_setup_claims_the_budget_reservation_for_an_async_call() -> None:
    reservation: Final = _budget_reservation()

    setup("aocr", (), _kwargs_with_a_budget_reservation(reservation), datetime.datetime.now(), asynchronous=True)

    assert reservation["callback_bound"] is True


def test_setup_claims_the_budget_reservation_a_supplied_logger_already_saw() -> None:
    reservation: Final = _budget_reservation()
    supplied: Final = _supplied_logger()
    supplied.update_environment_variables(
        litellm_params={"metadata": {"user_api_key_budget_reservation": reservation}}, optional_params={}
    )
    assert reservation["callback_bound"] is False

    setup("aocr", (), {**_OCR_KWARGS, "litellm_logging_obj": supplied}, datetime.datetime.now(), asynchronous=True)

    assert reservation["callback_bound"] is True


def test_setup_leaves_the_budget_reservation_alone_for_a_sync_call() -> None:
    reservation: Final = _budget_reservation()

    setup("aocr", (), _kwargs_with_a_budget_reservation(reservation), datetime.datetime.now(), asynchronous=False)

    assert reservation["callback_bound"] is False


def test_setup_leaves_the_budget_reservation_alone_for_an_internal_call() -> None:
    reservation: Final = _budget_reservation()
    token: Final = is_internal_call.set(True)
    try:
        setup("aocr", (), _kwargs_with_a_budget_reservation(reservation), datetime.datetime.now(), asynchronous=True)
    finally:
        is_internal_call.reset(token)

    assert reservation["callback_bound"] is False


def test_failure_handler_hands_the_budget_reservation_back_for_an_async_call() -> None:
    reservation: Final = _budget_reservation()
    now: Final = datetime.datetime.now()
    result: Final = setup("aocr", (), _kwargs_with_a_budget_reservation(reservation), now, asynchronous=True)
    assert reservation["callback_bound"] is True

    pending: Final = failure_handler(result.logger, RuntimeError("upstream refused"), now, now, asynchronous=True)

    assert reservation["callback_bound"] is False
    assert pending is not None
    pending.close()


def test_failure_handler_of_an_internal_call_leaves_the_outer_budget_reservation_claim_in_place() -> None:
    reservation: Final = _budget_reservation()
    now: Final = datetime.datetime.now()
    result: Final = setup("aocr", (), _kwargs_with_a_budget_reservation(reservation), now, asynchronous=True)
    token: Final = is_internal_call.set(True)
    try:
        pending: Final = failure_handler(result.logger, RuntimeError("inner step failed"), now, now, asynchronous=True)
    finally:
        is_internal_call.reset(token)

    assert reservation["callback_bound"] is True
    assert pending is not None
    pending.close()


CRATE: Final = Path(__file__).parents[4] / "litellm-rust/crates/callbacks-legacy-python"
CONTRACT_PATH: Final = CRATE / "python_contract.json"
CALLBACK_TABLE_PATH: Final = CRATE / "custom_logger_contract.json"


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {name: list(inspect.signature(getattr(legacy, name)).parameters) for name in contract}


def test_every_public_custom_logger_method_has_a_row_in_the_rust_callback_table() -> None:
    """A hook added to `CustomLogger` fails here until the Rust table says where a native call runs it."""
    table: Final = TypeAdapter(dict[str, list[dict[str, object]]]).validate_json(CALLBACK_TABLE_PATH.read_text())
    public: Final = {
        name for name, _member in inspect.getmembers_static(CustomLogger, callable) if not name.startswith("_")
    }

    assert set(table) == public
    assert all(rows for rows in table.values())


@pytest.mark.asyncio
async def test_pre_request_hooks_see_the_handler_view_and_their_edits_lay_over_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from contextvars import ContextVar

    caller_messages: Final = [{"role": "user", "content": "hello"}]
    tools: Final = [{"name": "replacement", "input_schema": {"type": "object"}}]
    tool_choice: Final = {"type": "tool", "name": "original"}
    marker: Final = ContextVar("messages-hook-marker", default="caller")
    caller: Final = asyncio.current_task()

    class Replace(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
        ) -> dict[str, object]:
            await asyncio.sleep(0)
            assert asyncio.current_task() is caller
            assert marker.get() == "caller"
            assert kwargs["tool_choice"] is tool_choice
            assert kwargs["opaque"] is request["opaque"]
            assert "max_tokens" not in kwargs
            marker.set("hook")
            return {**kwargs, "tools": tools, "temperature": 0.75}

    class Observe(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
        ) -> None:
            assert messages is caller_messages
            assert kwargs["tools"] is tools
            assert kwargs["temperature"] == 0.75
            assert marker.get() == "hook"

    monkeypatch.setattr(litellm, "callbacks", [Replace(), Observe()])
    request: Final[dict[str, object]] = {
        "model": "anthropic/claude-sonnet-5",
        "messages": caller_messages,
        "max_tokens": 16,
        "stream": False,
        "tool_choice": tool_choice,
        "temperature": 0.25,
        "opaque": object(),
    }

    prepared: Final = await legacy.prepare_request("messages", request)

    assert prepared["tools"] is tools
    assert prepared["temperature"] == 0.75
    assert prepared["opaque"] is request["opaque"]
    assert prepared["messages"] is caller_messages
    assert prepared["tool_choice"] is tool_choice
    assert prepared["max_tokens"] == 16
    assert prepared["custom_llm_provider"] == "anthropic"
    assert "litellm_params" not in prepared
    assert request["temperature"] == 0.25
    assert "tools" not in request
    assert marker.get() == "hook"


@pytest.mark.asyncio
async def test_pre_request_hook_failure_keeps_exception_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = RuntimeError("pre-request rejected")

    class Reject(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
        ) -> None:
            raise failure

    monkeypatch.setattr(litellm, "callbacks", [Reject()])
    with pytest.raises(RuntimeError) as raised:
        await legacy.prepare_request(
            "messages", {"model": "anthropic/claude-sonnet-5", "messages": [], "max_tokens": 8}
        )
    assert raised.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("run", "plan"))
async def test_agentic_response_hooks_see_the_logged_request_and_their_answer_is_selected(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import asyncio

    caller_messages: Final = [{"role": "user", "content": "hello"}]
    caller_tools: Final = [{"name": "lookup", "input_schema": {"type": "object"}}]
    original: Final = {"content": [{"type": "text", "text": "provider"}]}
    replacement: Final = {"content": [{"type": "text", "text": "callback"}]}
    caller: Final = asyncio.current_task()

    class Gate(CustomLogger):
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
            assert response is original
            assert model == "claude-sonnet-5"
            assert messages is caller_messages
            assert tools is caller_tools
            assert stream is False
            assert custom_llm_provider == "anthropic"
            assert kwargs["api_key"] == "caller-key"
            assert kwargs["api_base"] == "http://localhost:4000"
            return True, {}

        async def async_run_agentic_loop(
            self,
            tools: object,
            model: str,
            messages: object,
            response: object,
            anthropic_messages_provider_config: object,
            anthropic_messages_optional_request_params: Mapping[str, object],
            logging_obj: object,
            stream: bool,
            kwargs: dict[str, object],
        ) -> object:
            await asyncio.sleep(0)
            assert asyncio.current_task() is caller
            assert anthropic_messages_optional_request_params["max_tokens"] == 8
            assert logging_obj is logger
            return replacement

    class Planned(Gate):
        async def async_build_agentic_loop_plan(
            self,
            tools: object,
            model: str,
            messages: object,
            response: object,
            anthropic_messages_provider_config: object,
            anthropic_messages_optional_request_params: Mapping[str, object],
            logging_obj: object,
            stream: bool,
            kwargs: dict[str, object],
        ) -> AgenticLoopPlan:
            await asyncio.sleep(0)
            assert asyncio.current_task() is caller
            assert logging_obj is logger
            return AgenticLoopPlan(response_override=replacement)

    callback: Final = Gate() if mode == "run" else Planned()
    logger: Final = _supplied_logger()
    request: Final[dict[str, object]] = {
        "model": "anthropic/claude-sonnet-5",
        "messages": caller_messages,
        "litellm_logging_obj": logger,
        "api_key": "caller-key",
        "api_base": "http://localhost:4000",
    }
    legacy.update_logging(logger, request, "claude-sonnet-5", {"tools": caller_tools, "max_tokens": 8}, {}, "anthropic")
    monkeypatch.setattr(litellm, "callbacks", [callback])

    selected: Final = await legacy.transform_response("messages", original, request, False)

    assert selected is replacement


@pytest.mark.asyncio
async def test_agentic_response_hooks_that_do_not_run_keep_the_provider_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original: Final = {"content": [{"type": "text", "text": "provider"}]}
    logger: Final = _supplied_logger()
    request: Final[dict[str, object]] = {
        "model": "anthropic/claude-sonnet-5",
        "messages": [],
        "litellm_logging_obj": logger,
    }
    legacy.update_logging(logger, request, "claude-sonnet-5", {"max_tokens": 8}, {}, "anthropic")
    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])

    assert await legacy.transform_response("messages", original, request, False) is original


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("ocr", "completion", "responses"))
async def test_request_body_hooks_refuse_an_operation_whose_handler_has_none(operation: str) -> None:
    with pytest.raises(ValueError, match=operation):
        await legacy.prepare_request(operation, {"model": "m"})


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("ocr", "completion"))
async def test_response_body_hooks_refuse_an_operation_whose_handler_has_none(operation: str) -> None:
    with pytest.raises(ValueError, match=operation):
        await legacy.transform_response(operation, {}, {"model": "m"}, False)


def _responses_logger() -> Logging:
    return Logging(
        model="openai/gpt-6-sol",
        messages=[],
        stream=False,
        call_type="aresponses",
        start_time=datetime.datetime.now(),
        litellm_call_id="supplied",
        function_id="supplied",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("public_input", ("hello", [{"role": "user", "content": "hello"}]), ids=("text", "items"))
async def test_responses_agentic_hooks_see_the_responses_surface_and_their_answer_is_selected(
    monkeypatch: pytest.MonkeyPatch, public_input: object
) -> None:
    from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
    from litellm.types.integrations.custom_logger import RESPONSES_AGENTIC_SURFACE

    original: Final = {"output": [{"type": "function_call", "name": "lookup"}]}
    replacement: Final = {"output": [{"type": "message", "content": "callback"}]}
    seen: Final[list[Mapping[str, object]]] = []

    class Gate(CustomLogger):
        async def async_should_run_agentic_loop(
            self, response, model, messages, tools, stream, custom_llm_provider, kwargs
        ) -> tuple[bool, dict[str, object]]:
            seen.append(
                {
                    "response": response,
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "provider": custom_llm_provider,
                    "surface": kwargs.get("_agentic_loop_api_surface"),
                    "api_key": kwargs.get("api_key"),
                }
            )
            return True, {}

        async def async_run_agentic_loop(
            self,
            tools,
            model,
            messages,
            response,
            anthropic_messages_provider_config,
            anthropic_messages_optional_request_params,
            logging_obj,
            stream,
            kwargs,
        ) -> object:
            assert isinstance(anthropic_messages_provider_config, BaseResponsesAPIConfig)
            assert anthropic_messages_optional_request_params["max_output_tokens"] == 8
            assert logging_obj is logger
            return replacement

    logger: Final = _responses_logger()
    request: Final[dict[str, object]] = {
        "model": "openai/gpt-6-sol",
        "input": public_input,
        "litellm_logging_obj": logger,
        "api_key": "caller-key",
    }
    legacy.update_logging(
        logger,
        request,
        "gpt-6-sol",
        {"tools": [{"type": "function", "name": "lookup"}], "max_output_tokens": 8},
        {},
        "openai",
    )
    monkeypatch.setattr(litellm, "callbacks", [Gate()])

    selected: Final = await legacy.transform_response("responses", original, request, False)

    assert selected is replacement
    assert seen == [
        {
            "response": original,
            "model": "gpt-6-sol",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "name": "lookup"}],
            "provider": "openai",
            "surface": RESPONSES_AGENTIC_SURFACE,
            "api_key": "caller-key",
        }
    ]


def test_run_sync_runs_a_body_fan_out_to_completion_without_a_running_loop() -> None:
    async def fan_out() -> str:
        import asyncio

        await asyncio.sleep(0)
        return "settled"

    assert legacy.run_sync(fan_out()) == "settled"


class _ServerSideSearch(CustomLogger):
    server_fulfilled_tool_names: Final = frozenset({"litellm_web_search"})


@pytest.mark.parametrize(
    ("callbacks", "tools", "expected"),
    (
        ((), [{"name": "litellm_web_search"}], "relay"),
        ((_ServerSideSearch(),), [{"name": "get_weather"}], "relay"),
        ((_ServerSideSearch(),), [{"name": "litellm_web_search"}], "hold_back"),
        ((_ServerSideSearch(),), None, "relay"),
    ),
    ids=("no-callbacks", "other-tool", "server-fulfilled-tool", "no-tools"),
)
@pytest.mark.parametrize("operation", ("messages", "responses"))
def test_stream_policy_holds_back_exactly_when_a_callback_fulfils_a_requested_tool(
    monkeypatch: pytest.MonkeyPatch,
    callbacks: tuple[CustomLogger, ...],
    tools: object,
    expected: str,
    operation: str,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", list(callbacks))
    request: Final[dict[str, object]] = {
        "model": "m",
        "stream": True,
        "tools": tools,
        "litellm_logging_obj": _supplied_logger(),
    }

    assert legacy.stream_policy(operation, request) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("held_back", (True, False), ids=("held-back", "complete"))
async def test_a_held_back_stream_whose_server_fulfilled_tool_call_nobody_fulfilled_is_an_error(
    monkeypatch: pytest.MonkeyPatch, held_back: bool
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_ServerSideSearch()])
    logger: Final = _supplied_logger()
    request: Final[dict[str, object]] = {
        "model": "anthropic/claude-sonnet-5",
        "messages": [],
        "tools": [{"name": "litellm_web_search"}],
        "litellm_logging_obj": logger,
    }
    legacy.update_logging(logger, request, "claude-sonnet-5", {"tools": request["tools"]}, {}, "anthropic")
    leaking: Final = {"content": [{"type": "tool_use", "id": "t", "name": "litellm_web_search", "input": {}}]}

    if held_back:
        with pytest.raises(litellm.InternalServerError, match="Server-side tool retrieval failed"):
            await legacy.transform_response("messages", leaking, request, True)
    else:
        assert await legacy.transform_response("messages", leaking, request, False) is leaking


class _SearchAnswers(CustomLogger):
    """Stands in for the websearch logger: the handler's isinstance check is what gates it."""

    def __init__(self, answer: object) -> None:
        super().__init__()
        self.answer = answer
        self.seen: list[Mapping[str, object]] = []

    async def try_short_circuit_search(self, model, messages, tools, custom_llm_provider, kwargs=None):
        self.seen.append({"model": model, "tools": tools, "provider": custom_llm_provider, "kwargs": kwargs})
        return self.answer


class _OwnsAdvisor:
    def __init__(self, answer: object) -> None:
        self.answer = answer
        self.handled: list[dict[str, object]] = []

    def can_handle(self, tools, custom_llm_provider) -> bool:
        return bool(tools) and any(tool.get("type") == "advisor" for tool in tools)

    async def handle(self, **kwargs):
        self.handled.append(kwargs)
        return self.answer


@pytest.fixture
def websearch_isinstance(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.integrations.websearch_interception import handler as websearch

    monkeypatch.setattr(websearch, "WebSearchInterceptionLogger", _SearchAnswers)


@pytest.mark.asyncio
async def test_short_circuit_answers_from_the_websearch_logger_before_any_interceptor(
    monkeypatch: pytest.MonkeyPatch, websearch_isinstance: None
) -> None:
    from litellm.llms.anthropic.pass_through.messages import interceptors

    answer: Final = {"content": [{"type": "text", "text": "searched"}]}
    searcher: Final = _SearchAnswers(answer)
    advisor: Final = _OwnsAdvisor({"content": []})
    monkeypatch.setattr(litellm, "callbacks", [searcher])
    monkeypatch.setattr(interceptors, "_interceptors", [advisor])
    request: Final[dict[str, object]] = {
        "model": "github_copilot/claude-sonnet-5",
        "messages": [{"role": "user", "content": "search"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}, {"type": "advisor"}],
        "custom_llm_provider": "github_copilot",
        "stream": True,
        "max_tokens": 8,
        "metadata": {"user": "u"},
        "opaque": object(),
    }

    assert await legacy.short_circuit("messages", request) is answer
    assert searcher.seen == [
        {
            "model": "github_copilot/claude-sonnet-5",
            "tools": request["tools"],
            "provider": "github_copilot",
            "kwargs": {"opaque": request["opaque"], "metadata": {"user": "u"}},
        }
    ]
    assert advisor.handled == []


@pytest.mark.asyncio
async def test_short_circuit_hands_the_call_to_the_first_interceptor_that_owns_it_and_never_streams(
    monkeypatch: pytest.MonkeyPatch, websearch_isinstance: None
) -> None:
    from litellm.llms.anthropic.pass_through.messages import interceptors

    answer: Final = {"content": [{"type": "text", "text": "advised"}]}
    advisor: Final = _OwnsAdvisor(answer)
    monkeypatch.setattr(litellm, "callbacks", [_SearchAnswers(None)])
    monkeypatch.setattr(interceptors, "_interceptors", [_OwnsAdvisor({"content": []}) if False else advisor])
    request: Final[dict[str, object]] = {
        "model": "openai/gpt-6",
        "messages": [{"role": "user", "content": "plan"}],
        "tools": [{"type": "advisor"}],
        "custom_llm_provider": "openai",
        "stream": True,
        "max_tokens": 8,
        "api_key": "k",
        "client": object(),
        "opaque": object(),
    }

    assert await legacy.short_circuit("messages", request) is answer
    [handled] = advisor.handled
    assert handled["stream"] is False
    assert handled["model"] == "openai/gpt-6"
    assert handled["max_tokens"] == 8
    assert handled["api_key"] == "k"
    assert handled["opaque"] is request["opaque"]
    assert "client" not in handled


@pytest.mark.asyncio
async def test_short_circuit_continues_to_the_provider_when_nothing_owns_the_request(
    monkeypatch: pytest.MonkeyPatch, websearch_isinstance: None
) -> None:
    from litellm.llms.anthropic.pass_through.messages import interceptors

    monkeypatch.setattr(litellm, "callbacks", [_SearchAnswers(None)])
    monkeypatch.setattr(interceptors, "_interceptors", [_OwnsAdvisor({"content": []})])
    request: Final[dict[str, object]] = {
        "model": "anthropic/claude-sonnet-5",
        "messages": [],
        "tools": [{"name": "get_weather"}],
        "custom_llm_provider": "anthropic",
        "max_tokens": 8,
    }

    assert await legacy.short_circuit("messages", request) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("ocr", "completion", "responses"))
async def test_short_circuits_refuse_an_operation_whose_handler_has_none(operation: str) -> None:
    with pytest.raises(ValueError, match=operation):
        await legacy.short_circuit(operation, {"model": "m"})


class _StreamLogger:
    def __init__(self) -> None:
        self.start_time = datetime.datetime(2026, 1, 1)
        self.completion_start_time: datetime.datetime | None = None
        self.model_call_details: dict[str, object] = {}
        self.calls: list[tuple[str, object]] = []

    async def dispatch_success_handlers(self, result, start_time, end_time, cache_hit, prefer_async_handlers):
        self.calls.append(("dispatch", (result, start_time, end_time, cache_hit, prefer_async_handlers)))

    async def async_failure_handler(self, exception, traceback_exception, start_time, end_time):
        self.calls.append(("async_failure", (exception, start_time, end_time)))

    def failure_handler(self, exception, traceback_exception, start_time, end_time):
        self.calls.append(("failure", (exception, start_time, end_time)))


_RESPONSE: Final = ResponsesAPIResponse.model_validate(
    {"id": "resp_1", "object": "response", "created_at": 1, "model": "gpt", "status": "completed", "output": []}
)
_COMPLETED: Final = ResponseCompletedEvent(type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=_RESPONSE)
_DELTA: Final = OutputTextDeltaEvent(
    type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, item_id="msg", output_index=0, content_index=0, delta="hi"
)


@pytest.mark.asyncio
async def test_responses_stream_success_dispatches_a_copy_of_the_terminal_event_with_the_stream_times() -> None:
    logger: Final = _StreamLogger()
    logger.model_call_details["cache_hit"] = True
    start: Final = datetime.datetime(2026, 1, 1, 0, 0, 1)
    end: Final = datetime.datetime(2026, 1, 1, 0, 0, 2)
    first: Final = datetime.datetime(2026, 1, 1, 0, 0, 1, 500)

    legacy.stream_success(
        "responses",
        cast(Logging, logger),  # cast-ok: a recording stand-in for the Logging surface
        {},
        [_DELTA, _COMPLETED],
        start,
        end,
        first,
    )
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)

    assert logger.completion_start_time == first
    [(name, (logged, *rest))] = logger.calls
    assert name == "dispatch"
    assert rest == [start, end, True, True]
    assert logged == _COMPLETED
    assert logged is not _COMPLETED


def test_responses_stream_success_without_a_terminal_event_logs_nothing() -> None:
    logger: Final = _StreamLogger()

    legacy.stream_success(
        "responses",
        cast(Logging, logger),  # cast-ok: a recording stand-in for the Logging surface
        {},
        [_DELTA],
        datetime.datetime.now(),
        datetime.datetime.now(),
        None,
    )

    assert logger.calls == []


@pytest.mark.asyncio
async def test_responses_stream_failure_runs_the_async_then_the_sync_failure_handler() -> None:
    logger: Final = _StreamLogger()
    error: Final = RuntimeError("stream broke")

    await legacy.stream_failure(
        "responses",
        cast(Logging, logger),  # cast-ok: a recording stand-in for the Logging surface
        {},
        [_DELTA],
        error,
    )
    await asyncio.get_running_loop().run_in_executor(None, lambda: None)
    for _ in range(50):
        if len(logger.calls) == 2:
            break
        await asyncio.sleep(0.01)

    assert [name for name, _ in logger.calls] == ["async_failure", "failure"]
    assert all(call[1][0] is error and call[1][1] == logger.start_time for call in logger.calls)


@pytest.mark.parametrize("function", ("stream_success", "stream_failure"))
def test_stream_logging_rejects_an_operation_without_a_stream_iterator(function: str) -> None:
    logger: Final = _StreamLogger()
    arguments: Final = (
        (
            "ocr",
            cast(Logging, logger),
            {},
            [],
            datetime.datetime.now(),
            datetime.datetime.now(),
            None,
        )  # cast-ok: a recording stand-in for the Logging surface
        if function == "stream_success"
        else (
            "ocr",
            cast(Logging, logger),
            {},
            [],
            RuntimeError("x"),
        )  # cast-ok: a recording stand-in for the Logging surface
    )
    with pytest.raises(ValueError, match="no stream logging"):
        getattr(legacy, function)(*arguments)

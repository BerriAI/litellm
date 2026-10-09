import datetime
import inspect
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm._internal_context import is_internal_call
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge import callbacks_legacy_python as legacy
from litellm.rust_bridge.callbacks_legacy_python import failure_handler, setup
from litellm.types.integrations.custom_logger import AgenticLoopPlan
from litellm.types.utils import ModelResponse

_OCR_KWARGS: Final = MappingProxyType(
    {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
    }
)


@pytest.mark.asyncio
async def test_pre_request_hooks_keep_caller_identity_and_chain_replacements(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from contextvars import ContextVar

    messages: Final = [{"role": "user", "content": "hello"}]
    tools: Final = [{"name": "replacement", "input_schema": {"type": "object"}}]
    tool_choice: Final = {"type": "tool", "name": "original"}
    marker: Final = ContextVar("messages-hook-marker", default="caller")
    caller: Final = asyncio.current_task()

    class Replace(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: list[object], kwargs: dict[str, object]
        ) -> dict[str, object]:
            await asyncio.sleep(0)
            assert asyncio.current_task() is caller
            assert marker.get() == "caller"
            assert kwargs["tool_choice"] is tool_choice
            marker.set("hook")
            return {**kwargs, "tools": tools, "temperature": 0.75}

    class Observe(CustomLogger):
        async def async_pre_request_hook(self, model: str, received: list[object], kwargs: dict[str, object]) -> None:
            assert received is messages
            assert kwargs["tools"] is tools
            assert kwargs["temperature"] == 0.75
            assert kwargs["tool_choice"] is tool_choice
            assert marker.get() == "hook"

    monkeypatch.setattr(litellm, "callbacks", [Replace(), Observe()])
    request: Final = {
        "model": "anthropic/claude-sonnet-5",
        "messages": messages,
        "stream": False,
        "tool_choice": tool_choice,
        "temperature": 0.25,
        "opaque": object(),
    }

    prepared: Final = await legacy.prepare_messages_request(request)

    assert prepared["tools"] is tools
    assert prepared["temperature"] == 0.75
    assert prepared["opaque"] is request["opaque"]
    assert prepared["messages"] is messages
    assert prepared["tool_choice"] is tool_choice
    assert prepared["stream"] is False
    assert prepared["custom_llm_provider"] == "anthropic"
    assert request["temperature"] == 0.25
    assert "tools" not in request
    assert marker.get() == "hook"


@pytest.mark.asyncio
async def test_pre_request_hook_failure_keeps_exception_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = RuntimeError("pre-request rejected")

    class Reject(CustomLogger):
        async def async_pre_request_hook(self, model: str, messages: list[object], kwargs: dict[str, object]) -> None:
            raise failure

    monkeypatch.setattr(litellm, "callbacks", [Reject()])
    with pytest.raises(RuntimeError) as raised:
        await legacy.prepare_messages_request({"model": "anthropic/claude-sonnet-5", "messages": []})
    assert raised.value is failure


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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("run", "plan"))
async def test_agentic_response_hooks_keep_callback_inputs_and_selected_response(
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
            assert messages is caller_messages
            assert tools is caller_tools
            assert stream is False
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
    request: Final = {
        "model": "anthropic/claude-sonnet-5",
        "messages": caller_messages,
        "litellm_logging_obj": logger,
        "api_key": "caller-key",
        "api_base": "http://localhost:4000",
    }
    legacy.update_logging(logger, request, "claude-sonnet-5", {"tools": caller_tools, "max_tokens": 8}, {}, "anthropic")
    monkeypatch.setattr(litellm, "callbacks", [callback])

    selected: Final = await legacy.transform_messages_response(original, request)

    assert selected is replacement


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


CRATE: Final = Path(__file__).parents[3] / "litellm-rust/crates/callbacks-legacy-python"
CONTRACT_PATH: Final = CRATE / "python_contract.json"
CALLBACK_TABLE_PATH: Final = CRATE / "custom_logger_contract.json"


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {name: list(inspect.signature(getattr(legacy, name)).parameters) for name in contract}


def test_every_public_custom_logger_method_has_a_row_in_the_rust_callback_table() -> None:
    table: Final = TypeAdapter(dict[str, list[str]]).validate_json(CALLBACK_TABLE_PATH.read_text())
    public: Final = {
        name for name, _ in inspect.getmembers(CustomLogger, inspect.isroutine) if not name.startswith("_")
    }

    assert set(table) == public
    assert all(table[name] for name in public)

import asyncio
import datetime
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._internal_context import is_internal_call
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.pass_through.messages.streaming_iterator import GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from litellm.rust_bridge.host import callbacks_legacy_python as legacy
from litellm.rust_bridge.host.callbacks_legacy_python import failure_handler, setup
from litellm.rust_bridge.host.routes import messages as messages_host
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
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


class _LoggedHiddenParams(TypedDict):
    additional_headers: ReadOnly[Mapping[str, object]]


class _LoggedPayload(TypedDict):
    hidden_params: ReadOnly[_LoggedHiddenParams]


class _SuccessPayloadRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: Final[asyncio.Queue[_LoggedPayload]] = asyncio.Queue()

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.payloads.put_nowait(TypeAdapter(_LoggedPayload).validate_python(kwargs["standard_logging_object"]))


def _sse(event: str, data: Mapping[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


_MESSAGES_STREAM: Final = (
    _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_native",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
    ),
    _sse(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ),
    _sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
    ),
    _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
    _sse(
        "message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}
    ),
    _sse("message_stop", {"type": "message_stop"}),
)


@pytest.mark.asyncio
async def test_a_native_messages_stream_logs_the_rate_limit_from_the_provider_headers() -> None:
    recorder: Final = _SuccessPayloadRecorder()
    logger: Final = Logging(
        model="claude-sonnet-5-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.datetime(2026, 1, 1),
        litellm_call_id="native-stream-rate-limit",
        function_id="native-stream-rate-limit",
        dynamic_async_success_callbacks=[recorder],
    )
    logger.update_environment_variables(
        model="claude-sonnet-5-5",
        user="",
        optional_params={},
        litellm_params={"custom_llm_provider": "anthropic"},
        custom_llm_provider="anthropic",
    )
    head: Final = messages_host.stream_hidden_params(
        (
            ("request-id", "req_native"),
            ("anthropic-ratelimit-tokens-remaining", "11999000"),
            ("anthropic-ratelimit-requests-remaining", "12999"),
            ("content-type", "text/event-stream"),
        )
    )

    legacy.stream_opened(logger, head)
    handled: Final = AnthropicPassthroughLoggingHandler.handle_logging_anthropic_collected_chunks(
        litellm_logging_obj=logger,
        passthrough_success_handler_obj=GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
        url_route="/v1/messages",
        request_body={"model": "claude-sonnet-5-5", "stream": True},
        endpoint_type=EndpointType.ANTHROPIC,
        start_time=datetime.datetime(2026, 1, 1),
        all_chunks=list(_MESSAGES_STREAM),
        end_time=datetime.datetime(2026, 1, 1, 0, 0, 1),
    )
    await logger.dispatch_success_handlers(
        result=handled["result"],
        start_time=datetime.datetime(2026, 1, 1),
        end_time=datetime.datetime(2026, 1, 1, 0, 0, 1),
        cache_hit=False,
        prefer_async_handlers=True,
        **handled["kwargs"],
    )
    logged: Final = await asyncio.wait_for(recorder.payloads.get(), timeout=5)

    logged_headers: Final = logged["hidden_params"]["additional_headers"]
    assert logged_headers["x_ratelimit_remaining_tokens"] == 11999000
    assert logged_headers["x_ratelimit_remaining_requests"] == 12999
    assert not [name for name in logged_headers if name.startswith("llm_provider-llm_provider-")]


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


CONTRACT_PATH: Final = Path(__file__).parents[4] / "litellm-rust/crates/callbacks-legacy-python/python_contract.json"


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {name: list(inspect.signature(getattr(legacy, name)).parameters) for name in contract}

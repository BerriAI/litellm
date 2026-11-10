import asyncio
import datetime
import inspect
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

from litellm._internal_context import is_internal_call
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge.host import callbacks_legacy_python as legacy
from litellm.rust_bridge.host.callbacks_legacy_python import failure_handler, setup
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


def test_a_deferred_success_waits_for_the_proxy_release() -> None:
    logger: Final = _supplied_logger()
    released: Final[list[str]] = []
    start: Final = datetime.datetime.now()
    end: Final = start + datetime.timedelta(seconds=1)

    legacy.defer_success(logger, lambda: released.append("released"), "response", start, end)

    assert logger.deferred_success_result == ("response", start, end)
    assert released == []
    release: Final = logger.enqueue_deferred_logging
    assert release is not None
    release()
    assert released == ["released"]


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


class _Hooks(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.hooks: list[str] = []

    def log_success_event(self, kwargs: object, response_obj: object, start_time: object, end_time: object) -> None:
        self.hooks.append("sync_success")

    async def async_log_success_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.hooks.append("async_success")

    def log_failure_event(self, kwargs: object, response_obj: object, start_time: object, end_time: object) -> None:
        self.hooks.append("sync_failure")

    async def async_log_failure_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.hooks.append("async_failure")


def _logger_with_hooks(hooks: _Hooks) -> Logging:
    logger: Final = Logging(
        model="mistral/mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="aocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="hooks",
        function_id="hooks",
        dynamic_success_callbacks=[hooks],
        dynamic_async_success_callbacks=[hooks],
        dynamic_failure_callbacks=[hooks],
        dynamic_async_failure_callbacks=[hooks],
    )
    logger.update_environment_variables(litellm_params={"metadata": {}}, optional_params={})
    return logger


@pytest.mark.asyncio
async def test_failure_of_an_async_call_gives_a_logger_with_both_failure_hooks_only_the_async_one() -> None:
    hooks: Final = _Hooks()
    now: Final = datetime.datetime.now()
    kwargs: Final = {**_OCR_KWARGS, "litellm_logging_obj": _logger_with_hooks(hooks)}
    result: Final = setup("aocr", (), kwargs, now, asynchronous=True)

    assert failure_handler(result.logger, RuntimeError("upstream refused"), now, now, asynchronous=False) is None
    pending: Final = failure_handler(result.logger, RuntimeError("upstream refused"), now, now, asynchronous=True)
    assert pending is not None
    await pending

    assert hooks.hooks == ["async_failure"]


@pytest.mark.parametrize("earlier_async_call", [False, True])
def test_failure_of_a_sync_call_keeps_the_sync_failure_hook(earlier_async_call: bool) -> None:
    hooks: Final = _Hooks()
    now: Final = datetime.datetime.now()
    kwargs: Final = {**_OCR_KWARGS, "litellm_logging_obj": _logger_with_hooks(hooks)}
    if earlier_async_call:
        setup("aocr", (), kwargs, now, asynchronous=True)
    result: Final = setup("ocr", (), kwargs, now, asynchronous=False)

    assert failure_handler(result.logger, RuntimeError("upstream refused"), now, now, asynchronous=False) is None

    assert hooks.hooks == ["sync_failure"]


CONTRACT_PATH: Final = Path(__file__).parents[4] / "litellm-rust/crates/callbacks-legacy-python/python_contract.json"


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {name: list(inspect.signature(getattr(legacy, name)).parameters) for name in contract}


def test_messages_input_is_the_body_as_the_python_handler_builds_and_serializes_it() -> None:
    body: Final = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "system": "brief",
        "stream": True,
        "context_management": {"edits": []},
        "output_config": {"effort": "low"},
    }

    logged: Final = legacy.messages_input(
        body, {"output_config": {"effort": "low"}, "context_management": {"edits": []}}
    )

    assert logged == [
        {
            "role": "user",
            "content": '{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16, "model": "claude-sonnet-4-5", '
            '"stream": true, "system": "brief", "output_config": {"effort": "low"}, "context_management": {"edits": []}}',
        }
    ]


def _messages_logger() -> Logging:
    return Logging(
        model="claude-sonnet-4-5",
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="messages",
        function_id="messages",
    )


def test_native_provider_response_is_recorded_for_logging_replay() -> None:
    logger: Final = _messages_logger()

    legacy.record_response(logger, 200, (("request-id", "req_native"),), '{"id": "msg_native"}')

    recorded: Final[object] = logger.model_call_details["httpx_response"]
    assert isinstance(recorded, httpx.Response)
    assert recorded.status_code == 200
    assert recorded.content == b'{"id": "msg_native"}'
    assert logger.model_call_details["response_headers"]["request-id"] == "req_native"


def test_native_provider_exchange_window_is_recorded_like_the_timed_post() -> None:
    logger: Final = _messages_logger()
    logger.model_call_details["litellm_params"] = {"litellm_metadata": {"user_api_key": "hashed"}}

    legacy.record_api_window(logger, 1_700_000_000.0, 1_700_000_000.25)

    assert logger.model_call_details["llm_api_duration_ms"] == 250.0
    metadata: Final = logger.model_call_details["litellm_params"]["litellm_metadata"]
    assert metadata["llm_api_timing_windows"] == ((1_700_000_000.0, 1_700_000_000.25),)


def test_stream_head_is_stamped_in_place_with_the_call_hidden_params() -> None:
    logger: Final = _messages_logger()
    head: Final[dict[str, object]] = {"additional_headers": {"request-id": "req_native"}}
    started: Final = datetime.datetime.now()

    legacy.stamp_stream_head(
        head,
        logger,
        {"model": "anthropic/claude-sonnet-4-5", "litellm_call_id": "messages"},
        started,
        started + datetime.timedelta(milliseconds=5),
    )

    assert head["additional_headers"] == {"request-id": "req_native", "llm_provider-request-id": "req_native"}
    assert head["litellm_call_id"] == "messages"
    assert "_response_ms" in head


@pytest.mark.asyncio
async def test_a_native_stream_failure_logs_like_the_python_chunk_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler

    logged: Final[list[tuple[object, ...]]] = []
    scheduled: Final[list[Mapping[str, object]]] = []

    async def schedule(**kwargs: object) -> None:
        scheduled.append(MappingProxyType(dict(kwargs)))

    monkeypatch.setattr(verbose_proxy_logger, "error", lambda *args: logged.append(args))
    monkeypatch.setattr(PassThroughStreamingHandler, "schedule_stream_failure_logging", schedule)
    logger: Final = _supplied_logger()
    logger.model_call_details["cache_hit"] = False
    cut: Final = httpx.ReadError("cut")

    await legacy.stream_failure(logger, "anthropic", {}, [b"event: ping\n\n"], cut)

    assert logged == [("Error in chunk_processor: %s", cut)]
    assert "cache_hit" not in logger.model_call_details
    assert scheduled[0]["exception"] is cut


@pytest.mark.asyncio
async def test_a_proxy_stream_abandoned_mid_cancel_bills_after_the_disconnect_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler

    logger: Final = _messages_logger()
    logger.update_environment_variables(
        litellm_params={"metadata": {}, "proxy_server_request": {"url": "/v1/messages"}}, optional_params={}
    )

    async def billing() -> None:
        return None

    events: Final[list[str]] = []
    monkeypatch.setattr(PassThroughStreamingHandler, "_route_streaming_logging_to_handler", lambda **_: billing())
    monkeypatch.setattr(legacy, "enqueue_logging", lambda coroutine: (events.append("billed"), coroutine.close()))
    started: Final = asyncio.Event()
    now: Final = datetime.datetime.now()

    async def relay() -> None:
        try:
            started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            legacy.stream_success(logger, "/v1/messages", "anthropic", {}, [b"event: ping\n\n"], now, now, None, True)
            await asyncio.sleep(0)
            events.append("disconnect recorded")
            raise

    task: Final = asyncio.create_task(relay())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert events == ["disconnect recorded", "billed"]

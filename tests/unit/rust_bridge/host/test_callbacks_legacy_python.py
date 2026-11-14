import asyncio
import datetime
import inspect
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # a duck-typed logger double stands in for Logging

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


def test_recorded_messages_stream_keeps_the_provider_header_names() -> None:
    logger: Final = _supplied_logger()
    sent: Final[tuple[str, Mapping[str, str], bytes]] = ("http://upstream/v1/messages", {}, b'{"stream":true}')
    legacy.record_response(logger, 200, [("request-id", "req_native")], "", True, sent)
    legacy.stream_opened(logger, {"additional_headers": {"llm_provider-request-id": "req_native"}})

    assert logger.stream is True
    assert logger.model_call_details["response_headers"] == {"request-id": "req_native"}


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
    legacy.finalize(response, prepared.logger, prepared.kwargs, now, now, False)
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


CONTRACT_PATH: Final = Path(__file__).parents[4] / "litellm/rust/crates/callbacks-legacy-python/python_contract.json"


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {name: list(inspect.signature(getattr(legacy, name)).parameters) for name in contract}


def test_messages_input_order_is_the_key_order_the_python_handler_builds_its_body_in() -> None:
    body: Final = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "system": "brief",
        "stream": True,
        "context_management": {"edits": []},
        "output_config": {"effort": "low"},
    }

    order: Final = legacy.messages_input_order(
        body, {"output_config": {"effort": "low"}, "context_management": {"edits": []}, "absent": 1}
    )

    assert order == (
        "messages",
        "max_tokens",
        "model",
        "stream",
        "system",
        "output_config",
        "context_management",
    )


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

    legacy.record_response(logger, 200, (("request-id", "req_native"),), '{"id": "msg_native"}', False, None)

    recorded: Final[object] = logger.model_call_details["httpx_response"]
    assert isinstance(recorded, httpx.Response)
    assert recorded.status_code == 200
    assert recorded.content == b'{"id": "msg_native"}'
    assert logger.model_call_details["response_headers"]["request-id"] == "req_native"


def test_a_non_ascii_provider_header_is_recorded_as_httpx_decodes_it() -> None:
    logger: Final = _messages_logger()

    legacy.record_response(logger, 200, (("x-fz-unicode", "caf\u00e9"),), '{"id": "msg_native"}', False, None)

    assert logger.model_call_details["response_headers"]["x-fz-unicode"] == "caf\u00e9"


_HTTPX_RESPONSE: Final = TypeAdapter(httpx.Response, config={"arbitrary_types_allowed": True})
_SENT_HEADERS: Final = MappingProxyType({"x-api-key": "sk-ant", "anthropic-beta": "message-threads-2026-08-12"})


def test_native_provider_response_carries_the_request_that_was_sent() -> None:
    logger: Final = _messages_logger()
    wire: Final = '{"model":"claude-sonnet-4-5","system":"caf\u00e9","temperature":1e-7}'.encode()

    legacy.record_response(
        logger, 200, (), '{"id": "msg_native"}', False, ("https://api.anthropic.com/v1/messages", _SENT_HEADERS, wire)
    )

    recorded: Final = _HTTPX_RESPONSE.validate_python(logger.model_call_details["httpx_response"])
    assert recorded.request.method == "POST"
    assert str(recorded.request.url) == "https://api.anthropic.com/v1/messages"
    assert recorded.request.headers["anthropic-beta"] == "message-threads-2026-08-12"
    assert recorded.request.content == wire
    assert recorded.content == b'{"id": "msg_native"}'


def test_native_stream_response_is_recorded_unread_like_the_python_stream() -> None:
    logger: Final = _messages_logger()

    legacy.record_response(
        logger,
        200,
        (("request-id", "req_native"),),
        "",
        True,
        ("https://api.anthropic.com/v1/messages", _SENT_HEADERS, b'{"stream":true}'),
    )

    recorded: Final = _HTTPX_RESPONSE.validate_python(logger.model_call_details["httpx_response"])
    assert recorded.request.headers["anthropic-beta"] == "message-threads-2026-08-12"
    with pytest.raises(httpx.ResponseNotRead):
        _ = recorded.content


def test_native_provider_exchange_window_is_recorded_like_the_timed_post() -> None:
    logger: Final = _messages_logger()
    logger.model_call_details["litellm_params"] = {"litellm_metadata": {"user_api_key": "hashed"}}

    legacy.record_api_window(logger, 1_700_000_000.0, 1_700_000_000.25)

    assert logger.model_call_details["llm_api_duration_ms"] == 250.0
    metadata: Final = logger.model_call_details["litellm_params"]["litellm_metadata"]
    assert metadata["llm_api_timing_windows"] == ((1_700_000_000.0, 1_700_000_000.25),)


@pytest.mark.asyncio
async def test_a_retried_request_rewrites_the_logged_body_in_place_like_the_python_retry_loop() -> None:
    logger: Final = _messages_logger()
    metadata: Final = {"user_id": "u-1"}
    replayed: Final = [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "s"}]},
        {"role": "user", "content": "again"},
    ]
    logged: Final[dict[str, object]] = {
        "model": "claude-sonnet-4-5",
        "messages": replayed,
        "metadata": metadata,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    logger.model_call_details.update(logged)
    retried: Final = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "again"}],
        "metadata": {"user_id": "u-1"},
    }

    await legacy.retried_request(logger, logged, retried, ({"kind": "thinking_signature"},))

    assert logged == retried
    assert logged["metadata"] is metadata
    assert logger.model_call_details["messages"] == [{"role": "user", "content": "again"}]
    assert logger.model_call_details["thinking"] == {"type": "enabled", "budget_tokens": 1024}


class _BaselineLogger:
    def __init__(self) -> None:
        self.baseline_cache_context: object = object()
        self.model_call_details: dict[str, object] = {}
        self.invalidated: list[str] = []

    async def invalidate_baseline_cache_estimate(self, reason: str) -> None:
        self.invalidated.append(reason)


@pytest.mark.asyncio
async def test_a_retried_request_invalidates_the_baseline_cache_estimate() -> None:
    logger: Final = _BaselineLogger()

    await legacy.retried_request(cast(Logging, logger), {"messages": []}, {"messages": []}, ())  # cast-ok: duck-typed

    assert logger.invalidated == ["retried_request"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reasons", "warned"),
    [
        pytest.param(({"kind": "thinking_signature"},), [], id="thinking_signature"),
        pytest.param(({"kind": "strict_tools", "dropped": 2},), [2], id="strict_tools"),
        pytest.param(({"kind": "thinking_signature"}, {"kind": "strict_tools", "dropped": 3}), [3], id="both_reasons"),
    ],
)
async def test_a_retry_that_dropped_strict_tools_warns_with_the_dropped_count(
    monkeypatch: pytest.MonkeyPatch, reasons: tuple[Mapping[str, object], ...], warned: list[int]
) -> None:
    from litellm.llms.anthropic import common_utils

    warnings: Final[list[int]] = []
    monkeypatch.setattr(common_utils, "warn_claude_strict_tools_dropped", warnings.append)

    await legacy.retried_request(_messages_logger(), {"messages": []}, {"messages": []}, reasons)

    assert warnings == warned


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
        False,
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


@pytest.mark.parametrize(
    ("requested", "keeps_requested", "expected"),
    [
        pytest.param("false", True, "false", id="messages-keeps-a-truthy-string"),
        pytest.param(1, True, 1, id="messages-keeps-a-truthy-int"),
        pytest.param(False, True, True, id="messages-hook-streamed"),
        pytest.param("false", False, True, id="other-routes-stream-true"),
    ],
)
def test_a_stream_logs_the_flag_python_logs(requested: object, keeps_requested: bool, expected: object) -> None:
    logger: Final = _supplied_logger()
    logger.model_call_details["stream"] = requested

    legacy.stream_opened(logger, {}, keeps_requested)

    assert logger.model_call_details["stream"] == expected
    assert logger.stream == expected

import datetime
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm._internal_context import is_internal_call
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge.host import callbacks_legacy_python as legacy
from litellm.litellm_core_utils import litellm_logging
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


def _legacy_success_callback(*_: object) -> None:
    pass


class _InlineExecutor:
    def submit(self, fn: Callable[..., object], /, *args: object, **kwargs: object) -> None:
        fn(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("async_step", "expected"),
    [("enqueued", ["async_success"]), ("deferred", []), ("skipped", ["sync_success"])],
)
async def test_success_of_an_async_call_keeps_the_sync_hook_only_when_no_async_success_is_scheduled(
    monkeypatch: pytest.MonkeyPatch, async_step: str, expected: list[str]
) -> None:
    monkeypatch.setattr(litellm_logging, "executor", _InlineExecutor())
    monkeypatch.setattr(litellm, "success_callback", [_legacy_success_callback])
    hooks: Final = _Hooks()
    now: Final = datetime.datetime.now()
    kwargs: Final = {**_OCR_KWARGS, "litellm_logging_obj": _logger_with_hooks(hooks)}
    logger: Final = setup("aocr", (), kwargs, now, asynchronous=True).logger
    response: Final = ModelResponse()

    if async_step == "enqueued":
        await legacy.async_success_handler(logger, response, now, now)
    elif async_step == "deferred":
        legacy.defer_success(logger, object())
    legacy.sync_success_for_async_call(logger, response, now, now)

    assert hooks.hooks == expected


class _Pending:
    def __init__(self) -> None:
        self.released: list[bool] = []

    def release(self, success: bool) -> None:
        self.released.append(success)


@pytest.mark.parametrize(("exception_raised", "expected"), [(True, ["sync_success"]), (False, [])])
def test_success_the_proxy_drops_for_a_blocked_ocr_response_gives_the_logger_its_sync_hook(
    monkeypatch: pytest.MonkeyPatch, exception_raised: bool, expected: list[str]
) -> None:
    from litellm.litellm_core_utils import thread_pool_executor
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    monkeypatch.setattr(litellm_logging, "executor", _InlineExecutor())
    monkeypatch.setattr(thread_pool_executor, "executor", _InlineExecutor())
    monkeypatch.setattr(litellm, "success_callback", [_legacy_success_callback])
    hooks: Final = _Hooks()
    now: Final = datetime.datetime.now()
    kwargs: Final = {**_OCR_KWARGS, "litellm_logging_obj": _logger_with_hooks(hooks)}
    logger: Final = setup("aocr", (), kwargs, now, asynchronous=True).logger
    pending: Final = _Pending()
    legacy.defer_success(logger, pending)
    legacy.sync_success_for_async_call(logger, ModelResponse(), now, now)

    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logging_obj=logger, exception_raised=exception_raised)

    assert pending.released == [not exception_raised]
    assert hooks.hooks == expected


def test_success_of_a_later_call_on_the_same_logger_does_not_reuse_the_earlier_async_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm_logging, "executor", _InlineExecutor())
    monkeypatch.setattr(litellm, "success_callback", [_legacy_success_callback])
    hooks: Final = _Hooks()
    now: Final = datetime.datetime.now()
    kwargs: Final = {**_OCR_KWARGS, "litellm_logging_obj": _logger_with_hooks(hooks)}
    legacy.async_success_handler(
        setup("aocr", (), kwargs, now, asynchronous=True).logger, ModelResponse(), now, now
    ).close()
    logger: Final = setup("aocr", (), kwargs, now, asynchronous=True).logger

    legacy.sync_success_for_async_call(logger, ModelResponse(), now, now)

    assert hooks.hooks == ["sync_success"]


CONTRACT_PATH: Final = Path(__file__).parents[4] / "litellm-rust/crates/callbacks-legacy-python/python_contract.json"


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {name: list(inspect.signature(getattr(legacy, name)).parameters) for name in contract}

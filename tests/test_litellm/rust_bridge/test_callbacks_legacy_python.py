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
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge import callbacks_legacy_python as legacy
from litellm.rust_bridge.callbacks_legacy_python import check_limits, failure_handler, setup

_OCR_KWARGS: Final = MappingProxyType(
    {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
    }
)


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize(
    "cap, request_retry_count, refused",
    [(5, 5, True), (5, 4, False), (0, 0, False), (0, 1, True)],
    ids=[
        "cap-above-four-reached",
        "cap-above-four-not-reached",
        "first-attempt-passes-cap-of-zero",
        "cap-of-zero-refuses-first-retry",
    ],
)
def test_check_limits_reads_request_retry_count(
    monkeypatch: pytest.MonkeyPatch, metadata_key: str, cap: int, request_retry_count: int, refused: bool
) -> None:
    monkeypatch.setattr(litellm, "num_retries_per_request", cap)
    monkeypatch.setattr(litellm, "max_budget", None)
    kwargs: Final = {
        "model": "mistral/mistral-ocr-latest",
        metadata_key: {"request_retry_count": request_retry_count},
    }
    if refused:
        with pytest.raises(RuntimeError, match="Max retries per request hit!"):
            check_limits(kwargs)
    else:
        check_limits(kwargs)


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


def test_setup_reuses_a_supplied_logger() -> None:
    supplied: Final = _supplied_logger()
    result: Final = setup(
        "aocr", (), {**_OCR_KWARGS, "litellm_logging_obj": supplied}, datetime.datetime.now(), asynchronous=True
    )
    assert result.logger is supplied


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


CONTRACT_PATH: Final = (
    Path(__file__).parents[3] / "litellm-rust/crates/callbacks-legacy-python/python_contract.json"
)


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {name: list(inspect.signature(getattr(legacy, name)).parameters) for name in contract}

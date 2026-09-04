from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Final, cast

import pytest

from litellm.litellm_core_utils.logging_worker import LoggingWorker

from .billing import BillingObserver
from .models import BillingObservation, BillingUsageMetric

CALL_ID: Final = "billing-test-call"


def _usage(response: object) -> tuple[BillingUsageMetric, ...]:
    if not isinstance(response, dict):
        raise ValueError("response must be a dictionary")
    values: Final = cast(  # cast-ok: the fixture response is created with string keys below
        Mapping[str, object], response
    )
    pages: Final = values.get("pages_processed")
    if not isinstance(pages, int):
        raise ValueError("pages_processed must be an integer")
    return (BillingUsageMetric(name="pages_processed", value=pages),)


def _kwargs(
    *,
    call_id: str = CALL_ID,
    response_cost: float | None = 0.25,
    failure: object = None,
    volatile_id: str = "generated-1",
) -> dict[str, object]:
    return {
        "litellm_call_id": call_id,
        "model": "provider/ocr-model",
        "custom_llm_provider": "provider",
        "start_time": 100.0,
        "standard_logging_object": {
            "id": volatile_id,
            "startTime": 100.0,
            "endTime": 101.0,
            "call_type": "ocr",
            "response_cost": response_cost,
            "response_cost_failure_debug_info": failure,
            "cost_breakdown": {
                "input_cost": response_cost,
                "output_cost": 0,
                "total_cost": response_cost,
            },
        },
    }


def _capture(
    *,
    response_cost: float | None = 0.25,
    failure: object = None,
    volatile_id: str = "generated-1",
) -> BillingObservation:
    observer: Final = BillingObserver(CALL_ID, _usage)
    observer.log_success_event(
        _kwargs(response_cost=response_cost, failure=failure, volatile_id=volatile_id),
        {"pages_processed": 5, "request_id": volatile_id},
        object(),
        object(),
    )
    return observer.observation(timeout=0)


def test_billing_observation_excludes_volatile_callback_fields() -> None:
    assert _capture(volatile_id="generated-1") == _capture(volatile_id="generated-2")


@pytest.mark.parametrize(
    ("response_cost", "failure", "status", "diagnostic"),
    (
        (0.0, None, "calculated", None),
        (None, None, "unavailable", None),
        (None, {"error_str": "missing price"}, "failed", "response_cost_calculation_failed"),
    ),
)
def test_billing_observation_preserves_zero_none_and_failure(
    response_cost: float | None,
    failure: object,
    status: str,
    diagnostic: str | None,
) -> None:
    observation: Final = _capture(response_cost=response_cost, failure=failure)

    assert observation.response_cost == response_cost
    assert observation.cost_calculation_status == status
    assert observation.cost_failure_diagnostic == diagnostic


def test_billing_observer_rejects_missing_callback() -> None:
    observer: Final = BillingObserver(CALL_ID, _usage)

    with pytest.raises(AssertionError, match="did not complete"):
        observer.observation(timeout=0)


def test_billing_observer_rejects_duplicate_callback() -> None:
    observer: Final = BillingObserver(CALL_ID, _usage)
    kwargs: Final = _kwargs()
    observer.log_success_event(kwargs, {"pages_processed": 2}, object(), object())
    observer.log_success_event(kwargs, {"pages_processed": 2}, object(), object())

    with pytest.raises(AssertionError, match=r"exactly one.*got 2"):
        observer.observation(timeout=0)


def test_billing_observer_rejects_mismatched_call_id() -> None:
    observer: Final = BillingObserver(CALL_ID, _usage)
    observer.log_success_event(_kwargs(call_id="wrong-call"), {"pages_processed": 2}, object(), object())

    with pytest.raises(AssertionError, match="did not match"):
        observer.observation(timeout=0)


def test_billing_observer_surfaces_normalization_errors() -> None:
    observer: Final = BillingObserver(CALL_ID, _usage)
    observer.log_success_event(_kwargs(), {}, object(), object())

    with pytest.raises(AssertionError, match="pages_processed"):
        observer.observation(timeout=0)


def test_sync_billing_observation_waits_for_background_callback() -> None:
    observer: Final = BillingObserver(CALL_ID, _usage)
    with ThreadPoolExecutor(max_workers=1) as executor:
        completed: Final = executor.submit(
            observer.log_success_event,
            _kwargs(),
            {"pages_processed": 3},
            object(),
            object(),
        )
        observation: Final = observer.observation(timeout=1)
        completed.result(timeout=1)

    assert observation.callback_count == 1


def test_async_billing_observation_flushes_logging_worker() -> None:
    async def scenario() -> BillingObservation:
        observer: Final = BillingObserver(CALL_ID, _usage)
        worker: Final = LoggingWorker()
        worker.ensure_initialized_and_enqueue(  # pyright: ignore[reportUnknownMemberType]  # worker lacks typed coroutine generics
            observer.async_log_success_event(
                _kwargs(),
                {"pages_processed": 4},
                object(),
                object(),
            )
        )
        await worker.flush()
        observation: Final = observer.observation(timeout=0)
        await worker.stop()
        return observation

    observation: Final = asyncio.run(scenario())

    assert observation.callback_count == 1
    assert observation.billable_usage == (BillingUsageMetric(name="pages_processed", value=4),)

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Event, Lock
from typing import Final, Literal, Protocol, cast

from litellm.integrations.custom_logger import CustomLogger

from .models import (
    BillingCostBreakdown,
    BillingObservation,
    BillingUsageMetric,
)


class UsageNormalizer(Protocol):
    def __call__(self, response: object, /) -> tuple[BillingUsageMetric, ...]: ...


@dataclass(frozen=True, slots=True)
class _BillingCallbackValue:
    call_type: str
    pricing_model: str
    custom_llm_provider: str | None
    billable_usage: tuple[BillingUsageMetric, ...]
    response_cost: float | None
    cost_breakdown: BillingCostBreakdown | None
    cost_calculation_status: Literal["calculated", "unavailable", "failed"]
    cost_failure_diagnostic: Literal["response_cost_calculation_failed"] | None


class BillingObserver(CustomLogger):
    def __init__(self, expected_call_id: str, usage_normalizer: UsageNormalizer) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger has untyped legacy kwargs
        self._expected_call_id: Final = expected_call_id
        self._usage_normalizer: Final = usage_normalizer
        self._completed: Final = Event()
        self._lock: Final = Lock()
        self._callback_values: tuple[_BillingCallbackValue, ...] = ()
        self._errors: tuple[str, ...] = ()
        self._callback_count: int = 0

    def log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._capture(kwargs, response_obj)

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._capture(kwargs, response_obj)

    def _capture(self, kwargs: Mapping[str, object], response_obj: object) -> None:
        try:
            value: Final = self._normalize(kwargs, response_obj)
            with self._lock:
                self._callback_count += 1
                self._callback_values = (*self._callback_values, value)
        except Exception as error:
            with self._lock:
                self._callback_count += 1
                self._errors = (*self._errors, f"{type(error).__name__}: {error}")
        finally:
            self._completed.set()

    def _normalize(self, kwargs: Mapping[str, object], response_obj: object) -> _BillingCallbackValue:
        call_id: Final = _optional_string(kwargs.get("litellm_call_id"))
        if call_id != self._expected_call_id:
            raise ValueError(f"callback call ID {call_id!r} did not match {self._expected_call_id!r}")
        standard_logging_object: Final = kwargs.get("standard_logging_object")
        if not isinstance(standard_logging_object, Mapping):
            raise ValueError("callback did not contain a standard_logging_object")
        payload: Final = cast(  # cast-ok: only literal string keys are read from LiteLLM's callback mapping
            Mapping[str, object], standard_logging_object
        )
        call_type: Final = _required_string(payload, "call_type")
        pricing_model: Final = _required_string(kwargs, "model")
        custom_llm_provider: Final = _optional_string(kwargs.get("custom_llm_provider"))
        response_cost: Final = _optional_float(payload.get("response_cost"), "response_cost")
        raw_failure: Final = payload.get("response_cost_failure_debug_info")
        cost_calculation_status: Final = (
            "failed" if raw_failure is not None else "unavailable" if response_cost is None else "calculated"
        )
        return _BillingCallbackValue(
            call_type=call_type,
            pricing_model=pricing_model,
            custom_llm_provider=custom_llm_provider,
            billable_usage=self._usage_normalizer(response_obj),
            response_cost=response_cost,
            cost_breakdown=_cost_breakdown(payload.get("cost_breakdown")),
            cost_calculation_status=cost_calculation_status,
            cost_failure_diagnostic=("response_cost_calculation_failed" if raw_failure is not None else None),
        )

    def observation(self, timeout: float) -> BillingObservation:
        if not self._completed.wait(timeout):
            raise AssertionError(
                f"billing callback for call ID {self._expected_call_id!r} did not complete within {timeout}s"
            )
        with self._lock:
            callback_count: Final = self._callback_count
            values: Final = self._callback_values
            errors: Final = self._errors
        if errors:
            raise AssertionError(f"billing callback failed for call ID {self._expected_call_id!r}: {'; '.join(errors)}")
        if callback_count != 1 or len(values) != 1:
            raise AssertionError(
                f"expected exactly one billing callback for call ID {self._expected_call_id!r}, got {callback_count}"
            )
        value: Final = values[0]
        return BillingObservation(
            callback_count=callback_count,
            call_type=value.call_type,
            pricing_model=value.pricing_model,
            custom_llm_provider=value.custom_llm_provider,
            billable_usage=value.billable_usage,
            response_cost=value.response_cost,
            cost_breakdown=value.cost_breakdown,
            cost_calculation_status=value.cost_calculation_status,
            cost_failure_diagnostic=value.cost_failure_diagnostic,
        )

    def assert_no_success_callback(self) -> None:
        with self._lock:
            callback_count: Final = self._callback_count
            errors: Final = self._errors
        if callback_count or errors:
            raise AssertionError(
                f"expected no success billing callback for call ID {self._expected_call_id!r}, got {callback_count}"
            )


def _required_string(values: Mapping[str, object], key: str) -> str:
    value: Final = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected a string or None")
    return value


def _optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric or None")
    return float(value)


def _cost_breakdown(value: object) -> BillingCostBreakdown | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("cost_breakdown must be a mapping or None")
    breakdown: Final = cast(  # cast-ok: only literal string keys are read after the mapping check
        Mapping[str, object], value
    )
    return BillingCostBreakdown(
        input_cost=_optional_float(breakdown.get("input_cost"), "input_cost"),
        output_cost=_optional_float(breakdown.get("output_cost"), "output_cost"),
        total_cost=_optional_float(breakdown.get("total_cost"), "total_cost"),
        cache_read_cost=_optional_float(breakdown.get("cache_read_cost"), "cache_read_cost"),
        cache_creation_cost=_optional_float(breakdown.get("cache_creation_cost"), "cache_creation_cost"),
        reasoning_cost=_optional_float(breakdown.get("reasoning_cost"), "reasoning_cost"),
        tool_usage_cost=_optional_float(breakdown.get("tool_usage_cost"), "tool_usage_cost"),
        original_cost=_optional_float(breakdown.get("original_cost"), "original_cost"),
        discount_percent=_optional_float(breakdown.get("discount_percent"), "discount_percent"),
        discount_amount=_optional_float(breakdown.get("discount_amount"), "discount_amount"),
        margin_percent=_optional_float(breakdown.get("margin_percent"), "margin_percent"),
        margin_fixed_amount=_optional_float(breakdown.get("margin_fixed_amount"), "margin_fixed_amount"),
        margin_total_amount=_optional_float(breakdown.get("margin_total_amount"), "margin_total_amount"),
    )

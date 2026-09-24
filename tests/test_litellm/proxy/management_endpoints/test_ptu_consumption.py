"""Tests for attaching PTU-hours to a daily activity response."""

from typing import Final

import pytest

from litellm.litellm_core_utils.azure_ptu_capacity import PTUCapacity
from litellm.proxy.management_endpoints.ptu_consumption import attach_ptu_hours
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    BreakdownMetrics,
    DailySpendData,
    DailySpendMetadata,
    KeyMetricWithMetadata,
    MetricWithMetadata,
    SpendAnalyticsPaginatedResponse,
    SpendMetrics,
)

_ROW: Final = PTUCapacity(input_tpm_per_ptu=1_000, output_to_input_ratio=4.0)
_CACHED_ROW: Final = PTUCapacity(input_tpm_per_ptu=1_000, output_to_input_ratio=4.0, cached_input_ratio=0.1)
_CAPACITY: Final = {"gpt-4.1-ptu": _ROW, "gpt-6-ptu": _CACHED_ROW}


def _metrics(prompt: int, completion: int, cached: int = 0) -> SpendMetrics:
    return SpendMetrics(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cache_read_input_tokens=cached,
        total_tokens=prompt + completion,
        api_requests=1,
        successful_requests=1,
    )


def _bucket(metrics: SpendMetrics, keys: dict[str, SpendMetrics] | None = None) -> MetricWithMetadata:
    return MetricWithMetadata(
        metrics=metrics,
        metadata={},
        api_key_breakdown={key: KeyMetricWithMetadata(metrics=m, metadata={}) for key, m in (keys or {}).items()},
    )


def _day(date: str, model_groups: dict[str, MetricWithMetadata]) -> DailySpendData:
    total: Final = _metrics(
        sum(b.metrics.prompt_tokens for b in model_groups.values()),
        sum(b.metrics.completion_tokens for b in model_groups.values()),
    )
    return DailySpendData(date=date, metrics=total, breakdown=BreakdownMetrics(model_groups=model_groups))


def _response(*days: DailySpendData) -> SpendAnalyticsPaginatedResponse:
    return SpendAnalyticsPaginatedResponse(results=list(days), metadata=DailySpendMetadata())


def test_a_ptu_model_groups_tokens_become_ptu_hours_on_the_group_the_day_and_the_total():
    """60,000 normalized tokens on a 1,000 input-TPM row is one PTU-hour."""
    day: Final = _day("2026-09-23", {"gpt-4.1-ptu": _bucket(_metrics(prompt=40_000, completion=5_000))})

    attached: Final = attach_ptu_hours(_response(day), _CAPACITY.get)

    group: Final = attached.results[0].breakdown.model_groups["gpt-4.1-ptu"]
    assert group.metrics.ptu_hours == pytest.approx(1.0)
    assert attached.results[0].metrics.ptu_hours == pytest.approx(1.0)
    assert attached.metadata.total_ptu_hours == pytest.approx(1.0)


def test_a_model_group_without_a_sizing_row_keeps_zero_ptu_hours_and_the_rest_of_the_day_intact():
    day: Final = _day(
        "2026-09-23",
        {
            "gpt-4.1-ptu": _bucket(_metrics(prompt=30_000, completion=0)),
            "gpt-4o-mini": _bucket(_metrics(prompt=1_000_000, completion=1_000_000)),
        },
    )

    attached: Final = attach_ptu_hours(_response(day), _CAPACITY.get)

    groups: Final = attached.results[0].breakdown.model_groups
    assert groups["gpt-4o-mini"].metrics.ptu_hours == 0.0
    assert groups["gpt-4o-mini"].metrics.prompt_tokens == 1_000_000
    assert groups["gpt-4.1-ptu"].metrics.ptu_hours == pytest.approx(0.5)
    assert attached.results[0].metrics.ptu_hours == pytest.approx(0.5)
    assert attached.results[0].metrics.total_tokens == day.metrics.total_tokens


def test_cached_input_is_charged_at_the_rows_cached_ratio():
    day: Final = _day("2026-09-23", {"gpt-6-ptu": _bucket(_metrics(prompt=60_000, completion=0, cached=60_000))})

    attached: Final = attach_ptu_hours(_response(day), _CAPACITY.get)

    assert attached.results[0].breakdown.model_groups["gpt-6-ptu"].metrics.ptu_hours == pytest.approx(0.1)


def test_each_api_key_under_a_ptu_group_gets_its_own_ptu_hours():
    day: Final = _day(
        "2026-09-23",
        {
            "gpt-4.1-ptu": _bucket(
                _metrics(prompt=60_000, completion=0),
                keys={"key-a": _metrics(prompt=45_000, completion=0), "key-b": _metrics(prompt=15_000, completion=0)},
            )
        },
    )

    attached: Final = attach_ptu_hours(_response(day), _CAPACITY.get)

    breakdown: Final = attached.results[0].breakdown.model_groups["gpt-4.1-ptu"].api_key_breakdown
    assert breakdown["key-a"].metrics.ptu_hours == pytest.approx(0.75)
    assert breakdown["key-b"].metrics.ptu_hours == pytest.approx(0.25)


def test_the_total_sums_every_day_on_the_page():
    first: Final = _day("2026-09-22", {"gpt-4.1-ptu": _bucket(_metrics(prompt=60_000, completion=0))})
    second: Final = _day("2026-09-23", {"gpt-4.1-ptu": _bucket(_metrics(prompt=0, completion=15_000))})

    attached: Final = attach_ptu_hours(_response(first, second), _CAPACITY.get)

    assert [day.metrics.ptu_hours for day in attached.results] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert attached.metadata.total_ptu_hours == pytest.approx(2.0)


def test_a_page_with_no_ptu_group_is_returned_unchanged():
    day: Final = _day("2026-09-23", {"gpt-4o-mini": _bucket(_metrics(prompt=100, completion=100))})
    response: Final = _response(day)

    attached: Final = attach_ptu_hours(response, _CAPACITY.get)

    assert attached.results[0] is day
    assert attached.metadata.total_ptu_hours == 0.0
    assert response.metadata.total_ptu_hours == 0.0

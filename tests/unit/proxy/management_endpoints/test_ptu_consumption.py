"""Tests for attaching PTU-hours to a daily activity response."""

from typing import Final

import pytest

from litellm import Router
from litellm.llms.azure.ptu_capacity import AZURE_PTU_CAPACITY, PTUCapacity
from litellm.proxy.management_endpoints.ptu_consumption import attach_ptu_hours, with_ptu_consumption
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
_ONE_PTU_HOUR_OF_INPUT: Final = AZURE_PTU_CAPACITY["gpt-4.1"].normalized_tokens_per_ptu_hour


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


def _shared_ptu_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gpt-4.1-ptu",
                "litellm_params": {"model": "azure/gpt-4.1", "api_key": "sk-ptu", "api_base": "https://ptu.example"},
                "model_info": {
                    "id": "shared-ptu",
                    "base_model": "azure/gpt-4.1",
                    "ptu_count": 50,
                    "cost_per_ptu_per_hour": 1.0,
                    "ptu_effective_from": "2026-01-01T00:00:00Z",
                    "ptu_shares": {"team-a": 30, "team-b": 20},
                },
            }
        ],
        model_group_alias={"ptu-alias": {"model": "gpt-4.1-ptu", "hidden": True}},
    )


def test_rows_keyed_by_an_alias_a_deployment_id_or_a_provider_model_are_sized_like_the_group(monkeypatch):
    """The ceiling charges a request however it names the shared deployment, so the usage row
    that request lands in, keyed by the name it used, reports the same PTU-hours as the group."""
    monkeypatch.setenv("LITELLM_ENABLE_PTU_COST_ATTRIBUTION", "True")
    names: Final = ("gpt-4.1-ptu", "ptu-alias", "shared-ptu", "azure/gpt-4.1")
    one_hour_each: Final = {name: _bucket(_metrics(prompt=_ONE_PTU_HOUR_OF_INPUT, completion=0)) for name in names}

    attached: Final = with_ptu_consumption(_response(_day("2026-09-23", one_hour_each)), _shared_ptu_router(), "team-a")

    groups: Final = attached.results[0].breakdown.model_groups
    assert [groups[name].metrics.ptu_hours for name in names] == [pytest.approx(1.0)] * len(names)
    assert attached.results[0].metrics.ptu_hours == pytest.approx(float(len(names)))
    assert attached.metadata.total_ptu_hours == pytest.approx(float(len(names)))


def test_rows_keyed_by_a_name_a_wildcard_route_serves_are_sized_by_its_shared_deployment(monkeypatch):
    """A request the router served through a wildcard lands in a row keyed by the name it used,
    which reports the PTU-hours of the shared deployment behind the pattern."""
    monkeypatch.setenv("LITELLM_ENABLE_PTU_COST_ATTRIBUTION", "True")
    router: Final = Router(
        model_list=[
            {
                "model_name": "ptu-*",
                "litellm_params": {"model": "azure/gpt-4.1", "api_key": "sk-ptu", "api_base": "https://ptu.example"},
                "model_info": {
                    "id": "shared-ptu",
                    "base_model": "azure/gpt-4.1",
                    "ptu_count": 50,
                    "cost_per_ptu_per_hour": 1.0,
                    "ptu_effective_from": "2026-01-01T00:00:00Z",
                    "ptu_shares": {"team-a": 30, "team-b": 20},
                },
            }
        ]
    )
    one_hour: Final = _bucket(_metrics(prompt=_ONE_PTU_HOUR_OF_INPUT, completion=0))

    attached: Final = with_ptu_consumption(_response(_day("2026-09-23", {"ptu-chat": one_hour})), router, "team-a")

    assert attached.results[0].breakdown.model_groups["ptu-chat"].metrics.ptu_hours == pytest.approx(1.0)


def _mixed_ptu_router() -> Router:
    """One group split between team-a on a gpt-4.1 PTU deployment and team-b on a gpt-5.5 one,
    beside an open pay-as-you-go deployment of gpt-4.1 in its own group."""
    return Router(
        model_list=[
            {
                "model_name": "ptu",
                "litellm_params": {"model": "azure/gpt-4.1", "api_key": "sk-ptu", "api_base": "https://ptu.example"},
                "model_info": {
                    "id": "ptu-41",
                    "base_model": "azure/gpt-4.1",
                    "ptu_count": 50,
                    "cost_per_ptu_per_hour": 1.0,
                    "ptu_effective_from": "2026-01-01T00:00:00Z",
                    "ptu_shares": {"team-a": 50},
                },
            },
            {
                "model_name": "ptu",
                "litellm_params": {"model": "azure/gpt-5.5", "api_key": "sk-ptu", "api_base": "https://ptu.example"},
                "model_info": {
                    "id": "ptu-55",
                    "base_model": "azure/gpt-5.5",
                    "ptu_count": 50,
                    "cost_per_ptu_per_hour": 1.0,
                    "ptu_effective_from": "2026-01-01T00:00:00Z",
                    "ptu_shares": {"team-b": 50},
                },
            },
            {
                "model_name": "gpt-4.1",
                "litellm_params": {"model": "azure/gpt-4.1", "api_key": "sk-payg", "api_base": "https://payg.example"},
                "model_info": {"id": "payg"},
            },
        ]
    )


def _reserved_ptu_router() -> Router:
    """A gpt-4.1 PTU deployment reserved for team-x alone the single-team way, beside an open
    pay-as-you-go deployment of gpt-4.1 that serves every other team."""
    return Router(
        model_list=[
            {
                "model_name": "reserved",
                "litellm_params": {"model": "azure/gpt-4.1", "api_key": "sk-ptu", "api_base": "https://ptu.example"},
                "model_info": {
                    "id": "reserved-41",
                    "team_id": "team-x",
                    "ptu_count": 50,
                    "cost_per_ptu_per_hour": 1.0,
                    "ptu_effective_from": "2026-01-01T00:00:00Z",
                },
            },
            {
                "model_name": "gpt-4.1",
                "litellm_params": {"model": "azure/gpt-4.1", "api_key": "sk-payg", "api_base": "https://payg.example"},
                "model_info": {"id": "payg"},
            },
        ]
    )


@pytest.mark.parametrize(("team_id", "expected_on_the_provider_model"), [("team-x", 1.0), ("team-y", 0.0), (None, 1.0)])
def test_another_teams_single_team_reservation_sizes_nothing_on_a_teams_page(
    monkeypatch, team_id: str | None, expected_on_the_provider_model: float
):
    """A provider-model row reaches the reserved deployment only for the team it is reserved for: any
    other team was served by the open deployment, so its page counts no PTU-hours on that row, while
    a page spanning teams keeps the reserved deployment's sizing."""
    monkeypatch.setenv("LITELLM_ENABLE_PTU_COST_ATTRIBUTION", "True")
    one_hour: Final = {"azure/gpt-4.1": _bucket(_metrics(prompt=_ONE_PTU_HOUR_OF_INPUT, completion=0))}

    attached: Final = with_ptu_consumption(_response(_day("2026-09-23", one_hour)), _reserved_ptu_router(), team_id)

    row: Final = attached.results[0].breakdown.model_groups["azure/gpt-4.1"]
    assert row.metrics.ptu_hours == pytest.approx(expected_on_the_provider_model)
    assert attached.metadata.total_ptu_hours == pytest.approx(expected_on_the_provider_model)


_ONE_GPT_41_PTU_HOUR_ON_GPT_55: Final = _ONE_PTU_HOUR_OF_INPUT / AZURE_PTU_CAPACITY["gpt-5.5"].normalized_tokens_per_ptu_hour


@pytest.mark.parametrize(
    ("team_id", "expected_on_the_group", "expected_on_the_provider_model"),
    [
        ("team-a", 1.0, 1.0),
        ("team-b", _ONE_GPT_41_PTU_HOUR_ON_GPT_55, 0.0),
        ("team-c", 0.0, 0.0),
        (None, 1.0, 1.0),
    ],
)
def test_a_teams_rows_are_sized_by_the_deployment_it_is_served_from(
    monkeypatch, team_id: str | None, expected_on_the_group: float, expected_on_the_provider_model: float
):
    """A team's page converts its tokens through the deployment the ceiling served it from: team-b's
    share is on the gpt-5.5 deployment so its group row counts at that rate, its provider-model row
    reached only the open deployment so it counts nothing, team-c holds no share so it counts nothing,
    and a page spanning teams keeps the group's first sized deployment."""
    monkeypatch.setenv("LITELLM_ENABLE_PTU_COST_ATTRIBUTION", "True")
    one_hour_each: Final = {
        name: _bucket(_metrics(prompt=_ONE_PTU_HOUR_OF_INPUT, completion=0)) for name in ("ptu", "azure/gpt-4.1")
    }

    attached: Final = with_ptu_consumption(_response(_day("2026-09-23", one_hour_each)), _mixed_ptu_router(), team_id)

    groups: Final = attached.results[0].breakdown.model_groups
    assert groups["ptu"].metrics.ptu_hours == pytest.approx(expected_on_the_group)
    assert groups["azure/gpt-4.1"].metrics.ptu_hours == pytest.approx(expected_on_the_provider_model)
    assert attached.metadata.total_ptu_hours == pytest.approx(expected_on_the_group + expected_on_the_provider_model)

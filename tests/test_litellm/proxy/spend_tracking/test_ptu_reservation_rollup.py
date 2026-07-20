import types
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.proxy_server as ps
from litellm.proxy.spend_tracking.ptu_reservation_rollup import (
    PTU_SENTINEL_API_KEY,
    _compute_daily_flat_cost,
    _days_in_month,
    run_ptu_reservation_rollup,
)


@dataclass
class _Reservation:
    id: str
    team_id: str
    model: str
    cost_source: str
    ptu_count: int | None
    cost_per_ptu: float | None
    effective_from: datetime
    effective_to: datetime | None
    azure_resource_id: str | None = None


def _r(
    *,
    id: str = "res_1",
    team_id: str = "team_x",
    model: str = "gpt-4",
    cost_source: str = "manual",
    ptu_count: int | None = 1,
    cost_per_ptu: float | None = 200.0,
    effective_from: datetime | None = None,
    effective_to: datetime | None = None,
    azure_resource_id: str | None = None,
) -> _Reservation:
    return _Reservation(
        id=id,
        team_id=team_id,
        model=model,
        cost_source=cost_source,
        ptu_count=ptu_count,
        cost_per_ptu=cost_per_ptu,
        effective_from=effective_from or datetime(2026, 7, 1, tzinfo=timezone.utc),
        effective_to=effective_to,
        azure_resource_id=azure_resource_id,
    )


@pytest.fixture
def mock_prisma(monkeypatch):
    mock_daily = MagicMock()
    mock_daily.upsert = AsyncMock()
    mock_reservation = MagicMock()
    mock_reservation.find_many = AsyncMock(return_value=[])

    prisma = MagicMock()
    prisma.db = types.SimpleNamespace(
        litellm_dailyteamspend=mock_daily,
        litellm_ptureservation=mock_reservation,
    )
    monkeypatch.setattr(ps, "prisma_client", prisma)
    monkeypatch.setitem(ps.general_settings, "enable_ptu_cost_attribution", True)
    return prisma, mock_daily, mock_reservation


def test_days_in_month_covers_calendar_variants():
    assert _days_in_month(date(2026, 1, 15)) == 31
    assert _days_in_month(date(2026, 2, 15)) == 28
    assert _days_in_month(date(2024, 2, 15)) == 29
    assert _days_in_month(date(2026, 4, 1)) == 30
    assert _days_in_month(date(2026, 7, 31)) == 31


@pytest.mark.asyncio
async def test_compute_flat_cost_calendar_month_31():
    r = _r(ptu_count=1, cost_per_ptu=200.0)
    assert await _compute_daily_flat_cost(r, date(2026, 7, 15)) == pytest.approx(200.0 / 31)


@pytest.mark.asyncio
async def test_compute_flat_cost_calendar_month_28():
    r = _r(ptu_count=1, cost_per_ptu=200.0)
    assert await _compute_daily_flat_cost(r, date(2026, 2, 10)) == pytest.approx(200.0 / 28)


@pytest.mark.asyncio
async def test_compute_flat_cost_leap_february():
    r = _r(ptu_count=1, cost_per_ptu=200.0)
    assert await _compute_daily_flat_cost(r, date(2024, 2, 10)) == pytest.approx(200.0 / 29)


@pytest.mark.asyncio
async def test_compute_flat_cost_scales_with_ptu_count():
    small = await _compute_daily_flat_cost(_r(ptu_count=1, cost_per_ptu=200.0), date(2026, 7, 1))
    big = await _compute_daily_flat_cost(_r(ptu_count=100, cost_per_ptu=200.0), date(2026, 7, 1))
    assert big == pytest.approx(small * 100)


@pytest.mark.asyncio
async def test_compute_flat_cost_zero_when_azure_billing_and_no_fetcher():
    r = _r(cost_source="azure_billing", ptu_count=None, cost_per_ptu=None, azure_resource_id="/x")
    assert await _compute_daily_flat_cost(r, date(2026, 7, 1)) == 0.0


@pytest.mark.asyncio
async def test_compute_flat_cost_zero_when_manual_fields_missing():
    r = _r(ptu_count=None, cost_per_ptu=None)
    assert await _compute_daily_flat_cost(r, date(2026, 7, 1)) == 0.0


@pytest.mark.asyncio
async def test_compute_flat_cost_azure_billing_uses_fetcher_result():
    r = _r(cost_source="azure_billing", ptu_count=None, cost_per_ptu=None, azure_resource_id="/subs/x/deploy/y")
    fetcher = MagicMock()
    fetcher.get_daily_cost = AsyncMock(return_value=42.5)

    result = await _compute_daily_flat_cost(r, date(2026, 7, 15), azure_fetcher=fetcher)

    assert result == 42.5
    fetcher.get_daily_cost.assert_awaited_once_with("/subs/x/deploy/y", date(2026, 7, 15))


@pytest.mark.asyncio
async def test_compute_flat_cost_azure_billing_fetcher_error_returns_zero():
    r = _r(cost_source="azure_billing", ptu_count=None, cost_per_ptu=None, azure_resource_id="/subs/x/deploy/y")
    fetcher = MagicMock()
    fetcher.get_daily_cost = AsyncMock(side_effect=RuntimeError("azure boom"))

    result = await _compute_daily_flat_cost(r, date(2026, 7, 15), azure_fetcher=fetcher)

    assert result == 0.0


@pytest.mark.asyncio
async def test_compute_flat_cost_unknown_source_returns_zero():
    r = _r(cost_source="mystery")
    assert await _compute_daily_flat_cost(r, date(2026, 7, 1)) == 0.0


@pytest.mark.asyncio
async def test_rollup_azure_billing_reservation_writes_fetched_amount(mock_prisma):
    prisma, mock_daily, mock_reservation = mock_prisma
    reservation = _r(
        id="res_azure",
        team_id="team_x",
        model="gpt-4",
        cost_source="azure_billing",
        ptu_count=None,
        cost_per_ptu=None,
        azure_resource_id="/subs/x/deploy/y",
    )
    mock_reservation.find_many = AsyncMock(return_value=[reservation])
    fetcher = MagicMock()
    fetcher.get_daily_cost = AsyncMock(return_value=150.0)

    result = await run_ptu_reservation_rollup(
        prisma, target_date=date(2026, 7, 12), azure_fetcher=fetcher
    )

    assert result.rows_written == 1
    fetcher.get_daily_cost.assert_awaited_once_with("/subs/x/deploy/y", date(2026, 7, 12))
    create = mock_daily.upsert.await_args.kwargs["data"]["create"]
    assert create["ptu_flat_cost"] == 150.0
    assert create["ptu_reservation_id"] == "res_azure"


@pytest.mark.asyncio
async def test_rollup_flag_off_short_circuits(mock_prisma, monkeypatch):
    prisma, mock_daily, mock_reservation = mock_prisma
    monkeypatch.setitem(ps.general_settings, "enable_ptu_cost_attribution", False)

    result = await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    assert result.skipped_flag_off is True
    assert result.rows_written == 0
    mock_reservation.find_many.assert_not_awaited()
    mock_daily.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_rollup_prisma_none_returns_zero(monkeypatch):
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setitem(ps.general_settings, "enable_ptu_cost_attribution", True)

    result = await run_ptu_reservation_rollup(None, target_date=date(2026, 7, 12))

    assert result.rows_written == 0
    assert result.skipped_flag_off is False


@pytest.mark.asyncio
async def test_rollup_writes_expected_row(mock_prisma):
    prisma, mock_daily, mock_reservation = mock_prisma
    reservation = _r(
        id="res_1",
        team_id="team_x",
        model="gpt-4",
        ptu_count=1,
        cost_per_ptu=200.0,
        effective_from=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    mock_reservation.find_many = AsyncMock(return_value=[reservation])

    result = await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    assert result.rows_written == 1
    assert result.reservations_processed == 1
    mock_daily.upsert.assert_awaited_once()
    kwargs = mock_daily.upsert.await_args.kwargs
    where_key = "team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint"
    assert kwargs["where"][where_key] == {
        "team_id": "team_x",
        "date": "2026-07-12",
        "api_key": PTU_SENTINEL_API_KEY,
        "model": "gpt-4",
        "custom_llm_provider": "",
        "mcp_namespaced_tool_name": "",
        "endpoint": "",
    }
    create = kwargs["data"]["create"]
    assert create["team_id"] == "team_x"
    assert create["api_key"] == PTU_SENTINEL_API_KEY
    assert create["ptu_reservation_id"] == "res_1"
    assert create["ptu_flat_cost"] == pytest.approx(200.0 / 31)
    assert create.get("spend", 0) == 0 or "spend" not in create
    update = kwargs["data"]["update"]
    assert update["ptu_flat_cost"] == pytest.approx(200.0 / 31)
    assert update["ptu_reservation_id"] == "res_1"


@pytest.mark.asyncio
async def test_rollup_skips_azure_billing_reservations_when_pull_disabled(mock_prisma):
    """Regression: azure_billing rows must not corrupt sentinel table when pull flag off."""
    prisma, mock_daily, mock_reservation = mock_prisma
    mock_reservation.find_many = AsyncMock(
        return_value=[
            _r(cost_source="azure_billing", ptu_count=None, cost_per_ptu=None, azure_resource_id="/x"),
        ]
    )

    result = await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    assert result.reservations_processed == 1
    assert result.rows_written == 0
    mock_daily.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_rollup_processes_multiple_reservations(mock_prisma):
    prisma, mock_daily, mock_reservation = mock_prisma
    mock_reservation.find_many = AsyncMock(
        return_value=[
            _r(id="a", team_id="team_x", model="gpt-4"),
            _r(id="b", team_id="team_y", model="gpt-4"),
            _r(id="c", team_id="team_x", model="gpt-4o"),
        ]
    )

    result = await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    assert result.rows_written == 3
    assert mock_daily.upsert.await_count == 3


@pytest.mark.asyncio
async def test_rollup_defaults_target_date_to_yesterday_utc(mock_prisma):
    prisma, _, mock_reservation = mock_prisma
    mock_reservation.find_many = AsyncMock(return_value=[])

    result = await run_ptu_reservation_rollup(prisma)

    expected = datetime.now(timezone.utc).date() - timedelta(days=1)
    assert result.day == expected


@pytest.mark.asyncio
async def test_rollup_queries_active_reservations_at_day_start(mock_prisma):
    prisma, _, mock_reservation = mock_prisma
    mock_reservation.find_many = AsyncMock(return_value=[])

    await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    mock_reservation.find_many.assert_awaited_once()
    where = mock_reservation.find_many.await_args.kwargs["where"]
    day_start = datetime(2026, 7, 12, tzinfo=timezone.utc)
    assert where["effective_from"] == {"lte": day_start}
    assert {"effective_to": None} in where["OR"]
    assert {"effective_to": {"gt": day_start}} in where["OR"]


@pytest.mark.asyncio
async def test_rollup_idempotent_second_run_upserts_same_row(mock_prisma):
    prisma, mock_daily, mock_reservation = mock_prisma
    mock_reservation.find_many = AsyncMock(
        return_value=[_r(id="res_1", ptu_count=1, cost_per_ptu=200.0)]
    )

    r1 = await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))
    r2 = await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    assert r1.rows_written == 1
    assert r2.rows_written == 1
    assert mock_daily.upsert.await_count == 2
    calls = mock_daily.upsert.await_args_list
    where_key = "team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint"
    assert calls[0].kwargs["where"][where_key] == calls[1].kwargs["where"][where_key]


@pytest.mark.asyncio
async def test_rollup_continues_after_single_upsert_failure(mock_prisma):
    prisma, mock_daily, mock_reservation = mock_prisma
    mock_reservation.find_many = AsyncMock(
        return_value=[
            _r(id="a", team_id="team_x"),
            _r(id="b", team_id="team_y"),
            _r(id="c", team_id="team_z"),
        ]
    )

    call_count = {"n": 0}

    async def flaky_upsert(**_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated db failure")

    mock_daily.upsert = AsyncMock(side_effect=flaky_upsert)

    result = await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    assert result.reservations_processed == 3
    assert result.rows_written == 2


@pytest.mark.asyncio
async def test_rollup_writes_sentinel_api_key_not_real(mock_prisma):
    prisma, mock_daily, mock_reservation = mock_prisma
    mock_reservation.find_many = AsyncMock(return_value=[_r(id="res_1")])

    await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    kwargs = mock_daily.upsert.await_args.kwargs
    assert kwargs["data"]["create"]["api_key"] == PTU_SENTINEL_API_KEY
    assert kwargs["data"]["create"]["api_key"] != "sk-real-token"


@pytest.mark.asyncio
async def test_rollup_upserts_zero_spend_tokens_on_sentinel_row(mock_prisma):
    prisma, mock_daily, mock_reservation = mock_prisma
    mock_reservation.find_many = AsyncMock(return_value=[_r(id="res_1")])

    await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    create = mock_daily.upsert.await_args.kwargs["data"]["create"]
    update = mock_daily.upsert.await_args.kwargs["data"]["update"]

    for k in ("spend", "prompt_tokens", "completion_tokens", "api_requests"):
        assert k not in update, f"{k} must not be part of the PTU update payload"
        assert k not in create, f"{k} must not be part of the PTU create payload"


@pytest.mark.asyncio
async def test_rollup_boundary_effective_from_at_day_start_is_active(mock_prisma):
    prisma, _, mock_reservation = mock_prisma
    day_start = datetime(2026, 7, 12, tzinfo=timezone.utc)
    r = _r(id="edge", effective_from=day_start)
    mock_reservation.find_many = AsyncMock(return_value=[r])

    result = await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    assert result.rows_written == 1


@pytest.mark.asyncio
async def test_rollup_force_bypasses_flag_off(mock_prisma, monkeypatch):
    prisma, mock_daily, mock_reservation = mock_prisma
    monkeypatch.setitem(ps.general_settings, "enable_ptu_cost_attribution", False)
    mock_reservation.find_many = AsyncMock(return_value=[_r(id="res_1", ptu_count=1, cost_per_ptu=200.0)])

    result = await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12), force=True)

    assert result.skipped_flag_off is False
    assert result.rows_written == 1
    mock_reservation.find_many.assert_awaited_once()
    mock_daily.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_rollup_force_default_false_still_honors_flag(mock_prisma, monkeypatch):
    prisma, mock_daily, mock_reservation = mock_prisma
    monkeypatch.setitem(ps.general_settings, "enable_ptu_cost_attribution", False)
    mock_reservation.find_many = AsyncMock(return_value=[_r(id="res_1")])

    result = await run_ptu_reservation_rollup(prisma, target_date=date(2026, 7, 12))

    assert result.skipped_flag_off is True
    assert result.rows_written == 0
    mock_reservation.find_many.assert_not_awaited()
    mock_daily.upsert.assert_not_awaited()

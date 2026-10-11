"""Unit tests for BudgetReservationReconcileLogger (#45895)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from litellm.proxy.hooks.budget_reservation_reconcile_logger import (
    BudgetReservationReconcileLogger,
)


def _reservation(callback_bound: bool = True) -> dict:
    return {
        "reserved_cost": 0.00255,
        "entries": [{"counter_key": "spend:key:hashed", "reserved_cost": 0.00255}],
        "finalized": False,
        "callback_bound": callback_bound,
        "input_cost": 0.0,
        "input_tokens": None,
    }


@pytest.mark.asyncio
async def test_success_reconciles_reservation_to_response_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    reservation = _reservation()
    reconcile = AsyncMock(return_value=())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        reconcile,
    )
    logger = BudgetReservationReconcileLogger()

    await logger.async_log_success_event(
        {
            "litellm_params": {
                "metadata": {"user_api_key_budget_reservation": reservation},
            },
            "response_cost": 0.0,
        },
        response_obj=object(),
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
    )

    reconcile.assert_awaited_once()
    assert reconcile.await_args.kwargs["budget_reservation"] is reservation
    assert reconcile.await_args.kwargs["actual_cost"] == 0.0


@pytest.mark.asyncio
async def test_success_uses_standard_logging_response_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    reservation = _reservation()
    reconcile = AsyncMock(return_value=())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        reconcile,
    )
    logger = BudgetReservationReconcileLogger()

    await logger.async_log_success_event(
        {
            "litellm_params": {
                "litellm_metadata": {"user_api_key_budget_reservation": reservation},
            },
            "standard_logging_object": {"response_cost": 0.001},
            "response_cost": 9.9,  # ignored when SL payload present
        },
        response_obj=object(),
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
    )

    assert reconcile.await_args.kwargs["actual_cost"] == 0.001


@pytest.mark.asyncio
async def test_success_noop_when_no_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    reconcile = AsyncMock(return_value=())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        reconcile,
    )
    logger = BudgetReservationReconcileLogger()

    await logger.async_log_success_event(
        {"litellm_params": {"metadata": {}}},
        response_obj=object(),
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
    )

    reconcile.assert_not_awaited()


@pytest.mark.asyncio
async def test_success_noop_when_already_finalized(monkeypatch: pytest.MonkeyPatch) -> None:
    reservation = _reservation()
    reservation["finalized"] = True
    reconcile = AsyncMock(return_value=())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        reconcile,
    )
    logger = BudgetReservationReconcileLogger()

    await logger.async_log_success_event(
        {
            "litellm_params": {"metadata": {"user_api_key_budget_reservation": reservation}},
            "response_cost": 0.0,
        },
        response_obj=object(),
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
    )

    reconcile.assert_not_awaited()


@pytest.mark.asyncio
async def test_success_falls_back_to_release_when_reconcile_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation = _reservation()
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        AsyncMock(side_effect=RuntimeError("counter down")),
    )
    release = AsyncMock()
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.release_or_invalidate_budget_reservation",
        release,
    )
    logger = BudgetReservationReconcileLogger()

    await logger.async_log_success_event(
        {
            "litellm_params": {"metadata": {"user_api_key_budget_reservation": reservation}},
            "response_cost": 0.0,
        },
        response_obj=object(),
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
    )

    release.assert_awaited_once_with(budget_reservation=reservation)

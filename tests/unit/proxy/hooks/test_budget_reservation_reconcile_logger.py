"""Unit tests for BudgetReservationReconcileLogger (#45895)."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm.caching import DualCache
from litellm.proxy import proxy_server
from litellm.proxy.hooks.budget_reservation_reconcile_logger import (
    BudgetReservationReconcileLogger,
)
from litellm.types.interactions import InteractionsAPIResponse


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


@pytest.fixture
def spend_counter_cache(monkeypatch: pytest.MonkeyPatch) -> DualCache:
    cache: Final = DualCache()
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    return cache


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ("in_progress", "queued"))
@pytest.mark.parametrize("standard_logging", (False, True))
async def test_background_interaction_keeps_reservation_until_final_cost(
    spend_counter_cache: DualCache, status: str, standard_logging: bool
) -> None:
    reservation: Final = _reservation()
    counter_key: Final = "spend:key:hashed"
    spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=reservation["reserved_cost"])
    logger: Final = BudgetReservationReconcileLogger()
    now: Final = datetime(2026, 1, 1, tzinfo=UTC)
    kwargs: Final = {
        "litellm_params": {"metadata": {"user_api_key_budget_reservation": reservation}},
        **({"standard_logging_object": {"response_cost": None}, "response_cost": 9.9} if standard_logging else {}),
    }
    pending: Final = InteractionsAPIResponse(id="interactions/background", status=status)

    await logger.async_log_success_event(kwargs, pending, now, now)

    assert reservation["finalized"] is False
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(reservation["reserved_cost"])

    completed: Final = pending.model_copy(update={"status": "completed", "usage": {"total_tokens": 12}})
    final_kwargs: Final = {**kwargs, "standard_logging_object": {"response_cost": 0.004}}
    await logger.async_log_success_event(final_kwargs, completed, now, now)

    assert reservation["finalized"] is True
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(0.004)

    await logger.async_log_success_event(final_kwargs, completed, now, now)

    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(0.004)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "interaction_id", "polling_enabled", "has_usage"),
    (
        ("in_progress", "interactions/background", False, False),
        ("queued", "interactions/background", False, False),
        ("in_progress", None, True, False),
        ("queued", "", True, False),
        ("completed", "interactions/background", True, False),
        ("failed", "interactions/background", True, False),
        ("cancelled", "interactions/background", True, False),
        ("requires_action", "interactions/background", True, False),
        ("in_progress", "interactions/background", True, True),
    ),
)
async def test_missing_cost_releases_reservation_when_no_usage_poll_will_settle_it(
    spend_counter_cache: DualCache,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    interaction_id: str | None,
    polling_enabled: bool,
    has_usage: bool,
) -> None:
    monkeypatch.setattr(
        "litellm.proxy.hooks.budget_reservation_reconcile_logger.BACKGROUND_INTERACTION_COST_POLLING_ENABLED",
        polling_enabled,
    )
    reservation: Final = _reservation()
    counter_key: Final = "spend:key:hashed"
    spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=reservation["reserved_cost"])
    logger: Final = BudgetReservationReconcileLogger()
    now: Final = datetime(2026, 1, 1, tzinfo=UTC)

    await logger.async_log_success_event(
        {"litellm_params": {"metadata": {"user_api_key_budget_reservation": reservation}}},
        InteractionsAPIResponse(id=interaction_id, status=status, usage={"total_tokens": 12} if has_usage else None),
        now,
        now,
    )

    assert reservation["finalized"] is True
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_cost", "expected_cost"),
    ((0.0, 0.0), (0.004, 0.004), ("0.004", 0.004), (b"0.004", 0.004), (memoryview(b"0.004"), 0.004)),
)
async def test_reported_interaction_cost_is_settled_immediately(
    spend_counter_cache: DualCache, response_cost: object, expected_cost: float
) -> None:
    reservation: Final = _reservation()
    counter_key: Final = "spend:key:hashed"
    spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=reservation["reserved_cost"])
    logger: Final = BudgetReservationReconcileLogger()
    now: Final = datetime(2026, 1, 1, tzinfo=UTC)

    await logger.async_log_success_event(
        {
            "litellm_params": {"metadata": {"user_api_key_budget_reservation": reservation}},
            "response_cost": response_cost,
        },
        InteractionsAPIResponse(id="interactions/background", status="in_progress"),
        now,
        now,
    )

    assert reservation["finalized"] is True
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(expected_cost)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"litellm_params": {"metadata": None}},
        {"litellm_params": {"metadata": "invalid"}},
        {"litellm_params": {"metadata": {1: "ignored"}}},
    ),
)
async def test_missing_or_invalid_metadata_leaves_existing_spend_unchanged(
    spend_counter_cache: DualCache, kwargs: Mapping[str, object]
) -> None:
    counter_key: Final = "spend:key:hashed"
    spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=0.00255)
    logger: Final = BudgetReservationReconcileLogger()
    now: Final = datetime(2026, 1, 1, tzinfo=UTC)

    await logger.async_log_success_event(kwargs, object(), now, now)

    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(0.00255)

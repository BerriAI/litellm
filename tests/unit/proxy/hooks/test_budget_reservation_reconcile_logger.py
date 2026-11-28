"""Unit tests for _PROXY_BudgetReservationReconcileLogger (#45895)."""

import asyncio
import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.budget_reservation_reconcile_logger import (
    BudgetReservationReconcileLogger,
)
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.spend_tracking import native_budgets
from litellm.proxy.spend_tracking.budget_reservation import reserve_budget_for_request
from litellm.proxy.spend_tracking.native_budgets import reserve_native
from litellm.proxy.utils import ProxyLogging, ProxyUpdateSpend
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming
from litellm.types.interactions import InteractionsAPIResponse
from litellm.types.utils import CallTypes


def _reservation(callback_bound: bool = True) -> dict:  # mutable-ok: test helper returns mutable reservation dict
    return {
        "reserved_cost": 0.00255,
        "entries": [{"counter_key": "spend:key:hashed", "reserved_cost": 0.00255}],
        "finalized": False,
        "callback_bound": callback_bound,
        "input_cost": 0.0,
        "input_tokens": None,
    }


async def _native_reservation(
    *,
    token: str = "hashed",
    max_budget: float = 1.0,
    reservation_cost: float = 0.00255,
) -> dict:  # mutable-ok: test helper returns mutable reservation dict
    reservation: Final = await reserve_native(
        {
            "counters": [
                {
                    "counter_key": f"spend:key:{token}",
                    "max_budget": max_budget,
                    "fallback_spend": 0.0,
                    "entity_type": "Key",
                    "entity_id": token,
                    "spend_log_entity_id": token,
                }
            ],
            "reservation_cost": reservation_cost,
            "input_cost": 0.0,
            "input_tokens": None,
            "throttle_counter": None,
            "fail_closed": False,
            "status_code": 429,
            "reservation_id": f"res-{token}",
        },
        UserAPIKeyAuth(token=token, max_budget=max_budget, spend=0.0),
        {},
    )
    assert reservation is not None
    reservation["callback_bound"] = True
    return reservation


@pytest.mark.asyncio
async def test_success_reconciles_reservation_to_response_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    reservation: Final = _reservation()
    reconcile: Final = AsyncMock(return_value=())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        reconcile,
    )
    logger: Final = BudgetReservationReconcileLogger()

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
    assert reconcile.await_args is not None
    assert reconcile.await_args.kwargs["budget_reservation"] is reservation
    assert reconcile.await_args.kwargs["actual_cost"] == 0.0


@pytest.mark.asyncio
async def test_success_uses_standard_logging_response_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    reservation: Final = _reservation()
    reconcile: Final = AsyncMock(return_value=())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        reconcile,
    )
    logger: Final = BudgetReservationReconcileLogger()

    await logger.async_log_success_event(
        {
            "litellm_params": {
                "litellm_metadata": {"user_api_key_budget_reservation": reservation},
            },
            "standard_logging_object": {"response_cost": 0.001},
            "response_cost": 9.9,
        },
        response_obj=object(),
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
    )

    assert reconcile.await_args is not None
    assert reconcile.await_args.kwargs["actual_cost"] == 0.001


@pytest.mark.asyncio
async def test_success_falls_back_to_kwargs_cost_when_sl_payload_omits_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation: Final = _reservation()
    reconcile: Final = AsyncMock(return_value=())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        reconcile,
    )
    logger: Final = BudgetReservationReconcileLogger()

    await logger.async_log_success_event(
        {
            "litellm_params": {
                "metadata": {"user_api_key_budget_reservation": reservation},
            },
            "standard_logging_object": {"id": "sl-without-cost"},
            "response_cost": 0.004,
        },
        response_obj=object(),
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
    )

    reconcile.assert_awaited_once()
    assert reconcile.await_args is not None
    assert reconcile.await_args.kwargs["actual_cost"] == 0.004


@pytest.mark.asyncio
async def test_success_noop_when_no_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    reconcile: Final = AsyncMock(return_value=())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        reconcile,
    )
    logger: Final = BudgetReservationReconcileLogger()

    await logger.async_log_success_event(
        {"litellm_params": {"metadata": {}}},
        response_obj=object(),
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
    )

    reconcile.assert_not_awaited()


@pytest.mark.asyncio
async def test_success_noop_when_already_finalized(monkeypatch: pytest.MonkeyPatch) -> None:
    reservation: Final = _reservation()
    reservation["finalized"] = True
    reconcile: Final = AsyncMock(return_value=())
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        reconcile,
    )
    logger: Final = BudgetReservationReconcileLogger()

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
    reservation: Final = _reservation()
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        AsyncMock(side_effect=RuntimeError("counter down")),
    )
    release: Final = AsyncMock()
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.release_or_invalidate_budget_reservation",
        release,
    )
    logger: Final = BudgetReservationReconcileLogger()

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


@pytest.mark.asyncio
async def test_success_retains_known_spend_when_reconcile_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation: Final = _reservation()
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        AsyncMock(side_effect=RuntimeError("counter down")),
    )
    release: Final = AsyncMock()
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.release_or_invalidate_budget_reservation",
        release,
    )
    logger: Final = BudgetReservationReconcileLogger()

    await logger.async_log_success_event(
        {
            "litellm_params": {"metadata": {"user_api_key_budget_reservation": reservation}},
            "response_cost": 0.004,
        },
        response_obj=object(),
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
    )

    release.assert_not_awaited()
    assert reservation["finalized"] is False


@pytest.fixture
def spend_counter_store(monkeypatch: pytest.MonkeyPatch) -> Iterator[SpendCounterStore]:
    store: Final = SpendCounterStore(connect(None).spend_counters())
    monkeypatch.setattr(proxy_server, "spend_counter_store", store)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    native_budgets._engine.clear()  # pyright: ignore[reportPrivateUsage]  # isolate native handle per test
    yield store
    native_budgets._engine.clear()  # pyright: ignore[reportPrivateUsage]  # isolate native handle per test


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ("in_progress", "queued"))
@pytest.mark.parametrize("standard_logging", (False, True))
async def test_background_interaction_keeps_reservation_until_final_cost(
    spend_counter_store: SpendCounterStore, status: str, standard_logging: bool
) -> None:
    reservation: Final = await _native_reservation()
    counter_key: Final = "spend:key:hashed"
    logger: Final = BudgetReservationReconcileLogger()
    now: Final = datetime(2026, 1, 1, tzinfo=UTC)
    kwargs: Final = {
        "litellm_params": {"metadata": {"user_api_key_budget_reservation": reservation}},
        **({"standard_logging_object": {"response_cost": None}, "response_cost": 9.9} if standard_logging else {}),
    }
    pending: Final = InteractionsAPIResponse(id="interactions/background", status=status)

    await logger.async_log_success_event(kwargs, pending, now, now)

    assert reservation["finalized"] is False
    assert spend_counter_store.get_local(counter_key) == pytest.approx(reservation["reserved_cost"])

    completed: Final = pending.model_copy(update={"status": "completed", "usage": {"total_tokens": 12}})
    final_kwargs: Final = {**kwargs, "standard_logging_object": {"response_cost": 0.004}}
    await logger.async_log_success_event(final_kwargs, completed, now, now)

    assert reservation["finalized"] is True
    assert spend_counter_store.get_local(counter_key) == pytest.approx(0.004)

    await logger.async_log_success_event(final_kwargs, completed, now, now)

    assert spend_counter_store.get_local(counter_key) == pytest.approx(0.004)


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
    spend_counter_store: SpendCounterStore,
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
    reservation: Final = await _native_reservation()
    counter_key: Final = "spend:key:hashed"
    logger: Final = BudgetReservationReconcileLogger()
    now: Final = datetime(2026, 1, 1, tzinfo=UTC)

    await logger.async_log_success_event(
        {"litellm_params": {"metadata": {"user_api_key_budget_reservation": reservation}}},
        InteractionsAPIResponse(id=interaction_id, status=status, usage={"total_tokens": 12} if has_usage else None),
        now,
        now,
    )

    assert reservation["finalized"] is True
    assert spend_counter_store.get_local(counter_key) == pytest.approx(0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_cost", "expected_cost"),
    ((0.0, 0.0), (0.004, 0.004), ("0.004", 0.004), (b"0.004", 0.004), (memoryview(b"0.004"), 0.004)),
)
async def test_reported_interaction_cost_is_settled_immediately(
    spend_counter_store: SpendCounterStore, response_cost: object, expected_cost: float
) -> None:
    reservation: Final = await _native_reservation()
    counter_key: Final = "spend:key:hashed"
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
    assert spend_counter_store.get_local(counter_key) == pytest.approx(expected_cost)


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
    spend_counter_store: SpendCounterStore, kwargs: Mapping[str, object]
) -> None:
    counter_key: Final = "spend:key:hashed"
    spend_counter_store.set_local(counter_key, 0.00255)
    logger: Final = BudgetReservationReconcileLogger()
    now: Final = datetime(2026, 1, 1, tzinfo=UTC)

    await logger.async_log_success_event(kwargs, object(), now, now)

    assert spend_counter_store.get_local(counter_key) == pytest.approx(0.00255)


def test_no_log_opt_out_does_not_skip_budget_reconciler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settlement is mandatory accounting: no-log must not suppress this callback."""
    monkeypatch.setattr(litellm, "global_disable_no_log_param", False)
    logging_obj: Final = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="budget-reconcile-no-log",
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        function_id="1234",
    )
    logger: Final = BudgetReservationReconcileLogger()

    assert "_PROXY_" in type(logger).__name__
    assert (
        logging_obj.should_run_callback(
            callback=logger,
            litellm_params={"no-log": True},
            event_hook="async_success_handler",
        )
        is True
    )


@pytest.mark.asyncio
async def test_later_websocket_turns_charge_spend_counters_and_enforce_budget_without_db(
    spend_counter_store: SpendCounterStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(ProxyUpdateSpend, "disable_spend_updates", staticmethod(lambda: False))
    proxy_server.cost_tracking()

    auth_cache: Final = UserApiKeyCache()
    proxy_logging_obj: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    valid_token: Final = UserAPIKeyAuth(
        token="ws-custom-token",
        api_key="sk-raw-custom-key",
        user_id="ws-user-1",
        team_id="ws-team-1",
        max_budget=0.015,
        spend=0.0,
    )
    reservation: Final = await reserve_budget_for_request(
        request_body={"model": "gpt-4o", "input": "turn 1", "max_output_tokens": 10},
        route="/v1/responses",
        llm_router=None,
        valid_token=valid_token,
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=auth_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    assert reservation is not None
    reservation["callback_bound"] = True

    stamped: Final = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data={"litellm_metadata": {}},
        user_api_key_dict=valid_token,
        _metadata_variable_name="litellm_metadata",
    )
    connection_logging: Final = Logging(
        model="gpt-4o",
        messages=[],
        stream=False,
        call_type=CallTypes.aresponses_websocket.value,
        litellm_call_id="ws-conn-1",
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        function_id="ws-conn",
    )
    connection_logging.update_environment_variables(
        litellm_params={
            CallTypes.aresponses_websocket.value: True,
            "litellm_metadata": stamped["litellm_metadata"],
        },
        optional_params={},
    )

    streaming: Final = ResponsesWebSocketStreaming(
        websocket=MagicMock(send_text=AsyncMock()),
        backend_ws=MagicMock(send=AsyncMock()),
        logging_obj=connection_logging,
        user_api_key_dict=valid_token,
        authorized_model="gpt-4o",
        custom_llm_provider="openai",
    )

    turn_costs: Final = iter((0.004, 0.008, 0.005))

    def _mock_cost_calculator(*args: object, **kwargs: object) -> float:
        return next(turn_costs)

    monkeypatch.setattr(litellm, "response_cost_calculator", _mock_cost_calculator)

    for index in range(3):
        streaming._start_turn(f'{{"type":"response.create","input":"turn {index + 1}"}}')  # pyright: ignore[reportPrivateUsage]  # test exercises the real WS turn boundary
        await streaming._record_backend_event(  # pyright: ignore[reportPrivateUsage]  # drive the real terminal event dispatch
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": f"resp_{index + 1}",
                        "model": "gpt-4o",
                        "status": "completed",
                        "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
                    },
                }
            )
        )
        await asyncio.sleep(0)
        pending_tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
        if pending_tasks:
            await asyncio.gather(*pending_tasks)

    assert "user_api_key_budget_reservation" not in connection_logging.litellm_params["litellm_metadata"]
    assert spend_counter_store.get_local(f"spend:key:{valid_token.token}") == pytest.approx(0.017)
    assert spend_counter_store.get_local("spend:user:ws-user-1") == pytest.approx(0.013)
    assert spend_counter_store.get_local("spend:team:ws-team-1") == pytest.approx(0.013)

    with pytest.raises(litellm.BudgetExceededError):
        await reserve_budget_for_request(
            request_body={"model": "gpt-4o", "input": "turn 4", "max_output_tokens": 10},
            route="/v1/responses",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=auth_cache,
            proxy_logging_obj=proxy_logging_obj,
        )


@pytest.mark.asyncio
async def test_unreserved_success_charges_all_proxy_identity_scopes_without_db(
    spend_counter_store: SpendCounterStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProxyUpdateSpend, "disable_spend_updates", staticmethod(lambda: False))
    logger: Final = BudgetReservationReconcileLogger()
    now: Final = datetime(2026, 1, 1, tzinfo=UTC)
    auth: Final = UserAPIKeyAuth(
        token="hashed-tok-1",
        api_key="sk-raw-ignored",
        user_id="user-1",
        team_id="team-1",
        org_id="org-1",
        project_id="proj-1",
        end_user_id="end-user-1",
    )

    await logger.async_log_success_event(
        {
            "litellm_params": {
                "litellm_metadata": {
                    "user_api_key": "sk-raw-ignored",
                    "user_api_key_auth": auth,
                    "user_api_key_user_id": "user-1",
                    "user_api_key_team_id": "team-1",
                    "user_api_key_org_id": "org-1",
                    "user_api_key_project_id": "proj-1",
                    "user_api_key_end_user_id": "end-user-1",
                    "tags": ["ws", "tier-a"],
                },
            },
            "standard_logging_object": {"response_cost": 0.012, "request_tags": None},
        },
        response_obj=object(),
        start_time=now,
        end_time=now,
    )

    assert spend_counter_store.get_local(f"spend:key:{auth.token}") == pytest.approx(0.012)
    assert spend_counter_store.get_local("spend:user:user-1") == pytest.approx(0.012)
    assert spend_counter_store.get_local("spend:team:team-1") == pytest.approx(0.012)
    assert spend_counter_store.get_local("spend:team_member:user-1:team-1") == pytest.approx(0.012)
    assert spend_counter_store.get_local("spend:org:org-1") == pytest.approx(0.012)
    assert spend_counter_store.get_local("spend:project:proj-1") == pytest.approx(0.012)
    assert spend_counter_store.get_local("spend:end_user:end-user-1") == pytest.approx(0.012)
    assert spend_counter_store.get_local("spend:tag:ws") == pytest.approx(0.012)
    assert spend_counter_store.get_local("spend:tag:tier-a") == pytest.approx(0.012)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "response_cost", "disable_updates"),
    (
        ({"user_api_key": "tok-1"}, None, False),
        ({"user_api_key": "tok-1"}, 0.0, False),
        ({"user_api_key": "tok-1"}, -0.01, False),
        ({"user_api_key": "tok-1"}, True, False),
        ({"user_api_key": "tok-1"}, float("nan"), False),
        ({"user_api_key": "tok-1"}, float("inf"), False),
        ({"user_api_key": "tok-1"}, 0.05, True),
        ({"tags": ["unauthenticated-tag"]}, 0.05, False),
    ),
)
async def test_unreserved_success_skips_non_positive_disabled_or_unattributed_calls(
    spend_counter_store: SpendCounterStore,
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, object],  # mutable-ok: parametrized metadata payload
    response_cost: object,
    disable_updates: bool,
) -> None:
    monkeypatch.setattr(ProxyUpdateSpend, "disable_spend_updates", staticmethod(lambda: disable_updates))
    logger: Final = BudgetReservationReconcileLogger()
    now: Final = datetime(2026, 1, 1, tzinfo=UTC)

    await logger.async_log_success_event(
        {
            "litellm_params": {"metadata": metadata},
            "response_cost": response_cost,
        },
        response_obj=object(),
        start_time=now,
        end_time=now,
    )

    assert spend_counter_store.snapshot() == {}


@pytest.mark.asyncio
async def test_unreserved_success_logs_and_does_not_raise_when_increment_fails(
    spend_counter_store: SpendCounterStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProxyUpdateSpend, "disable_spend_updates", staticmethod(lambda: False))
    monkeypatch.setattr(proxy_server, "increment_spend_counters", AsyncMock(side_effect=RuntimeError("counter down")))
    logger: Final = BudgetReservationReconcileLogger()
    now: Final = datetime(2026, 1, 1, tzinfo=UTC)

    await logger.async_log_success_event(
        {
            "litellm_params": {"metadata": {"user_api_key": "tok-fail"}},
            "response_cost": 0.004,
        },
        response_obj=object(),
        start_time=now,
        end_time=now,
    )

    assert spend_counter_store.snapshot() == {}

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Final
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from pydantic import TypeAdapter

from litellm.models.performance_alert import ResourceUnreadableError, SweepOwnershipLostError, SweepState
from litellm.proxy.spend_tracking.performance_alerts import run_performance_alerts
from litellm.repositories.performance_alert_repository import PerformanceAlertRepository


def _raw_query_error(message: str, code: str = "") -> Exception:
    from prisma.errors import RawQueryError

    return RawQueryError({"user_facing_error": {"is_panic": False, "message": message, "meta": {"code": code}}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "code", "retryable"),
    (
        ("ERROR: canceling statement due to statement timeout", "57014", True),
        ("Query canceled", "57014", True),
        ("ERROR: canceling statement due to statement timeout", "", True),
        ("FATAL: sorry, too many clients already", "53300", False),
        ("FATAL: terminating connection due to administrator command", "57P01", False),
        ("ERROR: deadlock detected", "40P01", False),
    ),
)
async def test_only_a_statement_timeout_marks_a_resource_unreadable(message: str, code: str, retryable: bool) -> None:
    error: Final = _raw_query_error(message, code)
    repository, _query, execute = _repository(error)
    with pytest.raises(ResourceUnreadableError if retryable else type(error)):
        await repository.evaluate_page("worker-a", SweepState(day=date(2026, 1, 8), scope="model"), 25)
    execute.assert_not_awaited()


def _repository(
    rows: Sequence[Mapping[str, object]] | Exception,
    *,
    enabled: bool = True,
    used: int = 0,
    enterprise: bool = False,
) -> tuple[PerformanceAlertRepository, AsyncMock, AsyncMock]:
    query: Final = AsyncMock(side_effect=((), ({"enabled": enabled, "alerts_used": used},), rows))
    execute: Final = AsyncMock(return_value=1)
    transaction: Final = MagicMock(query_raw=query, execute_raw=execute)
    context: Final = MagicMock(__aenter__=AsyncMock(return_value=transaction), __aexit__=AsyncMock(return_value=False))
    client: Final = MagicMock(writer_db=MagicMock(tx=MagicMock(return_value=context)))
    return PerformanceAlertRepository(client, enterprise=enterprise), query, execute


@pytest.mark.asyncio
@pytest.mark.parametrize("day", (None, date(2026, 1, 8)))
async def test_claim_returns_the_saved_position_or_a_fresh_sweep(day: date | None) -> None:
    repository, query, _execute = _repository(
        (
            {
                "day": day,
                "scope": "team",
                "cursor": "team-a",
                "wrapped": True,
                "checked": 12,
                "skipped": 2,
                "completed": False,
                "start_scope": "team",
                "start_after": "team-z",
            },
        )
    )
    claim: Final = await repository.claim(timedelta(minutes=5))
    assert claim is not None
    assert claim.state == (
        None
        if day is None
        else SweepState(
            day=day,
            scope="team",
            cursor="team-a",
            wrapped=True,
            checked=12,
            skipped=2,
            start_scope="team",
            start_after="team-z",
        )
    )
    query.assert_awaited_with(ANY, claim.owner, "300 seconds")


@pytest.mark.asyncio
async def test_a_claim_taken_by_another_worker_returns_no_work() -> None:
    repository, _query, execute = _repository(())
    assert await repository.claim(timedelta(minutes=5)) is None
    execute.assert_not_awaited()


@pytest.fixture
def spend_window() -> Mapping[str, object]:
    return {
        "resource_id": "model-a",
        "spend": 100.0,
        "base_spend": 70.0,
        "requests": 100,
        "base_requests": 700,
        "failed": 0,
        "base_failed": 0,
        "response_ms": 0,
        "timed": 0,
        "base_response_ms": 0,
        "base_timed": 0,
        "active_days": 7,
        "timed_days": 0,
    }


@pytest.mark.asyncio
async def test_a_page_saves_the_detected_evidence_and_its_resume_position(spend_window: Mapping[str, object]) -> None:
    repository, _query, execute = _repository((spend_window,))
    state: Final = SweepState(day=date(2026, 1, 8), scope="model")
    result: Final = await repository.evaluate_page("worker-a", state, 1)
    assert result == state.model_copy(update={"cursor": "model-a", "checked": 1})
    payload: Final = TypeAdapter(str).validate_python(execute.await_args_list[0].args[1])
    records: Final = TypeAdapter(tuple[dict[str, object], ...]).validate_json(payload)
    assert len(records) == 1
    assert records[0]["resource_id"] == "model-a"
    assert records[0]["type"] == "spend"
    assert records[0]["evidence"] == {"value": 100.0, "baseline": 10.0, "unit": "usd", "sample_size": 100}
    execute.assert_awaited_with(ANY, "worker-a", "2026-01-08", "model", "model-a", False, 1, 0, False, None, "")


@pytest.mark.asyncio
async def test_lost_ownership_aborts_the_page_transaction(spend_window: Mapping[str, object]) -> None:
    repository, _query, execute = _repository((spend_window,))
    execute.side_effect = (1, 1, 0)
    with pytest.raises(SweepOwnershipLostError):
        await repository.evaluate_page("previous-worker", SweepState(day=date(2026, 1, 8), scope="model"), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("timeout", "invalid_data", "connection_lost"))
async def test_unreadable_data_is_retryable_but_database_failures_propagate(failure: str) -> None:
    error: Final = _raw_query_error("query failed", "57014" if failure == "timeout" else "57P01")
    repository, _query, execute = _repository(({},) if failure == "invalid_data" else error)
    with pytest.raises(ResourceUnreadableError if failure != "connection_lost" else type(error)):
        await repository.evaluate_page("worker-a", SweepState(day=date(2026, 1, 8), scope="model"), 25)
    execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", (False, True))
async def test_skipping_an_unreadable_resource_saves_progress_without_counting_it_as_checked(wrapped: bool) -> None:
    repository, query, execute = _repository(({"resource_id": "team-b"},))
    state: Final = SweepState(
        day=date(2026, 1, 8), scope="team", cursor="team-a", wrapped=wrapped, start_scope="team", start_after="team-z"
    )
    result: Final = await repository.skip_next("worker-a", state)
    assert result == state.model_copy(update={"cursor": "team-b", "skipped": 1})
    query.assert_awaited_with(ANY, "2026-01-08", "2026-01-01", 1, "team-a", *(("team-z",) if wrapped else ()))
    execute.assert_awaited_once_with(
        ANY, "worker-a", "2026-01-08", "team", "team-b", wrapped, 0, 1, False, "team", "team-z"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("enabled", "used"), ((True, 5), (False, 0)))
async def test_stopped_monitoring_never_reads_usage_or_claims_work(enabled: bool, used: int) -> None:
    repository, query, execute = _repository((), enabled=enabled, used=used)
    await run_performance_alerts(repository, now=lambda: datetime(2026, 1, 9, tzinfo=timezone.utc))
    assert query.await_count == 2
    execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("enterprise", (False, True))
async def test_first_read_initializes_the_allowance_once(enterprise: bool) -> None:
    repository, query, execute = _repository((), enterprise=enterprise)
    saved: Final = ({"enabled": True, "alerts_used": 3},)
    query.side_effect = ((), (), ({"enabled": True, "alerts_used": None},), saved, (), saved)
    first: Final = await repository.monitoring()
    assert first.alerts_used == 3
    assert first.alert_limit == (None if enterprise else 5)
    assert first.active
    assert await repository.monitoring() == first
    assert execute.await_count == 2
    execute.assert_awaited_with(ANY, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (False, True))
async def test_admin_toggle_persists_the_requested_value(enabled: bool) -> None:
    repository, query, execute = _repository(())
    query.side_effect = ((), ({"enabled": not enabled, "alerts_used": 2},), ({"enabled": enabled, "alerts_used": 2},))
    result: Final = await repository.set_enabled(enabled)
    assert result.enabled is enabled
    assert result.active is enabled
    assert result.alerts_used == 2
    execute.assert_awaited_once_with(ANY, enabled)


@pytest.mark.asyncio
async def test_last_free_alert_keeps_the_page_available_for_an_enterprise_upgrade(
    spend_window: Mapping[str, object],
) -> None:
    repository, _query, execute = _repository((spend_window,), used=4)
    state: Final = SweepState(day=date(2026, 1, 8), scope="model")
    assert await repository.evaluate_page("worker-a", state, 1) is None
    execute.assert_any_await(ANY, ANY, 1)
    execute.assert_any_await(ANY, 5, 1)
    execute.assert_awaited_with(ANY, "worker-a", "2026-01-08", "model", "", False, 0, 0, False, None, "")


@pytest.mark.asyncio
async def test_enterprise_keeps_saving_alerts_after_the_free_allowance(spend_window: Mapping[str, object]) -> None:
    repository, _query, execute = _repository((spend_window,), used=5, enterprise=True)
    state: Final = SweepState(day=date(2026, 1, 8), scope="model")
    result: Final = await repository.evaluate_page("worker-a", state, 1)
    assert result == state.model_copy(update={"cursor": "model-a", "checked": 1})
    execute.assert_any_await(ANY, ANY, None)

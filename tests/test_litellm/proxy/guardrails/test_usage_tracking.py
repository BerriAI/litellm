import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.proxy.guardrails.usage_tracking import (
    _MAX_PENDING_ROWS,
    PendingRollups,
    _capped,
    process_spend_logs_guardrail_usage,
)


def _prisma() -> MagicMock:
    client = MagicMock()
    db = client.db
    db.litellm_dailyguardrailmetrics.upsert = AsyncMock()
    db.litellm_dailyguardrailusageunits.upsert = AsyncMock()
    db.litellm_spendlogguardrailindex.create_many = AsyncMock()
    return client


def _payload(
    request_id: str,
    *,
    team_id: str | None = "team-a",
    api_key: str = "hashed-key-1",
    usage: dict[str, Any] | None = None,
    guardrail_status: str = "success",
    cost_by_unit: dict[str, Any] | None = None,
    cost_in_spend: bool | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "guardrail_id": "bedrock-guard",
        "guardrail_status": guardrail_status,
    }
    if usage is not None:
        entry["guardrail_usage"] = usage
    if cost_by_unit is not None:
        entry["guardrail_cost_by_unit"] = cost_by_unit
    if cost_in_spend is not None:
        entry["guardrail_cost_in_spend"] = cost_in_spend
    return {
        "request_id": request_id,
        "startTime": datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc),
        "team_id": team_id,
        "api_key": api_key,
        "metadata": json.dumps({"guardrail_information": [entry]}),
    }


def _units_upserts(prisma: MagicMock) -> dict[tuple, int]:
    calls = prisma.db.litellm_dailyguardrailusageunits.upsert.call_args_list
    out: dict[tuple, int] = {}
    for c in calls:
        where = c.kwargs["where"]["guardrail_id_date_team_id_api_key_usage_unit"]
        create = c.kwargs["data"]["create"]
        assert create["units"] == c.kwargs["data"]["update"]["units"]["increment"]
        assert {k: create[k] for k in where} == where
        out[tuple(where[k] for k in ("guardrail_id", "date", "team_id", "api_key", "usage_unit"))] = create["units"]
    return out


def _cost_upserts(prisma: MagicMock) -> dict[str, tuple[float | None, object]]:
    """usage_unit -> (cost written on create, cost clause sent on update)."""
    calls = prisma.db.litellm_dailyguardrailusageunits.upsert.call_args_list
    return {
        c.kwargs["data"]["create"]["usage_unit"]: (
            c.kwargs["data"]["create"]["cost"],
            c.kwargs["data"]["update"]["cost"],
        )
        for c in calls
    }


@pytest.mark.asyncio
async def test_usage_units_rolled_up_by_guardrail_team_key_and_date():
    """
    LIT-5650: billable units must aggregate per (guardrail, date, team, key,
    counter): same-key payloads sum into one upsert, a team-less payload gets
    its own empty-string-team row, and blocked invocations (which Bedrock
    still bills for) count exactly like passed ones.
    """
    prisma = _prisma()
    logs = [
        _payload("r1", usage={"topicPolicyUnits": 1, "contentPolicyUnits": 1}),
        _payload(
            "r2",
            usage={"topicPolicyUnits": 1, "contentPolicyUnits": 2},
            guardrail_status="guardrail_intervened",
        ),
        _payload("r3", team_id=None, api_key="hashed-key-2", usage={"topicPolicyUnits": 1}),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs)

    assert _units_upserts(prisma) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "topicPolicyUnits"): 2,
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "contentPolicyUnits"): 3,
        ("bedrock-guard", "2026-08-17", "", "hashed-key-2", "topicPolicyUnits"): 1,
    }


def _fake_sleep() -> tuple[AsyncMock, list[float]]:
    delays: list[float] = []
    sleep = AsyncMock(side_effect=lambda delay: delays.append(delay))
    return sleep, delays


@pytest.mark.asyncio
async def test_one_failing_upsert_does_not_drop_remaining_writes():
    """
    A DB error on one daily-metrics or usage-unit upsert must not cancel the
    remaining upserts in the flushed batch, or the usage endpoints would
    permanently under-report billable counters.
    """
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailmetrics.upsert.side_effect = httpx.ConnectError("db down")
    prisma.db.litellm_dailyguardrailusageunits.upsert.side_effect = [httpx.ConnectError("db down"), None, None]
    sleep, _ = _fake_sleep()
    logs = [
        _payload("r1", usage={"topicPolicyUnits": 1}),
        _payload("r2", team_id=None, api_key="hashed-key-2", usage={"topicPolicyUnits": 1}),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs, sleep=sleep, pending=PendingRollups())

    assert _units_upserts(prisma) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "topicPolicyUnits"): 1,
        ("bedrock-guard", "2026-08-17", "", "hashed-key-2", "topicPolicyUnits"): 1,
    }


@pytest.mark.asyncio
async def test_transient_upsert_failure_is_retried_with_backoff_for_failed_rows_only():
    """
    A connection error (the write provably never reached the database) must
    not permanently drop billed units from the aggregates: only the rows that
    failed are re-sent, after exponential backoff, and the batch ends once
    every row has landed.
    """
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailusageunits.upsert.side_effect = [httpx.ConnectError("blip"), None, None]
    sleep, delays = _fake_sleep()
    logs = [
        _payload("r1", usage={"topicPolicyUnits": 1}),
        _payload("r2", team_id=None, api_key="hashed-key-2", usage={"topicPolicyUnits": 1}),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs, sleep=sleep)

    calls = prisma.db.litellm_dailyguardrailusageunits.upsert.call_args_list
    assert len(calls) == 3
    assert calls[2].kwargs["where"] == calls[0].kwargs["where"]
    assert delays == [1]


@pytest.mark.asyncio
async def test_persistent_upsert_failure_stops_after_three_retries():
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailmetrics.upsert.side_effect = httpx.ConnectError("db down")
    sleep, delays = _fake_sleep()
    pending = PendingRollups()

    await process_spend_logs_guardrail_usage(
        prisma, [_payload("r1", usage={"topicPolicyUnits": 1})], sleep=sleep, pending=pending
    )

    assert prisma.db.litellm_dailyguardrailmetrics.upsert.call_count == 4
    assert delays == [1, 2, 4]
    assert prisma.db.litellm_dailyguardrailusageunits.upsert.call_count == 1
    assert dict(pending.metrics) == {
        ("bedrock-guard", "2026-08-17"): {
            "requests_evaluated": 1,
            "passed_count": 1,
            "blocked_count": 0,
            "flagged_count": 0,
        }
    }


@pytest.mark.asyncio
async def test_retry_exhausted_rows_are_requeued_and_land_on_the_next_flush():
    """
    LIT-5761: rollup rows whose connection-error retries exhaust must not be
    silently lost. They are requeued and merged into the next flushed batch,
    so the aggregates catch up once the database is reachable again.
    """
    pending = PendingRollups()
    down = _prisma()
    down.db.litellm_dailyguardrailmetrics.upsert.side_effect = httpx.ConnectError("db down")
    down.db.litellm_dailyguardrailusageunits.upsert.side_effect = httpx.ConnectError("db down")
    sleep, _ = _fake_sleep()

    await process_spend_logs_guardrail_usage(
        down, [_payload("r1", usage={"topicPolicyUnits": 2})], sleep=sleep, pending=pending
    )

    assert dict(pending.units) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "topicPolicyUnits"): (2, None)
    }

    recovered = _prisma()
    await process_spend_logs_guardrail_usage(
        recovered, [_payload("r2", usage={"topicPolicyUnits": 3})], sleep=sleep, pending=pending
    )

    assert _units_upserts(recovered) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "topicPolicyUnits"): 5,
    }
    metrics_create = recovered.db.litellm_dailyguardrailmetrics.upsert.call_args.kwargs["data"]["create"]
    assert metrics_create["requests_evaluated"] == 2
    assert not pending.units
    assert not pending.metrics


@pytest.mark.asyncio
async def test_ambiguous_failures_are_never_requeued():
    """
    A post-send failure (the increment may have committed) must stay dropped:
    requeueing it would re-send a possibly applied increment and double-count.
    """
    pending = PendingRollups()
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailusageunits.upsert.side_effect = httpx.ReadTimeout("maybe committed")
    sleep, delays = _fake_sleep()

    await process_spend_logs_guardrail_usage(
        prisma, [_payload("r1", usage={"topicPolicyUnits": 1})], sleep=sleep, pending=pending
    )

    assert delays == []
    assert not pending.units
    assert not pending.metrics


def test_pending_requeue_is_capped_dropping_oldest_rows():
    rows = {index: index for index in range(_MAX_PENDING_ROWS + 5)}

    capped = _capped(rows, "usage unit")

    assert len(capped) == _MAX_PENDING_ROWS
    assert 4 not in capped
    assert _MAX_PENDING_ROWS + 4 in capped


def _units_upsert_wheres(prisma: MagicMock) -> list[tuple]:
    return [
        tuple(
            c.kwargs["where"]["guardrail_id_date_team_id_api_key_usage_unit"][k]
            for k in ("guardrail_id", "date", "team_id", "api_key", "usage_unit")
        )
        for c in prisma.db.litellm_dailyguardrailusageunits.upsert.call_args_list
    ]


@pytest.mark.asyncio
async def test_post_send_failure_is_never_retried_so_increments_cannot_double_count():
    """
    Follow-up to #37225: the units upsert is a non-idempotent increment, so an
    ambiguous post-send failure (read timeout after the statement may have
    committed) must be attempted exactly once. Re-sending it stacks a second
    increment and inflates billable unit totals. Only a connection error proves
    the write never reached the database and may be retried; the other rows in
    the batch still land either way.
    """
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailusageunits.upsert.side_effect = [
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectError("refused"),
        None,
    ]
    sleep, delays = _fake_sleep()
    logs = [
        _payload("r1", usage={"topicPolicyUnits": 1}),
        _payload("r2", team_id=None, api_key="hashed-key-2", usage={"topicPolicyUnits": 1}),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs, sleep=sleep)

    timed_out_row = ("bedrock-guard", "2026-08-17", "", "hashed-key-2", "topicPolicyUnits")
    refused_row = ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "topicPolicyUnits")
    assert _units_upsert_wheres(prisma) == [timed_out_row, refused_row, refused_row]
    assert delays == [1]


@pytest.mark.asyncio
async def test_generic_upsert_exception_is_terminal_for_that_row_only():
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailmetrics.upsert.side_effect = RuntimeError("constraint violation")
    sleep, delays = _fake_sleep()

    await process_spend_logs_guardrail_usage(prisma, [_payload("r1", usage={"topicPolicyUnits": 1})], sleep=sleep)

    assert prisma.db.litellm_dailyguardrailmetrics.upsert.call_count == 1
    assert delays == []
    assert _units_upserts(prisma) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "topicPolicyUnits"): 1,
    }


@pytest.mark.asyncio
async def test_zero_and_non_int_usage_counters_are_skipped():
    prisma = _prisma()
    logs = [
        _payload(
            "r1",
            usage={
                "topicPolicyUnits": 1,
                "wordPolicyUnits": 0,
                "contentPolicyImageUnits": 0,
                "oddball": "not-an-int",
                "boolish": True,
            },
        ),
        _payload("r2", usage=None),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs)

    assert _units_upserts(prisma) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "topicPolicyUnits"): 1,
    }


@pytest.mark.asyncio
async def test_payload_without_request_id_is_skipped_like_the_metrics_path():
    prisma = _prisma()
    logs = [
        {**_payload("ignored", usage={"topicPolicyUnits": 5}), "request_id": None},
        _payload("r2", usage={"topicPolicyUnits": 1}),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs)

    assert _units_upserts(prisma) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "topicPolicyUnits"): 1,
    }
    assert prisma.db.litellm_dailyguardrailmetrics.upsert.call_args.kwargs["data"]["create"]["requests_evaluated"] == 1


@pytest.mark.asyncio
async def test_cost_rolled_up_per_counter_alongside_units():
    """LIT-5652: the hook's per-counter cost lands on the same daily row as the
    units it priced, summed across payloads exactly like the units are, and the
    update path increments it so a second flush on the same day keeps adding."""
    prisma = _prisma()
    logs = [
        _payload(
            "r1",
            usage={"contentPolicyUnits": 1000, "wordPolicyUnits": 50},
            cost_by_unit={"contentPolicyUnits": 0.15, "wordPolicyUnits": 0.0},
        ),
        _payload(
            "r2",
            usage={"contentPolicyUnits": 2000, "wordPolicyUnits": 10},
            cost_by_unit={"contentPolicyUnits": 0.3, "wordPolicyUnits": 0.0},
        ),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs)

    assert _units_upserts(prisma) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "contentPolicyUnits"): 3000,
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "wordPolicyUnits"): 60,
    }
    costs = _cost_upserts(prisma)
    assert costs["contentPolicyUnits"][0] == pytest.approx(0.45)
    assert costs["contentPolicyUnits"][1] == {"increment": pytest.approx(0.45)}
    assert costs["wordPolicyUnits"] == (0.0, {"increment": 0.0})


@pytest.mark.asyncio
async def test_unpriced_increment_makes_the_rows_cost_unknown_not_partial():
    """A payload with usage but no per-counter cost (a hook without pricing, a
    pre-upgrade proxy in a mixed fleet) must poison that row's cost to NULL on
    both create and update. Keeping the priced part would understate the day
    while looking exact."""
    prisma = _prisma()
    logs = [
        _payload("r1", usage={"contentPolicyUnits": 1000}, cost_by_unit={"contentPolicyUnits": 0.15}),
        _payload("r2", usage={"contentPolicyUnits": 1000}),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs)

    assert _units_upserts(prisma) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "contentPolicyUnits"): 2000,
    }
    assert _cost_upserts(prisma) == {"contentPolicyUnits": (None, None)}


@pytest.mark.asyncio
async def test_report_only_and_forged_costs_are_not_rolled_up_but_units_are():
    """guardrail_cost_in_spend=False (Azure Prompt Shield) keeps its cost out of
    spend, so the rollup must not record it either or the dashboard would show
    a number the budget never charged. A negative or non-finite per-counter cost
    is treated the same way rather than subtracting from the day."""
    prisma = _prisma()
    logs = [
        _payload("r1", usage={"text_records": 3}, cost_by_unit={"text_records": 0.5}, cost_in_spend=False),
        _payload("r2", usage={"contentPolicyUnits": 10}, cost_by_unit={"contentPolicyUnits": -0.5}),
        _payload("r3", usage={"topicPolicyUnits": 10}, cost_by_unit={"topicPolicyUnits": float("inf")}),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs)

    assert _units_upserts(prisma) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "text_records"): 3,
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "contentPolicyUnits"): 10,
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "topicPolicyUnits"): 10,
    }
    assert _cost_upserts(prisma) == {
        "text_records": (None, None),
        "contentPolicyUnits": (None, None),
        "topicPolicyUnits": (None, None),
    }


@pytest.mark.asyncio
async def test_requeued_cost_is_added_to_the_next_flush():
    """Cost must survive the connection-error requeue the same way units do, or
    a DB blip would silently drop dollars while keeping the units they bought."""
    pending = PendingRollups()
    down = _prisma()
    down.db.litellm_dailyguardrailmetrics.upsert.side_effect = httpx.ConnectError("db down")
    down.db.litellm_dailyguardrailusageunits.upsert.side_effect = httpx.ConnectError("db down")
    sleep, _ = _fake_sleep()

    await process_spend_logs_guardrail_usage(
        down,
        [_payload("r1", usage={"contentPolicyUnits": 1000}, cost_by_unit={"contentPolicyUnits": 0.15})],
        sleep=sleep,
        pending=pending,
    )
    recovered = _prisma()
    await process_spend_logs_guardrail_usage(
        recovered,
        [_payload("r2", usage={"contentPolicyUnits": 2000}, cost_by_unit={"contentPolicyUnits": 0.3})],
        sleep=sleep,
        pending=pending,
    )

    assert _units_upserts(recovered) == {
        ("bedrock-guard", "2026-08-17", "team-a", "hashed-key-1", "contentPolicyUnits"): 3000,
    }
    assert _cost_upserts(recovered)["contentPolicyUnits"][0] == pytest.approx(0.45)

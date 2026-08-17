import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.guardrails.usage_tracking import process_spend_logs_guardrail_usage


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
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "guardrail_id": "bedrock-guard",
        "guardrail_status": guardrail_status,
    }
    if usage is not None:
        entry["guardrail_usage"] = usage
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


@pytest.mark.asyncio
async def test_one_failing_upsert_does_not_drop_remaining_writes():
    """
    A DB error on one daily-metrics or usage-unit upsert must not cancel the
    remaining upserts in the flushed batch, or the usage endpoints would
    permanently under-report billable counters (batches are never retried).
    """
    prisma = _prisma()
    prisma.db.litellm_dailyguardrailmetrics.upsert.side_effect = RuntimeError("db down")
    prisma.db.litellm_dailyguardrailusageunits.upsert.side_effect = [RuntimeError("db down"), None]
    logs = [
        _payload("r1", usage={"topicPolicyUnits": 1}),
        _payload("r2", team_id=None, api_key="hashed-key-2", usage={"topicPolicyUnits": 1}),
    ]

    await process_spend_logs_guardrail_usage(prisma, logs)

    assert prisma.db.litellm_dailyguardrailusageunits.upsert.call_count == 2


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

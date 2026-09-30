from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.clickhouse.clickhouse_spend_logger import ClickHouseSpendLogger


def _payload(request_id: str, *, status: str, cost: float) -> dict[str, object]:
    return {
        "id": request_id,
        "call_type": "acompletion",
        "response_cost": cost,
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "startTime": 1_700_000_000.123,
        "endTime": 1_700_000_001.456,
        "metadata": {"user_api_key_hash": "key-a", "user_api_key_team_id": "team-a"},
        "model": "test-model",
        "status": status,
    }


@pytest.mark.asyncio
async def test_success_and_failure_events_write_scoped_spend_rows():
    storage = MagicMock()
    storage.ensure_schema = AsyncMock()
    storage.insert_rows = AsyncMock()
    logger = ClickHouseSpendLogger(storage=storage)
    now = datetime.now(timezone.utc)

    await logger.async_log_success_event(
        {"standard_logging_object": _payload("response-1", status="success", cost=0.25)}, None, now, now
    )
    await logger.async_log_failure_event(
        {"standard_logging_object": _payload("response-2_cache_hit123", status="failure", cost=0.0)},
        None,
        now,
        now,
    )
    await logger.flush_queue()
    if logger._flush_task is not None:
        logger._flush_task.cancel()

    storage.ensure_schema.assert_not_awaited()
    assert storage.insert_rows.await_count == 1
    table, rows = storage.insert_rows.await_args.args
    assert table == "spend_logs"
    assert rows == [
        {
            "request_id": "response-1",
            "response_id": "response-1",
            "call_type": "acompletion",
            "api_key": "key-a",
            "team_id": "team-a",
            "model": "test-model",
            "spend": 0.25,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
            "start_time": 1_700_000_000_123,
            "end_time": 1_700_000_001_456,
            "status": "success",
            "cache_hit": False,
        },
        {
            "request_id": "response-2_cache_hit123",
            "response_id": "response-2",
            "call_type": "acompletion",
            "api_key": "key-a",
            "team_id": "team-a",
            "model": "test-model",
            "spend": 0.0,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
            "start_time": 1_700_000_000_123,
            "end_time": 1_700_000_001_456,
            "status": "failure",
            "cache_hit": False,
        },
    ]


@pytest.mark.asyncio
async def test_trace_ingest_and_invalid_payload_do_not_write_spend():
    storage = MagicMock()
    storage.ensure_schema = AsyncMock()
    logger = ClickHouseSpendLogger(storage=storage)
    now = datetime.now(timezone.utc)

    await logger.async_log_success_event(
        {"standard_logging_object": {**_payload("trace", status="success", cost=0), "call_type": "/v1/traces"}},
        None,
        now,
        now,
    )
    await logger.async_log_success_event({"standard_logging_object": "invalid"}, None, now, now)

    assert logger.log_queue == []
    storage.ensure_schema.assert_not_awaited()

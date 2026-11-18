import asyncio
import contextlib
import hashlib
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Final
from unittest.mock import Mock
from litellm.proxy.utils import get_redoc_url, get_docs_url

import pytest
from fastapi import Request

import litellm
from unittest.mock import MagicMock, patch, AsyncMock


import httpx
import math
from litellm.constants import SPEND_LOG_WRITE_BATCH_MAX_ROWS
from litellm.proxy.utils import update_spend

# The flush chunks the queue by BATCH_SIZE and then splits each chunk by the row
# budget, so statement counts below are derived from both rather than hardcoded.
_OUTER_BATCH_SIZE = 1000


def _statements_for(rows: int) -> int:
    full, remainder = divmod(rows, _OUTER_BATCH_SIZE)
    chunks = [_OUTER_BATCH_SIZE] * full + ([remainder] if remainder else [])
    return sum(math.ceil(chunk / SPEND_LOG_WRITE_BATCH_MAX_ROWS) for chunk in chunks)


class MockPrismaClient:
    def __init__(self):
        # Create AsyncMock for db operations
        self.db = AsyncMock()
        self.db.litellm_spendlogs = AsyncMock()
        self.db.litellm_spendlogs.create_many = AsyncMock()

        # Initialize transaction lists
        self.spend_log_transactions = []
        self.daily_user_spend_transactions = {}
        self.tool_usage_transactions = []
        self.model_usage_transactions = []
        self.autorouter_turn_transactions = []
        self.baseline_accounting_transactions = []
        self.baseline_accounting_lock = asyncio.Lock()
        self.spend_log_flush_requested = None
        self.db.tx = MagicMock()
        self.db.tx.return_value.__aenter__ = AsyncMock(return_value=self.db)
        self.db.tx.return_value.__aexit__ = AsyncMock(return_value=None)
        self.db.query_raw.return_value = []

        # Add locks for the transaction queues (matches real PrismaClient)
        self._spend_log_transactions_lock = asyncio.Lock()
        self.spend_log_write_lock = asyncio.Lock()
        self._tool_usage_transactions_lock = asyncio.Lock()
        self._model_usage_transactions_lock = asyncio.Lock()
        self._autorouter_turn_transactions_lock = asyncio.Lock()

    def jsonify_object(self, obj):
        return obj

    def add_spend_log_transaction_to_daily_user_transaction(self, payload):
        # Mock implementation
        pass


def create_mock_proxy_logging():
    print("creating mock proxy logging")
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.failure_handler = AsyncMock()
    proxy_logging_obj.db_spend_update_writer = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler = AsyncMock()
    print("returning proxy logging obj")
    return proxy_logging_obj


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ConnectError("Failed to connect"),
        httpx.ReadError("Failed to read response"),
        httpx.ReadTimeout("Request timed out"),
    ],
)
async def test_update_spend_logs_connection_errors(error_type):
    """Test retry mechanism for different connection error types"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Create AsyncMock for db_spend_update_writer
    proxy_logging_obj.db_spend_update_writer = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler = AsyncMock()

    # Add test spend logs
    prisma_client.spend_log_transactions = [
        {"id": "1", "spend": 10},
        {"id": "2", "spend": 20},
    ]

    # Mock the database to fail with connection error twice then succeed
    create_many_mock = AsyncMock()
    create_many_mock.side_effect = [
        error_type,  # First attempt fails
        error_type,  # Second attempt fails
        error_type,  # Third attempt fails
        None,  # Fourth attempt succeeds
    ]

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute
    await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify
    assert create_many_mock.call_count == 4  # Should have tried 3 times
    assert len(prisma_client.spend_log_transactions) == 0  # Should have cleared after success


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ConnectError("Failed to connect"),
        httpx.ReadError("Failed to read response"),
        httpx.ReadTimeout("Request timed out"),
    ],
)
async def test_update_spend_logs_max_retries_exceeded(error_type):
    """Test that each connection error type properly fails after max retries"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Add test spend logs
    prisma_client.spend_log_transactions = [
        {"id": "1", "spend": 10},
        {"id": "2", "spend": 20},
    ]

    # Mock the database to always fail
    create_many_mock = AsyncMock(side_effect=error_type)

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute and verify it raises after max retries
    with pytest.raises(type(error_type)) as exc_info:
        await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify error message matches
    assert str(exc_info.value) == str(error_type)
    # Verify retry attempts (initial try + 4 retries)
    assert create_many_mock.call_count == 4

    await asyncio.sleep(2)
    # Verify failure handler was called
    assert proxy_logging_obj.failure_handler.call_count == 1


@pytest.mark.asyncio
async def test_update_spend_logs_non_connection_error():
    """Test handling of non-connection related errors"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Add test spend logs
    prisma_client.spend_log_transactions = [
        {"id": "1", "spend": 10},
        {"id": "2", "spend": 20},
    ]

    # Mock a different type of error (not connection-related)
    unexpected_error = ValueError("Unexpected database error")
    create_many_mock = AsyncMock(side_effect=unexpected_error)

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute and verify it raises immediately without retrying
    with pytest.raises(ValueError, match="Unexpected database error") as exc_info:
        await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify error message
    assert str(exc_info.value) == "Unexpected database error"
    # Verify only tried once (no retries for non-connection errors)
    assert create_many_mock.call_count == 1
    # Verify failure handler was called
    assert proxy_logging_obj.failure_handler.called


@pytest.mark.asyncio
async def test_update_spend_logs_exponential_backoff():
    """Test that exponential backoff is working correctly"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Add test spend logs
    prisma_client.spend_log_transactions = [{"id": "1", "spend": 10}]

    # Track sleep times
    sleep_times = []

    # Mock asyncio.sleep to track delay times
    async def mock_sleep(seconds):
        sleep_times.append(seconds)

    # Mock the database to fail with connection errors
    create_many_mock = AsyncMock(
        side_effect=[
            httpx.ConnectError("Failed to connect"),  # First attempt
            httpx.ConnectError("Failed to connect"),  # Second attempt
            None,  # Third attempt succeeds
        ]
    )

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Apply mocks
    with patch("asyncio.sleep", mock_sleep):
        await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify exponential backoff
    assert len(sleep_times) == 2  # Should have slept twice
    assert sleep_times[0] >= 1 and sleep_times[0] <= 2  # First retry after 2^0~2^1 seconds
    assert sleep_times[1] >= 2 and sleep_times[1] <= 4  # Second retry after 2^1~2^2 seconds


@pytest.mark.asyncio
async def test_update_spend_logs_multiple_batches_success():
    """
    Test successful processing of multiple batches of spend logs

    Code sets batch size to 1000. This test creates 1500 logs, so it should make 2 batches.
    """
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Create 1500 test spend logs (1.5x BATCH_SIZE)
    prisma_client.spend_log_transactions = [{"id": str(i), "spend": 10} for i in range(1500)]

    create_many_mock = AsyncMock(return_value=None)
    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute
    await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify
    assert create_many_mock.call_count == _statements_for(1500)

    # No statement may exceed the row budget, which is what bounds the query
    # engine's resident memory.
    batches = [call[1]["data"] for call in create_many_mock.call_args_list]
    assert all(len(batch) <= SPEND_LOG_WRITE_BATCH_MAX_ROWS for batch in batches)

    # Every row is written exactly once and in order, whatever the split.
    written_ids = [item["id"] for batch in batches for item in batch]
    assert written_ids == [str(i) for i in range(1500)]

    # Verify all logs were processed
    assert len(prisma_client.spend_log_transactions) == 0


@pytest.mark.asyncio
async def test_update_spend_logs_multiple_batches_with_failure():
    """
    Test processing of multiple batches where one batch fails.
    Creates 4000 logs (4 batches) with one batch failing but eventually succeeding after retry.
    """
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Create 4000 test spend logs (4x BATCH_SIZE)
    prisma_client.spend_log_transactions = [{"id": str(i), "spend": 10} for i in range(4000)]

    # Mock to fail on second batch first attempt, then succeed
    call_count = 0

    async def create_many_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        # Fail on the second batch's first attempt
        if call_count == 2:
            raise httpx.ConnectError("Failed to connect")
        return None

    create_many_mock = AsyncMock(side_effect=create_many_side_effect)
    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute
    await update_spend(prisma_client, None, proxy_logging_obj)

    # The first attempt aborts on its second statement, then the whole flush
    # replays, so the total is those two calls plus one complete pass.
    assert create_many_mock.call_count == 2 + _statements_for(4000)

    # Verify all batches were processed
    all_processed_logs = []
    for call in create_many_mock.call_args_list:
        all_processed_logs.extend(call[1]["data"])

    # Verify all IDs were processed
    processed_ids = {item["id"] for item in all_processed_logs}

    # these should have ids 0-3999
    print("all processed ids", sorted(processed_ids, key=int))
    expected_ids = {str(i) for i in range(4000)}
    assert processed_ids == expected_ids

    # Verify all logs were cleared from transactions
    assert len(prisma_client.spend_log_transactions) == 0


@pytest.mark.asyncio
async def test_update_spend_logs_drops_row_prisma_cannot_encode_and_keeps_the_rest():
    """A row the Prisma engine cannot utf-8 encode (a lone surrogate in a string
    column) is permanent, so it must be dropped like any other poison row instead
    of being requeued at the head of the queue where it would block every tenant."""
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()
    prisma_client.spend_log_transactions = [
        {"request_id": "good-1", "request_tags": '["a"]'},
        {"request_id": "poison", "request_tags": "\ud800"},
        {"request_id": "good-2", "request_tags": '["b"]'},
    ]
    written = []

    async def create_many(data, skip_duplicates):
        if any("\ud800" in row["request_tags"] for row in data):
            raise UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")
        written.extend(row["request_id"] for row in data)

    prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=create_many)

    await update_spend(prisma_client, None, proxy_logging_obj)

    assert sorted(written) == ["good-1", "good-2"]
    assert prisma_client.spend_log_transactions == []
    proxy_logging_obj.failure_handler.assert_not_called()


def _raw_query_error(sqlstate: str):
    from prisma.errors import RawQueryError

    return RawQueryError(
        {
            "user_facing_error": {
                "error_code": "P2010",
                "meta": {"code": sqlstate, "message": f"ERROR: sqlstate {sqlstate}"},
            }
        }
    )


class _RawInsertRepository:
    def __init__(self, sqlstate: str):
        self.sqlstate = sqlstate
        self.written: list[str] = []

    async def insert_with_deliveries(self, rows):
        if any(row["request_id"] == "rejected" for row in rows):
            raise _raw_query_error(self.sqlstate)
        self.written.extend(str(row["request_id"]) for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlstate", ["22P02", "23502", "54000"])
async def test_raw_insert_rejecting_one_row_drops_only_that_row(sqlstate):
    from litellm.proxy.utils import _create_spend_logs_with_poison_isolation

    repo = _RawInsertRepository(sqlstate)
    rows = [{"request_id": "good-1"}, {"request_id": "rejected"}, {"request_id": "good-2"}]

    await _create_spend_logs_with_poison_isolation(repo, rows, failure_budget=10)

    assert sorted(repo.written) == ["good-1", "good-2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlstate", ["42P01", "40P01"])
async def test_systemic_raw_insert_failure_is_retried_not_dropped(sqlstate):
    from prisma.errors import RawQueryError

    from litellm.proxy.utils import _create_spend_logs_with_poison_isolation

    repo = _RawInsertRepository(sqlstate)
    rows = [{"request_id": "good-1"}, {"request_id": "rejected"}]

    with pytest.raises(RawQueryError):
        await _create_spend_logs_with_poison_isolation(repo, rows, failure_budget=10)

    assert repo.written == []


class _RescueFailsRepository:
    def __init__(self, rescue_error: Exception):
        self.rescue_error = rescue_error

    async def insert_with_deliveries(self, rows):
        if any(row.get("accounting_log_disabled") for row in rows):
            raise self.rescue_error
        if any(row["request_id"] == "rejected" for row in rows):
            raise _raw_query_error("22P02")


@pytest.mark.asyncio
async def test_a_transient_failure_while_keeping_a_dropped_rows_charge_is_retried_not_lost():
    from litellm.proxy.utils import _create_spend_logs_with_poison_isolation

    outage = _raw_query_error("40P01")
    repo = _RescueFailsRepository(outage)
    rows = [{"request_id": "rejected", "accounting_payload": {"spend": 1.0}}]

    with pytest.raises(type(outage)):
        await _create_spend_logs_with_poison_isolation(repo, rows, failure_budget=10)


@pytest.mark.asyncio
async def test_a_rejected_charge_rescue_is_logged_and_the_flush_continues():
    from litellm.proxy.utils import _create_spend_logs_with_poison_isolation

    repo = _RescueFailsRepository(_raw_query_error("22P02"))
    rows = [{"request_id": "rejected", "accounting_payload": {"spend": 1.0}}]

    assert await _create_spend_logs_with_poison_isolation(repo, rows, failure_budget=10) == 9


class _StatementValidatingRepository:
    def __init__(self):
        self.written = []
        self.charges_kept = []

    async def insert_with_deliveries(self, rows):
        from litellm.repositories.spend_log_insert import spend_log_insert_statement

        if all(row.get("accounting_log_disabled") for row in rows):
            self.charges_kept.extend(row["request_id"] for row in rows)
            return len(rows)
        spend_log_insert_statement(rows)
        self.written.extend(row["request_id"] for row in rows)
        return len(rows)


@pytest.mark.asyncio
async def test_a_row_the_insert_statement_rejects_is_dropped_alone_and_keeps_its_charge():
    from litellm.proxy.utils import _create_spend_logs_with_poison_isolation

    repo = _StatementValidatingRepository()
    rows = [
        {"request_id": "good", "call_type": "acompletion", "messages": "[]"},
        {
            "request_id": "unparseable",
            "call_type": "acompletion",
            "messages": "not json",
            "accounting_payload": {"spend": 1.0},
        },
    ]

    assert await _create_spend_logs_with_poison_isolation(repo, rows, failure_budget=10) < 10
    assert repo.written == ["good"]
    assert repo.charges_kept == ["unparseable"]


class _RecordingLedgerTransaction:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []  # mutable-ok: records statements
        self.queried: list[tuple[str, tuple[object, ...]]] = []  # mutable-ok: records statements

    async def query_raw(self, query: str, *args: object) -> list[dict[str, object]]:
        self.queried.append((query, args))
        if 'INSERT INTO "LiteLLM_SpendLogDelivery"' in query:
            return [{"delivery_id": row["delivery_id"]} for row in json.loads(str(args[0]))]
        return []

    async def execute_raw(self, query: str, *args: object) -> int:
        self.executed.append((query, args))
        return 1

    def inserted(self, table: str) -> list[dict[str, object]]:
        batches = [json.loads(str(args[0])) for query, args in self.executed if f'INSERT INTO "{table}"' in query]
        return [row for batch in batches for row in batch]  # comprehension-ok: flattens recorded batches

    def local_spend_rows(self) -> list[dict[str, object]]:
        batches = [
            json.loads(str(args[0])) for query, args in self.queried if 'INSERT INTO "LiteLLM_SpendLogs"' in query
        ]
        return [row for batch in batches for row in batch]  # comprehension-ok: flattens recorded batches


class _LedgerWriter:
    def __init__(self, transaction: _RecordingLedgerTransaction) -> None:
        self.transaction = transaction

    @contextlib.asynccontextmanager
    async def tx(self, *, timeout: object) -> AsyncIterator[_RecordingLedgerTransaction]:
        yield self.transaction


class _FlushClient:
    """The parts of PrismaClient the spend log flush uses, with its real row normalization."""

    def __init__(self, transaction: _RecordingLedgerTransaction) -> None:
        self.writer_db = _LedgerWriter(transaction)
        self.spend_log_transactions: list[dict[str, object]] = []  # mutable-ok: the flush queue
        self._spend_log_transactions_lock = asyncio.Lock()

    def jsonify_object(self, data: dict[str, object]) -> dict[str, object]:
        from litellm.proxy.utils import PrismaClient

        return PrismaClient.jsonify_object(self, data)  # pyright: ignore[reportArgumentType]  # the method uses no client state


_START: Final = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
_BODY: Final = json.dumps([{"role": "user", "content": "雪 " * 2000}], ensure_ascii=False)


def _queued_row(delivery_id: str, *, envelope: object, hidden: bool = False) -> dict[str, object]:
    from litellm.proxy.utils import accounting_only_spend_log

    accounting: Final = {
        "request_id": f"request-{delivery_id}",
        "call_type": "acompletion",
        "api_key": "hashed-key",
        "spend": 0.25,
        "startTime": _START.isoformat(),
        "team_id": "team",
        "user": "user",
        "model": "gpt-5-mini",
    }
    row: Final = {
        "request_id": f"request-{delivery_id}",
        "call_type": "acompletion",
        "api_key": "hashed-key",
        "spend": 0.25,
        "startTime": _START.isoformat(),
        "endTime": _START.isoformat(),
        "messages": "{}",
        "metadata": "{}",
        "accounting_delivery_id": delivery_id,
        "accounting_payload": accounting,
        "lens_export": envelope,
    }
    return dict(accounting_only_spend_log(row)) if hidden else {**row, "accounting_log_disabled": False}


def _envelope(request_id: str) -> dict[str, object]:
    return {
        "record": {
            "request_id": request_id,
            "spend": 0.25,
            "team_id": "team",
            "api_key": "hashed-key",
            "start_time": int(_START.timestamp() * 1000),
            "end_time": int(_START.timestamp() * 1000),
            "messages": "preview",
        },
        "payloads": [
            {
                "field": "messages",
                "team_id": "team",
                "api_key": "hashed-key",
                "event_time_ms": int(_START.timestamp() * 1000),
                "content": _BODY,
                "bytes": len(_BODY.encode()),
                "sha256": hashlib.sha256(_BODY.encode()).hexdigest(),
            }
        ],
    }


@pytest.mark.asyncio
async def test_local_spend_flush_commits_lens_events_and_bodies_with_the_charge() -> None:
    from litellm.proxy.utils import ProxyUpdateSpend

    transaction: Final = _RecordingLedgerTransaction()
    rows: Final = [
        _queued_row("visible", envelope=_envelope("request-visible")),
        _queued_row("visible", envelope=_envelope("request-visible")),
        _queued_row("hidden", envelope=_envelope("request-hidden"), hidden=True),
        _queued_row("malformed", envelope="{not json"),
    ]

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=_FlushClient(transaction),  # pyright: ignore[reportArgumentType]  # flush-facing client
        db_writer_client=None,
        proxy_logging_obj=MagicMock(),
        logs_to_process=rows,
    )

    events: Final = {row["event_id"]: row for row in transaction.inserted("LiteLLM_LensLedgerExport")}
    assert sorted(events) == ["hidden", "malformed", "visible"]
    assert events["visible"]["event"]["request_id"] == "request-visible"
    assert events["hidden"]["event"]["request_id"] == "request-hidden"
    assert events["visible"]["event"]["spend"] == 0.25
    assert events["visible"]["event"]["start_time"] == int(_START.timestamp() * 1000)
    assert events["malformed"]["event"] == {
        "call_type": "acompletion",
        "api_key": "hashed-key",
        "team_id": "team",
        "user": "user",
        "model": "gpt-5-mini",
        "messages": "",
        "response": "",
        "metadata": "",
        "error_str": "",
        "request_id": "request-malformed",
        "spend": 0.25,
        "start_time": int(_START.timestamp() * 1000),
        "end_time": int(_START.timestamp() * 1000),
    }
    bodies: Final = transaction.inserted("LiteLLM_LensLedgerPayload")
    assert [(body["event_id"], body["content"]) for body in bodies] == [("hidden", _BODY), ("visible", _BODY)]
    local: Final = transaction.local_spend_rows()
    assert sorted(row["request_id"] for row in local) == ["request-malformed", "request-visible"]
    assert "雪" not in json.dumps(local, ensure_ascii=False)
    assert "lens_export" not in json.dumps(local)


@pytest.mark.asyncio
async def test_spend_logs_url_flush_commits_the_native_lens_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.utils import ProxyUpdateSpend

    transaction: Final = _RecordingLedgerTransaction()
    writer_client: Final = MagicMock()
    writer_client.post = AsyncMock(return_value=MagicMock(status_code=200))
    monkeypatch.setenv("SPEND_LOGS_URL", "http://spend-logs.test")

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=_FlushClient(transaction),  # pyright: ignore[reportArgumentType]  # flush-facing client
        db_writer_client=writer_client,
        proxy_logging_obj=MagicMock(),
        logs_to_process=[_queued_row("visible", envelope=_envelope("request-visible"))],
    )

    [event] = transaction.inserted("LiteLLM_LensLedgerExport")
    assert event["event_id"] == "visible"
    assert event["event"]["spend"] == 0.25
    posted: Final = json.loads(writer_client.post.await_args.kwargs["data"])
    assert "lens_export" not in posted[0]

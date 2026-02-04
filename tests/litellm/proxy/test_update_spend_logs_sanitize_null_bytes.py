import asyncio
from unittest.mock import AsyncMock

import pytest

from litellm.proxy.utils import ProxyUpdateSpend


@pytest.mark.asyncio
async def test_update_spend_logs_sanitizes_null_bytes_before_db_insert(monkeypatch):
    """Regression test for Postgres 22P05 (NULL byte) failures.

    Ensures spend logs are sanitized before JSON serialization / DB writes.
    """

    # Force DB write path (no external writer).
    monkeypatch.delenv("SPEND_LOGS_URL", raising=False)

    create_many = AsyncMock()

    class _DummySpendLogsTable:
        def __init__(self, create_many_mock: AsyncMock):
            self.create_many = create_many_mock

    class _DummyDB:
        def __init__(self, create_many_mock: AsyncMock):
            self.litellm_spendlogs = _DummySpendLogsTable(create_many_mock)

    class _DummyPrismaClient:
        _spend_log_transactions_lock = asyncio.Lock()

        def __init__(self):
            self.db = _DummyDB(create_many)
            self.spend_log_transactions = [
                {
                    "request": "hello\x00world",
                    "metadata": {"nested": "a\x00b"},
                }
            ]

        def jsonify_object(self, data: dict) -> dict:
            # Keep behavior minimal; `update_spend_logs` should have already sanitized strings.
            return data

    prisma_client = _DummyPrismaClient()
    proxy_logging_obj = AsyncMock()

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=prisma_client,  # type: ignore[arg-type]
        db_writer_client=None,
        proxy_logging_obj=proxy_logging_obj,  # type: ignore[arg-type]
    )

    assert create_many.await_count == 1
    kwargs = create_many.await_args.kwargs
    written = kwargs["data"]

    assert written[0]["request"] == "helloworld"
    assert written[0]["metadata"]["nested"] == "ab"

"""Tests for unique request_id stamping in ``ProxyUpdateSpend.update_spend_logs``.

Upstream providers may reuse short request IDs (e.g. ``chatcmpl-424``).
Because ``request_id`` is the LiteLLM_SpendLogs primary key and the batch
insert uses ``skip_duplicates=True``, reused IDs previously caused spend
logs to be silently dropped.

``update_spend_logs`` now resolves collisions before writing:
1. The FIRST entry with a given request_id keeps its ORIGINAL id
   (backward compatible - exact-match lookups keep working).
2. Subsequent colliding entries get a deterministic content-hash suffix,
   so both rows are preserved.
3. Retries / re-enqueues of identical batches regenerate the SAME ids,
   letting ``skip_duplicates=True`` dedupe already-committed rows.
"""

import asyncio
import os
import sys
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.utils import ProxyUpdateSpend


def _make_mock_prisma_client() -> MagicMock:
    client = MagicMock()
    client.spend_log_transactions = []
    client._spend_log_transactions_lock = asyncio.Lock()
    client.jsonify_object = lambda data: data
    client.db.litellm_spendlogs.create_many = AsyncMock()
    return client


def _make_spend_log_row(
    request_id: str = "req-1", spend: float = 0.01, **overrides: Any
) -> Dict[str, Any]:
    row = {
        "request_id": request_id,
        "spend": spend,
        "model": "gpt-4o-mini",
        "user": "user-1",
        "api_key": "hashed-key",
        "startTime": "2026-06-02T00:00:00Z",
        "endTime": "2026-06-02T00:00:01Z",
        "metadata": {},
    }
    row.update(overrides)
    return row


async def _run_update_spend_logs(
    mock_prisma_client: MagicMock, logs: List[Dict[str, Any]]
) -> List[str]:
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock()
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )
    kwargs = mock_prisma_client.db.litellm_spendlogs.create_many.await_args.kwargs
    assert kwargs["skip_duplicates"] is True
    return [row["request_id"] for row in kwargs["data"]]


@pytest.mark.asyncio
async def test_update_spend_logs_preserves_colliding_request_ids():
    """Two logs sharing an upstream request_id but with different content
    must be written under distinct request_ids (the original bug: one of
    them was silently dropped by skip_duplicates=True). The FIRST entry
    keeps its original id for backward compatibility."""
    mock_prisma_client = _make_mock_prisma_client()
    request_ids = await _run_update_spend_logs(
        mock_prisma_client,
        [
            _make_spend_log_row(request_id="chatcmpl-424", spend=1.0),
            _make_spend_log_row(request_id="chatcmpl-424", spend=2.0),
        ],
    )
    assert len(set(request_ids)) == 2
    assert request_ids[0] == "chatcmpl-424"  # first keeps original id
    assert request_ids[1].startswith("chatcmpl-424-")  # collision suffixed


@pytest.mark.asyncio
async def test_update_spend_logs_keeps_unique_request_ids_unchanged():
    """Backward compatibility: entries whose upstream request_ids are
    already unique must be written with their ORIGINAL ids, so exact-match
    queries, UI links, and external integrations keep working."""
    mock_prisma_client = _make_mock_prisma_client()
    request_ids = await _run_update_spend_logs(
        mock_prisma_client,
        [
            _make_spend_log_row(request_id="chatcmpl-abc123", spend=1.0),
            _make_spend_log_row(request_id="chatcmpl-def456", spend=2.0),
        ],
    )
    assert request_ids == ["chatcmpl-abc123", "chatcmpl-def456"]


@pytest.mark.asyncio
async def test_update_spend_logs_request_id_suffix_is_deterministic():
    """Identical colliding batches processed twice (e.g. a retried or
    re-enqueued batch) must be stamped with the SAME request_ids so that
    skip_duplicates=True can dedupe already-committed rows."""
    mock_prisma_client = _make_mock_prisma_client()

    def _colliding_batch() -> List[Dict[str, Any]]:
        return [
            _make_spend_log_row(request_id="chatcmpl-424", spend=1.0),
            _make_spend_log_row(request_id="chatcmpl-424", spend=2.0),
        ]

    ids_first = await _run_update_spend_logs(mock_prisma_client, _colliding_batch())
    ids_second = await _run_update_spend_logs(mock_prisma_client, _colliding_batch())
    assert ids_first == ids_second


@pytest.mark.asyncio
async def test_update_spend_logs_stamps_ids_before_retry_loop():
    """If a DB connection error fires after a partial commit, the retry pass
    must reuse the SAME stamped ids - otherwise already-committed batches
    would be rewritten under new ids, duplicating spend entries."""
    import httpx

    mock_prisma_client = _make_mock_prisma_client()
    seen_ids_per_attempt: List[List[str]] = []

    async def _create_many(data: List[Dict[str, Any]], skip_duplicates: bool) -> None:
        seen_ids_per_attempt.append([row["request_id"] for row in data])
        if len(seen_ids_per_attempt) == 1:
            raise httpx.ReadError("connection dropped mid-flush")

    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(
        side_effect=_create_many
    )
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=1,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=[
            _make_spend_log_row(request_id="chatcmpl-424", spend=1.0),
            _make_spend_log_row(request_id="chatcmpl-424", spend=2.0),
        ],
    )
    assert len(seen_ids_per_attempt) == 2
    assert seen_ids_per_attempt[0] == seen_ids_per_attempt[1]


@pytest.mark.asyncio
async def test_update_spend_logs_generates_id_when_request_id_missing():
    """Entries without any request_id should still get a usable id."""
    mock_prisma_client = _make_mock_prisma_client()
    row = _make_spend_log_row()
    row.pop("request_id")
    request_ids = await _run_update_spend_logs(mock_prisma_client, [row])
    assert len(request_ids) == 1
    assert request_ids[0]  # non-empty

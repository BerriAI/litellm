"""
Tests for TraceReceiver.ingest (litellm/tracing/receiver.py) with a fake storage.
"""

import asyncio
import gzip
import threading
from collections.abc import AsyncIterator
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.rust_bridge.trace.generated.types import TraceScope
from litellm.tracing import Tenant, TraceReceiver, TracingPayloadTooLargeError
from litellm.tracing import otlp_http
from litellm.tracing.otlp_http import InvalidOTLPPayloadError
from litellm.tracing.receiver import TracingOverloadedError

TENANT = Tenant(team_id="team-research", api_key_hash="hashed-key", org_id="org-1", user_id="user-1")


def _fake_storage() -> MagicMock:
    storage = MagicMock()
    storage.ingest = AsyncMock(return_value=6)
    storage.get_trace = AsyncMock(return_value=None)
    return storage


@pytest.mark.asyncio
@pytest.mark.parametrize("logs", (False, True))
async def test_ingest_decompresses_and_passes_the_authenticated_tenant(logs: bool) -> None:
    storage: Final = _fake_storage()
    count: Final = await TraceReceiver(storage).ingest(
        gzip.compress(b"export"), "application/json", "gzip", TENANT, logs=logs
    )
    assert count == 6
    storage.ingest.assert_awaited_once_with(b"export", "application/json", TENANT, logs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (OverflowError("ClickHouse insert exceeds the encoded size limit"), TracingPayloadTooLargeError),
        (ValueError("invalid OTLP trace payload"), InvalidOTLPPayloadError),
        (RuntimeError("ClickHouse unavailable"), RuntimeError),
    ),
)
async def test_storage_failures_map_to_ingest_errors(failure: Exception, expected: type[Exception]) -> None:
    storage: Final = _fake_storage()
    storage.ingest.side_effect = failure
    with pytest.raises(expected, match=str(failure)):
        await TraceReceiver(storage).ingest(b"{}", "application/json", None, TENANT)


@pytest.mark.asyncio
async def test_ingest_rejects_oversized_body_before_storage() -> None:
    storage: Final = _fake_storage()
    with patch.object(otlp_http, "OTLP_MAX_BODY_BYTES", 10):
        with pytest.raises(TracingPayloadTooLargeError):
            await TraceReceiver(storage).ingest(b"x" * 20, "application/json", None, TENANT)
    storage.ingest.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor,page_size", ((None, None), ("next", 200)))
async def test_reads_delegate_to_storage(cursor: str | None, page_size: int | None) -> None:
    storage: Final = _fake_storage()
    scope: Final[TraceScope] = {"all_teams": 0, "user_id": "", "team_ids": ("team-research",)}
    assert await TraceReceiver(storage).get_trace("t1", scope, "", cursor, page_size) is None
    storage.get_trace.assert_awaited_once_with("t1", scope, "", cursor, page_size)


@pytest.mark.asyncio
async def test_cancelled_request_keeps_its_worker_slot_until_decompression_finishes() -> None:
    loop: Final = asyncio.get_running_loop()
    owner: Final = threading.get_ident()
    started: Final = asyncio.Event()
    stored: Final = asyncio.Event()
    release: Final = threading.Event()

    def decompressor(body: bytes, content_encoding: str | None) -> bytes:
        assert threading.get_ident() != owner
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return b""

    storage: Final = _fake_storage()

    async def store(payload: bytes, content_type: str | None, tenant: Tenant, logs: bool) -> int:
        stored.set()
        return 0

    storage.ingest.side_effect = store
    tracing: Final = TraceReceiver(storage, max_concurrent_ingests=1, decompressor=decompressor)
    pending: Final = asyncio.create_task(tracing.ingest(b"small gzip", None, "gzip", TENANT))
    try:
        await asyncio.wait_for(started.wait(), 5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        with pytest.raises(TracingOverloadedError):
            await tracing.ingest(b"", None, None, TENANT)
    finally:
        release.set()
        await asyncio.wait_for(stored.wait(), 5)
        await asyncio.sleep(0)
    assert await tracing.ingest(b"", None, None, TENANT) == 0


@pytest.mark.asyncio
async def test_expired_upload_releases_ingestion_slot_without_writing() -> None:
    async def unfinished_body() -> AsyncIterator[bytes]:
        await asyncio.Event().wait()
        yield b""

    storage: Final = _fake_storage()
    receiver: Final = TraceReceiver(storage, max_concurrent_ingests=1, body_read_timeout=0)
    with pytest.raises(TracingOverloadedError, match="upload timed out"):
        await receiver.ingest(unfinished_body(), "application/json", None, TENANT)
    storage.ingest.assert_not_awaited()
    assert await receiver.ingest(b"{}", "application/json", None, TENANT) == 6

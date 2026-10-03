"""
Tests for TraceReceiver.ingest (litellm/tracing/receiver.py) with a fake store.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

from litellm.tracing import Tenant, TraceReceiver, TracingPayloadTooLargeError
from litellm.tracing import receiver as receiver_module
from litellm.tracing.types import TraceScope

pytestmark = pytest.mark.requires_rust_extension

FIXTURE = Path(__file__).parent / "fixtures" / "langsmith_deep_agent_export.json"
TENANT = Tenant(team_id="team-research", api_key_hash="hashed-key", org_id="org-1", user_id="user-1")


def _fake_store() -> MagicMock:
    store = MagicMock()
    store.insert_spans = AsyncMock()
    store.get_trace = AsyncMock(return_value=None)
    return store


def _spoofed_export() -> bytes:
    """A client that tries to claim another team via resource attributes."""
    resource_spans = ResourceSpans(scope_spans=[ScopeSpans(spans=[Span(trace_id=b"\x01" * 16, span_id=b"\x02" * 8)])])
    resource_spans.resource.attributes.extend(
        [
            KeyValue(key="service.name", value=AnyValue(string_value="svc")),
            KeyValue(key="litellm.team_id", value=AnyValue(string_value="someone-elses-team")),
            KeyValue(key="litellm.api_key_hash", value=AnyValue(string_value="someone-elses-key")),
            KeyValue(key="litellm.user_id", value=AnyValue(string_value="someone-elses-user")),
        ]
    )
    return ExportTraceServiceRequest(resource_spans=[resource_spans]).SerializeToString()


@pytest.mark.asyncio
async def test_ingest_returns_span_count_and_writes_stamped_rows():
    store = _fake_store()
    count = await TraceReceiver(store).ingest(FIXTURE.read_bytes(), "application/json", None, TENANT)
    assert count == 6
    (rows,) = store.insert_spans.await_args.args
    assert len(rows) == 6
    for row in rows:
        assert (row["TeamId"], row["ApiKeyHash"]) == ("team-research", "hashed-key")
        assert row["ResourceAttributes"]["litellm.org_id"] == "org-1"
        assert row["ResourceAttributes"]["service.name"] == "agent-demo"


@pytest.mark.asyncio
async def test_ingest_overwrites_client_supplied_tenant_attributes():
    store = _fake_store()
    await TraceReceiver(store).ingest(_spoofed_export(), "application/x-protobuf", None, TENANT)
    ((row,),) = store.insert_spans.await_args.args
    assert row["TeamId"] == "team-research"
    assert row["ResourceAttributes"]["litellm.team_id"] == "team-research"
    assert row["ResourceAttributes"]["litellm.api_key_hash"] == "hashed-key"
    assert row["UserId"] == TENANT.user_id
    assert row["ResourceAttributes"]["litellm.user_id"] == TENANT.user_id


@pytest.mark.asyncio
async def test_ingest_does_not_acknowledge_failed_clickhouse_write():
    store = _fake_store()
    store.insert_spans.side_effect = RuntimeError("ClickHouse unavailable")
    with pytest.raises(RuntimeError, match="ClickHouse unavailable"):
        await TraceReceiver(store).ingest(FIXTURE.read_bytes(), "application/json", None, TENANT)
    store.insert_spans.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_rejects_oversized_encoded_batch():
    store = _fake_store()
    store.insert_spans.side_effect = OverflowError("ClickHouse insert exceeds the encoded size limit")
    with pytest.raises(TracingPayloadTooLargeError, match="encoded size limit"):
        await TraceReceiver(store).ingest(FIXTURE.read_bytes(), "application/json", None, TENANT)


@pytest.mark.asyncio
async def test_ingest_rejects_oversized_body():
    store = _fake_store()
    with patch.object(receiver_module, "OTLP_MAX_BODY_BYTES", 10):
        with pytest.raises(TracingPayloadTooLargeError):
            await TraceReceiver(store).ingest(FIXTURE.read_bytes(), "application/json", None, TENANT)
    store.insert_spans.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_export_writes_nothing():
    store = _fake_store()
    assert await TraceReceiver(store).ingest(b"", "application/x-protobuf", None, TENANT) == 0
    store.insert_spans.assert_awaited_once_with(())


@pytest.mark.asyncio
async def test_reads_delegate_to_store():
    store = _fake_store()
    tracing = TraceReceiver(store)
    scope: Final[TraceScope] = {"all_teams": 0, "user_id": "", "team_ids": ("team-research",)}
    assert await tracing.get_trace("t1", scope) is None
    store.get_trace.assert_awaited_once_with("t1", scope, "")


@pytest.mark.asyncio
async def test_cancelled_request_keeps_its_worker_slot_until_decode_finishes():
    import asyncio
    import threading

    from litellm.tracing.receiver import TracingOverloadedError

    loop = asyncio.get_running_loop()
    owner = threading.get_ident()
    started = asyncio.Event()
    stored = asyncio.Event()
    release = threading.Event()

    def decoder(body, content_type, content_encoding):
        assert threading.get_ident() != owner
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return ()

    store = _fake_store()
    store.insert_spans.side_effect = lambda _: stored.set()
    tracing = TraceReceiver(store, max_concurrent_ingests=1, decoder=decoder)
    pending = asyncio.create_task(tracing.ingest(b"small gzip", None, "gzip", TENANT))
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
    from litellm.tracing.receiver import TracingOverloadedError

    async def unfinished_body() -> AsyncIterator[bytes]:
        await asyncio.Event().wait()
        yield b""

    store: Final = _fake_store()
    receiver: Final = TraceReceiver(store, max_concurrent_ingests=1, body_read_timeout=0)
    with pytest.raises(TracingOverloadedError, match="upload timed out"):
        await receiver.ingest(unfinished_body(), "application/json", None, TENANT)
    store.insert_spans.assert_not_awaited()
    assert await receiver.ingest(b"{}", "application/json", None, TENANT) == 0
    store.insert_spans.assert_awaited_once_with(())

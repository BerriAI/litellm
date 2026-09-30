"""
Tests for TraceReceiver.ingest (litellm/tracing/receiver.py) with a fake store.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

from litellm.tracing import Tenant, TraceReceiver, TracingPayloadTooLargeError
from litellm.tracing import receiver as receiver_module

pytestmark = pytest.mark.requires_rust_extension

FIXTURE = Path(__file__).parent / "fixtures" / "langsmith_deep_agent_export.json"
TENANT = Tenant(team_id="team-research", api_key_hash="hashed-key", org_id="org-1")


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
async def test_large_body_is_decoded_off_the_event_loop():
    store = _fake_store()
    with (
        patch.object(receiver_module, "OTLP_OFFLOAD_DECODE_BYTES", 0),
        patch.object(receiver_module.asyncio, "to_thread", wraps=receiver_module.asyncio.to_thread) as to_thread,
    ):
        count = await TraceReceiver(store).ingest(FIXTURE.read_bytes(), "application/json", None, TENANT)
    assert count == 6
    to_thread.assert_called_once()


@pytest.mark.asyncio
async def test_empty_export_writes_nothing():
    store = _fake_store()
    assert await TraceReceiver(store).ingest(b"", "application/x-protobuf", None, TENANT) == 0
    store.insert_spans.assert_awaited_once_with([])


@pytest.mark.asyncio
async def test_reads_delegate_to_store():
    store = _fake_store()
    tracing = TraceReceiver(store)
    scope = {"team_ids": ["team-research"], "api_key_hash": ""}
    assert await tracing.get_trace("t1", scope) is None  # type: ignore[arg-type]
    store.get_trace.assert_awaited_once_with("t1", scope)

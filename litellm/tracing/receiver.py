"""
`TraceReceiver`: the one entry point for agent tracing.

    tracing = TraceReceiver.from_env()          # or TraceReceiver(store=...)
    await tracing.start()                        # create tables if missing

    tracing.ingest(otlp_body, content_type, content_encoding, tenant)   # POST /v1/traces
    await tracing.list_traces(scope, start_ms, end_ms, cursor)          # GET  /v1/traces
    await tracing.get_trace(trace_id, scope)                            # GET  /v1/traces/{id}
    await tracing.get_span(trace_id, span_id, scope)                    # GET  /v1/traces/{id}/spans/{span_id}

The proxy endpoints are thin wrappers: auth -> build tenant/scope -> call one method.
"""

import asyncio
import os
from typing import Final

from litellm.constants import (
    AGENT_TRACING_RETENTION_DAYS,
    AGENT_TRACING_SPEND_LOG_RETENTION_DAYS,
    OTLP_MAX_BODY_BYTES,
    OTLP_OFFLOAD_DECODE_BYTES,
)
from litellm.integrations.clickhouse.clickhouse_client import ClickHouseClient
from litellm.integrations.clickhouse.schema import ensure_schema
from litellm.tracing.decode import decode_otlp
from litellm.tracing.markdown import trace_to_markdown
from litellm.tracing.store import ClickHouseTraceStore
from litellm.tracing.types import (
    SpanDetail,
    SpanRow,
    Trace,
    TracePage,
    TraceScope,
)


class TracingBackpressureError(Exception):
    """The span buffer is full. Callers should answer 429 so OTLP exporters retry."""


class TracingPayloadTooLargeError(Exception):
    pass


class Tenant:
    """Who sent the spans. Always taken from auth, never from span attributes."""

    def __init__(self, team_id: str, api_key_hash: str, org_id: str = ""):
        self.team_id = team_id
        self.api_key_hash = api_key_hash
        self.org_id = org_id

    def stamp(self, row: SpanRow) -> SpanRow:
        row["TeamId"] = self.team_id
        row["ApiKeyHash"] = self.api_key_hash
        row["ResourceAttributes"] = {
            **row["ResourceAttributes"],
            "litellm.team_id": self.team_id,
            "litellm.api_key_hash": self.api_key_hash,
            "litellm.org_id": self.org_id,
        }
        return row


class TraceReceiver:
    def __init__(self, store: ClickHouseTraceStore):
        self.store = store

    @classmethod
    def from_env(cls) -> "TraceReceiver":
        return cls(
            store=ClickHouseTraceStore(
                ClickHouseClient(
                    url=os.environ["CLICKHOUSE_URL"],
                    user=os.getenv("CLICKHOUSE_USER", "default"),
                    password=os.getenv("CLICKHOUSE_PASSWORD", ""),
                    database=os.getenv("CLICKHOUSE_DATABASE", "litellm"),
                )
            )
        )

    async def start(self) -> None:
        await ensure_schema(
            self.store.client,
            trace_retention_days=AGENT_TRACING_RETENTION_DAYS,
            spend_log_retention_days=AGENT_TRACING_SPEND_LOG_RETENTION_DAYS,
        )
        self.store.writer.start()

    async def flush(self) -> None:
        await self.store.flush()

    # ------------------------------------------------------------ write

    async def ingest(
        self,
        body: bytes,
        content_type: str | None,
        content_encoding: str | None,
        tenant: Tenant,
    ) -> int:
        """Decode an OTLP trace export, stamp the tenant, queue the spans. Returns span count."""
        if self.store.is_full():
            raise TracingBackpressureError()
        if len(body) > OTLP_MAX_BODY_BYTES:
            raise TracingPayloadTooLargeError(f"OTLP body exceeds {OTLP_MAX_BODY_BYTES} bytes")
        rows: Final = (
            await asyncio.to_thread(decode_otlp, body, content_type, content_encoding)
            if len(body) > OTLP_OFFLOAD_DECODE_BYTES
            else decode_otlp(body, content_type, content_encoding)
        )
        self.store.write_spans([tenant.stamp(r) for r in rows])
        return len(rows)

    # ------------------------------------------------------------ read

    async def list_traces(self, scope: TraceScope, start_ms: int, end_ms: int, cursor: str | None = None) -> TracePage:
        return await self.store.list_traces(scope, start_ms, end_ms, cursor)

    async def get_trace(self, trace_id: str, scope: TraceScope) -> Trace | None:
        return await self.store.get_trace(trace_id, scope)

    async def get_trace_markdown(self, trace_id: str, scope: TraceScope, span_id: str | None = None) -> str | None:
        """The trace (or the subtree under span_id) as Markdown, for pasting into Claude / Codex."""
        trace = await self.store.get_trace(trace_id, scope)
        if trace is None:
            return None
        return trace_to_markdown(trace, await self.store.get_span_io(trace_id, scope), span_id)

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope) -> SpanDetail | None:
        return await self.store.get_span(trace_id, span_id, scope)

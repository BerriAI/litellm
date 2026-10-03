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
from collections.abc import AsyncIterable, Callable, Mapping
from io import BytesIO
from threading import BoundedSemaphore
from types import MappingProxyType
from typing import Final

from litellm.constants import OTLP_MAX_BODY_BYTES, OTLP_MAX_CONCURRENT_INGESTS
from litellm.rust_bridge.traces import ClickHouseStorage
from litellm.tracing.config import trace_storage_config
from litellm.tracing.decode import OTLPPayloadTooLargeError, decode_otlp
from litellm.tracing.store import TraceStore
from litellm.tracing.types import (
    SpanDetail,
    SpanErrorPage,
    SpanRow,
    Trace,
    TracePage,
    TraceScope,
)


class TracingPayloadTooLargeError(Exception):
    pass


class TracingOverloadedError(RuntimeError):
    pass


class Tenant:
    """Who sent the spans. Always taken from auth, never from span attributes."""

    def __init__(self, team_id: str, api_key_hash: str, org_id: str = "", user_id: str = "") -> None:
        self.team_id = team_id
        self.api_key_hash = api_key_hash
        self.org_id = org_id
        self.user_id = user_id

    def stamp(self, row: SpanRow) -> SpanRow:
        return self.stamp_rows((row,))[0]

    def stamp_rows(self, rows: tuple[SpanRow, ...]) -> tuple[SpanRow, ...]:
        resources: Final = MappingProxyType({id(row["ResourceAttributes"]): row["ResourceAttributes"] for row in rows})
        stamped: Final = MappingProxyType(
            {
                identity: MappingProxyType(
                    {
                        **attributes,
                        "litellm.team_id": self.team_id,
                        "litellm.api_key_hash": self.api_key_hash,
                        "litellm.org_id": self.org_id,
                        "litellm.user_id": self.user_id,
                    }
                )
                for identity, attributes in resources.items()
            }
        )
        return tuple(self._stamp_row(row, stamped[id(row["ResourceAttributes"])]) for row in rows)

    def _stamp_row(self, row: SpanRow, resource: Mapping[str, str]) -> SpanRow:
        stamped: Final[SpanRow] = {
            **row,
            "TeamId": self.team_id,
            "ApiKeyHash": self.api_key_hash,
            "UserId": self.user_id,
            "ResourceAttributes": resource,
        }
        return stamped


class TraceReceiver:
    def __init__(
        self,
        store: TraceStore,
        max_concurrent_ingests: int = OTLP_MAX_CONCURRENT_INGESTS,
        decoder: Callable[[bytes, str | None, str | None], tuple[SpanRow, ...]] = decode_otlp,
        body_read_timeout: float = 30,
    ) -> None:
        if max_concurrent_ingests < 1:
            raise ValueError("OTLP ingestion concurrency must be positive")
        self.store = store
        self._decoder: Final = decoder
        self._body_read_timeout: Final = body_read_timeout
        self._ingest_slots: Final = BoundedSemaphore(max_concurrent_ingests)

    @classmethod
    def from_env(cls) -> "TraceReceiver":
        return cls.from_settings({})

    @classmethod
    def from_settings(cls, settings: Mapping[str, object]) -> "TraceReceiver":
        return cls(store=TraceStore(ClickHouseStorage(trace_storage_config(settings))))

    async def start(self) -> None:
        await self.store.storage.ensure_schema()

    async def ingest(
        self,
        body: bytes | AsyncIterable[bytes],
        content_type: str | None,
        content_encoding: str | None,
        tenant: Tenant,
    ) -> int:
        if not self._ingest_slots.acquire(blocking=False):
            raise TracingOverloadedError("OTLP ingestion is at capacity")
        task: Final = asyncio.create_task(self._ingest(body, content_type, content_encoding, tenant))
        task.add_done_callback(self._release_ingest)
        return await asyncio.shield(task)

    def _release_ingest(self, task: asyncio.Task[int]) -> None:
        self._ingest_slots.release()
        if not task.cancelled():
            task.exception()

    async def _ingest(
        self,
        body: bytes | AsyncIterable[bytes],
        content_type: str | None,
        content_encoding: str | None,
        tenant: Tenant,
    ) -> int:
        try:
            payload: Final = (
                body
                if isinstance(body, bytes)
                else await asyncio.wait_for(_read_body(body), timeout=self._body_read_timeout)
            )
        except asyncio.TimeoutError as error:
            raise TracingOverloadedError("OTLP body upload timed out") from error
        if len(payload) > OTLP_MAX_BODY_BYTES:
            raise TracingPayloadTooLargeError(f"OTLP body exceeds {OTLP_MAX_BODY_BYTES} bytes")
        try:
            rows: Final = await asyncio.to_thread(self._decoder, payload, content_type, content_encoding)
        except OTLPPayloadTooLargeError as error:
            raise TracingPayloadTooLargeError(str(error)) from error
        try:
            await self.store.insert_spans(tenant.stamp_rows(rows))
        except OverflowError as error:
            raise TracingPayloadTooLargeError(str(error)) from error
        return len(rows)

    async def list_traces(self, scope: TraceScope, start_ms: int, end_ms: int, cursor: str | None = None) -> TracePage:
        return await self.store.list_traces(scope, start_ms, end_ms, cursor)

    async def get_trace(self, trace_id: str, scope: TraceScope, trace_ref: str = "") -> Trace | None:
        return await self.store.get_trace(trace_id, scope, trace_ref)

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "") -> SpanDetail | None:
        return await self.store.get_span(trace_id, span_id, scope, trace_ref)

    async def get_span_error(
        self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "", cursor: str | None = None
    ) -> SpanErrorPage | None:
        return await self.store.get_span_error(trace_id, span_id, scope, trace_ref, cursor)


async def _read_body(chunks: AsyncIterable[bytes]) -> bytes:
    with BytesIO() as body:
        async for chunk in chunks:
            if body.tell() + len(chunk) > OTLP_MAX_BODY_BYTES:
                raise TracingPayloadTooLargeError(f"OTLP body exceeds {OTLP_MAX_BODY_BYTES} bytes")
            body.write(chunk)
        return body.getvalue()

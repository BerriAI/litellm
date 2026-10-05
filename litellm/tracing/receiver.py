"""
`TraceReceiver`: the one entry point for agent tracing.

    tracing = TraceReceiver.from_env()          # or TraceReceiver(storage=...)
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
from typing import Final

from litellm.constants import AGENT_TRACING_LIST_PAGE_SIZE, OTLP_MAX_BODY_BYTES, OTLP_MAX_CONCURRENT_INGESTS
from litellm.rust_bridge.trace.generated.types import SpanDetail, SpanErrorPage, Trace, TracePage, TraceScope
from litellm.rust_bridge.trace.storage import ClickHouseStorage, Tenant
from litellm.tracing.config import trace_storage_config
from litellm.tracing.otlp_http import InvalidOTLPPayloadError, TracingPayloadTooLargeError, decompress


class TracingOverloadedError(RuntimeError):
    pass


class TraceReceiver:
    def __init__(
        self,
        storage: ClickHouseStorage,
        max_concurrent_ingests: int = OTLP_MAX_CONCURRENT_INGESTS,
        decompressor: Callable[[bytes, str | None], bytes] = decompress,
        body_read_timeout: float = 30,
    ) -> None:
        if max_concurrent_ingests < 1:
            raise ValueError("OTLP ingestion concurrency must be positive")
        self.storage = storage
        self._decompressor: Final = decompressor
        self._body_read_timeout: Final = body_read_timeout
        self._ingest_slots: Final = BoundedSemaphore(max_concurrent_ingests)

    @classmethod
    def from_env(cls) -> "TraceReceiver":
        return cls.from_settings({})

    @classmethod
    def from_settings(cls, settings: Mapping[str, object]) -> "TraceReceiver":
        return cls(storage=ClickHouseStorage(trace_storage_config(settings)))

    async def start(self) -> None:
        await self.storage.ensure_schema()

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
            received: Final = (
                body
                if isinstance(body, bytes)
                else await asyncio.wait_for(_read_body(body), timeout=self._body_read_timeout)
            )
        except asyncio.TimeoutError as error:
            raise TracingOverloadedError("OTLP body upload timed out") from error
        payload: Final = await asyncio.to_thread(self._decompressor, received, content_encoding)
        try:
            return await self.storage.ingest(payload, content_type, tenant)
        except OverflowError as error:
            raise TracingPayloadTooLargeError(str(error)) from error
        except ValueError as error:
            raise InvalidOTLPPayloadError(str(error)) from error

    async def list_traces(self, scope: TraceScope, start_ms: int, end_ms: int, cursor: str | None = None) -> TracePage:
        return await self.storage.list_traces(scope, start_ms, end_ms, cursor, AGENT_TRACING_LIST_PAGE_SIZE)

    async def get_trace(
        self,
        trace_id: str,
        scope: TraceScope,
        trace_ref: str = "",
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Trace | None:
        return await self.storage.get_trace(trace_id, scope, trace_ref, cursor, page_size)

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "") -> SpanDetail | None:
        return await self.storage.get_span(trace_id, span_id, scope, trace_ref)

    async def get_span_error(
        self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "", cursor: str | None = None
    ) -> SpanErrorPage | None:
        return await self.storage.get_span_error(trace_id, span_id, scope, trace_ref, cursor)


async def _read_body(chunks: AsyncIterable[bytes]) -> bytes:
    with BytesIO() as body:
        async for chunk in chunks:
            if body.tell() + len(chunk) > OTLP_MAX_BODY_BYTES:
                raise TracingPayloadTooLargeError(f"OTLP body exceeds {OTLP_MAX_BODY_BYTES} bytes")
            body.write(chunk)
        return body.getvalue()

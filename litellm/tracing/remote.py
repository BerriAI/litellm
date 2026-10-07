import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

import httpx
from pydantic import JsonValue, TypeAdapter

from litellm.rust_bridge.trace.errors import TraceChanged
from litellm.rust_bridge.trace.generated.types import QueryScope, ReadQueryName, TraceScope

MAX_RESPONSE_BYTES: Final = 64 * 1024 * 1024
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True, repr=False)
class LensConnection:
    url: str
    token: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "LensConnection":
        url: Final = environ.get("LITELLM_LENS_URL", "").rstrip("/")
        token: Final = environ.get("LITELLM_LENS_SERVICE_TOKEN", "")
        parsed: Final = urlsplit(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Set LITELLM_LENS_URL to the Lens service URL")
        if len(token) < 32:
            raise ValueError("Set LITELLM_LENS_SERVICE_TOKEN to the same secret on LiteLLM and Lens")
        return cls(url, token)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(35, connect=3),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=10),
            follow_redirects=False,
        )


class RemoteTraceStore:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client: Final = client

    async def ensure_schema(self) -> None:
        return

    async def _read(self, request: Mapping[str, object]) -> JsonValue:
        try:
            async with self.client.stream("POST", "/internal/read", json=dict(request)) as response:
                if response.status_code == 400:
                    raise ValueError("Invalid trace query")
                if response.status_code == 409:
                    raise TraceChanged("Trace changed while paging; refresh the trace to continue")
                if response.status_code == 413:
                    raise OverflowError("Trace exceeds the interactive read budget")
                if response.status_code != 200:
                    raise RuntimeError("Lens trace storage is unavailable")
                return _JSON.validate_json(await bounded_response(response, MAX_RESPONSE_BYTES))
        except httpx.HTTPError as error:
            raise RuntimeError("Lens trace storage is unavailable") from error

    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        if table != "spend_logs":
            raise ValueError("Lens only accepts gateway request records on this endpoint")
        response: Final = await self.client.post("/internal/spend", json=tuple(dict(row) for row in rows))
        response.raise_for_status()

    async def ingest(
        self, payload: bytes, content_type: str | None, tenant: Mapping[str, str], logs: bool = False
    ) -> int:
        raise RuntimeError("Send OTLP directly to the Lens service")

    async def list_traces(
        self, scope: TraceScope, start_ms: int, end_ms: int, cursor: str | None, limit: int
    ) -> JsonValue:
        return await self._read(
            {
                "operation": "list",
                "scope": scope,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "cursor": cursor,
                "limit": limit,
            }
        )

    async def get_trace(
        self, trace_id: str, scope: TraceScope, trace_ref: str, cursor: str | None = None, page_size: int | None = None
    ) -> JsonValue:
        return await self._read(
            {
                "operation": "trace",
                "scope": scope,
                "trace_id": trace_id,
                "trace_ref": trace_ref,
                "cursor": cursor,
                "page_size": page_size,
            }
        )

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str) -> JsonValue:
        return await self._read(
            {
                "operation": "span",
                "scope": scope,
                "trace_id": trace_id,
                "trace_ref": trace_ref,
                "span_id": span_id,
            }
        )

    async def get_span_error(
        self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str, cursor: str | None
    ) -> JsonValue:
        return await self._read(
            {
                "operation": "span_error",
                "scope": scope,
                "trace_id": trace_id,
                "trace_ref": trace_ref,
                "span_id": span_id,
                "cursor": cursor,
            }
        )

    async def query_sql(self, sql: str, scope: QueryScope, secret: str) -> str:
        return json.dumps(await self._read({"operation": "sql", "sql": sql, "scope": scope}))

    async def query_help(self, scope: QueryScope, secret: str) -> JsonValue:
        return await self._read({"operation": "help", "scope": scope})

    async def query(self, name: ReadQueryName, parameters: Mapping[str, str | int | float | Sequence[str]]) -> str:
        return json.dumps(await self._read({"operation": "query", "name": name, "parameters": dict(parameters)}))


async def bounded_response(response: httpx.Response, limit: int) -> bytes:
    from io import BytesIO

    with BytesIO() as buffer:
        async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
            if buffer.tell() + len(chunk) > limit:
                raise RuntimeError("Lens response exceeds the size limit")
            buffer.write(chunk)
        return buffer.getvalue()

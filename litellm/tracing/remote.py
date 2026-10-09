import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final, NoReturn
from urllib.parse import urlsplit

import httpx
from pydantic import JsonValue, TypeAdapter
from typing_extensions import assert_never

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.tracing.errors import TraceChanged
from litellm.tracing.generated.types import QueryScope, ReadQueryName, TraceScope

MAX_RESPONSE_BYTES: Final = 64 * 1024 * 1024
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_INSERT_PATHS: Final[Mapping[str, str]] = {"spend_logs": "/internal/spend", "lens_feedback": "/internal/feedback"}


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

    def control_client(self) -> httpx.AsyncClient:
        return get_async_httpx_client(
            "lens-control",
            params={"timeout": httpx.Timeout(35, connect=3), "follow_redirects": False},
        ).client

    def endpoint(self, path: str) -> str:
        return self.url + path

    @property
    def headers(self) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def lifespan_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.url,
            headers=self.headers,
            timeout=httpx.Timeout(35, connect=3),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=10),
            follow_redirects=False,
        )


class _ReadFailure(Enum):
    INVALID_QUERY = "invalid_query"
    CHANGED = "changed"
    QUERY_TOO_LARGE = "query_too_large"
    UNAVAILABLE = "unavailable"
    RESPONSE_TOO_LARGE = "response_too_large"
    INVALID_RESPONSE = "invalid_response"


def _raise_read_failure(failure: _ReadFailure) -> NoReturn:
    match failure:
        case _ReadFailure.INVALID_QUERY:
            raise ValueError("Invalid trace query")
        case _ReadFailure.CHANGED:
            raise TraceChanged("Trace changed while paging; refresh the trace to continue")
        case _ReadFailure.QUERY_TOO_LARGE:
            raise OverflowError("Trace exceeds the interactive read budget")
        case _ReadFailure.UNAVAILABLE:
            raise RuntimeError("Lens trace storage is unavailable")
        case _ReadFailure.RESPONSE_TOO_LARGE:
            raise RuntimeError("Lens response exceeds the size limit")
        case _ReadFailure.INVALID_RESPONSE:
            raise ValueError("Invalid Lens response")
        case _:
            assert_never(failure)


class RemoteTraceStore:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client: Final = client

    async def _read(self, request: Mapping[str, object]) -> JsonValue:
        result: Final = await self._read_result(request)
        if isinstance(result, _ReadFailure):
            _raise_read_failure(result)
        return result

    async def _read_result(self, request: Mapping[str, object]) -> JsonValue | _ReadFailure:
        try:
            async with self.client.stream("POST", "/internal/read", json=dict(request)) as response:
                match response.status_code:
                    case 400:
                        return _ReadFailure.INVALID_QUERY
                    case 409:
                        return _ReadFailure.CHANGED
                    case 413:
                        return _ReadFailure.QUERY_TOO_LARGE
                    case 200:
                        return _JSON.validate_json(await bounded_response(response, MAX_RESPONSE_BYTES))
                    case _:
                        return _ReadFailure.UNAVAILABLE
        except httpx.HTTPError:
            return _ReadFailure.UNAVAILABLE
        except RuntimeError:
            return _ReadFailure.RESPONSE_TOO_LARGE
        except ValueError:
            return _ReadFailure.INVALID_RESPONSE

    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        path: Final = _INSERT_PATHS.get(table)
        if path is None:
            raise ValueError("Lens only accepts gateway request records and feedback on this endpoint")
        response: Final = await self.client.post(path, json=tuple(dict(row) for row in rows))
        response.raise_for_status()

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


async def bounded_response(
    response: httpx.Response, limit: int, *, reserve: Callable[[int], None] | None = None
) -> bytes:
    from io import BytesIO

    with BytesIO() as buffer:
        async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
            if buffer.tell() + len(chunk) > limit:
                raise RuntimeError("Lens response exceeds the size limit")
            if reserve is not None:
                reserve(len(chunk))
            buffer.write(chunk)
        return buffer.getvalue()

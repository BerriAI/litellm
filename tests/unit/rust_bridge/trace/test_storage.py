import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import Final, TypeVar

import pytest
from pydantic import JsonValue, ValidationError

from litellm.constants import TRACE_RESPONSE_INLINE_PARSE_BYTES
from litellm.rust_bridge import loader
from litellm.rust_bridge.trace.generated.types import QueryScope, TraceScope
from litellm.rust_bridge.trace.storage import ClickHouseStorage, RawJson, TraceStorageConfig

_T = TypeVar("_T")


class _NativeConfig:
    def __init__(self, database: str, url: str, retention_days: int, max_attribute_value_bytes: int) -> None:
        pass


class _NativeBridge(ModuleType):
    def __init__(self, response: str) -> None:
        super().__init__("native_traces")

        class Storage:
            def __init__(self, config: _NativeConfig) -> None:
                pass

            async def query_sql(self, sql: str, scope: QueryScope, secret: str) -> str:
                return response

        self.NativeTraceConfig: Final = _NativeConfig
        self.NativeTraceStorage: Final = Storage

    def trace_encode_error(self, message: str) -> bytes:
        return b""

    def trace_span_rows(
        self, body: bytes, content_type: str | None, tenant: Mapping[str, str], max_attribute_value_bytes: int
    ) -> list[dict[str, JsonValue]]:
        return []


@pytest.mark.parametrize(
    ("body", "rows"),
    (
        (
            json.dumps(
                {
                    "meta": [{"name": "value", "type": "UInt64"}],
                    "data": [
                        {
                            "integer": 9007199254740993,
                            "integer_string": "9007199254740993",
                            "fraction": 2.5,
                            "nested": {"values": [True, None, "text", {"count": 2}]},
                        }
                    ],
                    "rows": 1,
                    "statistics": {"elapsed": 0.01, "rows_read": 1, "bytes_read": 8},
                    "rows_before_limit_at_least": 1,
                    "unknown_extra": {"kept_by_internal_validator": True},
                }
            ),
            [
                {
                    "integer": 9007199254740993,
                    "integer_string": "9007199254740993",
                    "fraction": 2.5,
                    "nested": {"values": [True, None, "text", {"count": 2}]},
                }
            ],
        ),
        (
            json.dumps(
                {
                    "meta": [],
                    "data": [],
                    "rows": 0,
                    "statistics": {"elapsed": 0, "rows_read": 0, "bytes_read": 0},
                }
            ),
            [],
        ),
    ),
)
async def test_sql_query_returns_only_rows_without_normalizing_json_values(
    monkeypatch: pytest.MonkeyPatch, body: str, rows: list[dict[str, JsonValue]]
) -> None:
    monkeypatch.setattr(loader, "_cached_bridge", _NativeBridge(body))
    storage: Final = ClickHouseStorage(TraceStorageConfig("http://clickhouse:8123"))

    result: Final = await storage.query_sql("SELECT 1", {"kind": "all"}, "secret")

    assert result.model_dump(mode="json") == {"data": rows}
    if rows:
        row: Final = result.data[0]
        assert type(row["integer"]) is int
        assert type(row["integer_string"]) is str
        assert type(row["fraction"]) is float
    assert result.data == tuple(rows)


@pytest.mark.parametrize(
    "body",
    (
        '{"meta":[],"data":[],"rows":0}',
        '{"meta":[],"data":{},"rows":0,"statistics":{"elapsed":0,"rows_read":0,"bytes_read":0}}',
    ),
)
async def test_sql_query_rejects_malformed_clickhouse_envelopes(monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    monkeypatch.setattr(loader, "_cached_bridge", _NativeBridge(body))
    storage: Final = ClickHouseStorage(TraceStorageConfig("http://clickhouse:8123"))

    with pytest.raises(RuntimeError, match="Native trace query returned an invalid response"):
        await storage.query_sql("SELECT 1", {"kind": "all"}, "secret")


class _RawSpanStore:
    def __init__(self, body: bytes) -> None:
        self.body: Final = body

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str) -> RawJson:
        return RawJson(self.body)


@dataclass
class _RecordingOffload:
    bodies: list[bytes] = field(default_factory=list)

    async def __call__(self, func: Callable[[bytes], _T], body: bytes, /) -> _T:
        self.bodies.append(body)
        return await asyncio.to_thread(func, body)


def _span_body(output: str) -> bytes:
    text: Final = {"kind": "text", "text": output}
    return json.dumps(
        {
            "span_id": "span",
            "input_ui": text,
            "output_ui": text,
            "input": "question",
            "output": output,
            "attributes": {"k": "v"},
        }
    ).encode()


@pytest.mark.parametrize(
    ("output", "offloaded"),
    (("a", False), ("x" * TRACE_RESPONSE_INLINE_PARSE_BYTES, True)),
)
async def test_raw_responses_parse_once_into_the_typed_model_and_offload_large_bodies(
    output: str, offloaded: bool
) -> None:
    body: Final = _span_body(output)
    offload: Final = _RecordingOffload()
    storage: Final = ClickHouseStorage(_RawSpanStore(body), offload=offload)  # pyright: ignore[reportArgumentType]  # stub implements only the method under test

    span: Final = await storage.get_span("trace", "span", {"all_teams": 1, "user_id": "", "team_ids": ()})

    assert span is not None
    assert span["output"] == output
    assert span["output_ui"] == {"kind": "text", "text": output}
    assert offload.bodies == ([body] if offloaded else [])


async def test_invalid_raw_responses_fail_without_leaking_the_body() -> None:
    storage: Final = ClickHouseStorage(_RawSpanStore(b'{"span_id": "secret-body"}'))  # pyright: ignore[reportArgumentType]  # stub implements only the method under test

    with pytest.raises(RuntimeError, match="invalid response") as failure:
        await storage.get_span("trace", "span", {"all_teams": 1, "user_id": "", "team_ids": ()})
    assert "secret-body" not in str(failure.value)
    assert isinstance(failure.value.__cause__, ValidationError)

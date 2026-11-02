import json
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from litellm.tracing.remote import RemoteTraceStore
from litellm.tracing.storage import LensTraceStorage


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
    body: str, rows: list[dict[str, JsonValue]]
) -> None:
    async with httpx.AsyncClient(
        base_url="http://lens", transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        result: Final = await LensTraceStorage(RemoteTraceStore(client)).query_sql(
            "SELECT 1", {"kind": "all"}, "secret"
        )

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
async def test_sql_query_rejects_malformed_clickhouse_envelopes(body: str) -> None:
    async with httpx.AsyncClient(
        base_url="http://lens", transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        with pytest.raises(RuntimeError, match="Lens trace query returned an invalid response"):
            await LensTraceStorage(RemoteTraceStore(client)).query_sql("SELECT 1", {"kind": "all"}, "secret")

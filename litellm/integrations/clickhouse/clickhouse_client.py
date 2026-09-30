import gzip
import json
from collections.abc import Mapping
from typing import Any, Final

from litellm.rust_bridge.traces import query as query_traces
from pydantic import JsonValue, TypeAdapter

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

QUERY_PARAMETERS: Final = TypeAdapter(dict[str, str | int | list[str]])


class ClickHouseClient:
    def __init__(
        self,
        url: str,
        user: str,
        password: str,
        database: str,
        reader_user: str | None = None,
        reader_password: str | None = None,
    ) -> None:
        if not url:
            raise ValueError("ClickHouse url is required")
        self.url = url.rstrip("/") + "/"
        self.database = database
        self.reader_user = reader_user
        self.reader_password = reader_password
        self.auth_headers = {"X-ClickHouse-User": user, "X-ClickHouse-Key": password}
        self.http = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)

    async def execute(self, sql: str) -> None:
        await self.http.post(self.url, content=sql.encode(), headers=self.auth_headers)

    async def insert_json_each_row(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        body = gzip.compress("\n".join(json.dumps(r, default=str) for r in rows).encode())
        await self.http.post(
            self.url,
            params={
                "query": f"INSERT INTO {self.database}.{table} FORMAT JSONEachRow",
                "async_insert": "1",
                "wait_for_async_insert": "1",
                "input_format_skip_unknown_fields": "1",
                "date_time_input_format": "best_effort",
            },
            content=body,
            headers={**self.auth_headers, "Content-Encoding": "gzip"},
        )

    async def query(self, sql: str, params: Mapping[str, object] | None = None) -> list[dict[str, JsonValue]]:
        if self.reader_user is None or self.reader_password is None:
            raise RuntimeError("Trace reads require separate ClickHouse reader credentials")
        parameters: Final = QUERY_PARAMETERS.validate_python(params or {})
        return await query_traces(self.url, self.database, self.reader_user, self.reader_password, sql, parameters)

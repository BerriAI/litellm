"""
Minimal async ClickHouse client over the HTTP interface.

Uses LiteLLM's shared httpx client — no clickhouse driver dependency.
`AsyncHTTPHandler.post` raises on non-2xx, so callers see ClickHouse errors as exceptions.
"""

import gzip
import json
from typing import Any

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider


class ClickHouseClient:
    def __init__(self, url: str, user: str, password: str, database: str):
        if not url:
            raise ValueError("ClickHouse url is required")
        self.url = url.rstrip("/") + "/"
        self.database = database
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

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Parameterized SELECT. Use `{name:Type}` placeholders in `sql`."""
        query_params = {
            f"param_{k}": ("[" + ",".join(f"'{x}'" for x in v) + "]" if isinstance(v, list) else str(v))
            for k, v in (params or {}).items()
        }
        response = await self.http.post(
            self.url,
            params={
                "default_format": "JSON",
                "database": self.database,
                **query_params,
            },
            content=sql.encode(),
            headers=self.auth_headers,
        )
        return response.json()["data"]

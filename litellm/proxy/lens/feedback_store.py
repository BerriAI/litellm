import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import JsonValue, TypeAdapter

if TYPE_CHECKING:
    from litellm.rust_bridge.trace.storage import TraceStorageConfig

_ROWS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])


class FeedbackStore:
    def __init__(self, config: "TraceStorageConfig") -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_]+", config.database):
            raise ValueError("Invalid ClickHouse database")
        self.database: Final = config.database
        url: Final = httpx.URL(config.url)
        if url.scheme not in ("http", "https") or not url.host:
            raise ValueError("Invalid ClickHouse URL")
        self.url: Final = url.copy_with(query=None)
        self.settings: Final = {
            k: v for k, v in url.params.items() if k not in ("query", "database", "readonly", "async_insert")
        }

    async def execute(self, sql: str, parameters: Mapping[str, str | int], *, read: bool = True) -> str:
        settings: Final = {
            **self.settings,
            "database": self.database,
            "max_execution_time": "15",
            "max_result_rows": "10000",
            "max_result_bytes": "16777216",
            "result_overflow_mode": "throw",
            "max_rows_to_read": "1000000",
            "read_overflow_mode": "throw",
            "async_insert": "0",
            "optimize_move_to_prewhere_if_final": "0",
            **({"readonly": "1"} if read else {}),
            **{f"param_{key}": str(value) for key, value in parameters.items()},
        }
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
                response: Final = await client.post(self.url, params=settings, content=sql.encode())
                response.raise_for_status()
                return response.text
        except httpx.HTTPError as error:
            raise RuntimeError("ClickHouse feedback storage is unavailable") from error

    async def ensure_schema(self) -> None:
        schema: Final = Path(__file__).with_name("feedback.sql").read_text()
        await self.execute(schema.replace("{database}", f"`{self.database}`"), {}, read=False)

    async def rows(self, sql: str, parameters: Mapping[str, str | int]) -> tuple[dict[str, JsonValue], ...]:
        body: Final = await self.execute(sql + " FORMAT JSONEachRow", parameters)
        return _ROWS.validate_python(tuple(json.loads(line) for line in body.splitlines() if line))

    async def insert(self, row: Mapping[str, str | int]) -> None:
        await self.execute("INSERT INTO lens_feedback FORMAT JSONEachRow\n" + json.dumps(dict(row)), {}, read=False)

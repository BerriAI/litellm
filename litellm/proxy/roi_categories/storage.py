from collections.abc import Mapping
from datetime import datetime
from typing import Final, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter

from litellm.proxy.roi_categories.models import Assignment, Example, Settings, SpendBucket, Traffic, utc_milliseconds
from litellm.rust_bridge.clickhouse import SpendStorageConfig

Row: Final = TypeVar("Row", bound=BaseModel)
_ELIGIBLE: Final = """
    start_time >= fromUnixTimestamp64Milli({start:Int64})
    AND start_time < fromUnixTimestamp64Milli({end:Int64})
    AND ({team:String} = '' OR team_id = {team:String})
    AND NOT has(request_tags, 'litellm-roi-analysis')
    AND NOT has(request_tags, 'litellm-internal')
    AND JSONExtractString(metadata, 'internal_call_origin') = ''
"""
_JOIN: Final = """LEFT JOIN
    (SELECT request_id, category_id, attempts, completed
     FROM roi_category_assignments FINAL WHERE version = {version:String}) AS a USING request_id"""


class CategoryStorage:
    def __init__(self, config: SpendStorageConfig, client: httpx.AsyncClient) -> None:
        self.config: Final = config
        self.client: Final = client

    async def execute(self, sql: str, parameters: Mapping[str, str | int]) -> str:
        response: Final = await self.client.post(
            self.config.url,
            params={
                **{key: value for key, value in httpx.URL(self.config.url).params.items() if key != "query"},
                "database": self.config.database,
                "max_execution_time": "8",
                "max_result_bytes": "8388608",
                "result_overflow_mode": "throw",
                "join_use_nulls": "0",
                **{f"param_{key}": str(value) for key, value in parameters.items()},
            },
            content=sql.encode(),
            timeout=10,
        )
        response.raise_for_status()
        return response.text

    async def rows(self, sql: str, parameters: Mapping[str, str | int], model: type[Row]) -> tuple[Row, ...]:
        body: Final = await self.execute(sql + " FORMAT JSONEachRow", parameters)
        adapter: Final = TypeAdapter(model)
        return tuple(adapter.validate_json(line) for line in body.splitlines() if line.strip())

    async def ensure_schema(self) -> None:
        _ = await self.execute(
            """CREATE TABLE IF NOT EXISTS roi_category_assignments (
                request_id String, version String, category_id String,
                attempts UInt8, completed Bool, revision UInt16 DEFAULT attempts * 2 + completed, updated_at DateTime64(3) DEFAULT now64(3)
            ) ENGINE = ReplacingMergeTree(revision) ORDER BY (version, request_id)
            TTL toDateTime(updated_at) + INTERVAL {retention:UInt32} DAY""",
            {"retention": self.config.retention_days},
        )

    async def assign(self, assignment: Assignment) -> None:
        _ = await self.execute(
            "INSERT INTO roi_category_assignments (request_id, version, category_id, attempts, completed) "
            "FORMAT JSONEachRow\n" + assignment.model_dump_json(),
            {},
        )

    def parameters(self, settings: Settings, start: datetime, end: datetime) -> Mapping[str, str | int]:
        return {
            "start": utc_milliseconds(start),
            "end": utc_milliseconds(end),
            "team": settings.team_id,
            "version": settings.version,
        }

    async def sample(self, settings: Settings, now: datetime) -> tuple[Traffic, ...]:
        return await self.rows(
            "SELECT request_id, substringUTF8(messages, 1, 2000) AS excerpt FROM spend_logs FINAL WHERE "
            + _ELIGIBLE
            + " AND messages NOT IN ('', '{}', '[]', 'null') "
            "ORDER BY start_time, request_id LIMIT {limit:UInt32}",
            {**self.parameters(settings, settings.started_at, now), "limit": settings.sample_size},
            Traffic,
        )

    async def pending(self, settings: Settings, now: datetime) -> tuple[Traffic, ...]:
        return await self.rows(
            "SELECT request_id, substringUTF8(messages, 1, 2000) AS excerpt, a.attempts AS attempts "
            "FROM spend_logs FINAL "
            + _JOIN
            + " WHERE "
            + _ELIGIBLE
            + " AND messages NOT IN ('', '{}', '[]', 'null') AND a.completed = false AND a.attempts < 3 "
            "ORDER BY start_time, request_id LIMIT 5",
            self.parameters(settings, settings.activated_at or settings.started_at, now),
            Traffic,
        )

    async def report(
        self, settings: Settings, start: datetime, end: datetime, model: str, user: str = ""
    ) -> tuple[SpendBucket, ...]:
        return await self.rows(
            "SELECT toString(toDate(start_time, 'UTC')) AS day, "
            "if(a.category_id = '', 'unclassified', a.category_id) AS category_id, "
            "model, sum(ifNull(spend, 0)) AS spend, count() AS requests, countIf(isNull(spend)) AS unknown_cost_requests "
            "FROM spend_logs FINAL "
            + _JOIN
            + " WHERE "
            + _ELIGIBLE
            + " AND ({model:String} = '' OR model = {model:String}) AND ({user:String} = '' OR user = {user:String}) GROUP BY day, category_id, model "
            "ORDER BY day, category_id, model LIMIT 10001",
            {**self.parameters(settings, start, end), "model": model, "user": user},
            SpendBucket,
        )

    async def examples(
        self,
        settings: Settings,
        start: datetime,
        end: datetime,
        category_id: str,
        model: str,
        user: str = "",
    ) -> tuple[Example, ...]:
        return await self.rows(
            "SELECT request_id, model, spend, substringUTF8(messages, 1, 2000) AS excerpt "
            "FROM spend_logs FINAL "
            + _JOIN
            + " WHERE "
            + _ELIGIBLE
            + " AND if(a.category_id = '', 'unclassified', a.category_id) = {category:String} "
            "AND ({model:String} = '' OR model = {model:String}) AND ({user:String} = '' OR user = {user:String}) ORDER BY spend DESC, request_id LIMIT 5",
            {**self.parameters(settings, start, end), "category": category_id, "model": model, "user": user},
            Example,
        )

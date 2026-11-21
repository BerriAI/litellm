from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.utils import query_raw_rows

from .models import Speed

SPEED_QUERY: Final = """
WITH recent AS MATERIALIZED (
    SELECT model_group, "startTime", "endTime", "completionStartTime", completion_tokens
    FROM "LiteLLM_SpendLogs"
    WHERE "startTime" >= NOW() - INTERVAL '7 days'
    ORDER BY "startTime" DESC
    LIMIT 100000
)
SELECT model_group,
    percentile_cont(0.5) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM ("endTime" - "startTime"))
    ) AS latency_seconds,
    percentile_cont(0.5) WITHIN GROUP (
        ORDER BY completion_tokens / NULLIF(EXTRACT(EPOCH FROM (
            "endTime" - COALESCE("completionStartTime", "startTime")
        )), 0)
    ) FILTER (WHERE completion_tokens > 0) AS output_tokens_per_second
FROM recent
WHERE model_group = ANY($1::text[]) AND "endTime" > "startTime"
GROUP BY model_group
"""


class QueryClient(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


@dataclass(frozen=True)
class GatewaySpeedClient:
    database: PrismaWrapper | RoutingPrismaWrapper

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        return await query_raw_rows(self.database, query, *args)


async def observed_speed(client: QueryClient | None, groups: tuple[str, ...]) -> tuple[Speed, ...]:
    if client is None or not groups:
        return ()
    try:
        rows: Final = await client.query_raw(SPEED_QUERY, list(groups))
        return TypeAdapter(tuple[Speed, ...]).validate_python(rows)
    except Exception:  # noqa: BLE001  # Optional telemetry failures must not block recommendations.
        return ()

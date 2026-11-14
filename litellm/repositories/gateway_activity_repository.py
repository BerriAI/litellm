from typing import TYPE_CHECKING, Annotated, Final

from pydantic import Field, TypeAdapter

from litellm.models.gateway_activity import GatewayActivity

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_ACTIVITY_SQL: Final = """
    SELECT
        'route' AS kind, date, category, route,
        SUM(successful_requests)::bigint AS successful_requests,
        SUM(failed_requests)::bigint AS failed_requests,
        NULL::integer AS status_code
    FROM "LiteLLM_DailyGatewayRequests"
    WHERE date >= $1 AND date <= $2
    GROUP BY date, category, route
    UNION ALL
    SELECT
        'status' AS kind, NULL, NULL, NULL,
        0::bigint,
        SUM(failed_requests)::bigint,
        status_code
    FROM "LiteLLM_DailyGatewayFailedRequests"
    WHERE date >= $1 AND date <= $2
    GROUP BY status_code
"""

_ACTIVITY_ROWS_ADAPTER: Final = TypeAdapter(tuple[Annotated[GatewayActivity, Field(discriminator="kind")], ...])


class GatewayActivityRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._prisma_client = prisma_client

    async def get_activity(self, start_date: str, end_date: str) -> tuple[GatewayActivity, ...]:
        raw_rows: Final = await self._prisma_client.db.query_raw(  # pyright: ignore[reportAny]  # untyped Prisma client
            _ACTIVITY_SQL, start_date, end_date
        )
        return _ACTIVITY_ROWS_ADAPTER.validate_python(raw_rows or ())

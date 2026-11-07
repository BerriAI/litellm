"""Sample successful aggregate flushes without adding request-time telemetry."""

from datetime import datetime, timedelta, timezone
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypeAlias,
    cast,  # noqa: TID251  # generated Prisma delegate supplies the repository protocol
)

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.repositories.usage_alert_metrics import MetricsDatabase

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

FlushSource: TypeAlias = Literal["user", "team"]


class FlushCoverage(BaseModel):
    model_config = ConfigDict(frozen=True)
    hours: int
    after_grace: bool


_COVERAGE: Final = TypeAdapter(tuple[FlushCoverage, ...])


class UsageAlertHealthRepository:
    def __init__(self, database: MetricsDatabase) -> None:
        self._database = database

    @classmethod
    def from_prisma(cls, prisma: "PrismaClient") -> "UsageAlertHealthRepository":
        database: Final = cast(MetricsDatabase, prisma.writer_db)  # cast-ok: generated Prisma delegate
        return cls(database)

    async def record(self, source: FlushSource, now: datetime) -> bool:
        hour: Final = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
        rows: Final = await self._database.query_raw(
            """INSERT INTO "LiteLLM_UsageAlertFlush" (source,hour,observed_at)
            SELECT $1,$2::timestamp,$3::timestamp WHERE NOT EXISTS (
                SELECT 1 FROM "LiteLLM_SpendLogDelivery" WHERE state='pending' AND created_at<$2::timestamp LIMIT 1
            ) ON CONFLICT (source,hour) DO UPDATE SET observed_at=EXCLUDED.observed_at RETURNING source""",
            source,
            hour,
            now.astimezone(timezone.utc).replace(tzinfo=None),
        )
        await self._database.query_raw(
            'DELETE FROM "LiteLLM_UsageAlertFlush" WHERE hour<$1::timestamp RETURNING source', hour - timedelta(days=35)
        )
        return bool(rows)

    async def coverage(self, source: FlushSource, start: datetime) -> FlushCoverage:
        utc_start: Final = start.astimezone(timezone.utc).replace(tzinfo=None)
        end: Final = utc_start + timedelta(days=1)
        rows: Final = _COVERAGE.validate_python(
            await self._database.query_raw(
                """SELECT COUNT(*) FILTER(WHERE hour >= $2::timestamp AND hour < $3::timestamp)::integer AS hours,
                COALESCE(BOOL_OR(observed_at >= $3::timestamp + INTERVAL '1 hour'),FALSE) AS after_grace
                FROM "LiteLLM_UsageAlertFlush" WHERE source=$1 AND hour >= $2::timestamp""",
                source,
                utc_start,
                end,
            )
        )
        return rows[0] if rows else FlushCoverage(hours=0, after_grace=False)


class UsageAlertFlushRecorder:
    """At most one heartbeat attempt per source per worker per hour after success."""

    def __init__(self) -> None:
        self._user_hour: datetime | None = None
        self._team_hour: datetime | None = None

    async def record(self, prisma_client: "PrismaClient", source: FlushSource) -> None:
        from litellm._logging import verbose_proxy_logger
        from litellm.proxy.utils import ProxyUpdateSpend

        if ProxyUpdateSpend.disable_spend_updates():
            return
        now: Final = datetime.now(timezone.utc)
        hour: Final = now.replace(minute=0, second=0, microsecond=0)
        prior: Final = self._user_hour if source == "user" else self._team_hour
        if prior == hour:
            return
        try:
            database: Final = cast(MetricsDatabase, prisma_client.writer_db)  # cast-ok: generated Prisma delegate
            if not await UsageAlertHealthRepository(database).record(source, now):
                return
        except Exception:  # noqa: BLE001  # observability must not fail or requeue a committed spend flush
            verbose_proxy_logger.warning("Could not record usage-alert aggregate freshness")
            return
        if source == "user":
            self._user_hour = hour
        else:
            self._team_hour = hour

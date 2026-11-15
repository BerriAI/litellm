import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import date as Date
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # Prisma exposes generated table actions dynamically
)

from pydantic import BaseModel, ConfigDict, TypeAdapter
from typing_extensions import LiteralString

from litellm._logging import verbose_proxy_logger
from litellm.models.alert import (
    Alert,
    AlertDetection,
    AlertFilter,
    AlertHistoryPoint,
    AlertPage,
    AlertRecipient,
    AlertResourceType,
    AlertStatus,
    AlertSummary,
    AlertView,
)
from litellm.repositories.prisma_protocols import PrismaRecord, TableActions

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class _HistoryTransaction(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


class _HistoryDatabase(Protocol):
    def tx(self, *, timeout: timedelta) -> AbstractAsyncContextManager[_HistoryTransaction]: ...


class _HistoryRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    resource_name: str | None
    date: Date | None
    spend: float
    requests: int
    failed: int
    response_ms: int
    timed: int

    def point(self, kind: str) -> AlertHistoryPoint | None:
        if self.date is None or (kind == "latency" and not self.timed) or (kind == "reliability" and not self.requests):
            return None
        value: Final = {
            "spend": self.spend,
            "traffic": float(self.requests),
            "latency": self.response_ms / self.timed if self.timed else 0,
            "reliability": min(1.0, self.failed / self.requests) if self.requests else 0,
        }[kind]
        return AlertHistoryPoint(date=self.date, value=value)


_HISTORY_ROWS: Final = TypeAdapter(tuple[_HistoryRow, ...])
_SUMMARIES: Final = TypeAdapter(tuple[AlertSummary, ...])
_RECIPIENTS: Final = TypeAdapter(tuple[AlertRecipient, ...])


class _LatestAlert(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int


_LATEST_ALERT: Final = TypeAdapter(tuple[_LatestAlert, ...])
_HISTORY_SOURCES: Final[
    Mapping[AlertResourceType, tuple[LiteralString, LiteralString, LiteralString, LiteralString, LiteralString]]
] = {
    "gateway": ('"LiteLLM_DailyModelUsage"', "'gateway'", "request_count", "", "'Gateway'"),
    "model": ('"LiteLLM_DailyModelUsage"', "d.model_group", "request_count", "", "r.resource_id"),
    "team": (
        '"LiteLLM_DailyTeamSpend"',
        "d.team_id",
        "api_requests",
        'LEFT JOIN "LiteLLM_TeamTable" m ON m.team_id = r.resource_id',
        "m.team_alias",
    ),
    "user": (
        '"LiteLLM_DailyUserSpend"',
        "d.user_id",
        "api_requests",
        'LEFT JOIN "LiteLLM_UserTable" m ON m.user_id = r.resource_id',
        "m.user_alias",
    ),
    "key": (
        '"LiteLLM_DailyUserSpend"',
        "d.api_key",
        "api_requests",
        'LEFT JOIN "LiteLLM_VerificationToken" m ON m.token = r.resource_id',
        "m.key_alias",
    ),
}


def _history_sql(scope: AlertResourceType) -> LiteralString:
    table, resource, requests, metadata, name = _HISTORY_SOURCES[scope]
    return f"""SELECT r.id, LEFT({name}, 1024) AS resource_name, d.date,
        COALESCE(SUM(d.spend), 0)::float8 AS spend,
        COALESCE(SUM(d.{requests}), 0)::bigint AS requests,
        COALESCE(SUM(d.failed_requests), 0)::bigint AS failed,
        COALESCE(SUM(d.total_response_time_ms), 0)::bigint AS response_ms,
        COALESCE(SUM(d.timed_requests), 0)::bigint AS timed
    FROM requested r
    LEFT JOIN {table} d ON {resource} = r.resource_id AND d.date BETWEEN r.start AND r.day
    {metadata}
    WHERE r.resource_type = '{scope}'
    GROUP BY r.id, LEFT({name}, 1024), d.date"""


class AlertTransaction(Protocol):
    async def execute_raw(self, query: str, *args: object) -> int: ...


def _detection_key(detection: AlertDetection) -> str:
    return hashlib.sha256(json.dumps(detection.identity()).encode()).hexdigest()


async def record_alerts(
    transaction: AlertTransaction, detections: Sequence[AlertDetection], limit: int | None = None
) -> int:
    if not detections:
        return 0
    return await transaction.execute_raw(
        """INSERT INTO "LiteLLM_Alert"
            (detection_key, type, resource_type, resource_id, window_start, window_end, evidence, created_at, updated_at)
        SELECT detection_key, type, resource_type, resource_id, window_start, window_end, evidence,
            now() AT TIME ZONE 'UTC', now() AT TIME ZONE 'UTC'
        FROM jsonb_to_recordset($1::jsonb) AS rows(detection_key text, type text, resource_type text,
            resource_id text, window_start timestamp, window_end timestamp, evidence jsonb)
        WHERE NOT EXISTS (SELECT 1 FROM "LiteLLM_Alert" a WHERE a.detection_key = rows.detection_key)
        ORDER BY resource_type, resource_id, type
        LIMIT $2
        ON CONFLICT (detection_key) DO NOTHING""",
        json.dumps(
            tuple(
                {**detection.model_dump(mode="json"), "detection_key": _detection_key(detection)}
                for detection in detections
            )
        ),
        limit,
    )


def _alert(row: PrismaRecord) -> Alert:
    values: Final = row.dict()
    return Alert.model_validate(
        {**{key: value for key, value in values.items() if key != "detection_key"}, "id": str(values["id"])}
    )


class AlertRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._database: Final = cast(  # cast-ok: Prisma transaction exposes bounded raw query methods
            _HistoryDatabase, prisma_client.writer_db
        )
        self._table = cast(  # cast-ok: the generated Prisma client delegates typed table actions dynamically
            TableActions[PrismaRecord], prisma_client.writer_db.litellm_alert
        )

    async def latest_id(self) -> int:
        async with self._database.tx(timeout=timedelta(seconds=5)) as transaction:
            await transaction.query_raw("SELECT set_config('statement_timeout', '2s', true)")
            rows: Final = _LATEST_ALERT.validate_python(
                await transaction.query_raw('SELECT COALESCE(MAX(id), 0) AS id FROM "LiteLLM_Alert"')
            )
        return rows[0].id

    async def recipients(self, after: str) -> tuple[AlertRecipient, ...]:
        async with self._database.tx(timeout=timedelta(seconds=5)) as transaction:
            await transaction.query_raw("SELECT set_config('statement_timeout', '2s', true)")
            rows: Final = await transaction.query_raw(
                """SELECT user_id, LOWER(TRIM(user_email)) AS email FROM "LiteLLM_UserTable"
                WHERE user_role = 'proxy_admin' AND user_id > $1 AND LENGTH(TRIM(user_email)) BETWEEN 3 AND 320
                ORDER BY user_id LIMIT 100""",
                after,
            )
        return _RECIPIENTS.validate_python(rows)

    async def summaries(self, after: int, through: int) -> tuple[AlertSummary, ...]:
        async with self._database.tx(timeout=timedelta(seconds=5)) as transaction:
            await transaction.query_raw("SELECT set_config('statement_timeout', '2s', true)")
            rows: Final = await transaction.query_raw(
                """SELECT a.id::text, a.type, a.resource_type, a.created_at AT TIME ZONE 'UTC' AS created_at,
                    LEFT(COALESCE(k.key_alias, t.team_alias, u.user_alias,
                        CASE WHEN a.resource_type = 'model' THEN a.resource_id ELSE a.resource_type END), 120) AS resource_name,
                    jsonb_build_object('value', a.evidence->'value', 'baseline', a.evidence->'baseline',
                        'unit', a.evidence->'unit', 'sample_size', a.evidence->'sample_size') AS evidence
                FROM "LiteLLM_Alert" a
                LEFT JOIN "LiteLLM_VerificationToken" k ON a.resource_type = 'key' AND k.token = a.resource_id
                LEFT JOIN "LiteLLM_TeamTable" t ON a.resource_type = 'team' AND t.team_id = a.resource_id
                LEFT JOIN "LiteLLM_UserTable" u ON a.resource_type = 'user' AND u.user_id = a.resource_id
                WHERE a.id > $1 AND a.id <= $2 AND a.status = 'open' AND a.type != 'model_deprecation'
                ORDER BY a.id DESC LIMIT 5""",
                after,
                through,
            )
        return _SUMMARIES.validate_python(rows)

    async def record(self, detection: AlertDetection) -> Alert:
        from prisma import Json
        from prisma.errors import UniqueViolationError

        key: Final = _detection_key(detection)
        try:
            row: Final = await self._table.create(
                data={
                    **detection.model_dump(exclude={"evidence"}),
                    "detection_key": key,
                    "evidence": Json(detection.evidence.model_dump(mode="json")),
                }
            )
            return _alert(row)
        except UniqueViolationError:
            existing: Final = await self._table.find_unique(where={"detection_key": key})
            if existing is None:
                raise
            return _alert(existing)

    async def list(
        self, status: AlertStatus, limit: int, cursor: int | None, alert_type: AlertFilter | None = None
    ) -> AlertPage:
        size: Final = max(1, min(limit, 100))
        category: Final = {"not": "model_deprecation"} if alert_type == "performance" else alert_type
        rows: Final = await self._table.find_many(
            where={
                "status": status,
                **({"id": {"lt": cursor}} if cursor is not None else {}),
                **({"type": category} if category is not None else {}),
            },
            order={"id": "desc"},
            take=size + 1,
        )
        return AlertPage(
            data=tuple(_alert(row) for row in rows[:size]),
            next_cursor=str(rows[size - 1].dict()["id"]) if len(rows) > size else None,
        )

    async def get(self, alert_id: int) -> Alert | None:
        row: Final = await self._table.find_unique(where={"id": alert_id})
        return None if row is None else _alert(row)

    async def set_status(self, alert_id: int, status: AlertStatus, actor: str) -> Alert | None:
        row: Final = await self._table.update(where={"id": alert_id}, data={"status": status, "updated_by": actor})
        return None if row is None else _alert(row)

    async def enrich(self, alerts: tuple[Alert, ...]) -> tuple[AlertView, ...]:
        from prisma.errors import PrismaError

        requested: Final = tuple(
            {
                "id": alert.id,
                "resource_id": alert.resource_id,
                "resource_type": alert.resource_type,
                "day": alert.window_start.date().isoformat(),
                "start": (alert.window_start.date() - timedelta(days=7)).isoformat(),
            }
            for alert in alerts
            if alert.type != "model_deprecation"
        )
        if not requested:
            return tuple(AlertView.model_validate(alert.model_dump()) for alert in alerts)
        scopes: Final = frozenset[AlertResourceType](alert.resource_type for alert in alerts)
        histories: Final = " UNION ALL ".join(_history_sql(scope) for scope in scopes)
        query: Final = f"""WITH requested AS (
            SELECT * FROM jsonb_to_recordset($1::jsonb)
            AS r(id text, resource_id text, resource_type text, day text, start text)
        ) {histories} ORDER BY id, date"""
        try:
            async with self._database.tx(timeout=timedelta(seconds=3)) as transaction:
                await transaction.query_raw(
                    "SELECT set_config('statement_timeout', '2s', true), set_config('work_mem', '1MB', true), "
                    "set_config('max_parallel_workers_per_gather', '0', true)"
                )
                rows: Final = _HISTORY_ROWS.validate_python(await transaction.query_raw(query, json.dumps(requested)))
            return tuple(_view(alert, rows) for alert in alerts)
        except (PrismaError, TimeoutError, ValueError) as error:
            verbose_proxy_logger.warning("Alert Usage history unavailable (%s)", type(error).__name__)
            return tuple(AlertView.model_validate(alert.model_dump()) for alert in alerts)


def _view(alert: Alert, rows: tuple[_HistoryRow, ...]) -> AlertView:
    history: Final = tuple(row for row in rows if row.id == alert.id)
    points: Final = tuple(point for row in history if (point := row.point(alert.type)) is not None)
    return AlertView.model_validate(
        {
            **alert.model_dump(),
            "resource_name": history[0].resource_name if history else None,
            "history": points,
        }
    )

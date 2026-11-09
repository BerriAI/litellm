import hashlib
import json
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # Prisma exposes generated table actions dynamically
)

from litellm.models.alert import Alert, AlertDetection, AlertPage, AlertStatus
from litellm.repositories.prisma_protocols import PrismaRecord, TableActions

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class AlertTransaction(Protocol):
    async def execute_raw(self, query: str, *args: object) -> int: ...


def _detection_key(detection: AlertDetection) -> str:
    return hashlib.sha256(json.dumps(detection.identity()).encode()).hexdigest()


async def record_alerts(transaction: AlertTransaction, detections: Sequence[AlertDetection]) -> None:
    if not detections:
        return
    await transaction.execute_raw(
        """INSERT INTO "LiteLLM_Alert"
            (detection_key, type, resource_type, resource_id, window_start, window_end, evidence, created_at, updated_at)
        SELECT detection_key, type, resource_type, resource_id, window_start, window_end, evidence,
            now() AT TIME ZONE 'UTC', now() AT TIME ZONE 'UTC'
        FROM jsonb_to_recordset($1::jsonb) AS rows(detection_key text, type text, resource_type text,
            resource_id text, window_start timestamp, window_end timestamp, evidence jsonb)
        ON CONFLICT (detection_key) DO NOTHING""",
        json.dumps(
            tuple(
                {**detection.model_dump(mode="json"), "detection_key": _detection_key(detection)}
                for detection in detections
            )
        ),
    )


def _alert(row: PrismaRecord) -> Alert:
    values: Final = row.dict()
    return Alert.model_validate(
        {**{key: value for key, value in values.items() if key != "detection_key"}, "id": str(values["id"])}
    )


class AlertRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._table = cast(  # cast-ok: the generated Prisma client delegates typed table actions dynamically
            TableActions[PrismaRecord], prisma_client.writer_db.litellm_alert
        )

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

    async def list(self, status: AlertStatus, limit: int, cursor: int | None) -> AlertPage:
        size: Final = max(1, min(limit, 100))
        rows: Final = await self._table.find_many(
            where={"status": status, **({"id": {"lt": cursor}} if cursor is not None else {})},
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

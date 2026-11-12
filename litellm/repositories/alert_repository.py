import hashlib
import json
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # Prisma exposes generated table actions dynamically

from litellm.models.alert import Alert, AlertDetection, AlertPage, AlertStatus
from litellm.repositories.prisma_protocols import PrismaRecord, TableActions

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


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

        key: Final = hashlib.sha256(json.dumps(detection.identity()).encode()).hexdigest()
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

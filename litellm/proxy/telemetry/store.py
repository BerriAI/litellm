import json
import secrets
import uuid
from collections.abc import Awaitable
from typing import Final, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from typing_extensions import LiteralString

from litellm._logging import verbose_proxy_logger
from litellm.telemetry.consent import ConsentError, TelemetryConsent, parse_consent
from litellm.telemetry.report import Report, report_to_json
from litellm.telemetry.sink import ExportOutcome

INSTANCE_ID_PARAM: Final = "telemetry_instance_id"
HASH_SECRET_PARAM: Final = "telemetry_deployment_hash_secret"
SETTINGS_PARAM: Final = "telemetry_settings"


class Database(Protocol):
    def query_raw(self, query: LiteralString, *args: object) -> Awaitable[object]: ...
    def execute_raw(self, query: LiteralString, *args: object) -> Awaitable[int]: ...


class _ValueRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str


class StoredReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    window_start: float
    window_end: float
    report: JsonValue


class _StoredGroups(BaseModel):
    model_config = ConfigDict(frozen=True)

    groups: tuple[str, ...]


class _SettingsRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    param_value: _StoredGroups


_SETTINGS_ROWS: Final = TypeAdapter(tuple[_SettingsRow, ...])
_VALUE_ROWS: Final = TypeAdapter(tuple[_ValueRow, ...])
_REPORT_ROWS: Final = TypeAdapter(tuple[StoredReport, ...])


class TelemetryStore:
    """The proxy DB side of telemetry: the install's stable instance id, deployment hash secret and report table"""

    def __init__(self, db: Database, retention_days: int) -> None:
        self._db: Final = db
        self._retention_days: Final = retention_days

    async def instance_id(self) -> str:
        return await self._persisted(INSTANCE_ID_PARAM, uuid.uuid4().hex)

    async def deployment_hash_secret(self) -> bytes:
        return (await self._persisted(HASH_SECRET_PARAM, secrets.token_hex(32))).encode()

    async def _persisted(self, param: str, candidate: str) -> str:
        await self._db.execute_raw(
            """INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES ($1, to_jsonb($2::text))
            ON CONFLICT (param_name) DO NOTHING""",
            param,
            candidate,
        )
        rows: Final = _VALUE_ROWS.validate_python(
            await self._db.query_raw(
                """SELECT param_value #>> '{}' AS value FROM "LiteLLM_Config" WHERE param_name = $1""",
                param,
            )
        )
        return rows[0].value

    async def consent(self) -> TelemetryConsent | ConsentError | None:
        rows: Final = _SETTINGS_ROWS.validate_python(
            await self._db.query_raw(
                """SELECT param_value FROM "LiteLLM_Config" WHERE param_name = $1""", SETTINGS_PARAM
            )
        )
        return parse_consent(rows[0].param_value.groups) if rows else None

    async def save_consent(self, consent: TelemetryConsent) -> None:
        await self._db.execute_raw(
            """INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES ($1, $2::jsonb)
            ON CONFLICT (param_name) DO UPDATE SET param_value = EXCLUDED.param_value""",
            SETTINGS_PARAM,
            json.dumps({"groups": sorted(group.value for group in consent.groups)}),
        )

    async def save(self, report: Report) -> None:
        await self._db.execute_raw(
            """INSERT INTO "LiteLLM_TelemetryReport" (id, window_start, window_end, report)
            VALUES ($1, $2, $3, $4::jsonb)""",
            uuid.uuid4().hex,
            report.window_start,
            report.window_end,
            json.dumps(report_to_json(report)),
        )

    async def prune(self) -> None:
        await self._db.execute_raw(
            """DELETE FROM "LiteLLM_TelemetryReport" WHERE created_at < now() - make_interval(days => $1::int)""",
            self._retention_days,
        )

    async def reports_after(self, window_end: float, report_id: str, limit: int) -> tuple[StoredReport, ...]:
        return _REPORT_ROWS.validate_python(
            await self._db.query_raw(
                """SELECT id, window_start, window_end, report FROM "LiteLLM_TelemetryReport"
                WHERE (window_end, id) > ($1, $2) ORDER BY window_end, id LIMIT $3""",
                window_end,
                report_id,
                limit,
            )
        )


def store_read_errors() -> tuple[type[Exception], ...]:
    from prisma.errors import PrismaError
    from pydantic import ValidationError

    return (PrismaError, OSError, httpx.HTTPError, ValidationError)


class LocalTableExporter:
    """``Exporter`` that keeps reports in the proxy DB for manual export instead of sending them"""

    def __init__(self, store: TelemetryStore) -> None:
        self._store: Final = store

    async def export(self, report: Report) -> ExportOutcome:
        if not (report.requests or report.attempts or report.ui_events or report.dropped_records):
            return ExportOutcome.SENT
        try:
            await self._store.save(report)
        except store_read_errors() as e:
            verbose_proxy_logger.debug("telemetry: could not store the report locally: %s", e)
            return ExportOutcome.RETRY
        try:
            await self._store.prune()
        except store_read_errors() as e:
            verbose_proxy_logger.debug("telemetry: could not prune old local reports: %s", e)
        return ExportOutcome.SENT

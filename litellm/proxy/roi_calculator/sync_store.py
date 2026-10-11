from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, Protocol, cast  # noqa: TID251 - PrismaWrapper dynamically delegates database methods

from pydantic import ConfigDict, TypeAdapter

from litellm.proxy.utils import PrismaClient
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.roi_calculator import ROIReport, ROISyncStatus
from litellm.types.roi_observed import ObservedData

_SYNC_KEY: Final = "roi_calculator_sync"
_REPORT_KEY: Final = "roi_calculator_report"


class _SyncState(LiteLLMBaseModel):
    owner: str
    status: ROISyncStatus
    cancel: bool = False


class _StateRow(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore")
    param_value: _SyncState
    expired: bool = False
    last_run_at: datetime


class _SyncDatabase(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...
    async def execute_raw(self, query: str, *args: object) -> int: ...


class SyncStore:
    def __init__(
        self,
        prisma: PrismaClient,
        namespace: Literal["roi_calculator", "roi_observed", "roi_oauth_refresh"] = "roi_calculator",
    ) -> None:
        self._db: Final = cast(_SyncDatabase, prisma.writer_db)  # cast-ok: PrismaWrapper delegates methods dynamically
        self._sync_key: Final = namespace + "_sync"
        self._report_key: Final = namespace + "_report"

    async def acquire(self, owner: str, status: ROISyncStatus, scheduled_interval: float = 0) -> bool:
        rows: Final = await self._db.query_raw(
            """INSERT INTO "LiteLLM_Config" (param_name, param_value, last_run_at)
               VALUES ($1, $2::jsonb, NOW())
               ON CONFLICT (param_name) DO UPDATE
               SET param_value = EXCLUDED.param_value, last_run_at = NOW()
               WHERE ("LiteLLM_Config".last_run_at < NOW() - INTERVAL '60 seconds'
                  OR "LiteLLM_Config".param_value->'status'->>'running' = 'false')
                 AND ($3::text::double precision = 0 OR "LiteLLM_Config".last_run_at <= NOW() - $3::text::double precision * INTERVAL '1 minute')
               RETURNING param_name""",
            self._sync_key,
            _SyncState(owner=owner, status=status).model_dump_json(),
            str(scheduled_interval),
        )
        return bool(rows)

    async def heartbeat(self, owner: str, status: ROISyncStatus) -> bool:
        rows: Final = await self._db.query_raw(
            """UPDATE "LiteLLM_Config"
               SET param_value = jsonb_set(param_value, '{status}', $3::jsonb), last_run_at = NOW()
               WHERE param_name = $1 AND param_value->>'owner' = $2
                 AND param_value->>'cancel' = 'false'
                 AND param_value->'status'->>'running' = 'true'
                 AND last_run_at >= NOW() - INTERVAL '60 seconds'
               RETURNING param_name""",
            self._sync_key,
            owner,
            status.model_dump_json(),
        )
        return bool(rows)

    async def finish(self, owner: str, status: ROISyncStatus, report: ROIReport | ObservedData | None = None) -> bool:
        report_json: Final = (
            report.model_dump_json()
            if isinstance(report, ObservedData)
            else TypeAdapter(ROIReport).dump_json(report).decode()
            if report is not None
            else None
        )
        rows: Final = await self._db.query_raw(
            """WITH owned AS (
                   SELECT param_name FROM "LiteLLM_Config"
                   WHERE param_name = $1 AND param_value->>'owner' = $2
                     AND last_run_at >= NOW() - INTERVAL '60 seconds'
                     AND ($4::text IS NULL OR param_value->>'cancel' = 'false')
                   FOR UPDATE
               ), report_write AS (
                   INSERT INTO "LiteLLM_Config" (param_name, param_value)
                   SELECT $5, $4::jsonb FROM owned WHERE $4::text IS NOT NULL
                   ON CONFLICT (param_name) DO UPDATE SET param_value = EXCLUDED.param_value
               ), cache_cleanup AS (
                   DELETE FROM "LiteLLM_Config" cached
                   WHERE starts_with(cached.param_name, 'roi_calculator_pull_')
                     AND EXISTS (SELECT 1 FROM owned) AND $4::text IS NOT NULL
                     AND EXISTS (
                         SELECT 1 FROM jsonb_array_elements($4::jsonb->'pulls') pull
                         WHERE pull->>'url' = cached.param_value->>'url'
                           AND pull->'estimate'->>'status' = 'estimated'
                           AND pull->>'cache_key' IS NOT NULL
                           AND cached.param_name <> 'roi_calculator_pull_' || (pull->>'cache_key')
                     )
               )
               UPDATE "LiteLLM_Config" SET param_value = jsonb_set(param_value, '{status}', $3::jsonb),
                   last_run_at = NOW()
               WHERE param_name IN (SELECT param_name FROM owned) RETURNING param_name""",
            self._sync_key,
            owner,
            status.model_dump_json(),
            report_json,
            self._report_key,
        )
        return bool(rows)

    async def status(self) -> ROISyncStatus | None:
        rows: Final = TypeAdapter(tuple[_StateRow, ...]).validate_python(
            await self._db.query_raw(
                """SELECT param_value, last_run_at, last_run_at < NOW() - INTERVAL '60 seconds' AS expired
               FROM "LiteLLM_Config" WHERE param_name = $1""",
                self._sync_key,
            )
        )
        if not rows:
            return None
        status: Final = rows[0].param_value.status
        if rows[0].expired and status.running:
            return status.model_copy(
                update=MappingProxyType(
                    {
                        "running": False,
                        "phase": "error",
                        "finished_at": rows[0].last_run_at.replace(tzinfo=timezone.utc).isoformat(),
                        "stage": "Sync interrupted",
                        "error": "The worker stopped responding. Sync again to refresh the report.",
                    }
                )
            )
        return status

    async def cancel(self) -> None:
        await self._db.execute_raw(
            """UPDATE "LiteLLM_Config"
               SET param_value = param_value || jsonb_build_object(
                   'cancel', true, 'owner', '',
                   'status', (param_value->'status') || jsonb_build_object(
                       'running', false, 'phase', 'cancelled', 'stage', 'Sync cancelled',
                       'finished_at', to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"')
                   )
               ), last_run_at = NOW()
               WHERE param_name = $1 AND param_value->'status'->>'running' = 'true' """,
            self._sync_key,
        )

    async def clear_report(self) -> None:
        await self._db.execute_raw('DELETE FROM "LiteLLM_Config" WHERE param_name = $1', self._report_key)

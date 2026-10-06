import asyncio
import json
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.lens.models import Job, Lens, Scope, TraceFindingCount, TraceIdentity, Worker
from litellm.types.llms.base import LiteLLMBaseModel


class Database(Protocol):
    def query_raw(self, query: str, *args: object) -> Awaitable[object]: ...
    def execute_raw(self, query: str, *args: object) -> Awaitable[int]: ...


class Row(LiteLLMBaseModel):
    data: JsonValue


_ROWS: Final = TypeAdapter(tuple[Row, ...])
UPDATE_ATTEMPTS: Final = 40
UPDATE_BACKOFF_SECONDS: Final = 0.02


class LensRepository:
    def __init__(self, db: Database, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.db: Final = db
        self.sleep: Final = sleep

    async def lenses(self) -> tuple[Lens, ...]:
        rows: Final = _ROWS.validate_python(await self.db.query_raw('SELECT data FROM "LiteLLM_Lens" ORDER BY id'))
        return tuple(Lens.model_validate(row.data) for row in rows)

    async def get(self, lens_id: str) -> Lens | None:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                'SELECT data FROM "LiteLLM_Lens" WHERE id=$1',
                lens_id,
            )
        )
        return Lens.model_validate(rows[0].data) if rows else None

    async def create(self, lens: Lens) -> Lens:
        await self.db.execute_raw(
            'INSERT INTO "LiteLLM_Lens" (id, version, data) VALUES ($1,0,$2::jsonb)',
            lens.id,
            lens.model_dump_json(),
        )
        return lens

    async def update(
        self,
        lens_id: str,
        transform: Callable[[Lens], Lens],
        attempts: int = UPDATE_ATTEMPTS,
        *,
        changed_only: bool = False,
    ) -> Lens | None:
        for attempt in range(attempts):
            completed, updated = await self._try_update(lens_id, transform, changed_only)
            if completed:
                return updated
            await self.sleep(random.uniform(0, UPDATE_BACKOFF_SECONDS * min(attempt + 1, 8)))
        return None

    async def _try_update(
        self, lens_id: str, transform: Callable[[Lens], Lens], changed_only: bool
    ) -> tuple[bool, Lens | None]:
        previous: Final = await self.get(lens_id)
        if previous is None:
            return True, None
        candidate: Final = transform(previous)
        if candidate == previous:
            return True, None if changed_only else previous
        updated: Final = candidate.model_copy(update=MappingProxyType({"version": previous.version + 1}))
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """WITH previous AS MATERIALIZED (
                SELECT data FROM "LiteLLM_Lens" WHERE id=$2 AND version=$3 FOR UPDATE
            ), updated AS (
                UPDATE "LiteLLM_Lens" SET data=$1::jsonb, version=version+1
                WHERE id=$2 AND version=$3 AND EXISTS (SELECT 1 FROM previous) RETURNING id
            )
            , archived AS (INSERT INTO "LiteLLM_LensRun" (id, lens_id, created_at, data)
            SELECT job->>'id', $2, (job->>'created_at')::timestamp, job
            FROM previous, jsonb_array_elements(previous.data->'jobs') AS job
            WHERE EXISTS (SELECT 1 FROM updated)
              AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements(($1::jsonb)->'jobs') AS retained
                              WHERE retained->>'id'=job->>'id')
            ON CONFLICT (id) DO NOTHING)
            SELECT to_jsonb(count(*)) AS data FROM updated""",
                updated.model_dump_json(),
                lens_id,
                previous.version,
            )
        )
        return bool(rows and rows[0].data == 1), updated

    async def jobs(self, lens_id: str, offset: int = 0) -> tuple[Job, ...]:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """SELECT data FROM (
                SELECT data FROM "LiteLLM_LensRun" WHERE lens_id=$1
                UNION ALL
                SELECT jsonb_array_elements(data->'jobs') AS data FROM "LiteLLM_Lens" WHERE id=$1
            ) AS jobs ORDER BY data->>'created_at' DESC, data->>'id' DESC LIMIT 50 OFFSET $2""",
                lens_id,
                offset,
            )
        )
        return tuple(Job.model_validate(row.data) for row in rows)

    async def job(self, lens_id: str, job_id: str) -> Job | None:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """SELECT data FROM "LiteLLM_LensRun" WHERE lens_id=$1 AND id=$2
            UNION ALL SELECT job AS data FROM "LiteLLM_Lens", jsonb_array_elements(data->'jobs') AS job
            WHERE id=$1 AND job->>'id'=$2 LIMIT 1""",
                lens_id,
                job_id,
            )
        )
        return Job.model_validate(rows[0].data) if rows else None

    async def trace_findings(self, traces: tuple[TraceIdentity, ...]) -> tuple[TraceFindingCount, ...]:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """WITH targets AS (
                    SELECT DISTINCT trace_id, trace_ref,
                        jsonb_build_array(jsonb_build_object('source', 'traces', 'trace_id', trace_id)) AS executions
                    FROM jsonb_to_recordset($1::jsonb) AS target(trace_id text, trace_ref text)
                ), jobs AS (
                    SELECT target.trace_id, target.trace_ref, run.data AS job
                    FROM targets AS target JOIN "LiteLLM_LensRun" AS run
                        ON run.data->'sample'->'executions' @> target.executions
                    WHERE run.data->>'status'='completed'
                    UNION ALL
                    SELECT target.trace_id, target.trace_ref, job
                    FROM targets AS target JOIN "LiteLLM_Lens" AS lens
                        ON lens.data->'jobs' @> jsonb_build_array(jsonb_build_object(
                            'status', 'completed', 'sample', jsonb_build_object('executions', target.executions)))
                    CROSS JOIN LATERAL jsonb_array_elements(lens.data->'jobs') AS job
                    WHERE job->>'status'='completed'
                ), assessed AS (
                    SELECT jobs.trace_id, jobs.trace_ref, execution->>'id' AS execution_id, job
                    FROM jobs, jsonb_array_elements(job->'sample'->'executions') AS execution
                    WHERE execution->>'trace_id'=jobs.trace_id
                        AND COALESCE(execution->>'trace_ref', '')=jobs.trace_ref
                        AND execution->>'source'='traces' AND EXISTS (
                        SELECT 1 FROM jsonb_array_elements(job->'assessments') AS assessment
                        WHERE assessment->>'execution_id'=execution->>'id'
                            AND COALESCE((assessment->>'cannot_assess')::boolean, false)=false
                    )
                )
                SELECT jsonb_build_object(
                    'trace_id', target.trace_id, 'trace_ref', target.trace_ref,
                    'finding_count', CASE WHEN count(assessed.execution_id)=0 THEN NULL
                        ELSE count(DISTINCT finding->>'id') END
                ) AS data FROM targets AS target
                LEFT JOIN assessed USING (trace_id, trace_ref)
                LEFT JOIN LATERAL jsonb_array_elements(NULLIF(assessed.job->'findings', 'null'::jsonb)) AS finding
                    ON finding->'occurrences' ? assessed.execution_id
                GROUP BY target.trace_id, target.trace_ref""",
                json.dumps(tuple(trace.model_dump() for trace in traces)),
            )
        )
        return tuple(TraceFindingCount.model_validate(row.data) for row in rows)

    async def workers(self) -> tuple[Worker, ...]:
        rows: Final = _ROWS.validate_python(await self.db.query_raw('SELECT data FROM "LiteLLM_LensWorker"'))
        return tuple(Worker.model_validate(row.data) for row in rows)

    async def eligible_workers(self, scope: Scope) -> AsyncIterator[Worker]:
        scoped: Final = (
            {"all_teams": True}
            if scope.all_teams
            else {"team_id": scope.team_id}
            if scope.team_id
            else {"team_id": "", "api_key_hash": scope.api_key_hash}
        )
        cursor = ""  # rebind-ok: advance the keyset cursor after each bounded page
        while True:
            rows = _ROWS.validate_python(
                await self.db.query_raw(
                    """SELECT data FROM "LiteLLM_LensWorker"
                    WHERE data @> '{"revoked": false}'::jsonb AND id > $1
                    AND (data->'scope' @> '{"all_teams": true}'::jsonb OR data->'scope' @> $2::jsonb)
                    ORDER BY id LIMIT 50""",
                    cursor,
                    json.dumps(scoped),
                )
            )
            workers = tuple(Worker.model_validate(row.data) for row in rows)
            for worker in workers:
                yield worker
            if len(workers) < 50:
                return
            cursor = workers[-1].id

    async def worker(self, token_hash: str) -> Worker | None:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                'SELECT data FROM "LiteLLM_LensWorker" WHERE token_hash=$1',
                token_hash,
            )
        )
        return Worker.model_validate(rows[0].data) if rows else None

    async def save_worker(self, worker: Worker, token_hash: str | None = None) -> None:
        if token_hash is not None:
            await self.db.execute_raw(
                'INSERT INTO "LiteLLM_LensWorker" (id,token_hash,data) VALUES ($1,$2,$3::jsonb)',
                worker.id,
                token_hash,
                worker.model_dump_json(),
            )
            return
        await self.db.execute_raw(
            'UPDATE "LiteLLM_LensWorker" SET data=$1::jsonb WHERE id=$2', worker.model_dump_json(), worker.id
        )

    async def set_worker_billing(self, worker_id: str, key_id: str) -> Worker | None:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """UPDATE "LiteLLM_LensWorker"
                SET data=jsonb_set(data, '{analysis_key_id}', to_jsonb($1::text))
                WHERE id=$2 AND COALESCE((data->>'revoked')::boolean, false)=false RETURNING data""",
                key_id,
                worker_id,
            )
        )
        return Worker.model_validate(rows[0].data) if rows else None

    async def revoke_worker(self, worker_id: str) -> None:
        await self.db.execute_raw(
            """UPDATE "LiteLLM_LensWorker" SET data=jsonb_set(data, '{revoked}', 'true') WHERE id=$1""",
            worker_id,
        )

    async def heartbeat(self, worker_id: str, now: str) -> None:
        await self.db.execute_raw(
            """UPDATE "LiteLLM_LensWorker" SET data=jsonb_set(data, '{last_seen}', to_jsonb($1::text)) WHERE id=$2""",
            now,
            worker_id,
        )


class WriterDatabase:
    def __init__(self, writer: PrismaWrapper) -> None:
        self.writer: Final = writer

    async def query_raw(self, query: str, *args: object) -> object:
        return _ROWS.validate_python(await self.writer.query_raw(query, *args))  # pyright: ignore[reportAny]  # Prisma forwards dynamically; validate rows here.

    async def execute_raw(self, query: str, *args: object) -> int:
        return TypeAdapter(int).validate_python(await self.writer.execute_raw(query, *args))  # pyright: ignore[reportAny]  # Prisma forwards dynamically; validate the count here.

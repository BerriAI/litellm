import asyncio
import json
import random
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter
from typing_extensions import LiteralString

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.lens.models import (
    Job,
    Lens,
    Progress,
    Review,
    ReviewVersion,
    Scope,
    TraceFindingCount,
    TraceIdentity,
    Worker,
)
from litellm.proxy.lens.reviews import criteria_key
from litellm.proxy.lens.state import apply_progress, current_job, due_at, replace_job
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from prisma import Prisma


class Database(Protocol):
    def query_raw(self, query: LiteralString, *args: object) -> Awaitable[object]: ...
    def execute_raw(self, query: LiteralString, *args: object) -> Awaitable[int]: ...
    def transaction(self) -> AbstractAsyncContextManager["Database"]: ...


class Row(LiteLLMBaseModel):
    data: JsonValue
    due_at: datetime | None = None


class DueRow(LiteLLMBaseModel):
    data: JsonValue
    due_at: datetime


@dataclass(frozen=True, slots=True)
class DueLens:
    lens: Lens
    due_at: datetime


class FindingRun(LiteLLMBaseModel):
    finding_id: str
    job_id: str


_ROWS: Final = TypeAdapter(tuple[Row, ...])
_DUE_ROWS: Final = TypeAdapter(tuple[DueRow, ...])
_DUE_QUERY: Final[LiteralString] = """SELECT data, due_at FROM "LiteLLM_Lens"
WHERE due_at IS NOT NULL AND due_at <= ($4::timestamptz AT TIME ZONE 'UTC')
AND ($1::boolean OR (
    COALESCE((data->'scope'->>'all_teams')::boolean, false) IS NOT TRUE
    AND COALESCE(data->'scope'->>'team_id', '')=$2
    AND ($2 <> '' OR COALESCE(data->'scope'->>'api_key_hash', '')=$3)
))
ORDER BY due_at, id
LIMIT $5"""
_DUE_AFTER_QUERY: Final[LiteralString] = """SELECT data, due_at FROM "LiteLLM_Lens"
WHERE due_at IS NOT NULL AND due_at <= ($4::timestamptz AT TIME ZONE 'UTC')
AND (due_at, id) > ($6::timestamp, $7)
AND ($1::boolean OR (
    COALESCE((data->'scope'->>'all_teams')::boolean, false) IS NOT TRUE
    AND COALESCE(data->'scope'->>'team_id', '')=$2
    AND ($2 <> '' OR COALESCE(data->'scope'->>'api_key_hash', '')=$3)
))
ORDER BY due_at, id
LIMIT $5"""
UPDATE_ATTEMPTS: Final = 40
UPDATE_BACKOFF_SECONDS: Final = 0.02


class LensRepository:
    def __init__(self, db: Database, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.db: Final = db
        self.sleep: Final = sleep

    async def finding_runs(self, lens_id: str, finding_ids: tuple[str, ...]) -> tuple[FindingRun, ...]:
        if not finding_ids:
            return ()
        rows: Final = await self.db.query_raw(
            """WITH jobs AS (
                SELECT data FROM "LiteLLM_LensRun" WHERE lens_id=$1
                UNION ALL
                SELECT jsonb_array_elements(data->'jobs') FROM "LiteLLM_Lens" WHERE id=$1
            )
            SELECT DISTINCT jsonb_build_object('finding_id', finding->>'id', 'job_id', jobs.data->>'id') AS data
            FROM jobs, jsonb_array_elements(NULLIF(jobs.data->'findings', 'null'::jsonb)) AS finding
            WHERE finding->>'id'=ANY($2::text[])""",
            lens_id,
            finding_ids,
        )
        return tuple(FindingRun.model_validate(row.data) for row in _ROWS.validate_python(rows))

    async def reviews(self, lens_id: str, job: Job) -> tuple[Review, ...]:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                'SELECT data FROM "LiteLLM_LensReview" WHERE lens_id=$1 AND criteria_key=$2 '
                "AND execution_id=ANY($3::text[])",
                lens_id,
                criteria_key(job.settings),
                tuple(e.id for e in job.sample.executions) if job.sample else (),
            )
        )
        return tuple(Review.model_validate(row.data) for row in rows)

    @asynccontextmanager
    async def locked(self, lens_id: str) -> AsyncGenerator["LensRepository"]:
        async with self.db.transaction() as db:
            await db.query_raw('SELECT data FROM "LiteLLM_Lens" WHERE id=$1 FOR UPDATE', lens_id)
            yield LensRepository(db, self.sleep)

    async def update_locked(self, lens_id: str, transform: Callable[[Lens], Lens]) -> Lens | None:
        async with self.locked(lens_id) as repo:
            return await repo.update(lens_id, transform, attempts=1)

    async def progress(self, lens_id: str, assigned: Job, body: Progress) -> Lens | None:
        async with self.locked(lens_id) as repo:

            def renew(lens: Lens) -> Lens:
                job: Final = current_job(lens)
                now: Final = datetime.now(timezone.utc)
                if (
                    job is None
                    or job.id != assigned.id
                    or job.worker_id != assigned.worker_id
                    or job.attempts != assigned.attempts
                    or job.status != "running"
                    or job.lease_until is None
                    or job.lease_until <= now
                ):
                    raise HTTPException(409, "This worker no longer owns the job")
                return replace_job(lens, apply_progress(job, body, now))

            updated: Final = await repo.update(lens_id, renew, attempts=1)
            if updated is not None and body.review is not None:
                await repo._save_review(lens_id, assigned, body.review)
            return updated

    async def _save_review(self, lens_id: str, job: Job, review: Review) -> None:
        if review.reused or review.extraction is None or not review.content_version:
            return
        await self.db.execute_raw(
            'INSERT INTO "LiteLLM_LensReview" (lens_id, criteria_key, execution_id, data) '
            "VALUES ($1,$2,$3,$4::jsonb) ON CONFLICT (lens_id, criteria_key, execution_id) "
            "DO UPDATE SET data=EXCLUDED.data",
            lens_id,
            criteria_key(job.settings),
            review.execution_id,
            review.model_dump_json(),
        )

    async def complete_reviews(self, lens_id: str, job: Job, versions: tuple[ReviewVersion, ...]) -> None:
        await self.db.execute_raw(
            """UPDATE "LiteLLM_LensReview" AS review SET data=jsonb_set(data, '{consolidated}', 'true')
            FROM jsonb_to_recordset($3::jsonb) AS version(execution_id text, content_version text)
            WHERE review.lens_id=$1 AND review.criteria_key=$2 AND review.execution_id=version.execution_id
            AND review.data->>'content_version'=version.content_version""",
            lens_id,
            criteria_key(job.settings),
            json.dumps(tuple(version.model_dump() for version in versions)),
        )

    async def lenses(self) -> tuple[Lens, ...]:
        rows: Final = _ROWS.validate_python(await self.db.query_raw('SELECT data FROM "LiteLLM_Lens" ORDER BY id'))
        return tuple(Lens.model_validate(row.data) for row in rows)

    async def due(self, scope: Scope, now: datetime, limit: int, after: DueLens | None = None) -> tuple[DueLens, ...]:
        query: Final[LiteralString] = _DUE_QUERY if after is None else _DUE_AFTER_QUERY
        parameters: Final[tuple[object, ...]] = (
            (
                scope.all_teams,
                scope.team_id,
                scope.api_key_hash,
                now.isoformat(),
                limit,
            )
            if after is None
            else (
                scope.all_teams,
                scope.team_id,
                scope.api_key_hash,
                now.isoformat(),
                limit,
                after.due_at,
                after.lens.id,
            )
        )
        rows: Final = _DUE_ROWS.validate_python(await self.db.query_raw(query, *parameters), from_attributes=True)
        return tuple(DueLens(lens=Lens.model_validate(row.data), due_at=row.due_at) for row in rows)

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
            """INSERT INTO "LiteLLM_Lens" (id, version, data, due_at)
            VALUES ($1,0,$2::jsonb,($3::timestamptz AT TIME ZONE 'UTC'))""",
            lens.id,
            lens.model_dump_json(),
            scheduled_at.isoformat() if (scheduled_at := due_at(lens)) else None,
        )
        return lens

    async def sync_due(self, lens: Lens) -> None:
        await self.db.execute_raw(
            """UPDATE "LiteLLM_Lens"
            SET due_at=($3::timestamptz AT TIME ZONE 'UTC')
            WHERE id=$1 AND version=$2
              AND due_at IS DISTINCT FROM ($3::timestamptz AT TIME ZONE 'UTC')""",
            lens.id,
            lens.version,
            scheduled_at.isoformat() if (scheduled_at := due_at(lens)) else None,
        )

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
                UPDATE "LiteLLM_Lens" SET data=$1::jsonb, version=version+1,
                    due_at=($4::timestamptz AT TIME ZONE 'UTC')
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
                scheduled_at.isoformat() if (scheduled_at := due_at(updated)) else None,
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
    def __init__(self, writer: "PrismaWrapper | Prisma") -> None:
        self.writer: Final = writer

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Database]:
        async with self.writer.tx(max_wait=timedelta(seconds=30), timeout=timedelta(seconds=30)) as tx:
            yield WriterDatabase(tx)

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        return _ROWS.validate_python(await self.writer.query_raw(query, *args))  # pyright: ignore[reportAny]  # Prisma forwards dynamically; validate rows here.

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
        return TypeAdapter(int).validate_python(await self.writer.execute_raw(query, *args))  # pyright: ignore[reportAny]  # Prisma forwards dynamically; validate the count here.

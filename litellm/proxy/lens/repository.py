import asyncio
import json
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import Final, Protocol
from uuid import uuid4

from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.lens.models import MAX_STEPS, Job, Lens, Scope, Step, Worker
from litellm.proxy.lens.search import FIELD_VALUES, LensField, LensSearch, like_literal, search_predicate


class Database(Protocol):
    def query_raw(self, query: str, *args: object) -> Awaitable[object]: ...
    def execute_raw(self, query: str, *args: object) -> Awaitable[int]: ...


class Row(BaseModel):
    data: JsonValue


_ROWS: Final = TypeAdapter(tuple[Row, ...])


class RunLedger(BaseModel):
    run_id: str
    cost: float
    steps: tuple[Step, ...]


class StepRecord(BaseModel):
    """A `LiteLLM_LensRun` row holding one step of a run; archived run rows hold a `Job` instead."""

    run_id: str
    cost: float
    settled: bool
    step: Step


@dataclass(frozen=True, slots=True)
class Reservation:
    id: str
    lens_id: str
    month: str
    estimate: float


_DOCUMENT_SHAPE: Final = {"spent": True, "budget_month": True, "jobs": {"__all__": {"cost", "steps"}}}
_SPENT: Final = (
    """CASE WHEN data->>'budget_month'={month} THEN COALESCE((data->>'spent')::double precision, 0) ELSE 0 END"""
)


def document(lens: Lens) -> str:
    """The document shape: the server-owned spend counter and the run steps are kept out of it."""
    return lens.model_dump_json(exclude=_DOCUMENT_SHAPE)


def month_of(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m")


class LensRepository:
    def __init__(self, db: Database, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.db: Final = db
        self.sleep: Final = sleep

    async def lenses(self) -> tuple[Lens, ...]:
        rows: Final = _ROWS.validate_python(await self.db.query_raw('SELECT data FROM "LiteLLM_Lens" ORDER BY id'))
        return await self.hydrate_lenses(tuple(Lens.model_validate(row.data) for row in rows))

    async def search(self, search: LensSearch) -> tuple[Lens, ...]:
        where, args = search_predicate(search)
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                f"""SELECT data FROM "LiteLLM_Lens" WHERE {where}
                ORDER BY (data->>'created_at')::timestamptz DESC, id""",
                *args,
            )
        )
        return await self.hydrate_lenses(tuple(Lens.model_validate(row.data) for row in rows))

    async def values(self, field: LensField, contains: str, limit: int) -> tuple[str, ...]:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                f"""SELECT v AS data FROM "LiteLLM_Lens", unnest(array_remove({FIELD_VALUES[field]}, '')) AS v
                WHERE v ILIKE $1 GROUP BY v ORDER BY count(*) DESC, v LIMIT $2""",
                f"%{like_literal(contains)}%",
                limit,
            )
        )
        return tuple(TypeAdapter(str).validate_python(row.data) for row in rows)

    async def get(self, lens_id: str) -> Lens | None:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw('SELECT data FROM "LiteLLM_Lens" WHERE id=$1', lens_id)
        )
        if not rows:
            return None
        hydrated: Final = await self.hydrate_lenses((Lens.model_validate(rows[0].data),))
        return hydrated[0]

    async def create(self, lens: Lens) -> Lens:
        await self.db.execute_raw(
            """INSERT INTO "LiteLLM_Lens" (id, version, data)
            VALUES ($1, 0, $2::jsonb || jsonb_build_object('spent', $3::double precision, 'budget_month', $4::text))""",
            lens.id,
            document(lens),
            lens.spent,
            lens.budget_month,
        )
        return lens

    async def update(
        self, lens_id: str, transform: Callable[[Lens], Lens], attempts: int = 8, *, changed_only: bool = False
    ) -> Lens | None:
        for attempt in range(attempts):
            if attempt:
                await self.sleep(random.uniform(0, 0.02 * (1 << attempt)))
            completed, updated = await self._try_update(lens_id, transform, changed_only)
            if completed:
                return updated
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
                UPDATE "LiteLLM_Lens"
                SET data=$1::jsonb || jsonb_build_object(
                    'spent', COALESCE(data->'spent', '0'::jsonb), 'budget_month', COALESCE(data->'budget_month', '""'::jsonb)
                ), version=version+1
                WHERE id=$2 AND version=$3 AND EXISTS (SELECT 1 FROM previous) RETURNING data
            )
            , archived AS (INSERT INTO "LiteLLM_LensRun" (id, lens_id, created_at, data)
            SELECT job->>'id', $2, (job->>'created_at')::timestamp, job
            FROM previous, jsonb_array_elements(previous.data->'jobs') AS job
            WHERE EXISTS (SELECT 1 FROM updated)
              AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements(($1::jsonb)->'jobs') AS retained
                              WHERE retained->>'id'=job->>'id')
            ON CONFLICT (id) DO NOTHING)
            SELECT data FROM updated""",
                document(updated),
                lens_id,
                previous.version,
            )
        )
        if not rows:
            return False, None
        hydrated: Final = await self.hydrate_lenses((Lens.model_validate(rows[0].data),))
        return True, hydrated[0]

    async def jobs(self, lens_id: str, offset: int = 0) -> tuple[Job, ...]:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """SELECT data FROM (
                SELECT data FROM "LiteLLM_LensRun" WHERE lens_id=$1 AND data ? 'status'
                UNION ALL
                SELECT jsonb_array_elements(data->'jobs') AS data FROM "LiteLLM_Lens" WHERE id=$1
            ) AS jobs ORDER BY data->>'created_at' DESC, data->>'id' DESC LIMIT 50 OFFSET $2""",
                lens_id,
                offset,
            )
        )
        return await self.hydrate_jobs(lens_id, tuple(Job.model_validate(row.data) for row in rows))

    async def job(self, lens_id: str, job_id: str) -> Job | None:
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """SELECT data FROM "LiteLLM_LensRun" WHERE lens_id=$1 AND id=$2 AND data ? 'status'
            UNION ALL SELECT job AS data FROM "LiteLLM_Lens", jsonb_array_elements(data->'jobs') AS job
            WHERE id=$1 AND job->>'id'=$2 LIMIT 1""",
                lens_id,
                job_id,
            )
        )
        if not rows:
            return None
        hydrated: Final = await self.hydrate_jobs(lens_id, (Job.model_validate(rows[0].data),))
        return hydrated[0]

    async def hydrate_lenses(self, lenses: tuple[Lens, ...]) -> tuple[Lens, ...]:
        ledgers: Final = await self.ledgers(tuple(chain.from_iterable(embedded_runs(lens) for lens in lenses)))
        return tuple(
            lens.model_copy(update=MappingProxyType({"jobs": tuple(with_ledger(job, ledgers) for job in lens.jobs)}))
            for lens in lenses
        )

    async def hydrate_jobs(self, lens_id: str, jobs: tuple[Job, ...]) -> tuple[Job, ...]:
        ledgers: Final = await self.ledgers(tuple((lens_id, job.id) for job in jobs))
        return tuple(with_ledger(job, ledgers) for job in jobs)

    async def ledgers(self, runs: tuple[tuple[str, str], ...]) -> Mapping[str, RunLedger]:
        """Cost and the latest `MAX_STEPS` steps of each `(lens_id, run_id)`."""
        if not runs:
            return MappingProxyType({})
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                """SELECT jsonb_build_object(
                'run_id', run.id,
                'cost', (SELECT COALESCE(sum((r.data->>'cost')::double precision), 0) FROM "LiteLLM_LensRun" r
                         WHERE r.lens_id=run.lens_id AND r.data->>'run_id'=run.id),
                'steps', (SELECT COALESCE(jsonb_agg(latest.step ORDER BY latest.at, latest.id), '[]'::jsonb)
                          FROM (SELECT r.data->'step' AS step, r.created_at AS at, r.id FROM "LiteLLM_LensRun" r
                                WHERE r.lens_id=run.lens_id AND r.data->>'run_id'=run.id
                                ORDER BY r.created_at DESC, r.id DESC LIMIT $2) AS latest)
            ) AS data FROM jsonb_to_recordset($1::jsonb) AS run(lens_id text, id text)""",
                json.dumps([{"lens_id": lens_id, "id": run_id} for lens_id, run_id in runs]),
                MAX_STEPS,
            )
        )
        return MappingProxyType({ledger.run_id: ledger for ledger in (RunLedger.model_validate(r.data) for r in rows)})

    async def reserve(self, lens_id: str, run_id: str, step: Step) -> Reservation | None:
        """Charge `step.cost` against the month's budget and record the pending step; None when the budget is reached."""
        reservation: Final = Reservation(id=str(uuid4()), lens_id=lens_id, month=month_of(step.at), estimate=step.cost)
        record: Final = StepRecord(run_id=run_id, cost=step.cost, settled=False, step=step)
        rows: Final = _ROWS.validate_python(
            await self.db.query_raw(
                f"""WITH reserved AS (
                UPDATE "LiteLLM_Lens"
                SET data = data || jsonb_build_object('spent', {_SPENT.format(month="$2::text")} + $3, 'budget_month', $2::text)
                WHERE id=$1 AND {_SPENT.format(month="$2::text")} + $3
                    <= (data->'settings'->>'monthly_budget')::double precision
                RETURNING id
            )
            INSERT INTO "LiteLLM_LensRun" (id, lens_id, created_at, data)
            SELECT $4, $1, $5::timestamptz AT TIME ZONE 'UTC', $6::jsonb FROM reserved
            RETURNING to_jsonb(id) AS data""",
                lens_id,
                reservation.month,
                step.cost,
                reservation.id,
                step.at.isoformat(),
                record.model_dump_json(),
            )
        )
        return reservation if rows else None

    async def settle(self, reservation: Reservation, step: Step) -> None:
        await self.db.execute_raw(
            """WITH settled AS (
                UPDATE "LiteLLM_LensRun"
                SET data = data || jsonb_build_object('cost', $2::double precision, 'settled', true, 'step', $3::jsonb)
                WHERE id=$1 AND data->>'settled'='false' RETURNING lens_id
            )
            UPDATE "LiteLLM_Lens" l
            SET data = l.data || jsonb_build_object('spent', (l.data->>'spent')::double precision - $4 + $2)
            FROM settled WHERE l.id=settled.lens_id AND l.data->>'budget_month'=$5""",
            reservation.id,
            step.cost,
            step.model_dump_json(),
            reservation.estimate,
            reservation.month,
        )

    async def release(self, reservation: Reservation) -> None:
        await self.db.execute_raw(
            """WITH released AS (
                DELETE FROM "LiteLLM_LensRun" WHERE id=$1 AND data->>'settled'='false' RETURNING lens_id
            )
            UPDATE "LiteLLM_Lens" l
            SET data = l.data || jsonb_build_object('spent', (l.data->>'spent')::double precision - $2)
            FROM released WHERE l.id=released.lens_id AND l.data->>'budget_month'=$3""",
            reservation.id,
            reservation.estimate,
            reservation.month,
        )

    async def record_step(self, lens_id: str, run_id: str, step: Step) -> None:
        record: Final = StepRecord(run_id=run_id, cost=step.cost, settled=True, step=step)
        await self.db.execute_raw(
            """INSERT INTO "LiteLLM_LensRun" (id, lens_id, created_at, data)
            VALUES ($1, $2, $3::timestamptz AT TIME ZONE 'UTC', $4::jsonb)""",
            str(uuid4()),
            lens_id,
            step.at.isoformat(),
            record.model_dump_json(),
        )

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


def embedded_runs(lens: Lens) -> tuple[tuple[str, str], ...]:
    return tuple((lens.id, job.id) for job in lens.jobs)


def with_ledger(job: Job, ledgers: Mapping[str, RunLedger]) -> Job:
    """Runs recorded before the ledger existed keep the steps embedded in their document."""
    ledger: Final = ledgers.get(job.id)
    if ledger is None or not ledger.steps:
        return job
    return job.model_copy(update=MappingProxyType({"cost": ledger.cost, "steps": ledger.steps}))


class WriterDatabase:
    def __init__(self, writer: PrismaWrapper) -> None:
        self.writer: Final = writer

    async def query_raw(self, query: str, *args: object) -> object:
        return _ROWS.validate_python(await self.writer.query_raw(query, *args))  # pyright: ignore[reportAny]  # Prisma forwards dynamically; validate rows here.

    async def execute_raw(self, query: str, *args: object) -> int:
        return TypeAdapter(int).validate_python(await self.writer.execute_raw(query, *args))  # pyright: ignore[reportAny]  # Prisma forwards dynamically; validate the count here.

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, TypeAdapter

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.routing_prisma_wrapper import writer_wrapper
from litellm.proxy.engine.models import (
    Claim,
    Engine,
    EngineList,
    EngineSettings,
    Execution,
    ExecutionContent,
    FindingDraft,
    FindingUpdate,
    Job,
    ModelRequest,
    ModelResult,
    Progress,
    Result,
    RunRequest,
    Sample,
    Scope,
    Worker,
    WorkerCreated,
)
from litellm.proxy.engine.repository import EngineRepository, WriterDatabase
from litellm.proxy.engine.sources import SourceReader, parse_execution
from litellm.proxy.engine.state import can_access, claim_job, current_job, merge_finding, queue_job, replace_job

router: Final = APIRouter(prefix="/engine", tags=["Lens"])  # mutable-ok: FastAPI requires list
_bearer: Final = HTTPBearer()
Auth: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]


def repository() -> EngineRepository:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(503, "Lens needs a connected Postgres database")
    return EngineRepository(WriterDatabase(writer_wrapper(prisma_client.db)))


def source_reader() -> SourceReader:
    from litellm.proxy.tracing_endpoints import get_receiver

    return SourceReader(get_receiver().store.storage)


def user_scope(auth: UserAPIKeyAuth, write: bool = False) -> Scope:
    if write and auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(403, "Only proxy admins can configure or run Lens")
    if auth.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        return Scope(all_teams=True)
    if auth.team_id:
        return Scope(team_id=auth.team_id)
    if auth.token:
        return Scope(api_key_hash=auth.token)
    raise HTTPException(403, "A team or API key is required")


async def get_engine(engine_id: str, scope: Scope) -> Engine:
    engine: Final = await repository().get(engine_id)
    if engine is None or not can_access(scope, engine.scope):
        raise HTTPException(404, "Lens not found")
    return engine


async def worker_auth(credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)]) -> Worker:
    worker: Final = await repository().worker(hashlib.sha256(credentials.credentials.encode()).hexdigest())
    if worker is None or worker.revoked:
        raise HTTPException(401, "Worker credential is invalid or revoked")
    return worker


WorkerAuth: TypeAlias = Annotated[Worker, Depends(worker_auth)]


async def assigned(engine_id: str, job_id: str, worker: Worker) -> tuple[Engine, Job]:
    engine: Final = await get_engine(engine_id, worker.scope)
    job: Final = current_job(engine)
    if (
        job is None
        or job.id != job_id
        or job.status != "running"
        or job.worker_id != worker.id
        or job.lease_until is None
        or job.lease_until <= datetime.now(UTC)
    ):
        raise HTTPException(409, "This worker no longer owns the job")
    return engine, job


def required(engine: Engine | None) -> Engine:
    if engine is None:
        raise HTTPException(409, "Lens changed concurrently; retry the operation")
    return engine


def validate_model(settings: EngineSettings, auth: UserAPIKeyAuth) -> None:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None or settings.model not in llm_router.get_model_names(team_id=auth.team_id):
        raise HTTPException(400, "Choose a model configured on this LiteLLM instance")
    allowed_models: Final = TypeAdapter(tuple[str, ...]).validate_python(auth.model_dump().get("models") or ())
    if (
        auth.user_role != LitellmUserRoles.PROXY_ADMIN
        and allowed_models
        and settings.model not in allowed_models
        and "all-proxy-models" not in allowed_models
    ):
        raise HTTPException(403, "This key does not have access to the analysis model")


@router.get("", response_model=EngineList)
async def list_engines(auth: Auth) -> EngineList:
    from litellm.proxy import tracing_endpoints

    scope: Final = user_scope(auth)
    return EngineList(
        engines=tuple(e for e in await repository().engines() if can_access(scope, e.scope)),
        workers=tuple(w for w in await repository().workers() if can_access(scope, w.scope)),
        tracing_enabled=tracing_endpoints.receiver is not None,
    )


@router.post("", response_model=Engine)
async def create_engine(settings: EngineSettings, auth: Auth) -> Engine:
    scope: Final = user_scope(auth, write=True)
    validate_model(settings, auth)
    now: Final = datetime.now(UTC)
    engine: Final = Engine(
        id=str(uuid4()),
        scope=scope,
        settings=settings,
        created_at=now,
        next_run_at=now,
        budget_month=now.strftime("%Y-%m"),
    )
    return await repository().create(queue_job(engine, now, str(uuid4())))


@router.put("/{engine_id}", response_model=Engine)
async def update_engine(engine_id: str, settings: EngineSettings, auth: Auth) -> Engine:
    await get_engine(engine_id, user_scope(auth, write=True))
    validate_model(settings, auth)
    return required(
        await repository().update(
            engine_id,
            lambda e: e.model_copy(
                update=MappingProxyType(
                    {
                        "settings": settings,
                        "revision": e.revision + 1,
                    }
                )
            ),
        )
    )


@router.post("/{engine_id}/runs", response_model=Engine)
async def run_engine(engine_id: str, body: RunRequest, auth: Auth) -> Engine:
    await get_engine(engine_id, user_scope(auth, write=True))
    now: Final = datetime.now(UTC)
    job_id: Final = str(uuid4())
    return required(await repository().update(engine_id, lambda e: queue_job(e, now, job_id, body.lookback_hours)))


@router.post("/{engine_id}/cancel", response_model=Engine)
async def cancel_engine(engine_id: str, auth: Auth) -> Engine:
    await get_engine(engine_id, user_scope(auth, write=True))
    now: Final = datetime.now(UTC)

    def cancel(e: Engine) -> Engine:
        job: Final = current_job(e)
        if job is None:
            return e
        cancelled: Final = job.model_copy(
            update=MappingProxyType({"status": "cancelled", "stage": "Cancelled", "finished_at": now})
        )
        return replace_job(e, cancelled).model_copy(
            update=MappingProxyType({"next_run_at": now + timedelta(minutes=e.settings.interval_minutes)})
        )

    return required(await repository().update(engine_id, cancel))


@router.patch("/{engine_id}/findings/{finding_id}", response_model=Engine)
async def update_finding(engine_id: str, finding_id: str, body: FindingUpdate, auth: Auth) -> Engine:
    await get_engine(engine_id, user_scope(auth, write=True))
    return required(
        await repository().update(
            engine_id,
            lambda e: e.model_copy(
                update=MappingProxyType(
                    {
                        "findings": tuple(
                            f.model_copy(update=body.model_dump()) if f.id == finding_id else f for f in e.findings
                        ),
                    }
                )
            ),
        )
    )


class Preview(BaseModel):
    settings: EngineSettings
    lookback_hours: int = Field(default=24, ge=1, le=720)


@router.post("/preview/sample", response_model=Sample)
async def preview_sample(body: Preview, auth: Auth) -> Sample:
    now: Final = datetime.now(UTC)
    return await source_reader().sample(
        user_scope(auth),
        body.settings,
        int((now - timedelta(hours=body.lookback_hours)).timestamp() * 1000),
        int((now - timedelta(minutes=2)).timestamp() * 1000),
    )


class WorkerName(BaseModel):
    name: str = Field(default="Lens worker", min_length=1, max_length=100)


@router.post("/workers/register", response_model=WorkerCreated)
async def register_worker(body: WorkerName, auth: Auth) -> WorkerCreated:
    scope: Final = user_scope(auth, write=True)
    token: Final = "lens-" + secrets.token_urlsafe(40)
    worker: Final = Worker(id=str(uuid4()), name=body.name, scope=scope, last_seen=datetime(1970, 1, 1, tzinfo=UTC))
    await repository().save_worker(worker, hashlib.sha256(token.encode()).hexdigest())
    return WorkerCreated(worker=worker, token=token)


@router.delete("/workers/{worker_id}")
async def revoke_worker(worker_id: str, auth: Auth) -> bool:
    scope: Final = user_scope(auth, write=True)
    worker: Final = next((w for w in await repository().workers() if w.id == worker_id), None)
    if worker is None or not can_access(scope, worker.scope):
        raise HTTPException(404, "Worker not found")
    await repository().save_worker(worker.model_copy(update=MappingProxyType({"revoked": True})))
    return True


@router.post("/worker/claim", response_model=Claim | None)
async def claim(worker: WorkerAuth) -> Claim | None:
    now: Final = datetime.now(UTC)
    await repository().heartbeat(worker.id, now.isoformat())
    for candidate in await repository().engines():
        if not can_access(worker.scope, candidate.scope):
            continue
        if claimed := await claim_candidate(candidate, worker, now):
            return claimed
    return None


@router.post("/worker/{engine_id}/{job_id}/progress", response_model=bool)
async def progress(engine_id: str, job_id: str, body: Progress, worker: WorkerAuth) -> bool:
    await assigned(engine_id, job_id, worker)
    now: Final = datetime.now(UTC)

    def renew(e: Engine) -> Engine:
        job: Final = current_job(e)
        if job is None or job.id != job_id or job.worker_id != worker.id:
            return e
        return replace_job(
            e,
            job.model_copy(
                update=MappingProxyType(
                    {"stage": body.stage, "coverage": body.coverage, "lease_until": now + timedelta(minutes=5)}
                )
            ),
        )

    required(await repository().update(engine_id, renew))
    await repository().heartbeat(worker.id, now.isoformat())
    return True


@router.get("/worker/{engine_id}/{job_id}/sample", response_model=Sample)
async def sample(engine_id: str, job_id: str, worker: WorkerAuth) -> Sample:
    engine, job = await assigned(engine_id, job_id, worker)
    if job.sample is not None:
        return job.sample
    selected: Final = await source_reader().sample(
        engine.scope, job.settings, int(job.start.timestamp() * 1000), int(job.end.timestamp() * 1000)
    )

    def freeze(e: Engine) -> Engine:
        active: Final = current_job(e)
        if active is None or active.id != job_id or active.worker_id != worker.id:
            raise HTTPException(409, "Job was cancelled or reassigned")
        return (
            replace_job(e, active.model_copy(update=MappingProxyType({"sample": selected})))
            if active.sample is None
            else e
        )

    updated: Final = required(await repository().update(engine_id, freeze))
    frozen: Final = next(j for j in updated.jobs if j.id == job_id).sample
    if frozen is None:
        raise HTTPException(409, "Could not freeze the sample")
    return frozen


@router.get("/worker/{engine_id}/{job_id}/content", response_model=ExecutionContent)
async def content(
    engine_id: str,
    job_id: str,
    execution_id: str,
    worker: WorkerAuth,
    cursor: str = "",
    offset: int = Query(default=0, ge=0, le=1000000),
) -> ExecutionContent:
    engine, job = await assigned(engine_id, job_id, worker)
    selected: Final = job.sample or Sample(executions=(), eligible=0)
    execution: Final = next((e for e in selected.executions if e.id == execution_id), None)
    if execution is None:
        raise HTTPException(404, "Execution is outside this job's sample")
    return await source_reader().content(engine.scope, execution, cursor, offset)


@router.post("/worker/{engine_id}/{job_id}/model", response_model=ModelResult)
async def model(engine_id: str, job_id: str, body: ModelRequest, worker: WorkerAuth) -> ModelResult:
    from litellm.proxy.engine.inference import analyze

    engine, job = await assigned(engine_id, job_id, worker)
    return await analyze(repository(), engine, job, worker.id, body)


@router.post("/worker/{engine_id}/{job_id}/result", response_model=Engine)
async def result(engine_id: str, job_id: str, body: Result, worker: WorkerAuth) -> Engine:
    engine: Final = await get_engine(engine_id, worker.scope)
    old: Final = next((j for j in engine.jobs if j.id == job_id), None)
    if old and old.status in ("completed", "failed") and old.worker_id == worker.id:
        return engine
    _, job = await assigned(engine_id, job_id, worker)
    now: Final = datetime.now(UTC)
    selected: Final = job.sample or Sample(executions=(), eligible=0)
    allowed: Final = frozenset(e.id for e in selected.executions)
    check_ids: Final = frozenset(c.id for c in job.settings.checks if c.enabled)
    if any(
        f.check_id not in check_ids or any(e.execution_id not in allowed for e in f.evidence) for f in body.findings
    ):
        raise HTTPException(422, "Finding references evidence outside the job")

    for finding in body.findings:
        await validate_finding(engine, selected, finding)

    def finish(e: Engine) -> Engine:
        active: Final = current_job(e)
        if active is None or active.id != job_id or active.worker_id != worker.id:
            return e
        merged: Final = merge_results(e, body, job.revision, now).findings
        merged_ids: Final = frozenset(f.id for f in merged)
        return replace_job(
            e,
            active.model_copy(
                update=MappingProxyType(
                    {
                        "status": "failed" if body.error else "completed",
                        "stage": "Failed" if body.error else "Complete",
                        "finished_at": now,
                        "coverage": active.coverage if body.error else body.coverage,
                        "error": body.error,
                    }
                )
            ),
        ).model_copy(
            update=MappingProxyType(
                {
                    "findings": (*merged, *(f for f in e.findings if f.id not in merged_ids)),
                    "last_scan_at": e.last_scan_at if body.error else max(e.last_scan_at or job.end, job.end),
                    "next_run_at": now + timedelta(minutes=e.settings.interval_minutes),
                }
            )
        )

    return required(await repository().update(engine_id, finish))


def merge_results(engine: Engine, result: Result, revision: int, now: datetime) -> Engine:
    if not result.findings:
        return engine
    finding: Final = merge_finding(engine, result.findings[0], revision, now)
    updated: Final = engine.model_copy(
        update=MappingProxyType({"findings": (finding, *(f for f in engine.findings if f.id != finding.id))})
    )
    return merge_results(
        updated, result.model_copy(update=MappingProxyType({"findings": result.findings[1:]})), revision, now
    )


@router.post("/worker/{engine_id}/{job_id}/heartbeat", response_model=bool)
async def heartbeat(engine_id: str, job_id: str, worker: WorkerAuth) -> bool:
    _, job = await assigned(engine_id, job_id, worker)
    return await progress(engine_id, job_id, Progress(stage=job.stage, coverage=job.coverage), worker)


async def claim_candidate(candidate: Engine, worker: Worker, now: datetime) -> Claim | None:
    job_id: Final = str(uuid4())

    def schedule(e: Engine) -> Engine:
        scheduled: Final = queue_job(e, now, job_id) if e.settings.enabled and e.next_run_at <= now else e
        return claim_job(scheduled, worker, now)

    updated: Final = required(await repository().update(candidate.id, schedule))
    job: Final = current_job(updated)
    if job and job.worker_id == worker.id and job.status == "running" and job != current_job(candidate):
        return Claim(engine_id=updated.id, job=job, findings=updated.findings)
    return None


async def validate_finding(engine: Engine, selected: Sample, finding: FindingDraft) -> None:
    previous: Final = next((f for f in engine.findings if f.id == finding.existing_finding_id), None)
    if finding.existing_finding_id and (previous is None or previous.check_id != finding.check_id):
        raise HTTPException(422, "Existing finding must belong to the same check")
    for evidence in finding.evidence:
        if not await source_reader().verify_evidence(
            engine.scope, next(e for e in selected.executions if e.id == evidence.execution_id), evidence
        ):
            raise HTTPException(422, "Evidence quote does not match stored content")


@router.get("/{engine_id}/executions/{execution_id}", response_model=ExecutionContent)
async def evidence_content(
    engine_id: str, execution_id: str, auth: Auth, cursor: str = "", offset: int = Query(default=0, ge=0, le=1000000)
) -> ExecutionContent:
    engine: Final = await get_engine(engine_id, user_scope(auth))
    try:
        source, team, trace_id, trace_ref = parse_execution(execution_id)
    except ValueError:
        raise HTTPException(404, "Execution not found")
    if source not in ("traces", "requests") or (not engine.scope.all_teams and team != engine.scope.team_id):
        raise HTTPException(404, "Execution not found")
    execution: Final = Execution(
        id=execution_id,
        source="traces" if source == "traces" else "requests",
        trace_id=trace_id,
        trace_ref=trace_ref,
        team_id=team,
        name=trace_id,
        start_time="",
        span_count=1,
        root_seen=source == "requests",
    )
    return await source_reader().content(engine.scope, execution, cursor, offset)

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AwareDatetime, BaseModel, Field, TypeAdapter

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.routing_prisma_wrapper import writer_wrapper
from litellm.proxy.lens.billing import validate_key
from litellm.proxy.lens.models import (
    Claim,
    Execution,
    ExecutionContent,
    FindingDraft,
    FindingUpdate,
    Job,
    Lens,
    LensList,
    LensSettings,
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
from litellm.proxy.lens.repository import LensRepository, WriterDatabase
from litellm.proxy.lens.sources import ActivityAvailability, SourceReader, Storage, parse_execution
from litellm.proxy.lens.state import (
    can_access,
    claim_job,
    current_job,
    merge_finding,
    queue_job,
    replace_job,
    snapshot_finding,
)
from litellm.proxy.tracing_runtime import provide_storage

router: Final = APIRouter(prefix="/lens", tags=["Lens"])
_bearer: Final = HTTPBearer()
Auth: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
StorageDep: TypeAlias = Annotated[Storage | None, Depends(provide_storage)]


def repository() -> LensRepository:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(503, "Lens needs a connected Postgres database")
    return LensRepository(WriterDatabase(writer_wrapper(prisma_client.db)))


def source_reader(storage: Storage | None) -> SourceReader:
    if storage is None:
        raise HTTPException(
            status_code=501,
            detail="Agent tracing is not enabled. Set `tracing:` in general_settings and CLICKHOUSE_URL.",
        )
    return SourceReader(storage)


def user_scope(auth: UserAPIKeyAuth, write: bool = False) -> Scope:
    if write and auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(403, "Only proxy admins can configure or run Lens")
    if auth.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        return Scope(all_teams=True)
    raise HTTPException(403, "Lens requires proxy administrator access")


async def get_lens(lens_id: str, scope: Scope) -> Lens:
    lens: Final = await repository().get(lens_id)
    if lens is None or not can_access(scope, lens.scope):
        raise HTTPException(404, "Lens not found")
    return lens


async def worker_auth(credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)]) -> Worker:
    worker: Final = await repository().worker(hashlib.sha256(credentials.credentials.encode()).hexdigest())
    if worker is None or worker.revoked:
        raise HTTPException(401, "Worker credential is invalid or revoked")
    return worker


WorkerAuth: TypeAlias = Annotated[Worker, Depends(worker_auth)]


async def assigned(lens_id: str, job_id: str, worker: Worker) -> tuple[Lens, Job]:
    lens: Final = await get_lens(lens_id, worker.scope)
    job: Final = current_job(lens)
    if (
        job is None
        or job.id != job_id
        or job.status != "running"
        or job.worker_id != worker.id
        or job.lease_until is None
        or job.lease_until <= datetime.now(timezone.utc)
    ):
        raise HTTPException(409, "This worker no longer owns the job")
    return lens, job


def required(lens: Lens | None) -> Lens:
    if lens is None:
        raise HTTPException(409, "Lens changed concurrently; retry the operation")
    return lens


def validate_selection(settings: LensSettings) -> None:
    for identity in settings.execution_ids:
        try:
            source, _, _, _ = parse_execution(identity)
            if source not in ("traces", "requests"):
                raise ValueError("Unsupported source")
        except ValueError:
            raise HTTPException(422, "Choose execution IDs returned by the activity preview")


def validate_model(settings: LensSettings, auth: UserAPIKeyAuth) -> None:
    from litellm.proxy.proxy_server import llm_router

    validate_selection(settings)
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


@router.get("", response_model=LensList)
async def list_lenses(auth: Auth, storage: StorageDep) -> LensList:
    scope: Final = user_scope(auth)
    return LensList(
        lenses=tuple(e for e in await repository().lenses() if can_access(scope, e.scope)),
        workers=tuple(w for w in await repository().workers() if can_access(scope, w.scope)),
        tracing_enabled=storage is not None,
    )


@router.post("", response_model=Lens)
async def create_lens(settings: LensSettings, auth: Auth) -> Lens:
    scope: Final = user_scope(auth, write=True)
    validate_model(settings, auth)
    now: Final = datetime.now(timezone.utc)
    lens: Final = Lens(
        id=str(uuid4()),
        scope=scope,
        settings=settings,
        created_at=now,
        next_run_at=now,
        budget_month=now.strftime("%Y-%m"),
    )
    return await repository().create(queue_job(lens, now, str(uuid4())))


@router.get("/activity/available", response_model=ActivityAvailability)
async def activity_available(auth: Auth, storage: StorageDep) -> ActivityAvailability:
    scope: Final = user_scope(auth)
    return await source_reader(storage).availability(scope) if storage is not None else ActivityAvailability()


@router.get("/agents", response_model=tuple[str, ...])
async def list_agents(auth: Auth, storage: StorageDep) -> tuple[str, ...]:
    scope: Final = user_scope(auth)
    return await source_reader(storage).agents(scope) if storage is not None else ()


@router.put("/{lens_id}", response_model=Lens)
async def update_lens(lens_id: str, settings: LensSettings, auth: Auth) -> Lens:
    await get_lens(lens_id, user_scope(auth, write=True))
    validate_model(settings, auth)
    return required(
        await repository().update(
            lens_id,
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


@router.post("/{lens_id}/runs", response_model=Lens)
async def run_lens(lens_id: str, body: RunRequest, auth: Auth) -> Lens:
    await get_lens(lens_id, user_scope(auth, write=True))
    if body.settings is not None:
        validate_model(body.settings, auth)
    now: Final = datetime.now(timezone.utc)
    job_id: Final = str(uuid4())
    return required(
        await repository().update(lens_id, lambda e: queue_job(e, now, job_id, body.lookback_hours, body.settings))
    )


@router.get("/{lens_id}", response_model=Lens)
async def read_lens(lens_id: str, auth: Auth) -> Lens:
    return await get_lens(lens_id, user_scope(auth))


@router.get("/{lens_id}/runs", response_model=tuple[Job, ...])
async def list_runs(lens_id: str, auth: Auth, offset: int = Query(default=0, ge=0)) -> tuple[Job, ...]:
    await get_lens(lens_id, user_scope(auth))
    return tuple(
        j.model_copy(update=MappingProxyType({"sample": None, "findings": None, "assessments": ()}))
        for j in await repository().jobs(lens_id, offset)
    )


@router.get("/{lens_id}/runs/{job_id}", response_model=Job)
async def read_run(lens_id: str, job_id: str, auth: Auth) -> Job:
    await get_lens(lens_id, user_scope(auth))
    job: Final = await repository().job(lens_id, job_id)
    if job is None:
        raise HTTPException(404, "Investigation not found")
    return job


@router.post("/{lens_id}/cancel", response_model=Lens)
async def cancel_lens(lens_id: str, auth: Auth) -> Lens:
    await get_lens(lens_id, user_scope(auth, write=True))
    now: Final = datetime.now(timezone.utc)

    def cancel(e: Lens) -> Lens:
        job: Final = current_job(e)
        if job is None:
            return e
        cancelled: Final = job.model_copy(
            update=MappingProxyType({"status": "cancelled", "stage": "Cancelled", "finished_at": now})
        )
        return replace_job(e, cancelled).model_copy(
            update=MappingProxyType({"next_run_at": now + timedelta(minutes=e.settings.interval_minutes)})
        )

    return required(await repository().update(lens_id, cancel))


@router.patch("/{lens_id}/findings/{finding_id}", response_model=Lens)
async def update_finding(lens_id: str, finding_id: str, body: FindingUpdate, auth: Auth) -> Lens:
    await get_lens(lens_id, user_scope(auth, write=True))
    return required(
        await repository().update(
            lens_id,
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
    as_of: AwareDatetime | None = None
    offset: int = Field(default=0, ge=0)
    settings: LensSettings
    lookback_hours: int = Field(default=24, ge=1, le=8760)


@router.post("/preview/sample", response_model=Sample)
async def preview_sample(body: Preview, auth: Auth, storage: StorageDep) -> Sample:
    validate_selection(body.settings)
    now: Final = min(body.as_of or datetime.now(timezone.utc), datetime.now(timezone.utc))
    return await source_reader(storage).sample(
        user_scope(auth),
        body.settings,
        int((now - timedelta(hours=body.lookback_hours)).timestamp() * 1000),
        int((now - timedelta(minutes=2)).timestamp() * 1000),
        offset=body.offset,
        preview=True,
    )


class WorkerBilling(BaseModel):
    analysis_key_id: str = Field(pattern=r"^[a-f0-9]{64}$")


class WorkerName(WorkerBilling):
    name: str = Field(default="Lens worker", min_length=1, max_length=100)


@router.post("/workers/register", response_model=WorkerCreated)
async def register_worker(body: WorkerName, auth: Auth) -> WorkerCreated:
    scope: Final = user_scope(auth, write=True)
    await validate_key(body.analysis_key_id)
    token: Final = "lens-" + secrets.token_urlsafe(40)
    worker: Final = Worker(
        id=str(uuid4()),
        name=body.name,
        scope=scope,
        analysis_key_id=body.analysis_key_id,
        last_seen=datetime(1970, 1, 1, tzinfo=timezone.utc),
    )
    await repository().save_worker(worker, hashlib.sha256(token.encode()).hexdigest())
    return WorkerCreated(worker=worker, token=token)


@router.put("/workers/{worker_id}/billing-key", response_model=Worker)
async def set_worker_billing(worker_id: str, body: WorkerBilling, auth: Auth) -> Worker:
    scope: Final = user_scope(auth, write=True)
    worker: Final = next((w for w in await repository().workers() if w.id == worker_id), None)
    if worker is None or not can_access(scope, worker.scope):
        raise HTTPException(404, "Worker not found")
    if worker.revoked:
        raise HTTPException(409, "Register a new worker instead of updating revoked access")
    await validate_key(body.analysis_key_id)
    updated: Final = await repository().set_worker_billing(worker.id, body.analysis_key_id)
    if updated is None:
        raise HTTPException(409, "Worker access was revoked")
    return updated


@router.delete("/workers/{worker_id}")
async def revoke_worker(worker_id: str, auth: Auth) -> bool:
    scope: Final = user_scope(auth, write=True)
    worker: Final = next((w for w in await repository().workers() if w.id == worker_id), None)
    if worker is None or not can_access(scope, worker.scope):
        raise HTTPException(404, "Worker not found")
    jobs: Final = chain.from_iterable(lens.jobs for lens in await repository().lenses())
    if any(job.status == "running" and job.worker_id == worker.id for job in jobs):
        raise HTTPException(409, "Wait for this worker's investigation to finish or cancel it before revoking access")
    await repository().revoke_worker(worker.id)
    return True


@router.post("/worker/claim", response_model=Claim | None)
async def claim(worker: WorkerAuth, protocol_version: int = 1) -> Claim | None:
    if protocol_version != 2:
        raise HTTPException(409, "Upgrade the Lens worker using the current Connect worker command")
    if worker.analysis_key_id is None:
        raise HTTPException(409, "Assign an analysis key to this worker in Lens setup")
    now: Final = datetime.now(timezone.utc)
    await repository().heartbeat(worker.id, now.isoformat())
    for candidate in await repository().lenses():
        if not can_access(worker.scope, candidate.scope):
            continue
        if claimed := await claim_candidate(candidate, worker, now):
            return claimed
    return None


@router.post("/worker/{lens_id}/{job_id}/progress", response_model=bool)
async def progress(lens_id: str, job_id: str, body: Progress, worker: WorkerAuth) -> bool:
    await assigned(lens_id, job_id, worker)
    now: Final = datetime.now(timezone.utc)

    def renew(e: Lens) -> Lens:
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

    required(await repository().update(lens_id, renew))
    await repository().heartbeat(worker.id, now.isoformat())
    return True


@router.get("/worker/{lens_id}/{job_id}/sample", response_model=Sample)
async def sample(lens_id: str, job_id: str, worker: WorkerAuth, storage: StorageDep) -> Sample:
    lens, job = await assigned(lens_id, job_id, worker)
    if job.sample is not None:
        return job.sample
    pages: list[Sample] = []  # mutable-ok: freeze selection after stable cursor traversal
    cursor = ""  # rebind-ok: advance by immutable identity, never by shifting row positions
    while True:
        page = await source_reader(storage).sample(
            lens.scope,
            job.settings,
            int(job.start.timestamp() * 1000),
            int(job.end.timestamp() * 1000),
            cursor=cursor,
        )
        pages.append(page)
        if not page.next_cursor or sum(len(p.executions) for p in pages) >= pages[0].selected:
            break
        cursor = page.next_cursor
    executions: Final = tuple(
        execution for p in pages for execution in p.executions
    )  # comprehension-ok: flatten query pages
    selected: Final = Sample(executions=executions, eligible=pages[0].eligible, selected=len(executions))

    def freeze(e: Lens) -> Lens:
        active: Final = current_job(e)
        if active is None or active.id != job_id or active.worker_id != worker.id:
            raise HTTPException(409, "Job was cancelled or reassigned")
        return (
            replace_job(e, active.model_copy(update=MappingProxyType({"sample": selected})))
            if active.sample is None
            else e
        )

    updated: Final = required(await repository().update(lens_id, freeze))
    frozen: Final = next(j for j in updated.jobs if j.id == job_id).sample
    if frozen is None:
        raise HTTPException(409, "Could not freeze the sample")
    return frozen


@router.get("/worker/{lens_id}/{job_id}/content", response_model=ExecutionContent)
async def content(
    lens_id: str,
    job_id: str,
    execution_id: str,
    worker: WorkerAuth,
    storage: StorageDep,
    cursor: str = "",
    offset: int = Query(default=0, ge=0),
) -> ExecutionContent:
    lens, job = await assigned(lens_id, job_id, worker)
    selected: Final = job.sample or Sample(executions=(), eligible=0)
    execution: Final = next((e for e in selected.executions if e.id == execution_id), None)
    if execution is None:
        raise HTTPException(404, "Execution is outside this job's sample")
    return await source_reader(storage).content(lens.scope, execution, cursor, offset)


@router.post("/worker/{lens_id}/{job_id}/model", response_model=ModelResult)
async def model(lens_id: str, job_id: str, body: ModelRequest, worker: WorkerAuth, request: Request) -> ModelResult:
    from litellm.proxy.lens.inference import analyze

    lens, job = await assigned(lens_id, job_id, worker)
    return await analyze(repository(), lens, job, worker, body, request)


@router.post("/worker/{lens_id}/{job_id}/result", response_model=Lens)
async def result(lens_id: str, job_id: str, body: Result, worker: WorkerAuth, storage: StorageDep) -> Lens:
    lens: Final = await get_lens(lens_id, worker.scope)
    old: Final = next((j for j in lens.jobs if j.id == job_id), None)
    if old and old.status in ("completed", "failed") and old.worker_id == worker.id:
        return lens
    _, job = await assigned(lens_id, job_id, worker)
    now: Final = datetime.now(timezone.utc)
    selected: Final = job.sample or Sample(executions=(), eligible=0)
    allowed: Final = frozenset(e.id for e in selected.executions)
    if len(frozenset(a.execution_id for a in body.assessments)) != len(body.assessments):
        raise HTTPException(422, "Each run must have one assessment")
    if any(a.execution_id not in allowed for a in body.assessments):
        raise HTTPException(422, "Assessment references a run outside this job")
    check_ids: Final = frozenset(c.id for c in job.settings.analysis_checks)
    if any(not check_ids.issuperset((*a.issue_checks, *a.pattern_checks)) for a in body.assessments):
        raise HTTPException(422, "Assessment references an unknown check")
    if any(
        f.check_id not in check_ids or any(e.execution_id not in allowed for e in f.evidence) for f in body.findings
    ):
        raise HTTPException(422, "Finding references evidence outside the job")

    for finding in body.findings:
        await validate_finding(lens, selected, finding, storage)

    def finish(e: Lens) -> Lens:
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
                        "assessments": body.assessments,
                        "findings": tuple(snapshot_finding(e, f, job.revision, now) for f in body.findings),
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

    return required(await repository().update(lens_id, finish))


def merge_results(lens: Lens, result: Result, revision: int, now: datetime) -> Lens:
    def merge_one(current: Lens, draft: FindingDraft) -> Lens:
        finding: Final = merge_finding(current, draft, revision, now)
        return current.model_copy(
            update=MappingProxyType({"findings": (finding, *(f for f in current.findings if f.id != finding.id))})
        )

    return reduce(merge_one, result.findings, lens)


@router.post("/worker/{lens_id}/{job_id}/heartbeat", response_model=bool)
async def heartbeat(lens_id: str, job_id: str, worker: WorkerAuth) -> bool:
    _, job = await assigned(lens_id, job_id, worker)
    return await progress(lens_id, job_id, Progress(stage=job.stage, coverage=job.coverage), worker)


async def claim_candidate(candidate: Lens, worker: Worker, now: datetime) -> Claim | None:
    job_id: Final = str(uuid4())

    def schedule(e: Lens) -> Lens:
        scheduled: Final = queue_job(e, now, job_id) if e.settings.enabled and e.next_run_at <= now else e
        return claim_job(scheduled, worker, now)

    updated: Final = await repository().update(candidate.id, schedule, changed_only=True)
    if updated is None:
        return None
    job: Final = current_job(updated)
    if job and job.worker_id == worker.id and job.status == "running" and job != current_job(candidate):
        return Claim(lens_id=updated.id, job=job, findings=updated.findings)
    return None


async def validate_finding(lens: Lens, selected: Sample, finding: FindingDraft, storage: Storage | None) -> None:
    previous: Final = next((f for f in lens.findings if f.id == finding.existing_finding_id), None)
    if finding.existing_finding_id and (previous is None or previous.check_id != finding.check_id):
        raise HTTPException(422, "Existing finding must belong to the same check")
    for evidence in finding.evidence:
        if not await source_reader(storage).verify_evidence(
            lens.scope, next(e for e in selected.executions if e.id == evidence.execution_id), evidence
        ):
            raise HTTPException(422, "Evidence quote does not match stored content")


@router.get("/{lens_id}/executions/{execution_id}", response_model=ExecutionContent)
async def evidence_content(
    lens_id: str,
    execution_id: str,
    auth: Auth,
    storage: StorageDep,
    cursor: str = "",
    offset: int = Query(default=0, ge=0),
) -> ExecutionContent:
    lens: Final = await get_lens(lens_id, user_scope(auth))
    try:
        source, team, trace_id, trace_ref = parse_execution(execution_id)
    except ValueError:
        raise HTTPException(404, "Execution not found")
    if source not in ("traces", "requests") or (not lens.scope.all_teams and team != lens.scope.team_id):
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
    return await source_reader(storage).content(lens.scope, execution, cursor, offset)

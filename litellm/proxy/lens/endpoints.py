import hashlib
import secrets
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Final, Protocol, TypeAlias
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AwareDatetime, Field

from litellm.litellm_core_utils.secret_redaction import redact_internal_details
from litellm.proxy._types import LitellmUserRoles, ModelAccessDeniedProxyException, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_model
from litellm.proxy.auth.resolvers.exceptions import KeyNotFoundError
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.routing_prisma_wrapper import writer_wrapper
from litellm.proxy.lens.billing import validate_key
from litellm.proxy.lens.inference import Deployment, deployment_prices
from litellm.proxy.lens.ingestion import (
    IngestionCredential,
    IngestionKey,
    IngestionKeyCreated,
    IngestionKeyRequest,
    IngestionSnapshot,
    InvalidExpiry,
    ServiceConnection,
    ServiceStatus,
    new_key,
)
from litellm.proxy.lens.models import (
    ActivitySelection,
    Claim,
    Coverage,
    Execution,
    ExecutionContent,
    FindingDraft,
    FindingUpdate,
    Job,
    Lens,
    LensList,
    LensSettings,
    LookbackHours,
    ModelRequest,
    ModelResult,
    Progress,
    Result,
    Review,
    ReviewPage,
    RunRequest,
    Sample,
    Scope,
    TraceFindingCount,
    TraceFindingsRequest,
    WatchAllResult,
    WatchSkipped,
    Worker,
    WorkerCreated,
)
from litellm.proxy.lens.release import PROTOCOL_VERSION, release_tag, worker_image
from litellm.proxy.lens.repository import DueLens, LensRepository, WriterDatabase
from litellm.proxy.lens.reviews import criteria_key
from litellm.proxy.lens.signal_repository import SignalRepository
from litellm.proxy.lens.signals import SignalConfig, TraceSignals, trace_signals
from litellm.proxy.lens.sources import ActivityAvailability, SourceReader, Storage, parse_execution
from litellm.proxy.lens.state import (
    can_access,
    cancel_job,
    claim_job,
    current_job,
    end_job,
    merge_finding,
    next_scan_start,
    queue_job,
    replace_job,
    result_status,
    reviews_after,
    scheduled_window,
    summarized,
)
from litellm.proxy.tracing_runtime import provide_storage
from litellm.router import Router
from litellm.tracing.remote import LensConnection, bounded_response
from litellm.types.llms.base import LiteLLMBaseModel

router: Final = APIRouter(prefix="/lens", tags=["Lens"])
CLAIM_CANDIDATES: Final = 20
_bearer: Final = HTTPBearer()
Auth: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
StorageDep: TypeAlias = Annotated[Storage | None, Depends(provide_storage)]
SAMPLE_PAGE_SIZE: Final = 10_000
SAMPLE_PAGE_SIZES: Final = (SAMPLE_PAGE_SIZE, 5_000, 2_500, 1_250, 625, 312, 156, 100)
SAMPLE_RESPONSE_TOO_LARGE: Final = "ClickHouse query exceeded the response size limit"


class _ClaimRepository(Protocol):
    async def due(
        self, scope: Scope, now: datetime, limit: int, after: DueLens | None = None
    ) -> tuple[DueLens, ...]: ...

    async def sync_due(self, lens: Lens) -> None: ...

    async def update(
        self, lens_id: str, transform: Callable[[Lens], Lens], attempts: int, *, changed_only: bool
    ) -> Lens | None: ...


def repository() -> LensRepository:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(503, "Lens needs a connected Postgres database")
    return LensRepository(WriterDatabase(writer_wrapper(prisma_client.db)))


def signals_repository() -> SignalRepository:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(503, "Lens needs a connected Postgres database")
    return SignalRepository(WriterDatabase(writer_wrapper(prisma_client.db)))


def source_reader(storage: Storage | None) -> SourceReader:
    if storage is None:
        raise HTTPException(
            status_code=501,
            detail="Agent tracing is not enabled. Configure the Lens service and LITELLM_LENS_URL.",
        )
    return SourceReader(storage)


def user_scope(auth: UserAPIKeyAuth, write: bool = False) -> Scope:
    if write and auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(403, "Only proxy admins can configure or run Lens")
    if auth.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        return Scope(all_teams=True)
    raise HTTPException(403, "Lens requires proxy administrator access")


def validate_signal_model(config: SignalConfig, llm_router: Router | None) -> None:
    if not config.model:
        return
    message: Final = "Choose a System 1 model (evaluation mode) configured on this proxy"
    if llm_router is None:
        raise HTTPException(400, message)
    try:
        model_group: Final = llm_router.get_model_group_info(model_group=config.model)
    except Exception as error:
        raise HTTPException(400, message) from error
    if model_group is None or model_group.mode != "evaluation":
        raise HTTPException(400, message)


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
Attempt: TypeAlias = Annotated[int, Header(alias="X-LiteLLM-Lens-Attempt", ge=1)]


async def service_auth(credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)]) -> None:
    try:
        connection: Final = LensConnection.from_env()
    except ValueError as error:
        raise HTTPException(503, "Configure the Lens service connection") from error
    if not secrets.compare_digest(credentials.credentials, connection.token):
        raise HTTPException(401, "Invalid Lens service credential")


ServiceAuth: TypeAlias = Annotated[None, Depends(service_auth)]


@router.get("/service", response_model=ServiceConnection)
async def service_connection(auth: Auth) -> ServiceConnection:
    import os

    import httpx

    public_url: Final = os.environ.get("LITELLM_LENS_PUBLIC_URL", "").rstrip("/")
    try:
        connection: Final = LensConnection.from_env()
    except ValueError:
        return ServiceConnection(
            url=public_url,
            connected=False,
            status=ServiceStatus(),
            configured=bool(os.environ.get("LITELLM_LENS_URL")),
            release=release_tag(),
        )
    try:
        client: Final = connection.control_client()
        async with client.stream(
            "GET", connection.endpoint("/internal/status"), headers=connection.headers, timeout=2
        ) as response:
            if response.status_code == 200:
                status: Final = ServiceStatus.model_validate_json(await bounded_response(response, 16 * 1024))
                return ServiceConnection(
                    url=public_url,
                    connected=True,
                    status=status,
                    configured=True,
                    release=release_tag(),
                )
    except (ValueError, RuntimeError, httpx.HTTPError):
        pass
    return ServiceConnection(
        url=public_url,
        connected=False,
        status=ServiceStatus(),
        configured=True,
        release=release_tag(),
    )


async def credential_snapshot() -> IngestionSnapshot:
    now: Final = int(datetime.now(timezone.utc).timestamp())
    keys: Final = await repository().ingestion_keys()
    return IngestionSnapshot(
        issued_at=now,
        keys=tuple(
            IngestionCredential(token_hash=key.tenant.api_key_hash, tenant=key.tenant, expires_at=key.expires_at)
            for key in keys
            if key.expires_at is None or key.expires_at > now
        ),
    )


async def publish_credentials() -> bool:
    import httpx

    try:
        connection: Final = LensConnection.from_env()
        snapshot: Final = await credential_snapshot()
        response: Final = await connection.control_client().post(
            connection.endpoint("/internal/credentials"),
            headers=connection.headers,
            json=snapshot.model_dump(mode="json"),
            timeout=2,
        )
        return response.status_code == 204
    except (ValueError, httpx.HTTPError):
        return False


@router.post("/tracing/keys", response_model=IngestionKeyCreated)
async def create_ingestion_key(body: IngestionKeyRequest, auth: Auth) -> IngestionKeyCreated:
    user_scope(auth, write=True)
    created: Final = new_key(body, auth.user_id or "")
    if isinstance(created, InvalidExpiry):
        raise HTTPException(422, "Choose an expiry in the future")
    await repository().save_ingestion_key(created.record)
    return created.model_copy(update={"active": await publish_credentials()})


@router.get("/tracing/keys", response_model=tuple[IngestionKey, ...])
async def list_ingestion_keys(auth: Auth) -> tuple[IngestionKey, ...]:
    user_scope(auth)
    return await repository().ingestion_keys()


@router.delete("/tracing/keys/{key_id}")
async def revoke_ingestion_key(key_id: str, auth: Auth) -> bool:
    user_scope(auth, write=True)
    await repository().revoke_ingestion_key(key_id)
    await publish_credentials()
    return True


@router.get("/internal/ingestion-credentials", response_model=IngestionSnapshot)
async def ingestion_credentials(service: ServiceAuth, response: Response) -> IngestionSnapshot:
    response.headers["Cache-Control"] = "no-store"
    return await credential_snapshot()


async def assigned(lens_id: str, job_id: str, worker: Worker, attempt: int = 1) -> tuple[Lens, Job]:
    lens: Final = await get_lens(lens_id, worker.scope)
    job: Final = current_job(lens)
    if (
        job is None
        or job.id != job_id
        or job.status != "running"
        or job.worker_id != worker.id
        or job.attempts != attempt
        or job.lease_until is None
        or job.lease_until <= datetime.now(timezone.utc)
    ):
        raise HTTPException(409, "This worker no longer owns the job")
    return lens, job


def required(lens: Lens | None) -> Lens:
    if lens is None:
        raise HTTPException(409, "Lens changed concurrently; retry the operation")
    return lens


def validate_selection(settings: ActivitySelection) -> None:
    for identity in settings.execution_ids:
        try:
            source, _, _, _ = parse_execution(identity)
            if source not in ("traces", "requests"):
                raise ValueError("Unsupported source")
        except ValueError:
            raise HTTPException(422, "Choose execution IDs returned by the activity preview")


async def validate_model(settings: LensSettings, auth: UserAPIKeyAuth) -> None:
    from litellm.proxy.proxy_server import llm_router, prisma_client

    validate_selection(settings)
    deployments: Final = (
        llm_router.get_model_list(model_name=settings.model, team_id=auth.team_id) if llm_router else ()
    )
    if not deployments:
        raise HTTPException(400, "Choose a model configured on this LiteLLM instance")
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        try:
            await can_key_call_model(
                model=settings.model,
                llm_model_list=deployments,
                valid_token=auth,
                llm_router=llm_router,
                prisma_client=prisma_client,
            )
        except ModelAccessDeniedProxyException as exc:
            raise HTTPException(403, "This key does not have access to the analysis model") from exc
    for deployment in deployments:
        deployment_prices(Deployment.model_validate(deployment))


async def worker_supports_model(worker: Worker, settings: LensSettings) -> bool:
    if worker.revoked or worker.analysis_key_id is None:
        return False
    try:
        auth: Final = await validate_key(worker.analysis_key_id)
        if auth is None:
            return False
        await validate_model(settings, auth)
    except KeyNotFoundError:
        return False
    except HTTPException as exc:
        if exc.status_code not in (400, 401, 403):
            raise
        return False
    return True


async def validate_workers(settings: LensSettings, scope: Scope) -> None:
    workers: Final = repository().eligible_workers(scope)
    first: Final = await anext(workers, None)
    if first is None or await worker_supports_model(first, settings):
        return
    async for worker in workers:
        if await worker_supports_model(worker, settings):
            return
    raise HTTPException(
        400,
        "No worker can use this analysis model. Choose a model available to the worker's virtual key, "
        "or update its model access and pricing.",
    )


@router.get("", response_model=LensList)
async def list_lenses(auth: Auth, storage: StorageDep) -> LensList:
    scope: Final = user_scope(auth)
    return LensList(
        lenses=tuple(summarized(e) for e in await repository().lenses() if can_access(scope, e.scope)),
        workers=tuple(w for w in await repository().workers() if can_access(scope, w.scope)),
        tracing_enabled=storage is not None,
    )


@router.post("", response_model=Lens)
async def create_lens(settings: LensSettings, auth: Auth) -> Lens:
    scope: Final = user_scope(auth, write=True)
    await validate_model(settings, auth)
    await validate_workers(settings, scope)
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


@router.get("/signals", response_model=SignalConfig)
async def get_signals(auth: Auth) -> SignalConfig:
    user_scope(auth)
    return await signals_repository().get_config()


@router.put("/signals", response_model=SignalConfig)
async def put_signals(body: SignalConfig, auth: Auth) -> SignalConfig:
    user_scope(auth, write=True)
    from litellm.proxy.proxy_server import llm_router

    validate_signal_model(body, llm_router)
    await signals_repository().save_config(body)
    return body


@router.post("/traces/signals", response_model=tuple[TraceSignals, ...])
async def trace_signal_statuses(body: TraceFindingsRequest, auth: Auth) -> tuple[TraceSignals, ...]:
    user_scope(auth)
    repo: Final = signals_repository()
    config: Final = await repo.get_config()
    existing: Final = await repo.traces(body.traces)
    rows: Final = MappingProxyType({(row.trace_id, row.trace_ref): row for row in existing})
    return tuple(
        trace_signals(
            trace,
            rows.get((trace.trace_id, trace.trace_ref)),
            config,
        )
        for trace in body.traces
    )


@router.post("/traces/findings", response_model=tuple[TraceFindingCount, ...])
async def trace_findings(body: TraceFindingsRequest, auth: Auth) -> tuple[TraceFindingCount, ...]:
    user_scope(auth)
    return await repository().trace_findings(body.traces)


def watching(lens: Lens) -> Lens:
    if lens.settings.enabled:
        return lens
    return lens.model_copy(
        update=MappingProxyType(
            {
                "settings": lens.settings.model_copy(update=MappingProxyType({"enabled": True})),
                "revision": lens.revision + 1,
            }
        )
    )


async def watchable(lens: Lens, auth: UserAPIKeyAuth) -> WatchSkipped | None:
    try:
        await validate_model(lens.settings.model_copy(update=MappingProxyType({"enabled": True})), auth)
    except HTTPException as exc:
        return WatchSkipped(id=lens.id, name=lens.settings.name, reason=str(exc.detail))
    return None


@router.post("/watch-all", response_model=WatchAllResult)
async def watch_all(auth: Auth) -> WatchAllResult:
    scope: Final = user_scope(auth, write=True)
    paused: Final = tuple(
        e for e in await repository().lenses() if can_access(scope, e.scope) and not e.settings.enabled
    )
    checks: Final = tuple([(lens, await watchable(lens, auth)) for lens in paused])
    skipped: Final = tuple(skip for _, skip in checks if skip is not None)
    ready: Final = tuple(lens for lens, skip in checks if skip is None)
    updated: Final = tuple([await repository().update(lens.id, watching) for lens in ready])
    return WatchAllResult(watching=tuple(u.id for u in updated if u is not None), skipped=skipped)


@router.put("/{lens_id}", response_model=Lens)
async def update_lens(lens_id: str, settings: LensSettings, auth: Auth) -> Lens:
    lens: Final = await get_lens(lens_id, user_scope(auth, write=True))
    validate_selection(settings)
    if settings.model != lens.settings.model or (settings.enabled and not lens.settings.enabled):
        await validate_model(settings, auth)
    return required(
        await repository().update(
            lens_id,
            lambda e: e.model_copy(
                update=MappingProxyType(
                    {
                        "settings": settings,
                        "revision": e.revision + 1,
                        "criteria_updated_at": datetime.now(timezone.utc)
                        if criteria_key(e.settings) != criteria_key(settings)
                        else e.criteria_updated_at,
                        "last_scan_at": None if criteria_key(e.settings) != criteria_key(settings) else e.last_scan_at,
                    }
                )
            ),
        )
    )


def run_window(lens: Lens, body: RunRequest, now: datetime) -> tuple[datetime, datetime] | None:
    if body.start is not None and body.end is not None:
        return body.start, body.end
    if body.lookback_hours is None and body.settings is None:
        return scheduled_window(lens, now)
    return None


def run_settings(lens: Lens, body: RunRequest) -> LensSettings | None:
    if body.agent_name is None:
        return body.settings
    return (body.settings or lens.settings).model_copy(update=MappingProxyType({"agent_name": body.agent_name}))


@router.post("/{lens_id}/runs", response_model=Lens)
async def run_lens(lens_id: str, body: RunRequest, auth: Auth) -> Lens:
    lens: Final = await get_lens(lens_id, user_scope(auth, write=True))
    settings: Final = body.settings or lens.settings
    await validate_model(settings, auth)
    await validate_workers(settings, lens.scope)
    now: Final = datetime.now(timezone.utc)
    job_id: Final = str(uuid4())
    return required(
        await repository().update(
            lens_id,
            lambda e: queue_job(
                e,
                now,
                job_id,
                body.lookback_hours,
                run_settings(e, body),
                run_window(e, body, now),
                "manual",
            ),
        )
    )


@router.get("/{lens_id}", response_model=Lens)
async def read_lens(lens_id: str, auth: Auth) -> Lens:
    return summarized(await get_lens(lens_id, user_scope(auth)))


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


@router.get("/{lens_id}/runs/{job_id}/reviews", response_model=ReviewPage)
async def read_reviews(lens_id: str, job_id: str, auth: Auth, after: int = Query(default=0, ge=0)) -> ReviewPage:
    return reviews_after(await read_run(lens_id, job_id, auth), after)


@router.post("/{lens_id}/cancel", response_model=Lens)
async def cancel_lens(lens_id: str, auth: Auth) -> Lens:
    await get_lens(lens_id, user_scope(auth, write=True))
    now: Final = datetime.now(timezone.utc)

    return required(await repository().update(lens_id, lambda e: cancel_job(e, now)))


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
                            f.model_copy(update=body.model_dump()) if finding_id in (f.id, *f.merged_finding_ids) else f
                            for f in e.findings
                        ),
                    }
                )
            ),
        )
    )


class Preview(LiteLLMBaseModel):
    as_of: AwareDatetime | None = None
    offset: int = Field(default=0, ge=0)
    selection: ActivitySelection
    lookback_hours: LookbackHours = 24


@router.post("/preview/sample", response_model=Sample)
async def preview_sample(body: Preview, auth: Auth, storage: StorageDep) -> Sample:
    validate_selection(body.selection)
    now: Final = min(body.as_of or datetime.now(timezone.utc), datetime.now(timezone.utc))
    try:
        start: Final = int((now - timedelta(hours=body.lookback_hours)).timestamp() * 1000)
        end: Final = int((now - timedelta(minutes=2)).timestamp() * 1000)
    except (OverflowError, ValueError) as error:
        raise HTTPException(422, "Preview window exceeds the supported calendar range") from error
    return await source_reader(storage).sample(
        user_scope(auth),
        body.selection,
        start,
        end,
        offset=body.offset,
        preview=True,
    )


class WorkerBilling(LiteLLMBaseModel):
    analysis_key_id: str = Field(pattern=r"^[a-f0-9]{64}$")


class WorkerName(WorkerBilling):
    name: str = Field(default="Lens worker", min_length=1)
    managed: bool = False


def configured_worker_image() -> str:
    if image := worker_image():
        return image
    raise HTTPException(
        503,
        "This LiteLLM build has no release identity. Use a published release, make lens-dev, "
        "or build the gateway and worker from the same commit with the same LITELLM_RELEASE_TAG.",
    )


@router.post("/workers/register", response_model=WorkerCreated)
async def register_worker(body: WorkerName, auth: Auth) -> WorkerCreated:
    scope: Final = user_scope(auth, write=True)
    image: Final = configured_worker_image()
    await validate_key(body.analysis_key_id)
    try:
        token: Final = LensConnection.from_env().token if body.managed else "lens-" + secrets.token_urlsafe(40)
    except ValueError as error:
        raise HTTPException(503, "Configure the Lens service before enabling investigations") from error
    token_hash: Final = hashlib.sha256(token.encode()).hexdigest()
    worker: Final = Worker(
        id=str(uuid4()),
        name=body.name,
        scope=scope,
        analysis_key_id=body.analysis_key_id,
        last_seen=datetime(1970, 1, 1, tzinfo=timezone.utc),
    )
    if body.managed:
        managed: Final = await repository().configure_service_worker(worker, token_hash)
        return WorkerCreated(worker=managed, token="", image=image, managed=True)
    await repository().save_worker(worker, token_hash)
    return WorkerCreated(worker=worker, token=token, image=image)


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
async def claim(worker: WorkerAuth, protocol_version: int = 1, worker_release: str = "") -> Claim | None:
    image: Final = configured_worker_image()
    expected: Final = release_tag()
    if protocol_version != PROTOCOL_VERSION or worker_release != expected:
        raise HTTPException(409, f"Upgrade the Lens worker to {image} and retry")
    if worker.analysis_key_id is None:
        return None
    now: Final = datetime.now(timezone.utc)
    lens_repository: Final = repository()
    await lens_repository.heartbeat(worker.id, now.isoformat())
    return await claim_due(worker, now, lens_repository)


async def claim_due(
    worker: Worker,
    now: datetime,
    lens_repository: _ClaimRepository,
    supports_model: Callable[[Worker, LensSettings], Awaitable[bool]] = worker_supports_model,
) -> Claim | None:
    after: DueLens | None = None  # rebind-ok: keyset cursor advances one page at a time
    while True:
        page = await lens_repository.due(worker.scope, now, CLAIM_CANDIDATES, after)
        for candidate in page:
            if not can_access(worker.scope, candidate.lens.scope):
                continue
            if claimed := await claim_candidate(candidate.lens, worker, now, lens_repository, supports_model):
                return claimed
            await lens_repository.sync_due(candidate.lens)
        if len(page) < CLAIM_CANDIDATES:
            return None
        after = page[-1]


@router.post("/worker/{lens_id}/{job_id}/progress", response_model=bool)
async def progress(lens_id: str, job_id: str, body: Progress, worker: WorkerAuth, attempt: Attempt = 1) -> bool:
    _, assigned_job = await assigned(lens_id, job_id, worker, attempt)
    if body.review is not None:
        if assigned_job.sample is None or body.review.execution_id not in frozenset(
            execution.id for execution in assigned_job.sample.executions
        ):
            raise HTTPException(422, "Review references a trace outside this job")
        if body.review.extraction is not None and any(
            observation.check_id not in frozenset(check.id for check in assigned_job.settings.analysis_checks)
            or any(quote.execution_id != body.review.execution_id for quote in observation.evidence)
            for observation in body.review.extraction.observations
        ):
            raise HTTPException(422, "Cached review must use enabled checks and only its assigned trace")
    required(await repository().progress(lens_id, assigned_job, body))
    await repository().heartbeat(worker.id, datetime.now(timezone.utc).isoformat())
    return True


@router.get("/worker/{lens_id}/{job_id}/reviews", response_model=tuple[Review, ...])
async def cached_reviews(lens_id: str, job_id: str, worker: WorkerAuth, attempt: Attempt = 1) -> tuple[Review, ...]:
    _, job = await assigned(lens_id, job_id, worker, attempt)
    return await repository().reviews(lens_id, job)


@router.get("/worker/{lens_id}/{job_id}/sample", response_model=Sample)
async def sample(lens_id: str, job_id: str, worker: WorkerAuth, storage: StorageDep, attempt: Attempt = 1) -> Sample:
    lens, job = await assigned(lens_id, job_id, worker, attempt)
    if job.sample is not None:
        return job.sample

    async def read_page(cursor: str, sizes: tuple[int, ...]) -> tuple[Sample, tuple[int, ...]]:
        page_size: Final = sizes[0]
        try:
            page: Final = await source_reader(storage).sample(
                lens.scope,
                job.settings,
                int(job.start.timestamp() * 1000),
                int(job.end.timestamp() * 1000),
                page_size=page_size,
                cursor=cursor,
            )
        except RuntimeError as error:
            if type(error) is not RuntimeError or str(error) != SAMPLE_RESPONSE_TOO_LARGE or len(sizes) == 1:
                raise
            return await read_page(cursor, sizes[1:])
        return page, sizes

    pages: list[tuple[Sample, tuple[int, ...]]] = []  # mutable-ok: freeze selection after stable cursor traversal
    cursor = ""  # rebind-ok: advance by immutable identity, never by shifting row positions
    while True:
        sizes: Final = pages[-1][1] if pages else SAMPLE_PAGE_SIZES
        page, usable_sizes = await read_page(cursor, sizes)
        pages.append((page, usable_sizes))
        if not page.next_cursor or sum(len(p.executions) for p, _ in pages) >= pages[0][0].selected:
            break
        cursor = page.next_cursor
    executions: Final = tuple(
        execution for p, _ in pages for execution in p.executions
    )  # comprehension-ok: flatten query pages
    selected: Final = Sample(executions=executions, eligible=pages[0][0].eligible, selected=len(executions))

    def freeze(e: Lens) -> Lens:
        active: Final = current_job(e)
        if (
            active is None
            or active.id != job_id
            or active.worker_id != worker.id
            or active.attempts != attempt
            or active.status != "running"
            or active.lease_until is None
            or active.lease_until <= datetime.now(timezone.utc)
        ):
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
    attempt: Attempt = 1,
) -> ExecutionContent:
    lens, job = await assigned(lens_id, job_id, worker, attempt)
    selected: Final = job.sample or Sample(executions=(), eligible=0)
    execution: Final = next((e for e in selected.executions if e.id == execution_id), None)
    if execution is None:
        raise HTTPException(404, "Execution is outside this job's sample")
    return await source_reader(storage).content(lens.scope, execution, cursor, offset)


def model_failure(error: HTTPException | ProxyException) -> HTTPException:
    if isinstance(error, ProxyException):
        status: Final = int(error.code) if error.code.isdigit() else 500
        return HTTPException(status, {"lens_error": redact_internal_details(error.message)}, headers=error.headers)
    if isinstance(error.detail, str):
        return HTTPException(
            error.status_code, {"lens_error": redact_internal_details(error.detail)}, headers=error.headers
        )
    return error


@router.post("/worker/{lens_id}/{job_id}/model", response_model=ModelResult)
async def model(
    lens_id: str,
    job_id: str,
    body: ModelRequest,
    worker: WorkerAuth,
    request: Request,
    response: Response,
    attempt: Attempt = 1,
) -> ModelResult:
    from litellm.proxy.lens.inference import analyze

    lens, job = await assigned(lens_id, job_id, worker, attempt)
    try:
        completion: Final = await analyze(repository(), lens, job, worker, body, request)
    except (ProxyException, HTTPException) as error:
        raise model_failure(error) from error
    if completion.finish_reason:
        response.headers["x-litellm-lens-finish-reason"] = completion.finish_reason
    return completion


@router.post("/worker/{lens_id}/{job_id}/result", response_model=Lens)
async def result(
    lens_id: str, job_id: str, body: Result, worker: WorkerAuth, storage: StorageDep, attempt: Attempt = 1
) -> Lens:
    lens: Final = await get_lens(lens_id, worker.scope)
    old: Final = next((j for j in lens.jobs if j.id == job_id), None)
    if old and old.status in ("completed", "failed") and old.worker_id == worker.id and old.attempts == attempt:
        if old.review_versions and old.status == "completed":
            await repository().complete_reviews(lens_id, old, old.review_versions)
        return lens
    _, job = await assigned(lens_id, job_id, worker, attempt)
    now: Final = datetime.now(timezone.utc)
    selected: Final = job.sample or Sample(executions=(), eligible=0)
    allowed: Final = frozenset(e.id for e in selected.executions)
    if len(frozenset(a.execution_id for a in body.assessments)) != len(body.assessments):
        raise HTTPException(422, "Each run must have one assessment")
    if any(a.execution_id not in allowed for a in body.assessments):
        raise HTTPException(422, "Assessment references a run outside this job")
    if any(version.execution_id not in allowed for version in body.review_versions):
        raise HTTPException(422, "Review checkpoint references a trace outside this job")
    check_ids: Final = frozenset(c.id for c in job.settings.analysis_checks)
    if any(not check_ids.issuperset((*a.issue_checks, *a.pattern_checks)) for a in body.assessments):
        raise HTTPException(422, "Assessment references an unknown check")
    if any(
        not check_ids.issuperset((f.check_id, *f.check_ids)) or any(e.execution_id not in allowed for e in f.evidence)
        for f in body.findings
    ):
        raise HTTPException(422, "Finding references evidence outside the job")

    for finding in body.findings:
        await validate_finding(lens, selected, finding, storage)

    historical_runs: Final = await repository().finding_runs(
        lens_id,
        tuple(finding.id for finding in lens.findings if not finding.investigation_runs),
    )

    def finish(e: Lens) -> Lens:
        active: Final = current_job(e)
        if (
            active is None
            or active.id != job_id
            or active.worker_id != worker.id
            or active.attempts != attempt
            or active.status != "running"
            or active.lease_until is None
            or active.lease_until <= datetime.now(timezone.utc)
        ):
            return e
        restored: Final = e.model_copy(
            update=MappingProxyType(
                {
                    "findings": tuple(
                        finding.model_copy(
                            update=MappingProxyType(
                                {
                                    "investigation_runs": tuple(
                                        sorted(
                                            frozenset(
                                                run.job_id for run in historical_runs if run.finding_id == finding.id
                                            )
                                        )
                                    )
                                }
                            )
                        )
                        if not finding.investigation_runs
                        else finding
                        for finding in e.findings
                    )
                }
            )
        )
        merged: Final = merge_results(restored, body, job.revision, now, job.id).findings
        return replace_job(
            e,
            end_job(active, result_status(body), now).model_copy(
                update=MappingProxyType(
                    {
                        "coverage": active.coverage if body.error and body.coverage == Coverage() else body.coverage,
                        "error": body.error,
                        "assessments": body.assessments,
                        "review_versions": body.review_versions,
                        "findings": tuple(
                            finding.model_copy(
                                update=MappingProxyType(
                                    {
                                        "evidence": tuple(
                                            quote for quote in finding.evidence if quote.execution_id in allowed
                                        ),
                                        "occurrences": tuple(
                                            identity for identity in finding.occurrences if identity in allowed
                                        ),
                                    }
                                )
                            )
                            for finding in merged
                            if finding not in restored.findings
                            and any(quote.execution_id in allowed for quote in finding.evidence)
                        ),
                    }
                )
            ),
        ).model_copy(
            update=MappingProxyType(
                {
                    "findings": merged,
                    "last_scan_at": next_scan_start(e, job, failed=bool(body.error)),
                    "next_run_at": now + timedelta(minutes=e.settings.interval_minutes),
                }
            )
        )

    finished: Final = required(await repository().update(lens_id, finish))
    if body.review_versions and any(
        j.id == job_id and j.status == "completed" and j.attempts == attempt and j.worker_id == worker.id
        for j in finished.jobs
    ):
        await repository().complete_reviews(lens_id, job, body.review_versions)
    return finished


def merge_results(lens: Lens, result: Result, revision: int, now: datetime, job_id: str | None = None) -> Lens:
    def merge_one(current: Lens, draft: FindingDraft) -> Lens:
        finding: Final = merge_finding(current, draft, revision, now, job_id, match_titles=False)
        return current.model_copy(
            update=MappingProxyType(
                {
                    "findings": (
                        finding,
                        *(f for f in current.findings if f.id not in (finding.id, *finding.merged_finding_ids)),
                    )
                }
            )
        )

    return reduce(merge_one, result.findings, lens)


@router.post("/worker/{lens_id}/{job_id}/heartbeat", response_model=bool)
async def heartbeat(lens_id: str, job_id: str, worker: WorkerAuth, attempt: Attempt = 1) -> bool:
    return await progress(lens_id, job_id, Progress(), worker, attempt)


async def claim_candidate(
    candidate: Lens,
    worker: Worker,
    now: datetime,
    lens_repository: _ClaimRepository,
    supports_model: Callable[[Worker, LensSettings], Awaitable[bool]] = worker_supports_model,
) -> Claim | None:
    active: Final = current_job(candidate)
    if not await supports_model(worker, active.settings if active else candidate.settings):
        return None
    job_id: Final = str(uuid4())

    def schedule(e: Lens) -> Lens:
        scheduled: Final = queue_job(e, now, job_id) if e.settings.enabled and e.next_run_at <= now else e
        job: Final = current_job(scheduled)
        if job and job.settings.model != (active.settings.model if active else candidate.settings.model):
            return e
        return claim_job(scheduled, worker, now)

    updated: Final = await lens_repository.update(candidate.id, schedule, attempts=1, changed_only=True)
    if updated is None:
        return None
    job: Final = current_job(updated)
    if job and job.worker_id == worker.id and job.status == "running" and job != current_job(candidate):
        return Claim(lens_id=updated.id, job=job, findings=updated.findings)
    return None


async def validate_finding(lens: Lens, selected: Sample, finding: FindingDraft, storage: Storage | None) -> None:
    previous: Final = next((f for f in lens.findings if f.id == finding.existing_finding_id), None)
    if finding.existing_finding_id and (previous is None or previous.kind != finding.kind):
        raise HTTPException(422, "Existing finding must belong to the same kind")
    if any(
        not any(prior.id == identity and prior.kind == finding.kind for prior in lens.findings)
        for identity in finding.merged_finding_ids
    ):
        raise HTTPException(422, "Merged finding must belong to this investigation and kind")
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

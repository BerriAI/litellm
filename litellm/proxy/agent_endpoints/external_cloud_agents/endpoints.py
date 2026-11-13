import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, Final, TypeAlias, TypeVar

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # legacy factory has untyped optional parameters
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.external_cloud_agents.billing import HTTPClient, SpendSync, utc_now
from litellm.proxy.agent_endpoints.external_cloud_agents.database import Database, ProxyDatabase
from litellm.proxy.agent_endpoints.external_cloud_agents.inventory import (
    Frame,
    Inventory,
    SourceCache,
    authenticated,
    get_source,
    parse_frame,
)
from litellm.proxy.agent_endpoints.external_cloud_agents.management import Management
from litellm.proxy.agent_endpoints.external_cloud_agents.models import (
    AdminKey,
    AgentPage,
    Allow,
    Connection,
    Failure,
    Source,
    SpendReport,
    SyncResult,
    Updated,
)
from litellm.proxy.agent_endpoints.external_cloud_agents.spend import SpendReader
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.custom_openapi_spec import inline_request_body

router: Final = APIRouter(tags=["External Cloud Agents"])
source_cache: Final = SourceCache()
T: Final = TypeVar("T")
NO_BODY: Final = {
    "requestBody": {
        "required": False,
        "content": {"application/json": {"schema": {"type": "object", "maxProperties": 0}}},
    }
}


def raise_public(result: T | Failure) -> T:
    if isinstance(result, Failure):
        raise HTTPException(status_code=result.status, detail=result.message)
    return result


def database() -> Database:
    from litellm.proxy.proxy_server import prisma_client

    return ProxyDatabase(prisma_client.db if prisma_client else None)


def billing_client() -> HTTPClient:
    return get_async_httpx_client(llm_provider="external_cloud_agents_billing").client


async def admin(auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> UserAPIKeyAuth:
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Proxy admin access required")
    return auth


DB: TypeAlias = Annotated[Database, Depends(database)]
Admin: TypeAlias = Annotated[UserAPIKeyAuth, Depends(admin)]
Client: TypeAlias = Annotated[HTTPClient, Depends(billing_client)]


class SyncJobs:
    def __init__(self) -> None:
        self.active: frozenset[str] = frozenset()

    def reserve(self, source_id: str) -> bool:
        if source_id in self.active or len(self.active) >= 64:
            return False
        self.active = self.active | {source_id}
        return True

    async def finish(self, source_id: str, run: Callable[[], Awaitable[SyncResult | Failure]]) -> SyncResult | Failure:
        try:
            return await run()
        finally:
            self.active = self.active - {source_id}


sync_jobs: Final = SyncJobs()


@router.post(
    "/grok_bot/inference_hook/{source_id}",
    response_model=Allow,
    openapi_extra={
        "requestBody": inline_request_body(Frame, {"type": "prompt", "tenant_id": "tenant", "session_id": "bot"})
    },
)
async def inference_hook(source_id: str, request: Request, db: DB) -> Allow:
    """Observe a vendor bot without proxying or enforcing its inference requests."""
    authorization: Final = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer ") or not authorization.removeprefix("Bearer "):
        raise_public(Failure(status=401, message="Invalid ingest key"))
    try:
        async with asyncio.timeout(1.0):
            source: Final = await source_cache.source(db, source_id)
            if source is None or not authenticated(source, authorization):
                return raise_public(Failure(status=401, message="Invalid ingest key"))
            failure: Final = await Inventory(db, source_cache).record(source, parse_frame(await request.body()))
            raise_public(failure)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001  # inventory must fail open even for unexpected storage failures
        verbose_proxy_logger.error("External cloud agent inventory write failed (%s)", type(exc).__name__)
    return Allow()


@router.post("/v1/external_cloud_agents/sources", response_model=Connection, openapi_extra=NO_BODY)
async def create_source(db: DB, auth: Admin) -> Connection:
    """Generate a GrokBot inventory connection. The ingest key is returned here and on the connection endpoint."""
    return await Management(db, source_cache).create(auth.user_id or "proxy-admin")


@router.get("/v1/external_cloud_agents/sources", response_model=tuple[Source, ...])
async def list_sources(db: DB, auth: Admin) -> tuple[Source, ...]:
    """List connections and inventory/sync status without returning either credential."""
    return await Management(db, source_cache).sources()


@router.get("/v1/external_cloud_agents/sources/{source_id}/connection", response_model=Connection)
async def connection(source_id: str, db: DB, auth: Admin) -> Connection:
    """Re-show the hook path and ingest key to a proxy admin."""
    return raise_public(await Management(db, source_cache).connection(source_id))


@router.delete("/v1/external_cloud_agents/sources/{source_id}", response_model=Updated)
async def delete_source(source_id: str, db: DB, auth: Admin) -> Updated:
    """Delete a connection and its inventory and spend rows."""
    await Management(db, source_cache).delete(source_id)
    return Updated()


@router.put("/v1/external_cloud_agents/sources/{source_id}/admin_api_key", response_model=SyncResult)
async def save_admin_key(source_id: str, body: AdminKey, db: DB, auth: Admin, client: Client) -> SyncResult:
    """Validate and encrypt an optional vendor billing key, then sync its spend."""
    if not 0 < len(body.api_key.get_secret_value()) <= 8192:
        return raise_public(Failure(status=400, message="Invalid GrokBot Admin API key"))
    if not sync_jobs.reserve(source_id):
        return raise_public(Failure(status=409, message="A spend sync is already running. Try again shortly"))
    return raise_public(
        await sync_jobs.finish(
            source_id, lambda: SpendSync(db, client).save_key(source_id, body.api_key.get_secret_value())
        )
    )


@router.delete("/v1/external_cloud_agents/sources/{source_id}/admin_api_key", response_model=Updated)
async def delete_admin_key(source_id: str, db: DB, auth: Admin) -> Updated:
    """Remove a billing credential without removing previously collected inventory or spend."""
    raise_public(await Management(db, source_cache).remove_key(source_id))
    return Updated()


@router.post(
    "/v1/external_cloud_agents/sources/{source_id}/sync_spend", response_model=SyncResult, openapi_extra=NO_BODY
)
async def sync_spend(source_id: str, db: DB, auth: Admin, client: Client) -> SyncResult:
    """Refresh the last 30 UTC days of vendor spend."""
    if not sync_jobs.reserve(source_id):
        return SyncResult(status="already_running")
    return raise_public(await sync_jobs.finish(source_id, lambda: SpendSync(db, client).run(source_id)))


@router.get("/v1/external_cloud_agents/agents", response_model=AgentPage)
async def list_agents(
    db: DB,
    auth: Admin,
    source_id: str,
    search: Annotated[str, Query(max_length=200)] = "",
    page: Annotated[int, Query(ge=1, le=1000000)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 50,
) -> AgentPage:
    """List discovered bots, searching their vendor session identifiers."""
    return await Management(db, source_cache).agents(source_id, search, page, page_size)


@router.get("/v1/external_cloud_agents/spend", response_model=SpendReport)
async def get_spend(
    db: DB,
    auth: Admin,
    client: Client,
    background_tasks: BackgroundTasks,
    now: Annotated[datetime, Depends(utc_now)],
    source_id: str,
    days: Annotated[int, Query(ge=1, le=30)] = 30,
) -> SpendReport:
    """Report persisted member/team spend and asynchronously refresh stale vendor data when a key is configured."""
    source: Final = await get_source(db, source_id)
    if source is None:
        return raise_public(Failure(status=404, message="Unknown source"))
    rows: Final = await SpendReader(db, lambda: now).read(source_id, days)
    if (
        source.admin_api_key
        and (source.last_spend_sync_at is None or (now - source.last_spend_sync_at).total_seconds() > 3600)
        and sync_jobs.reserve(source_id)
    ):
        background_tasks.add_task(sync_jobs.finish, source_id, lambda: SpendSync(db, client).run(source_id))
    return rows

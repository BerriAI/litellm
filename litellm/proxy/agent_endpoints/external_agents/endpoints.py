import time
from typing import Annotated, Final, Literal, TypeAlias, TypeVar

from anyio import fail_after
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.external_agents.cache import metadata
from litellm.proxy.agent_endpoints.external_agents.database import Database, ProxyDatabase
from litellm.proxy.agent_endpoints.external_agents.governance import ExternalAgentGovernance, ExternalAgentStore
from litellm.proxy.agent_endpoints.external_agents.management import ExternalAgentManagement
from litellm.proxy.agent_endpoints.external_agents.models import (
    ActorsPut,
    Agent,
    AgentPatch,
    Decision,
    ExternalAgentSummary,
    HookFailure,
    ObservedAgentEvent,
    Policy,
    PolicyPut,
    Source,
    SourceConnection,
    SourceCreate,
    SourcePatch,
    SourceSaved,
    Updated,
    Verdict,
)
from litellm.proxy.agent_endpoints.external_agents.source import GrokBotHookSource, verify_ingest
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper

router: Final = APIRouter(tags=["External agent governance"])
T: Final = TypeVar("T")


def raise_public(result: T | HookFailure) -> T:
    if isinstance(result, HookFailure):
        raise HTTPException(status_code=result.status, detail=result.reason)
    return result


def database() -> Database:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return raise_public(HookFailure(status=503, reason="Database is not connected"))
    return ProxyDatabase(prisma_client.writer_db)


def admin(auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> UserAPIKeyAuth:
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        return raise_public(HookFailure(status=403, reason="Proxy admin access required"))
    return auth


Admin: TypeAlias = Annotated[UserAPIKeyAuth, Depends(admin)]
DB: TypeAlias = Annotated[Database, Depends(database)]


def management(db: DB) -> ExternalAgentManagement:
    return ExternalAgentManagement(db, metadata)


Management: TypeAlias = Annotated[ExternalAgentManagement, Depends(management)]


@router.post(
    "/grok_bot/inference_hook/{source_id}",
    response_model=Verdict,
    response_model_exclude_none=True,
    openapi_extra={"requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}}},
)
async def inference_hook(source_id: str, request: Request, db: DB) -> Verdict:
    """Authenticate a signed GrokBot frame and return a content-free governance verdict."""
    adapter: Final = GrokBotHookSource()
    with fail_after(0.7):
        source: Final = await ExternalAgentStore(db, metadata).source(source_id)
        if source is None:
            return raise_public(HookFailure(status=404, reason="Unknown source"))
        key: Final = decrypt_value_helper(source.ingest_key, "ingest_key") if source.ingest_key else None
        authentication: Final = verify_ingest(request.headers.get("authorization", ""), key)
        if authentication:
            return raise_public(authentication)
        body: Final = await request.body()
        current: Final = (
            decrypt_value_helper(source.signing_secret, "signing_secret") if source.signing_secret else None
        )
        previous: Final = (
            decrypt_value_helper(source.previous_signing_secret, "previous_signing_secret")
            if source.previous_signing_secret
            else None
        )
        secrets: Final = tuple(secret for secret in (current, previous) if secret)
        failure: Final = adapter.verify(body, request.headers, secrets, time.time()) if source.signing_secret else None
        if failure:
            return raise_public(failure)
        event: Final[ObservedAgentEvent] = raise_public(adapter.parse(body, request.headers.get("webhook-id", "")))
    if event.application == "config-test":
        await db.execute_raw(
            'UPDATE "LiteLLM_ExternalAgentSource" SET last_event_at=NOW() WHERE source_id=$1', source_id
        )
    return adapter.format_verdict(raise_public(await ExternalAgentGovernance(db, metadata).evaluate(source, event)))


@router.post("/v1/external_agents/sources", response_model=SourceConnection)
async def create_source(auth: Admin, service: Management, data: SourceCreate | None = None) -> SourceConnection:
    """Register a signed hook source without exposing its signing secrets."""
    return raise_public(await service.create_source(data or SourceCreate(), auth.user_id or "proxy_admin"))


@router.get("/v1/external_agents/sources/{id}/connection", response_model=SourceConnection)
async def source_connection(id: str, auth: Admin, service: Management) -> SourceConnection:
    """Show connection credentials to a proxy administrator."""
    return raise_public(await service.connection(id))


@router.get("/v1/external_agents/sources", response_model=tuple[Source, ...])
async def list_sources(auth: Admin, db: DB) -> tuple[Source, ...]:
    """List registered sources with secrets redacted."""
    return await ExternalAgentManagement(db).sources()


@router.patch("/v1/external_agents/sources/{id}", response_model=SourceSaved)
async def patch_source(id: str, data: SourcePatch, auth: Admin, service: Management) -> SourceSaved:
    """Update a source and retain its previous secret during signing key rotation."""
    return raise_public(await service.patch_source(id, data, auth.user_id or "proxy_admin"))


@router.get("/v1/external_agents", response_model=tuple[Agent, ...], include_in_schema=False)
@router.get("/v1/external_agents/", response_model=tuple[Agent, ...])
async def list_agents(
    auth: Admin,
    db: DB,
    source_id: str | None = None,
    team_id: str | None = None,
    exceptions_only: bool = False,
    search: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> tuple[Agent, ...]:
    """List discovered bots, optionally restricting results to governance exceptions."""
    return await ExternalAgentManagement(db).agents(source_id, team_id, exceptions_only, search, offset, limit)


@router.get("/v1/external_agents/summary", response_model=ExternalAgentSummary)
async def summary(source_id: str, auth: Admin, db: DB) -> ExternalAgentSummary:
    """Summarize discovered bots and unresolved governance exceptions."""
    return await ExternalAgentManagement(db).summary(source_id)


@router.patch("/v1/external_agents/bulk", response_model=Updated)
async def bulk_agents(data: AgentPatch, auth: Admin, db: DB) -> Updated:
    """Assign bot teams, confirm ownership, and update status in bulk."""
    return Updated(updated=await ExternalAgentManagement(db).bulk(data))


@router.get("/v1/external_agents/policies", response_model=tuple[Policy, ...])
async def get_policies(source_id: str, auth: Admin, db: DB) -> tuple[Policy, ...]:
    """List the source default and team policies."""
    return await ExternalAgentManagement(db).policies(source_id)


@router.put("/v1/external_agents/policies", response_model=Policy)
async def put_policy(data: PolicyPut, auth: Admin, db: DB) -> Policy:
    """Upsert the team policy or source-wide default when team_id is null."""
    return raise_public(await ExternalAgentManagement(db, metadata).put_policy(data))


@router.post("/v1/external_agents/actors/bulk", response_model=Updated)
async def bulk_actors(data: ActorsPut, auth: Admin, db: DB) -> Updated:
    """Map source-scoped external actor ids to existing LiteLLM users."""
    return Updated(updated=raise_public(await ExternalAgentManagement(db, metadata).actors(data)))


@router.get("/v1/external_agents/decisions", response_model=tuple[Decision, ...])
async def list_decisions(
    auth: Admin,
    db: DB,
    source_id: str | None = None,
    external_agent_id: str | None = None,
    verdict: Literal["allow", "deny"] | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> tuple[Decision, ...]:
    """List content-free governance decisions with stable pagination ordering."""
    return await ExternalAgentManagement(db).decisions(source_id, external_agent_id, verdict, offset, limit)

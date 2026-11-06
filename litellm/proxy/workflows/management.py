from typing import Annotated, Final, NoReturn, TypeAlias

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from pydantic import ConfigDict

from litellm.models.workflow import WorkflowConflict, WorkflowDraft, WorkflowStore
from litellm.models.workflow_bundle import (
    ChatCompletionNode,
    WorkflowBundle,
    WorkflowFailure,
    canonical_bundle,
    workflow_digest,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model, get_team_object
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import WorkflowRepositoryDep
from litellm.proxy.management.teams.authz import TEAM_OR_ORG_ADMIN, is_team_member
from litellm.proxy.management.teams.dependencies import get_team_access
from litellm.proxy.workflows.validation import validate_bundle
from litellm.types.llms.base import LiteLLMBaseModel

router: Final = APIRouter(tags=["Workflows"])
Auth: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]


class EmptyWorkflowAction(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RegisterWorkflow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle: WorkflowBundle


class PublishWorkflow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    digest: str


class DraftView(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    name: str
    team_id: str
    etag: str
    digest: str
    bundle: WorkflowBundle
    published_snapshot_id: str | None


class DraftList(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    workflows: tuple[DraftView, ...]
    next_cursor: str | None


class ValidationView(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    digest: str


class PublicationView(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    model: str
    digest: str


def raise_public(failure: WorkflowFailure) -> NoReturn:
    raise HTTPException(
        status_code=failure.status_code,
        detail={
            "type": "workflow_error",
            "code": failure.code,
            "message": failure.message,
            "param": failure.param,
        },
    )


def require_store(store: WorkflowStore | None) -> WorkflowStore:
    if store is None:
        raise_public(
            WorkflowFailure(code="database_required", message="Workflow execution requires a database", status_code=503)
        )
    return store


async def require_team(auth: UserAPIKeyAuth, team_id: str | None, *, admin: bool = False, write: bool = False) -> str:
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    resolved_id: Final = team_id if team_id is not None else auth.team_id
    if write and auth.user_role in {LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY}:
        raise_public(
            WorkflowFailure(
                code="write_denied", message="View-only users cannot change or execute drafts", status_code=403
            )
        )
    if resolved_id is None or not auth.user_id:
        raise_public(
            WorkflowFailure(
                code="team_identity_required", message="A team and a bound user are required", status_code=403
            )
        )
    team: Final = await get_team_object(
        resolved_id,
        prisma_client,
        user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        check_db_only=True,
    )
    allowed_admin: Final = await get_team_access().allows(auth, team, TEAM_OR_ORG_ADMIN)
    if not allowed_admin and (admin or not is_team_member(auth, team)):
        raise_public(WorkflowFailure(code="not_found", message="Workflow not found", status_code=404))
    return resolved_id


async def require_draft(
    store: WorkflowStore, workflow_id: str, auth: UserAPIKeyAuth, *, admin: bool = False, write: bool = False
) -> WorkflowDraft:
    draft: Final = await store.get_id(workflow_id)
    if draft is None:
        raise_public(WorkflowFailure(code="not_found", message="Workflow not found", status_code=404))
    await require_team(auth, draft.team_id, admin=admin, write=write)
    return draft


def draft_view(draft: WorkflowDraft) -> DraftView:
    bundle: Final = WorkflowBundle.model_validate_json(draft.bundle_json)
    return DraftView(
        workflow_id=draft.workflow_id,
        name=draft.name,
        team_id=draft.team_id,
        etag=f'"{draft.etag}"',
        digest=workflow_digest(bundle),
        bundle=bundle,
        published_snapshot_id=draft.published_snapshot_id,
    )


def require_etag(draft: WorkflowDraft, if_match: str | None) -> None:
    if if_match != f'"{draft.etag}"':
        raise_public(
            WorkflowFailure(code="stale_draft", message="If-Match must match the reviewed draft ETag", status_code=412)
        )


async def check_bundle(bundle: WorkflowBundle, auth: UserAPIKeyAuth) -> None:
    from litellm.proxy.proxy_server import llm_model_list_view, llm_router

    failure: Final = validate_bundle(bundle)
    if failure is not None:
        raise_public(failure)
    for node in bundle.nodes:
        if isinstance(node, ChatCompletionNode):
            if llm_router is None or node.model not in llm_router.registry().names(team_id=auth.team_id):
                raise_public(
                    WorkflowFailure(code="unknown_model", message="Use a configured gateway model alias", param=node.id)
                )
            await can_key_call_resolved_model(node.model, llm_model_list_view(), auth, llm_router)


@router.post("/v1/workflows", status_code=201, response_model=DraftView)
async def create_workflow(
    data: RegisterWorkflow, response: Response, auth: Auth, workflows: WorkflowRepositoryDep, team_id: str | None = None
) -> DraftView:
    """Register a team-scoped draft without publishing or executing it."""
    team: Final = await require_team(auth, team_id, write=True)
    await check_bundle(data.bundle, auth)
    result: Final = await require_store(workflows).create(
        team, data.bundle.name, canonical_bundle(data.bundle), auth.user_id or ""
    )
    if isinstance(result, WorkflowConflict):
        raise_public(WorkflowFailure(code="workflow_conflict", message=result.message, status_code=result.status_code))
    view: Final = draft_view(result)
    response.headers["ETag"] = view.etag  # rebind-ok: FastAPI injects the response to set protocol headers
    return view


@router.get("/v1/workflows", response_model=DraftList)
async def list_workflows(
    auth: Auth,
    workflows: WorkflowRepositoryDep,
    team_id: str | None = None,
    after: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
) -> DraftList:
    team: Final = await require_team(auth, team_id)
    rows: Final = await require_store(workflows).list(team, after, limit + 1)
    return DraftList(
        workflows=tuple(draft_view(row) for row in rows[:limit]),
        next_cursor=rows[limit - 1].workflow_id if len(rows) > limit else None,
    )


@router.get("/v1/workflows/{workflow_id}", response_model=DraftView)
async def get_workflow(workflow_id: str, response: Response, auth: Auth, workflows: WorkflowRepositoryDep) -> DraftView:
    view: Final = draft_view(await require_draft(require_store(workflows), workflow_id, auth))
    response.headers["ETag"] = view.etag  # rebind-ok: FastAPI injects the response to set protocol headers
    return view


@router.put("/v1/workflows/{workflow_id}", response_model=DraftView)
async def replace_workflow(
    workflow_id: str,
    data: RegisterWorkflow,
    response: Response,
    auth: Auth,
    workflows: WorkflowRepositoryDep,
    if_match: Annotated[str | None, Header()] = None,
) -> DraftView:
    store: Final = require_store(workflows)
    draft: Final = await require_draft(store, workflow_id, auth, write=True)
    require_etag(draft, if_match)
    if data.bundle.name != draft.name:
        raise_public(WorkflowFailure(code="immutable_name", message="Workflow names cannot be changed"))
    await check_bundle(data.bundle, auth)
    result: Final = await store.replace(
        draft.team_id, draft.name, draft.etag, canonical_bundle(data.bundle), auth.user_id or ""
    )
    if isinstance(result, WorkflowConflict):
        raise_public(WorkflowFailure(code="stale_draft", message=result.message, status_code=result.status_code))
    view: Final = draft_view(result)
    response.headers["ETag"] = view.etag  # rebind-ok: FastAPI injects the response to set protocol headers
    return view


@router.post("/v1/workflows/{workflow_id}/validate", response_model=ValidationView)
async def validate_workflow(
    workflow_id: str,
    auth: Auth,
    workflows: WorkflowRepositoryDep,
    if_match: Annotated[str | None, Header()] = None,
    action: EmptyWorkflowAction | None = None,
) -> ValidationView:
    """Validate the reviewed draft without executing Python or calling models."""
    draft: Final = await require_draft(require_store(workflows), workflow_id, auth)
    require_etag(draft, if_match)
    bundle: Final = WorkflowBundle.model_validate_json(draft.bundle_json)
    await check_bundle(bundle, auth)
    return ValidationView(valid=True, digest=workflow_digest(bundle))


@router.post("/v1/workflows/{workflow_id}/publish", response_model=PublicationView)
async def publish_workflow(
    workflow_id: str,
    data: PublishWorkflow,
    auth: Auth,
    workflows: WorkflowRepositoryDep,
    if_match: Annotated[str | None, Header()] = None,
) -> PublicationView:
    """Publish an immutable snapshot after independent approval of the current ETag."""
    store: Final = require_store(workflows)
    draft: Final = await require_draft(store, workflow_id, auth, admin=True, write=True)
    require_etag(draft, if_match)
    bundle: Final = WorkflowBundle.model_validate_json(draft.bundle_json)
    if data.digest != workflow_digest(bundle):
        raise_public(
            WorkflowFailure(code="stale_draft", message="Reviewed digest does not match the draft", status_code=412)
        )
    await check_bundle(bundle, auth)
    result: Final = await store.publish(draft.team_id, draft.name, draft.etag, auth.user_id or "")
    if isinstance(result, WorkflowConflict):
        raise_public(WorkflowFailure(code="publication_denied", message=result.message, status_code=result.status_code))
    return PublicationView(snapshot_id=result.snapshot_id, model=f"workflow/{draft.name}", digest=data.digest)

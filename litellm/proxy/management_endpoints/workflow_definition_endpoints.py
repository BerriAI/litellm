from typing import Annotated, Final, Literal, NoReturn, TypeAlias, TypeVar

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from typing_extensions import assert_never

from litellm.models.workflow_definition import (
    DefinitionError,
    DefinitionErrorCode,
    DefinitionPage,
    Digest,
    WorkflowBundle,
    WorkflowDefinitionStore,
    WorkflowDraft,
)
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import WorkflowDefinitionRepositoryDep
from litellm.proxy.workflows.definitions import (
    authorize_team,
    authorized_draft,
    check_etag,
    requested_team,
    validate_bundle,
)

router: Final = APIRouter(tags=["workflow management"])
_Caller: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
_Team: TypeAlias = Annotated[str | None, Query(min_length=1, max_length=256)]
_ETag: TypeAlias = Annotated[str | None, Header(alias="If-Match")]
_Result: Final = TypeVar("_Result")


class WorkflowRegistration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle: WorkflowBundle


class WorkflowPublication(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    digest: Digest


class WorkflowValidationRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WorkflowValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: Literal[True] = True
    workflow_id: str
    etag: str
    digest: str


class WorkflowPublicationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    snapshot_id: str
    model: str
    digest: str


def _error_status(code: DefinitionErrorCode) -> int:
    match code:
        case "not_found":
            return 404
        case "forbidden":
            return 403
        case "team_required" | "invalid_bundle":
            return 400
        case "conflict":
            return 409
        case "stale_draft":
            return 412
        case "precondition_required":
            return 412
        case _:
            assert_never(code)


def raise_public(error: DefinitionError) -> NoReturn:
    raise HTTPException(
        status_code=_error_status(error.code),
        detail={
            "error": {
                "type": "workflow_definition_error",
                "code": error.code,
                "message": error.message,
                "param": error.param,
            }
        },
    )


def _result(result: _Result | DefinitionError) -> _Result:
    if isinstance(result, DefinitionError):
        raise_public(result)
    return result


def _store(store: WorkflowDefinitionStore | None) -> WorkflowDefinitionStore:
    if store is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    return store


def _author(caller: UserAPIKeyAuth) -> str:
    if not caller.user_id or caller.user_role not in {
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.ORG_ADMIN,
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.TEAM,
    }:
        raise_public(DefinitionError("forbidden", "Workflow authoring requires a team member identity"))
    return caller.user_id


def _draft_response(draft: WorkflowDraft, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=draft.model_dump(mode="json", by_alias=True, exclude_unset=True),
        status_code=status_code,
        headers={"ETag": draft.etag},
    )


@router.post("/v1/workflows", status_code=201, response_model=WorkflowDraft, response_model_exclude_unset=True)
async def register_workflow(
    data: WorkflowRegistration,
    repository: WorkflowDefinitionRepositoryDep,
    caller: _Caller,
    team_id: _Team = None,
) -> JSONResponse:
    """Create a team draft from a sequential LiteFlow v1 bundle."""
    contributor: Final = _author(caller)
    store: Final = _store(repository)
    scope: Final = _result(requested_team(caller, team_id))
    _result(await authorize_team(store, caller, scope, "write"))
    if (failure := validate_bundle(data.bundle)) is not None:
        raise_public(failure)
    draft: Final = _result(await store.create(data.bundle, scope, contributor))
    return _draft_response(draft, 201)


@router.get("/v1/workflows", response_model=DefinitionPage, response_model_exclude_unset=True)
async def list_workflow_definitions(
    repository: WorkflowDefinitionRepositoryDep,
    caller: _Caller,
    team_id: _Team = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[
        str | None, Query(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    ] = None,
) -> DefinitionPage:
    """List readable team drafts in immutable workflow ID order."""
    store: Final = _store(repository)
    scope: Final = _result(requested_team(caller, team_id))
    _result(await authorize_team(store, caller, scope, "read"))
    return await store.list_drafts(scope, limit, cursor)


@router.get("/v1/workflows/{workflow_id}", response_model=WorkflowDraft, response_model_exclude_unset=True)
async def get_workflow_definition(
    workflow_id: str,
    repository: WorkflowDefinitionRepositoryDep,
    caller: _Caller,
    team_id: _Team = None,
) -> JSONResponse:
    """Read the current draft and its publication pointer."""
    draft: Final = _result(await authorized_draft(_store(repository), caller, workflow_id, team_id))
    return _draft_response(draft)


@router.put("/v1/workflows/{workflow_id}", response_model=WorkflowDraft, response_model_exclude_unset=True)
async def replace_workflow_definition(
    workflow_id: str,
    data: WorkflowRegistration,
    repository: WorkflowDefinitionRepositoryDep,
    caller: _Caller,
    team_id: _Team = None,
    if_match: _ETag = None,
) -> JSONResponse:
    """Replace a draft only when If-Match identifies its current revision."""
    contributor: Final = _author(caller)
    store: Final = _store(repository)
    current: Final = _result(await authorized_draft(store, caller, workflow_id, team_id))
    _result(await authorize_team(store, caller, current.team_id, "write"))
    if (precondition := check_etag(current, if_match)) is not None:
        raise_public(precondition)
    if (failure := validate_bundle(data.bundle)) is not None:
        raise_public(failure)
    draft: Final = _result(await store.replace(workflow_id, current.team_id, current.etag, data.bundle, contributor))
    return _draft_response(draft)


@router.post("/v1/workflows/{workflow_id}/validate", response_model=WorkflowValidationResult)
async def validate_workflow_definition(
    workflow_id: str,
    repository: WorkflowDefinitionRepositoryDep,
    caller: _Caller,
    team_id: _Team = None,
    if_match: _ETag = None,
    data: WorkflowValidationRequest | None = None,
) -> WorkflowValidationResult:
    """Validate a draft without model calls or Python execution."""
    _author(caller)
    store: Final = _store(repository)
    draft: Final = _result(await authorized_draft(store, caller, workflow_id, team_id))
    _result(await authorize_team(store, caller, draft.team_id, "write"))
    if (precondition := check_etag(draft, if_match)) is not None:
        raise_public(precondition)
    if (failure := validate_bundle(draft.bundle)) is not None:
        raise_public(failure)
    return WorkflowValidationResult(workflow_id=workflow_id, etag=draft.etag, digest=draft.digest)


@router.post("/v1/workflows/{workflow_id}/publish", response_model=WorkflowPublicationResult)
async def publish_workflow_definition(
    workflow_id: str,
    data: WorkflowPublication,
    repository: WorkflowDefinitionRepositoryDep,
    caller: _Caller,
    team_id: _Team = None,
    if_match: _ETag = None,
) -> WorkflowPublicationResult:
    """Publish the reviewed revision as an immutable snapshot."""
    publisher: Final = _author(caller)
    store: Final = _store(repository)
    draft: Final = _result(await authorized_draft(store, caller, workflow_id, team_id))
    _result(await authorize_team(store, caller, draft.team_id, "publish"))
    if (precondition := check_etag(draft, if_match)) is not None:
        raise_public(precondition)
    if (failure := validate_bundle(draft.bundle)) is not None:
        raise_public(failure)
    snapshot: Final = _result(await store.publish(workflow_id, draft.team_id, draft.etag, data.digest, publisher))
    return WorkflowPublicationResult(
        workflow_id=workflow_id,
        snapshot_id=snapshot.snapshot_id,
        model=f"workflow/{snapshot.name}",
        digest=snapshot.digest,
    )

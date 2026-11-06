import asyncio
import hashlib
import json
from collections.abc import Mapping
from datetime import timedelta
from typing import Annotated, Final, Literal, NoReturn, TypeAlias

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, TypeAdapter, ValidationError

from litellm.models.workflow_definition import DefinitionError, WorkflowDefinitionStore, WorkflowSnapshot
from litellm.models.workflow_execution import (
    EngineRun,
    ExecutionError,
    RunInput,
    RunPage,
    RunState,
    RunSteps,
    WorkflowExecutionStore,
)
from litellm.proxy._types import UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import WorkflowDefinitionRepositoryDep, WorkflowExecutionRepositoryDep
from litellm.proxy.management.teams.authz import is_team_admin
from litellm.proxy.management_endpoints.workflow_definition_endpoints import raise_public
from litellm.proxy.workflows.definitions import (
    authorize_team,
    authorized_draft,
    prepare_draft,
    published_workflow,
)
from litellm.proxy.workflows.lifespan import require_workflows_enabled
from litellm.proxy.workflows.monty.models import MontyFailure
from litellm.proxy.workflows.monty.runtime import validate_json
from litellm.proxy.workflows.runner import utc_now

router: Final = APIRouter(tags=["workflow management"], dependencies=[Depends(require_workflows_enabled)])
_Caller: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
_Idempotency: TypeAlias = Annotated[str | None, Header(alias="Idempotency-Key", min_length=1, max_length=256)]
_MESSAGES: Final = TypeAdapter(list[ChatCompletionMessageParam])


class TestRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[Mapping[str, JsonValue], ...] = Field(min_length=1, max_length=256)
    authorize_billing: StrictBool


class AcceptedRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    snapshot_id: str
    trace_url: str


class EmptyRunAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WorkflowInvocation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    messages: tuple[Mapping[str, JsonValue], ...] = Field(min_length=1, max_length=256)
    stream: Literal[False] = False
    user: str | None = Field(default=None, max_length=256)
    metadata: Mapping[str, JsonValue] | None = None


def _error(error: ExecutionError, run_id: str | None = None) -> NoReturn:
    code: Final = {
        "not_found": 404,
        "forbidden": 403,
        "idempotency_conflict": 409,
        "capacity_exceeded": 429,
        "wait_timeout": 504,
        "execution_failed": 422,
        "database_unavailable": 503,
    }.get(error.code, 400)
    raise HTTPException(
        status_code=code,
        detail={"error": {"type": "workflow_error", "code": error.code, "message": error.message}},
        headers={"x-litellm-workflow-run-id": run_id} if run_id is not None else None,
    )


def _stores(
    definitions: WorkflowDefinitionStore | None, runs: WorkflowExecutionStore | None
) -> tuple[WorkflowDefinitionStore, WorkflowExecutionStore]:
    if definitions is None or runs is None:
        _error(ExecutionError("database_unavailable", "Workflow execution requires a database"))
    return definitions, runs


def _execution_key(caller: UserAPIKeyAuth, team_id: str) -> str:
    if not caller.token or not caller.via_virtual_key or caller.is_session_token or caller.team_id != team_id:
        _error(ExecutionError("forbidden", "Execution requires a virtual key assigned to the workflow team"))
    return caller.token


def _idempotency_key(operation: str, key: str | None) -> str | None:
    return None if key is None else hashlib.sha256(f"{operation}:{key}".encode()).hexdigest()


def _digest(messages: tuple[Mapping[str, JsonValue], ...], user: str | None, draft_etag: str | None) -> str:
    return hashlib.sha256(
        json.dumps(
            {"messages": tuple(dict(message) for message in messages), "user": user, "etag": draft_etag},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _validate_messages(messages: tuple[Mapping[str, JsonValue], ...]) -> None:
    bounded: Final = validate_json([dict(message) for message in messages], max_bytes=1024 * 1024, max_depth=64)
    if isinstance(bounded, MontyFailure):
        _error(ExecutionError("invalid_input", "Workflow messages exceed the JSON size or depth limit"))
    try:
        _MESSAGES.validate_python(bounded, strict=True)
    except ValidationError:
        _error(ExecutionError("invalid_input", "Expected standard Chat Completions messages"))


async def _recover_admission(
    runs: WorkflowExecutionStore,
    workflow_id: str,
    team_id: str,
    key_hash: str,
    key: str | None,
    digest: str,
) -> EngineRun | None:
    if key is None:
        return None
    existing: Final = await runs.find_idempotent(workflow_id, team_id, key_hash, key)
    if existing is not None and existing.request_digest != digest:
        _error(ExecutionError("idempotency_conflict", "Idempotency key was used with different input"))
    return existing


async def _admit(
    runs: WorkflowExecutionStore,
    snapshot: WorkflowSnapshot,
    caller: UserAPIKeyAuth,
    messages: tuple[Mapping[str, JsonValue], ...],
    user: str | None,
    key: str | None,
    digest: str,
    *,
    draft: bool,
) -> EngineRun:
    admitted: Final = await runs.admit(
        workflow_id=snapshot.workflow_id,
        snapshot_id=snapshot.snapshot_id,
        team_id=snapshot.team_id,
        key_hash=_execution_key(caller, snapshot.team_id),
        input=RunInput(
            messages=messages,
            end_user_id=user,
            deadline=utc_now() + timedelta(seconds=snapshot.bundle.timeout_seconds),
            draft=draft,
        ),
        digest=digest,
        idempotency_key=key,
        draft_snapshot=snapshot if draft else None,
    )
    if isinstance(admitted, ExecutionError):
        _error(admitted)
    return admitted.run


def _accepted(run: EngineRun) -> JSONResponse:
    return JSONResponse(
        status_code=202,
        content=AcceptedRun(
            run_id=run.run_id,
            snapshot_id=run.snapshot_id,
            trace_url=f"/v1/workflows/runs/{run.run_id}/steps",
        ).model_dump(),
        headers={"x-litellm-workflow-run-id": run.run_id},
    )


@router.post("/v1/workflows/{workflow_id}/test", status_code=202, response_model=AcceptedRun)
async def test_workflow(
    workflow_id: str,
    data: TestRunRequest,
    caller: _Caller,
    definitions: WorkflowDefinitionRepositoryDep,
    executions: WorkflowExecutionRepositoryDep,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
    idempotency_key: _Idempotency = None,
) -> JSONResponse:
    """Start a paid test of the exact draft revision and return its run ID immediately."""
    store, runs = _stores(definitions, executions)
    if not data.authorize_billing:
        _error(ExecutionError("billing_authorization_required", "Set authorize_billing to true for a paid test run"))
    draft: Final = await authorized_draft(store, caller, workflow_id)
    if isinstance(draft, DefinitionError):
        raise_public(draft)
    permission: Final = await authorize_team(store, caller, draft.team_id, "write")
    if isinstance(permission, DefinitionError):
        raise_public(permission)
    key_hash: Final = _execution_key(caller, draft.team_id)
    _validate_messages(data.messages)
    digest: Final = _digest(data.messages, caller.end_user_id, if_match)
    key: Final = _idempotency_key("test", idempotency_key)
    previous: Final = await _recover_admission(runs, workflow_id, draft.team_id, key_hash, key, digest)
    if previous is not None:
        return _accepted(previous)
    snapshot: Final = await prepare_draft(store, caller, workflow_id, if_match)
    if isinstance(snapshot, DefinitionError):
        raise_public(snapshot)
    return _accepted(await _admit(runs, snapshot, caller, data.messages, caller.end_user_id, key, digest, draft=True))


async def _read_run(
    store: WorkflowDefinitionStore, runs: WorkflowExecutionStore, caller: UserAPIKeyAuth, run_id: str
) -> EngineRun:
    run: Final = await runs.get(run_id)
    if run is None:
        _error(ExecutionError("not_found", "Run not found"))
    team: Final = await authorize_team(store, caller, run.team_id, "invoke")
    if isinstance(team, DefinitionError):
        _error(ExecutionError("not_found", "Run not found"))
    owns_run: Final = bool(caller.token) and caller.token == run.created_by and caller.team_id == run.team_id
    if not owns_run and not user_api_key_has_admin_view(caller) and not is_team_admin(caller, team):
        _error(ExecutionError("not_found", "Run not found"))
    return run


@router.get("/v1/workflows/runs/{run_id}/steps")
async def get_workflow_steps(
    run_id: str,
    caller: _Caller,
    definitions: WorkflowDefinitionRepositoryDep,
    executions: WorkflowExecutionRepositoryDep,
) -> RunSteps:
    """Read run progress, typed node inputs and outputs, and linked model-call receipts."""
    store, runs = _stores(definitions, executions)
    run: Final = await _read_run(store, runs, caller, run_id)
    error: Final = (run.metadata or {}).get("error")
    return RunSteps(
        run_id=run_id,
        snapshot_id=run.snapshot_id,
        state=run.status,
        cancel_requested=run.cancel_requested,
        steps=await runs.steps(run_id),
        error=error if isinstance(error, str) else None,
    )


@router.get("/v1/workflows/runs/{run_id}/definition")
async def get_run_definition(
    run_id: str,
    caller: _Caller,
    definitions: WorkflowDefinitionRepositoryDep,
    executions: WorkflowExecutionRepositoryDep,
) -> WorkflowSnapshot:
    """Read the immutable workflow definition captured when this run was admitted."""
    store, runs = _stores(definitions, executions)
    run: Final = await _read_run(store, runs, caller, run_id)
    snapshot: Final = await store.get_snapshot(run.snapshot_id, run.team_id)
    if snapshot is None or snapshot.workflow_id != run.workflow_id:
        _error(ExecutionError("not_found", "Pinned workflow definition not found"))
    return snapshot


@router.post("/v1/workflows/runs/{run_id}/cancel", status_code=202, response_model=AcceptedRun)
async def cancel_workflow_run(
    run_id: str,
    caller: _Caller,
    definitions: WorkflowDefinitionRepositoryDep,
    executions: WorkflowExecutionRepositoryDep,
    data: EmptyRunAction | None = None,
) -> JSONResponse:
    """Request cancellation and return the run ID to poll until execution stops."""
    store, runs = _stores(definitions, executions)
    run: Final = await _read_run(store, runs, caller, run_id)
    if caller.token != run.created_by:
        permission: Final = await authorize_team(store, caller, run.team_id, "write")
        if isinstance(permission, DefinitionError):
            raise_public(permission)
    await runs.cancel(run_id)
    return _accepted(run)


async def list_engine_runs(
    *,
    workflow_id: str,
    team_id: str | None,
    status: str | None,
    limit: int,
    after: str | None,
    caller: UserAPIKeyAuth,
    definitions: WorkflowDefinitionStore | None,
    executions: WorkflowExecutionStore | None,
) -> RunPage:
    require_workflows_enabled()
    store, runs = _stores(definitions, executions)
    definition: Final = await store.get_draft(workflow_id, team_id)
    if definition is None:
        _error(ExecutionError("not_found", "Workflow not found"))
    permission: Final = await authorize_team(store, caller, definition.draft.team_id, "invoke")
    if isinstance(permission, DefinitionError):
        raise_public(permission)
    may_read_team: Final = user_api_key_has_admin_view(caller) or is_team_admin(caller, permission)
    if not may_read_team and not caller.token:
        _error(ExecutionError("forbidden", "A run reader identity is required"))
    try:
        state: Final = RunState(status) if status is not None else None
    except ValueError:
        _error(ExecutionError("invalid_status", "Unknown workflow run status"))
    result: Final = await runs.list_runs(
        workflow_id,
        definition.draft.team_id,
        None if may_read_team else caller.token,
        state,
        min(limit, 100),
        after,
    )
    if isinstance(result, ExecutionError):
        _error(result)
    return result


@router.get("/v1/workflows/{workflow_id}/runs")
async def list_definition_runs(
    workflow_id: str,
    caller: _Caller,
    definitions: WorkflowDefinitionRepositoryDep,
    executions: WorkflowExecutionRepositoryDep,
    team_id: str | None = None,
    status: RunState | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    after: Annotated[str | None, Query(max_length=512)] = None,
) -> RunPage:
    return await list_engine_runs(
        workflow_id=workflow_id,
        team_id=team_id,
        status=status.value if status is not None else None,
        limit=limit,
        after=after,
        caller=caller,
        definitions=definitions,
        executions=executions,
    )


async def _wait_for_run(runs: WorkflowExecutionStore, run_id: str) -> EngineRun:
    while True:
        current: EngineRun | None = await runs.get(run_id)
        if current is None:
            _error(ExecutionError("not_found", "Run not found"), run_id)
        if current.status in {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}:
            return current
        await asyncio.sleep(0.2)


async def invoke_workflow(
    request: Request,
    response: Response,
    caller: UserAPIKeyAuth,
    raw: object,
    definitions: WorkflowDefinitionStore | None,
    executions: WorkflowExecutionStore | None,
) -> Mapping[str, JsonValue] | JSONResponse:
    store, runs = _stores(definitions, executions)
    try:
        data: Final = WorkflowInvocation.model_validate(raw)
    except ValidationError:
        _error(ExecutionError("unsupported_request", "Use model, messages, user, metadata and stream=false"))
    snapshot: Final = await published_workflow(store, caller, data.model.removeprefix("workflow/"))
    if isinstance(snapshot, DefinitionError):
        raise_public(snapshot)
    key: Final = request.headers.get("Idempotency-Key")
    if key is not None and not 1 <= len(key) <= 256:
        _error(ExecutionError("invalid_idempotency_key", "Idempotency-Key must contain 1 to 256 characters"))
    _validate_messages(data.messages)
    admitted: Final = await _admit(
        runs,
        snapshot,
        caller,
        data.messages,
        data.user or caller.end_user_id,
        _idempotency_key("invoke", key),
        _digest(data.messages, data.user or caller.end_user_id, None),
        draft=False,
    )
    response.headers["x-litellm-workflow-run-id"] = admitted.run_id  # rebind-ok: FastAPI response header
    if request.headers.get("Prefer") == "respond-async":
        return _accepted(admitted)
    try:
        completed: Final = await asyncio.wait_for(_wait_for_run(runs, admitted.run_id), timeout=60)
    except (TimeoutError, asyncio.TimeoutError):
        _error(
            ExecutionError("wait_timeout", "The run is still executing. Inspect its trace or retry with the same key"),
            admitted.run_id,
        )
    if completed.status is not RunState.COMPLETED:
        _error(
            ExecutionError("execution_failed", "Workflow execution did not complete. Inspect the run trace"),
            admitted.run_id,
        )
    traces: Final = await runs.steps(admitted.run_id)
    prompt_tokens: Final = sum(step.prompt_tokens for step in traces)
    completion_tokens: Final = sum(step.completion_tokens for step in traces)
    content: Final = (
        completed.output
        if isinstance(completed.output, str)
        else json.dumps(completed.output, ensure_ascii=False, allow_nan=False)
    )
    return {
        "id": f"chatcmpl-workflow-{completed.run_id}",
        "object": "chat.completion",
        "created": int(completed.created_at.timestamp()),
        "model": data.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

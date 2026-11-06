from typing import Annotated, Final, Literal

from fastapi import APIRouter, Header, Request
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ConfigDict

from litellm.models.workflow import WorkflowConflict, WorkflowExecution, WorkflowRunState, WorkflowStep, WorkflowStore
from litellm.models.workflow_bundle import WorkflowBundle, WorkflowFailure
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.deps.repositories import WorkflowRepositoryDep
from litellm.proxy.workflows.invocation import InvocationMetadata, WorkflowInvocation, admit, principal_hash
from litellm.proxy.workflows.management import (
    Auth,
    EmptyWorkflowAction,
    check_bundle,
    raise_public,
    require_draft,
    require_etag,
    require_store,
    require_team,
)
from litellm.types.llms.base import LiteLLMBaseModel

router: Final = APIRouter(tags=["Workflows"])


class DraftTest(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[ChatCompletionMessageParam, ...]
    user: str | None = None
    metadata: InvocationMetadata | None = None
    authorize_billing: Literal[True]


class RunAccepted(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    snapshot_id: str
    trace_url: str


class RunSteps(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    snapshot_id: str
    state: WorkflowRunState
    cancel_requested: bool
    steps: tuple[WorkflowStep, ...]


class CancellationView(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    cancel_requested: bool


async def require_execution(
    store: WorkflowStore, run_id: str, auth: UserAPIKeyAuth, *, write: bool = False
) -> WorkflowExecution:
    execution: Final = await store.execution(run_id)
    if execution is None:
        raise_public(WorkflowFailure(code="not_found", message="Workflow run not found", status_code=404))
    if principal_hash(auth) != execution.principal:
        await require_team(auth, execution.team_id, admin=True, write=write)
    elif auth.team_id != execution.team_id:
        raise_public(WorkflowFailure(code="not_found", message="Workflow run not found", status_code=404))
    return execution


@router.get("/v1/workflows/runs/{run_id}/steps", response_model=RunSteps)
async def read_steps(run_id: str, auth: Auth, workflows: WorkflowRepositoryDep) -> RunSteps:
    store: Final = require_store(workflows)
    execution: Final = await require_execution(store, run_id, auth)
    return RunSteps(
        run_id=run_id,
        snapshot_id=execution.snapshot_id,
        state=execution.state,
        cancel_requested=execution.cancel_requested,
        steps=await store.steps(run_id),
    )


@router.get("/v1/workflows/runs/{run_id}/steps/{step_id}", response_model=WorkflowStep)
async def read_step(run_id: str, step_id: str, auth: Auth, workflows: WorkflowRepositoryDep) -> WorkflowStep:
    store: Final = require_store(workflows)
    await require_execution(store, run_id, auth)
    steps: Final = await store.steps(run_id)
    selected: Final = next((step for step in steps if step.step_id == step_id), None)
    if selected is None:
        raise_public(WorkflowFailure(code="not_found", message="Workflow step not found", status_code=404))
    return selected


@router.post("/v1/workflows/runs/{run_id}/cancel", status_code=202, response_model=CancellationView)
async def cancel_run(
    run_id: str, auth: Auth, workflows: WorkflowRepositoryDep, action: EmptyWorkflowAction | None = None
) -> CancellationView:
    """Request cancellation and prevent further node dispatch for this run."""
    store: Final = require_store(workflows)
    await require_execution(store, run_id, auth, write=True)
    await store.cancel(run_id)
    return CancellationView(run_id=run_id, cancel_requested=True)


@router.post("/v1/workflows/{workflow_id}/test", status_code=202, response_model=RunAccepted)
async def test_draft(
    workflow_id: str,
    data: DraftTest,
    request: Request,
    auth: Auth,
    workflows: WorkflowRepositoryDep,
    if_match: Annotated[str | None, Header()] = None,
) -> RunAccepted:
    """Execute the reviewed draft with explicit billing authorization."""
    from litellm.proxy.proxy_server import general_settings_view

    if general_settings_view().get("enable_workflows") is not True:
        raise_public(
            WorkflowFailure(code="workflow_disabled", message="Workflow execution is disabled", status_code=503)
        )
    store: Final = require_store(workflows)
    draft: Final = await require_draft(store, workflow_id, auth, write=True)
    require_etag(draft, if_match)
    bundle: Final = WorkflowBundle.model_validate_json(draft.bundle_json)
    await check_bundle(bundle, auth)
    snapshot: Final = await store.pin_draft(draft.workflow_id, draft.etag, auth.user_id or "")
    if isinstance(snapshot, WorkflowConflict):
        raise_public(WorkflowFailure(code="stale_draft", message=snapshot.message, status_code=snapshot.status_code))
    accepted: Final = await admit(
        store,
        auth,
        WorkflowInvocation(
            model=f"workflow/{draft.name}", messages=data.messages, user=data.user, metadata=data.metadata
        ),
        request.headers.get("Idempotency-Key"),
        snapshot=snapshot,
        operation="draft_test",
    )
    return RunAccepted(
        run_id=accepted.run_id,
        snapshot_id=accepted.snapshot_id,
        trace_url=f"/v1/workflows/runs/{accepted.run_id}/steps",
    )

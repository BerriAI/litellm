import asyncio
import hashlib
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final, Literal

from fastapi import Request
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ConfigDict, Field, JsonValue, TypeAdapter, ValidationError
from starlette.responses import JSONResponse

from litellm._uuid import uuid
from litellm.models.workflow import WorkflowExecution, WorkflowRunState, WorkflowSnapshot, WorkflowStore
from litellm.models.workflow_bundle import WorkflowFailure
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import hash_token
from litellm.proxy.workflows.management import raise_public, require_store
from litellm.proxy.workflows.validation import canonical_request
from litellm.types.llms.base import LiteLLMBaseModel


class InvocationMetadata(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tags: tuple[str, ...] = Field(default=(), max_length=20)


class WorkflowInvocation(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(pattern=r"^workflow/[a-z][a-z0-9-]{0,62}$")
    messages: tuple[ChatCompletionMessageParam, ...] = Field(min_length=1, max_length=128)
    user: str | None = Field(default=None, max_length=256)
    metadata: InvocationMetadata | None = None
    stream: Literal[False] = False


_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_RUN_HEADER: Final = "x-litellm-workflow-run-id"


def principal_hash(auth: UserAPIKeyAuth) -> str:
    from litellm.proxy.proxy_server import user_custom_auth

    if (
        not auth.token
        or not auth.team_id
        or not auth.via_virtual_key
        or auth.is_session_token
        or user_custom_auth is not None
    ):
        raise_public(
            WorkflowFailure(
                code="virtual_key_required", message="Workflow invocation requires a team virtual key", status_code=403
            )
        )
    return hash_token(auth.token) if auth.token.startswith("sk-") else auth.token


async def admit(
    store: WorkflowStore,
    auth: UserAPIKeyAuth,
    invocation: WorkflowInvocation,
    idempotency_key: str | None,
    *,
    snapshot: WorkflowSnapshot | None = None,
    operation: str = "invoke",
) -> WorkflowExecution:
    principal: Final = principal_hash(auth)
    if snapshot is not None and snapshot.team_id != auth.team_id:
        raise_public(
            WorkflowFailure(
                code="workflow_team_mismatch",
                message="Workflow execution requires a virtual key for its owning team",
                status_code=403,
            )
        )
    if idempotency_key is not None and not 1 <= len(idempotency_key) <= 256:
        raise_public(
            WorkflowFailure(code="invalid_idempotency_key", message="Idempotency-Key must contain 1 to 256 characters")
        )
    request_json: Final = canonical_request(
        _JSON_OBJECT.validate_python(invocation.model_dump(mode="json", exclude_none=True))
    )
    digest: Final = hashlib.sha256(request_json.encode()).hexdigest()
    name: Final = invocation.model.removeprefix("workflow/")
    dispatch_key: Final = hashlib.sha256(
        canonical_request(
            {
                "principal": principal,
                "team": auth.team_id,
                "operation": operation,
                "workflow": name,
                "idempotency_key": idempotency_key if idempotency_key is not None else str(uuid.uuid4()),
            }
        ).encode()
    ).hexdigest()
    existing: Final = await store.by_dispatch_key(dispatch_key)
    if existing is not None:
        if existing.request_digest != digest:
            raise_public(
                WorkflowFailure(
                    code="idempotency_conflict",
                    message="Idempotency-Key was accepted with a different request",
                    status_code=409,
                )
            )
        return existing
    pinned: Final = snapshot if snapshot is not None else await store.published(auth.team_id or "", name)
    if pinned is None:
        raise_public(WorkflowFailure(code="not_found", message="Published workflow not found", status_code=404))
    from litellm.models.workflow_bundle import WorkflowBundle

    bundle: Final = WorkflowBundle.model_validate_json(pinned.bundle_json)
    accepted: Final = await store.dispatch(
        WorkflowExecution(
            run_id=str(uuid.uuid4()),
            snapshot_id=pinned.snapshot_id,
            team_id=pinned.team_id,
            name=name,
            principal=principal,
            dispatch_key=dispatch_key,
            request_digest=digest,
            request_json=request_json,
            state=WorkflowRunState.PENDING,
            deadline=datetime.now(timezone.utc) + timedelta(seconds=bundle.timeout_seconds),
            cancel_requested=False,
        )
    )
    if accepted.request_digest != digest:
        raise_public(
            WorkflowFailure(
                code="idempotency_conflict",
                message="Idempotency-Key was accepted with a different request",
                status_code=409,
            )
        )
    return accepted


def error_response(failure: WorkflowFailure, run_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=failure.status_code,
        content={
            "error": {
                "type": "workflow_error",
                "code": failure.code,
                "message": failure.message,
                "param": failure.param,
            }
        },
        headers={_RUN_HEADER: run_id},
    )


async def wait_for_run(store: WorkflowStore, execution: WorkflowExecution, wait_seconds: float = 60.0) -> JSONResponse:
    try:
        async with asyncio.timeout(wait_seconds):
            while True:
                current = await store.execution(execution.run_id)
                if current is None:
                    return error_response(
                        WorkflowFailure(
                            code="run_unavailable", message="Accepted run is temporarily unavailable", status_code=503
                        ),
                        execution.run_id,
                    )
                if current.state is WorkflowRunState.COMPLETED and current.response_json is not None:
                    return JSONResponse(
                        content=_JSON_OBJECT.validate_json(current.response_json), headers={_RUN_HEADER: current.run_id}
                    )
                if current.state is WorkflowRunState.FAILED:
                    failure = (
                        WorkflowFailure.model_validate_json(current.error)
                        if current.error is not None
                        else WorkflowFailure(
                            code="execution_failed",
                            message="Workflow execution failed",
                            status_code=502,
                        )
                    )
                    return error_response(failure, current.run_id)
                if current.state is WorkflowRunState.CANCELLED:
                    return error_response(
                        WorkflowFailure(code="cancelled", message="Run was cancelled", status_code=409), current.run_id
                    )
                await asyncio.sleep(0.05)
    except TimeoutError:
        return error_response(
            WorkflowFailure(
                code="wait_timeout",
                message="Run continues; retry with the same Idempotency-Key or inspect it",
                status_code=504,
            ),
            execution.run_id,
        )


async def invoke_workflow(
    request: Request, auth: UserAPIKeyAuth, data: Mapping[str, object], workflows: WorkflowStore | None
) -> JSONResponse:
    from litellm.proxy.proxy_server import general_settings_view

    if general_settings_view().get("enable_workflows") is not True:
        raise_public(
            WorkflowFailure(code="workflow_disabled", message="Workflow execution is disabled", status_code=503)
        )
    try:
        invocation: Final = WorkflowInvocation.model_validate(data)
    except ValidationError:
        raise_public(
            WorkflowFailure(
                code="unsupported_workflow_request",
                message="Workflow requests accept messages, user, metadata.tags and non-streaming invocation only",
            )
        )
    store: Final = require_store(workflows)
    accepted: Final = await admit(store, auth, invocation, request.headers.get("Idempotency-Key"))
    return await wait_for_run(store, accepted)

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import JsonValue, TypeAdapter

from litellm.models.workflow_definition import WorkflowDraft, WorkflowSnapshot
from litellm.models.workflow_execution import RunPage, RunState, StepTrace
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import get_workflow_definition_repository, get_workflow_execution_repository
from litellm.proxy.management_endpoints.workflow_definition_endpoints import router as definitions_router
from litellm.proxy.management_endpoints.workflow_execution_endpoints import invoke_workflow, router
from litellm.proxy.management_endpoints.workflow_management_endpoints import router as tracking_router
from litellm.repositories.workflow_definition_repository import WorkflowDefinitionRepository
from litellm.repositories.workflow_execution_repository import WorkflowExecutionRepository
from tests.unit.proxy.workflow_definition_rows import DefinitionRows, bundle, caller
from tests.unit.proxy.workflows.test_runner import two_nodes
from tests.unit.repositories.test_workflow_execution_repository import NOW, ExecutionRows, pending_run

_MESSAGES: Final = ({"role": "user", "content": "plan"},)
_ACCEPTED: Final = TypeAdapter(dict[str, str])


@pytest.fixture(autouse=True)
def enable_workflows(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"enable_workflows": True})


def virtual_key(user: str | None = "author", team: str = "team", key: str = "owner-key") -> UserAPIKeyAuth:
    return caller(user, team).model_copy(update={"token": key, "via_virtual_key": True})


def client_for(definitions: DefinitionRows, executions: ExecutionRows, identity: UserAPIKeyAuth) -> TestClient:
    executions.litellm_workflowsnapshot = definitions.litellm_workflowsnapshot
    app: Final = FastAPI()
    app.include_router(tracking_router)
    app.include_router(definitions_router)
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: identity
    app.dependency_overrides[get_workflow_definition_repository] = lambda: WorkflowDefinitionRepository(
        definitions.database
    )
    app.dependency_overrides[get_workflow_execution_repository] = lambda: WorkflowExecutionRepository(executions)
    return TestClient(app)


def register(client: TestClient) -> WorkflowDraft:
    response: Final = client.post(
        "/v1/workflows", json={"bundle": bundle().model_dump(mode="json", by_alias=True, exclude_unset=True)}
    )
    assert response.status_code == 201, response.text
    return WorkflowDraft.model_validate(response.json())


def test_paid_test_returns_202_and_persists_pending_admission_without_executing_inline() -> None:
    definitions: Final = DefinitionRows()
    executions: Final = ExecutionRows()
    client: Final = client_for(definitions, executions, virtual_key())
    draft: Final = register(client)
    response: Final = client.post(
        f"/v1/workflows/{draft.workflow_id}/test",
        json={"messages": _MESSAGES, "authorize_billing": True},
        headers={"If-Match": draft.etag},
    )
    assert response.status_code == 202, response.text
    assert len(executions.runs) == 1
    admitted: Final = executions.runs[0]
    assert response.json() == {
        "run_id": admitted.run_id,
        "snapshot_id": admitted.snapshot_id,
        "trace_url": f"/v1/workflows/runs/{admitted.run_id}/steps",
    }
    assert response.headers["x-litellm-workflow-run-id"] == admitted.run_id
    assert (admitted.status, admitted.output, admitted.lease_token) == (RunState.PENDING, None, None)
    assert admitted.input.messages == _MESSAGES and admitted.input.draft
    assert (admitted.created_by, admitted.team_id) == ("owner-key", "team")
    assert executions.events == ()
    trace: Final = client.get(f"/v1/workflows/runs/{admitted.run_id}/steps")
    assert trace.status_code == 200, trace.text
    assert trace.json() == {
        "run_id": admitted.run_id,
        "snapshot_id": admitted.snapshot_id,
        "state": "pending",
        "cancel_requested": False,
        "steps": [],
        "error": None,
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_execution_routes_require_explicit_opt_in(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    from litellm.proxy import proxy_server

    definitions: Final = DefinitionRows()
    executions: Final = ExecutionRows()
    client: Final = client_for(definitions, executions, virtual_key())
    draft: Final = register(client)
    monkeypatch.setattr(proxy_server, "general_settings", {"enable_workflows": True} if enabled else {})
    response: Final = client.post(
        f"/v1/workflows/{draft.workflow_id}/test",
        json={"messages": _MESSAGES, "authorize_billing": True},
        headers={"If-Match": draft.etag},
    )
    assert response.status_code == (202 if enabled else 404), response.text
    assert len(executions.runs) == int(enabled)
    history: Final = client.get("/v1/workflows/runs", params={"workflow_id": draft.workflow_id})
    assert history.status_code == (200 if enabled else 404), history.text


def test_idempotent_test_after_draft_edit_returns_original_snapshot_and_conflict_for_changed_input() -> None:
    definitions: Final = DefinitionRows()
    executions: Final = ExecutionRows()
    client: Final = client_for(definitions, executions, virtual_key())
    draft: Final = register(client)
    path: Final = f"/v1/workflows/{draft.workflow_id}"
    headers: Final = {"If-Match": draft.etag, "Idempotency-Key": "retry-1"}
    body: Final = {"messages": _MESSAGES, "authorize_billing": True}
    admitted: Final = client.post(f"{path}/test", json=body, headers=headers)
    assert admitted.status_code == 202, admitted.text
    edited: Final = client.put(
        path,
        json={"bundle": bundle(model="changed-model").model_dump(mode="json", by_alias=True, exclude_unset=True)},
        headers={"If-Match": draft.etag},
    )
    assert edited.status_code == 200, edited.text
    replay: Final = client.post(f"{path}/test", json=body, headers=headers)
    assert replay.status_code == 202 and replay.json() == admitted.json(), replay.text
    frozen: Final = client.get(f"/v1/workflows/runs/{admitted.json()['run_id']}/definition")
    assert frozen.status_code == 200, frozen.text
    assert WorkflowSnapshot.model_validate(frozen.json()).bundle == draft.bundle
    assert len(definitions.litellm_workflowsnapshot.rows) == 1 and len(executions.runs) == 1
    conflict: Final = client.post(
        f"{path}/test", json={**body, "messages": [{"role": "user", "content": "different"}]}, headers=headers
    )
    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["detail"]["error"]["code"] == "idempotency_conflict"
    assert len(executions.runs) == 1 and len(definitions.litellm_workflowsnapshot.rows) == 1


@pytest.mark.parametrize("billing_authorized", [False, True])
def test_billing_consent_and_virtual_key_are_required_before_creating_snapshot_or_run(billing_authorized: bool) -> None:
    definitions: Final = DefinitionRows()
    executions: Final = ExecutionRows()
    author: Final = client_for(definitions, executions, virtual_key())
    draft: Final = register(author)
    session: Final = virtual_key().model_copy(update={"is_session_token": True})
    client: Final = client_for(definitions, executions, session if billing_authorized else virtual_key())
    response: Final = client.post(
        f"/v1/workflows/{draft.workflow_id}/test",
        json={"messages": _MESSAGES, "authorize_billing": billing_authorized},
        headers={"If-Match": draft.etag},
    )
    assert response.status_code == (403 if billing_authorized else 400), response.text
    assert response.json()["detail"]["error"]["code"] == (
        "forbidden" if billing_authorized else "billing_authorization_required"
    )
    assert executions.runs == () and definitions.litellm_workflowsnapshot.rows == ()


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "invalid", "content": "plan"}],
        [{"role": "user", "content": "x" * (1024 * 1024)}],
    ],
)
def test_invalid_test_input_does_not_create_snapshot_or_run(messages: list[dict[str, str]]) -> None:
    definitions: Final = DefinitionRows()
    executions: Final = ExecutionRows()
    client: Final = client_for(definitions, executions, virtual_key())
    draft: Final = register(client)
    response: Final = client.post(
        f"/v1/workflows/{draft.workflow_id}/test",
        json={"messages": messages, "authorize_billing": True},
        headers={"If-Match": draft.etag},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["error"]["code"] == "invalid_input"
    assert definitions.litellm_workflowsnapshot.rows == ()
    assert executions.runs == ()


def test_full_team_queue_does_not_create_draft_test_snapshots() -> None:
    definitions: Final = DefinitionRows()
    existing: Final = tuple(pending_run(f"run-{index}") for index in range(64))
    executions: Final = ExecutionRows(existing)
    client: Final = client_for(definitions, executions, virtual_key())
    draft: Final = register(client)
    for _ in range(2):
        response: Final = client.post(
            f"/v1/workflows/{draft.workflow_id}/test",
            json={"messages": _MESSAGES, "authorize_billing": True},
            headers={"If-Match": draft.etag},
        )
        assert response.status_code == 429, response.text
        assert response.json()["detail"]["error"]["code"] == "capacity_exceeded"
    assert definitions.litellm_workflowsnapshot.rows == ()
    assert executions.runs == existing


@pytest.mark.parametrize("identity", [virtual_key("other", key="other-key"), virtual_key("outsider", "other-team")])
def test_foreign_run_denies_trace_definition_and_cancellation_without_side_effects(identity: UserAPIKeyAuth) -> None:
    definitions: Final = DefinitionRows()
    executions: Final = ExecutionRows()
    owner: Final = client_for(definitions, executions, virtual_key())
    draft: Final = register(owner)
    accepted: Final = owner.post(
        f"/v1/workflows/{draft.workflow_id}/test",
        json={"messages": _MESSAGES, "authorize_billing": True},
        headers={"If-Match": draft.etag},
    )
    assert accepted.status_code == 202, accepted.text
    path: Final = f"/v1/workflows/runs/{accepted.json()['run_id']}"
    foreign: Final = client_for(definitions, executions, identity)
    assert foreign.get(f"{path}/steps").status_code == 404
    assert foreign.get(f"{path}/definition").status_code == 404
    denied: Final = foreign.post(f"{path}/cancel")
    assert denied.status_code == 404, denied.text
    assert not executions.runs[0].cancel_requested
    assert owner.get(f"{path}/steps").status_code == 200
    assert owner.post(f"{path}/cancel").status_code == 202
    assert executions.runs[0].cancel_requested


@pytest.mark.parametrize(
    ("identity", "expected_status"),
    [
        (virtual_key("reviewer", "other-team", "other-team-admin-key"), 404),
        (virtual_key("reviewer", "team", "team-admin-key"), 200),
        (virtual_key("reviewer", "other-team").model_copy(update={"is_session_token": True}), 200),
        (
            virtual_key("proxy-viewer", "other-team").model_copy(
                update={"user_role": LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY}
            ),
            200,
        ),
    ],
)
def test_run_admin_reads_honor_virtual_key_team_scope(identity: UserAPIKeyAuth, expected_status: int) -> None:
    definitions: Final = DefinitionRows()
    executions: Final = ExecutionRows()
    owner: Final = client_for(definitions, executions, virtual_key())
    draft: Final = register(owner)
    accepted: Final = owner.post(
        f"/v1/workflows/{draft.workflow_id}/test",
        json={"messages": _MESSAGES, "authorize_billing": True},
        headers={"If-Match": draft.etag},
    )
    assert accepted.status_code == 202, accepted.text
    path: Final = f"/v1/workflows/runs/{accepted.json()['run_id']}"
    reader: Final = client_for(definitions, executions, identity)
    for suffix in ("steps", "definition"):
        response: Final = reader.get(f"{path}/{suffix}")
        assert response.status_code == expected_status, response.text


def test_history_caps_page_size_and_scopes_ordinary_key_but_allows_team_admin() -> None:
    definitions: Final = DefinitionRows()
    executions: Final = ExecutionRows()
    owner: Final = client_for(definitions, executions, virtual_key())
    draft: Final = register(owner)
    executions.runs = (
        *(pending_run(f"run-{index:03}", workflow_id=draft.workflow_id) for index in range(101)),
        pending_run("foreign-key", workflow_id=draft.workflow_id, key_hash="other-key"),
        pending_run("foreign-team", workflow_id=draft.workflow_id, team_id="other-team"),
    )
    response: Final = owner.get("/v1/workflows/runs", params={"workflow_id": draft.workflow_id, "limit": 250})
    assert response.status_code == 200, response.text
    page: Final = RunPage.model_validate(response.json())
    assert page.count == 100 and len(page.runs) == 100 and page.next_cursor is not None
    assert tuple(run.run_id for run in page.runs) == tuple(f"run-{index:03}" for index in range(100, 0, -1))
    tail_response: Final = owner.get(
        f"/v1/workflows/{draft.workflow_id}/runs", params={"limit": 100, "after": page.next_cursor}
    )
    tail: Final = RunPage.model_validate(tail_response.json())
    assert tuple(run.run_id for run in tail.runs) == ("run-000",)
    assert tail.next_cursor is None
    admin: Final = client_for(definitions, executions, virtual_key("reviewer", key="admin-key"))
    team_response: Final = admin.get("/v1/workflows/runs", params={"workflow_id": draft.workflow_id, "limit": 250})
    assert team_response.status_code == 200, team_response.text
    team_page: Final = RunPage.model_validate(team_response.json())
    team_tail_response: Final = admin.get(
        f"/v1/workflows/{draft.workflow_id}/runs", params={"limit": 100, "after": team_page.next_cursor}
    )
    assert team_tail_response.status_code == 200, team_tail_response.text
    team_tail: Final = RunPage.model_validate(team_tail_response.json())
    assert tuple(run.run_id for run in team_tail.runs) == ("run-000", "foreign-key")
    assert team_tail.next_cursor is None
    outsider: Final = client_for(definitions, executions, virtual_key("outsider", "other-team"))
    denied: Final = outsider.get("/v1/workflows/runs", params={"workflow_id": draft.workflow_id, "team_id": "team"})
    assert denied.status_code == 404, denied.text
    assert owner.get(f"/v1/workflows/{draft.workflow_id}/runs", params={"limit": 101}).status_code == 422


@pytest.mark.asyncio
async def test_application_key_can_cancel_its_own_published_run() -> None:
    rows: Final = DefinitionRows()
    definitions: Final = WorkflowDefinitionRepository(rows.database)
    draft: Final = await definitions.create(bundle(), "team", "author")
    assert isinstance(draft, WorkflowDraft)
    snapshot: Final = await definitions.publish(draft.workflow_id, "team", draft.etag, draft.digest, "reviewer")
    assert isinstance(snapshot, WorkflowSnapshot)
    executions: Final = ExecutionRows()
    application: Final = virtual_key(None, key="application-key")
    accepted: Final = await invoke_workflow(
        Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [(b"prefer", b"respond-async")],
            }
        ),
        Response(),
        application,
        {"model": "workflow/example", "messages": _MESSAGES},
        definitions,
        WorkflowExecutionRepository(executions),
    )
    assert isinstance(accepted, JSONResponse) and accepted.status_code == 202
    run_id: Final = executions.runs[0].run_id
    foreign: Final = client_for(rows, executions, virtual_key(None, key="different-application-key"))
    assert foreign.post(f"/v1/workflows/runs/{run_id}/cancel").status_code == 404
    client: Final = client_for(rows, executions, application)
    response: Final = client.post(f"/v1/workflows/runs/{run_id}/cancel")
    assert response.status_code == 202, response.text
    assert executions.runs[0].cancel_requested


@pytest.mark.asyncio
async def test_prefer_respond_async_returns_published_run_id_without_waiting_for_execution() -> None:
    definitions: Final = WorkflowDefinitionRepository(DefinitionRows().database)
    draft: Final = await definitions.create(bundle(), "team", "author")
    assert isinstance(draft, WorkflowDraft)
    snapshot: Final = await definitions.publish(draft.workflow_id, "team", draft.etag, draft.digest, "reviewer")
    assert isinstance(snapshot, WorkflowSnapshot)
    database: Final = ExecutionRows()
    request: Final = Request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": [(b"prefer", b"respond-async")]}
    )
    response: Final = await invoke_workflow(
        request,
        Response(),
        virtual_key(),
        {"model": "workflow/example", "messages": _MESSAGES},
        definitions,
        WorkflowExecutionRepository(database),
    )
    assert isinstance(response, JSONResponse) and response.status_code == 202
    assert len(database.runs) == 1
    run: Final = database.runs[0]
    assert _ACCEPTED.validate_json(bytes(response.body)) == {
        "run_id": run.run_id,
        "snapshot_id": snapshot.snapshot_id,
        "trace_url": f"/v1/workflows/runs/{run.run_id}/steps",
    }
    assert response.headers["x-litellm-workflow-run-id"] == run.run_id
    assert run.status is RunState.PENDING and not run.input.draft and run.output is None
    assert database.events == ()


@pytest.mark.asyncio
async def test_sdk_retry_after_republish_reuses_pinned_run_and_sums_node_usage() -> None:
    definitions: Final = WorkflowDefinitionRepository(DefinitionRows().database)
    original: Final = two_nodes()
    draft: Final = await definitions.create(original, "team", "author")
    assert isinstance(draft, WorkflowDraft)
    snapshot: Final = await definitions.publish(draft.workflow_id, "team", draft.etag, draft.digest, "reviewer")
    assert isinstance(snapshot, WorkflowSnapshot)
    database: Final = ExecutionRows()
    executions: Final = WorkflowExecutionRepository(database)
    body: Final = {"model": "workflow/example", "messages": _MESSAGES}
    headers: Final = [(b"idempotency-key", b"published-retry")]
    scope: Final = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}
    accepted: Final = await invoke_workflow(
        Request({**scope, "headers": [*headers, (b"prefer", b"respond-async")]}),
        Response(),
        virtual_key(),
        body,
        definitions,
        executions,
    )
    assert isinstance(accepted, JSONResponse) and accepted.status_code == 202
    run: Final = database.runs[0]
    assert await executions.claim(run.run_id, "worker", datetime.now(timezone.utc) + timedelta(seconds=15)) is not None
    for node_id, prompt, completion, cost in (("node_1", 7, 2, 0.1), ("node_2", 11, 3, 0.2)):
        assert await executions.checkpoint(
            run.run_id,
            "worker",
            StepTrace(
                step_id=node_id,
                run_id=run.run_id,
                node_id=node_id,
                state=RunState.COMPLETED,
                input=[dict(message) for message in run.input.messages],
                output={"answer": "approved"},
                attempts=1,
                deadline=run.input.deadline,
                started_at=NOW,
                finished_at=NOW,
                prompt_tokens=prompt,
                completion_tokens=completion,
                response_cost=cost,
                request_id=f"receipt-{node_id}",
            ),
        )
    assert await executions.finish(run.run_id, "worker", RunState.COMPLETED, {"answer": "approved"}, None)
    changed: Final = await definitions.replace(
        draft.workflow_id,
        "team",
        draft.etag,
        original.model_copy(update={"timeout_seconds": 120}),
        "author",
    )
    assert isinstance(changed, WorkflowDraft)
    published: Final = await definitions.publish(changed.workflow_id, "team", changed.etag, changed.digest, "reviewer")
    assert isinstance(published, WorkflowSnapshot) and published.snapshot_id != snapshot.snapshot_id
    response: Final = Response()
    result: Final = await invoke_workflow(
        Request({**scope, "headers": headers}),
        response,
        virtual_key(),
        body,
        definitions,
        executions,
    )
    assert result == {
        "id": f"chatcmpl-workflow-{run.run_id}",
        "object": "chat.completion",
        "created": int(run.created_at.timestamp()),
        "model": "workflow/example",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": '{"answer": "approved"}'}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 18, "completion_tokens": 5, "total_tokens": 23},
    }
    assert response.headers["x-litellm-workflow-run-id"] == run.run_id
    assert len(database.runs) == 1 and database.runs[0].snapshot_id == snapshot.snapshot_id


@pytest.mark.parametrize(
    ("metadata", "error"),
    [
        ({"error": "Pinned input schema rejected messages"}, "Pinned input schema rejected messages"),
        ({"error": {"internal": "malformed metadata"}}, None),
    ],
)
def test_trace_exposes_run_level_failure_without_steps_and_only_string_errors(
    metadata: Mapping[str, JsonValue],
    error: str | None,
) -> None:
    run: Final = pending_run().model_copy(update={"status": RunState.FAILED, "metadata": metadata})
    response: Final = client_for(DefinitionRows(), ExecutionRows((run,)), virtual_key()).get(
        f"/v1/workflows/runs/{run.run_id}/steps"
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "run_id": run.run_id,
        "snapshot_id": run.snapshot_id,
        "state": "failed",
        "cancel_requested": False,
        "steps": [],
        "error": error,
    }

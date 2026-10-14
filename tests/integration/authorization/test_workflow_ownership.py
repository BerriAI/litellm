import hashlib
from typing import Final

import httpx
from integration._support.client import JSON_OBJECT, Gateway, object_value, string_value
from integration._support.database import read_rows, write_rows
from pydantic import BaseModel, JsonValue

_WORKFLOW_ROUTE: Final = ["/v1/workflows"]


class WorkflowRunResponse(BaseModel):
    run_id: str
    session_id: str
    workflow_type: str
    status: str
    created_by: str | None = None
    input: JsonValue | None = None


class WorkflowEventResponse(BaseModel):
    event_id: str
    run_id: str
    event_type: str
    step_name: str
    sequence_number: int


class WorkflowMessageResponse(BaseModel):
    message_id: str
    run_id: str
    role: str
    content: str
    sequence_number: int


def _delete_workflow_rows(run_id: str) -> None:
    write_rows('DELETE FROM "LiteLLM_WorkflowEvent" WHERE run_id = %s', (run_id,))
    write_rows('DELETE FROM "LiteLLM_WorkflowMessage" WHERE run_id = %s', (run_id,))
    write_rows('DELETE FROM "LiteLLM_WorkflowRun" WHERE run_id = %s', (run_id,))


def _run_rows(run_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT run_id, workflow_type, status, created_by, input FROM "LiteLLM_WorkflowRun" WHERE run_id = %s',
        (run_id,),
    )


def _route_refusal(response: httpx.Response, user_id: str) -> None:
    masked_user_id: Final = f"{user_id[:6]}{'*' * (len(user_id) - 8)}{user_id[-2:]}"
    assert response.status_code == 401, response.text
    assert JSON_OBJECT.validate_json(response.content) == {
        "error": {
            "message": (
                "Authentication Error, Only proxy admin can be used to generate, delete, update info for new "
                f"keys/users/teams. Route=/v1/workflows/runs. Your role=internal_user. Your user_id={masked_user_id}"
            ),
            "type": "auth_error",
            "param": "None",
            "code": "401",
        }
    }, response.text


def test_workflow_runs_events_and_messages_are_key_owned(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        default_user: Final = scenario.user(user_role="internal_user")
        default_key: Final = scenario.key(user_id=default_user)
        default_access: Final = gateway.request(
            "POST",
            "/v1/workflows/runs",
            {"workflow_type": f"workflow-{default_user}", "input": {"source": "default"}},
            key=default_key,
        )
        _route_refusal(default_access, default_user)

        user_one: Final = scenario.user(user_role="internal_user")
        user_two: Final = scenario.user(user_role="internal_user")
        key_one: Final = scenario.key(user_id=user_one, allowed_routes=_WORKFLOW_ROUTE)
        key_two: Final = scenario.key(user_id=user_two, allowed_routes=_WORKFLOW_ROUTE)
        workflow_type: Final = f"workflow-{user_one}-{user_two}"

        first_create: Final = gateway.request(
            "POST",
            "/v1/workflows/runs",
            {"workflow_type": workflow_type, "input": {"owner": "one"}, "metadata": {"case": "ownership"}},
            key=key_one,
        )
        second_create: Final = gateway.request(
            "POST",
            "/v1/workflows/runs",
            {"workflow_type": workflow_type, "input": {"owner": "two"}},
            key=key_two,
        )
        assert first_create.status_code == 200, first_create.text
        assert second_create.status_code == 200, second_create.text
        first_run: Final = WorkflowRunResponse.model_validate_json(first_create.content)
        second_run: Final = WorkflowRunResponse.model_validate_json(second_create.content)
        scenario.cleanups.callback(_delete_workflow_rows, first_run.run_id)
        scenario.cleanups.callback(_delete_workflow_rows, second_run.run_id)
        first_digest: Final = hashlib.sha256(key_one.encode()).hexdigest()
        second_digest: Final = hashlib.sha256(key_two.encode()).hexdigest()
        assert (first_run.status, first_run.created_by, first_run.input) == (
            "pending",
            first_digest,
            {"owner": "one"},
        ), first_create.text
        assert (second_run.status, second_run.created_by, second_run.input) == (
            "pending",
            second_digest,
            {"owner": "two"},
        ), second_create.text
        assert _run_rows(first_run.run_id) == [
            {
                "run_id": first_run.run_id,
                "workflow_type": workflow_type,
                "status": "pending",
                "created_by": first_digest,
                "input": {"owner": "one"},
            }
        ]
        assert _run_rows(second_run.run_id) == [
            {
                "run_id": second_run.run_id,
                "workflow_type": workflow_type,
                "status": "pending",
                "created_by": second_digest,
                "input": {"owner": "two"},
            }
        ]

        cross_key_operations: Final = (
            gateway.request("GET", f"/v1/workflows/runs/{first_run.run_id}", key=key_two),
            gateway.request(
                "PATCH",
                f"/v1/workflows/runs/{first_run.run_id}",
                {"status": "completed"},
                key=key_two,
            ),
            gateway.request(
                "POST",
                f"/v1/workflows/runs/{first_run.run_id}/events",
                {"event_type": "step.started", "step_name": "cross-key"},
                key=key_two,
            ),
            gateway.request(
                "POST",
                f"/v1/workflows/runs/{first_run.run_id}/messages",
                {"role": "user", "content": "cross-key message"},
                key=key_two,
            ),
            gateway.request("GET", f"/v1/workflows/runs/{first_run.run_id}/events", key=key_two),
            gateway.request("GET", f"/v1/workflows/runs/{first_run.run_id}/messages", key=key_two),
        )
        assert tuple((response.status_code, response.text) for response in cross_key_operations) == (
            (404, f"""{{"detail":"Run '{first_run.run_id}' not found"}}"""),
        ) * len(cross_key_operations), tuple(response.text for response in cross_key_operations)
        assert _run_rows(first_run.run_id)[0]["status"] == "pending"
        assert (
            read_rows(
                'SELECT event_id FROM "LiteLLM_WorkflowEvent" WHERE run_id = %s',
                (first_run.run_id,),
            )
            == []
        )
        assert (
            read_rows(
                'SELECT message_id FROM "LiteLLM_WorkflowMessage" WHERE run_id = %s',
                (first_run.run_id,),
            )
            == []
        )

        first_list: Final = gateway.request(
            "GET",
            "/v1/workflows/runs",
            params={"workflow_type": workflow_type},
            key=key_one,
        )
        second_list: Final = gateway.request(
            "GET",
            "/v1/workflows/runs",
            params={"workflow_type": workflow_type},
            key=key_two,
        )
        assert first_list.status_code == 200, first_list.text
        assert second_list.status_code == 200, second_list.text
        assert tuple(string_value(object_value(run)["run_id"]) for run in object_value(first_list.json())["runs"]) == (
            first_run.run_id,
        ), first_list.text
        assert tuple(string_value(object_value(run)["run_id"]) for run in object_value(second_list.json())["runs"]) == (
            second_run.run_id,
        ), second_list.text

        owner_update: Final = gateway.request(
            "PATCH",
            f"/v1/workflows/runs/{first_run.run_id}",
            {"status": "completed"},
            key=key_one,
        )
        assert owner_update.status_code == 200, owner_update.text
        assert WorkflowRunResponse.model_validate_json(owner_update.content).status == "completed", owner_update.text
        owner_event: Final = gateway.request(
            "POST",
            f"/v1/workflows/runs/{first_run.run_id}/events",
            {"event_type": "step.started", "step_name": "owner-step", "data": {"attempt": 1}},
            key=key_one,
        )
        assert owner_event.status_code == 200, owner_event.text
        assert WorkflowEventResponse.model_validate_json(owner_event.content).sequence_number == 0, owner_event.text
        owner_message: Final = gateway.request(
            "POST",
            f"/v1/workflows/runs/{first_run.run_id}/messages",
            {"role": "assistant", "content": "owner message", "session_id": "synthetic-session"},
            key=key_one,
        )
        assert owner_message.status_code == 200, owner_message.text
        assert WorkflowMessageResponse.model_validate_json(owner_message.content).content == "owner message", (
            owner_message.text
        )
        assert _run_rows(first_run.run_id)[0]["status"] == "running"

        owner_events: Final = gateway.request(
            "GET",
            f"/v1/workflows/runs/{first_run.run_id}/events",
            key=key_one,
        )
        owner_messages: Final = gateway.request(
            "GET",
            f"/v1/workflows/runs/{first_run.run_id}/messages",
            key=key_one,
        )
        assert owner_events.status_code == 200, owner_events.text
        assert owner_messages.status_code == 200, owner_messages.text
        assert object_value(owner_events.json())["count"] == 1, owner_events.text
        assert object_value(owner_messages.json())["count"] == 1, owner_messages.text

        admin_list: Final = gateway.request(
            "GET",
            "/v1/workflows/runs",
            params={"workflow_type": workflow_type},
        )
        assert admin_list.status_code == 200, admin_list.text
        admin_runs: Final = tuple(
            WorkflowRunResponse.model_validate(run) for run in object_value(admin_list.json())["runs"]
        )
        assert frozenset(run.run_id for run in admin_runs) == frozenset((first_run.run_id, second_run.run_id)), (
            admin_list.text
        )
        for run_id in (first_run.run_id, second_run.run_id):
            visible_to_admin: Final = gateway.request("GET", f"/v1/workflows/runs/{run_id}")
            assert visible_to_admin.status_code == 200, visible_to_admin.text
            assert WorkflowRunResponse.model_validate_json(visible_to_admin.content).run_id == run_id, (
                visible_to_admin.text
            )
        admin_update: Final = gateway.request(
            "PATCH",
            f"/v1/workflows/runs/{second_run.run_id}",
            {"status": "completed"},
        )
        assert admin_update.status_code == 200, admin_update.text
        assert _run_rows(second_run.run_id)[0]["status"] == "completed"

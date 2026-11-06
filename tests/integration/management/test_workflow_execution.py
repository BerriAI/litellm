import hashlib
import json
import threading
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_SOURCE: Final = (
    "def prepare_followup(text: str) -> list[dict[str, str]]:\n"
    '    return [{"role": "user", "content": "Review this draft: " + text}]\n'
)
_MESSAGES: Final[Mapping[str, JsonValue]] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"role": {"type": "string", "enum": ["user"]}, "content": {"type": "string"}},
        "required": ["role", "content"],
        "additionalProperties": False,
    },
}
_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class Draft(BaseModel):
    model_config = ConfigDict(frozen=True)
    workflow_id: str
    etag: str
    digest: str


class Step(BaseModel):
    model_config = ConfigDict(frozen=True)
    step_id: str
    state: str
    input: JsonValue
    output: JsonValue
    attempts: int
    response_cost: float | None


class Trace(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: str
    snapshot_id: str
    state: str
    steps: tuple[Step, ...]


@dataclass(frozen=True, slots=True)
class Actors:
    author: str
    publisher: str
    caller: str


@pytest.fixture(scope="module")
def workflow_gateway(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with owned_proxy(
        session_gateway,
        tmp_path_factory.mktemp("workflows"),
        {},
        workers=2,
        config=Path(__file__).with_name("workflow_proxy_config.yaml"),
    ) as gateway:
        yield gateway


def _actors(scenario: Scenario, model: str) -> Actors:
    author: Final = scenario.user(user_role="internal_user")
    publisher: Final = scenario.user(user_role="internal_user")
    team: Final = scenario.team(
        models=[model, "workflow/email-reply"],
        members_with_roles=[{"role": "admin", "user_id": author}, {"role": "admin", "user_id": publisher}],
    )
    return Actors(
        scenario.key(team_id=team, user_id=author, models=[model]),
        scenario.key(team_id=team, user_id=publisher, models=[model]),
        scenario.key(team_id=team, user_id=author, models=["workflow/email-reply"]),
    )


def _bundle(model: str, *, structured: bool = False, timeout: float = 60) -> Mapping[str, JsonValue]:
    return {
        "contract_version": "1",
        "name": "email-reply",
        "timeout_seconds": timeout,
        "input_schema": dict(_MESSAGES),
        "nodes": [
            {
                "id": "draft",
                "type": "chat_completion",
                "model": model,
                "messages": {"from": "$input", "output": "messages"},
                "timeout_seconds": timeout,
            },
            {
                "id": "prepare",
                "type": "python",
                "program": {
                    "contract_version": "1",
                    "source": _SOURCE,
                    "function_name": "prepare_followup",
                    "parameter_names": ["text"],
                    "source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
                    "input_schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                    "output_schema": dict(_MESSAGES),
                },
                "inputs": {"text": {"from": "draft", "output": "result"}},
            },
            {
                "id": "review",
                "type": "chat_completion",
                "model": model,
                "messages": {"from": "prepare", "output": "result"},
                **(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "Review",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {"approved": {"type": "boolean"}},
                                    "required": ["approved"],
                                    "additionalProperties": False,
                                },
                            },
                        }
                    }
                    if structured
                    else {}
                ),
            },
        ],
        "output": {"from": "review", "output": "result"},
    }


def _completion(request: Request) -> Reply:
    body: Final = _OBJECT.validate_json(request.body)
    assert request.target == "/v1/chat/completions", request.target
    assert body["messages"] in (
        [{"role": "user", "content": "Help me update my account email."}],
        [{"role": "user", "content": "Review this draft: synthetic draft"}],
    ), body
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-controlled-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"approved":true}' if "response_format" in body else "synthetic draft",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        ).encode()
    )


def _register(gateway: Gateway, actors: Actors, model: str, *, structured: bool = False, timeout: float = 60) -> Draft:
    response: Final = gateway.request(
        "POST",
        "/v1/workflows",
        {"bundle": dict(_bundle(model, structured=structured, timeout=timeout))},
        key=actors.author,
    )
    assert response.status_code == 201, response.text
    return Draft.model_validate_json(response.content)


def _publish(gateway: Gateway, actors: Actors, draft: Draft) -> None:
    response: Final = gateway.request(
        "POST",
        f"/v1/workflows/{draft.workflow_id}/publish",
        {"digest": draft.digest},
        key=actors.publisher,
        headers={"If-Match": draft.etag},
    )
    assert response.status_code == 200, response.text


def _request() -> Mapping[str, JsonValue]:
    return {
        "model": "workflow/email-reply",
        "messages": [{"role": "user", "content": "Help me update my account email."}],
    }


def _trace(gateway: Gateway, actors: Actors, run_id: str) -> Trace:
    response: Final = gateway.request("GET", f"/v1/workflows/runs/{run_id}/steps", key=actors.caller)
    assert response.status_code == 200, response.text
    return Trace.model_validate_json(response.content)


def test_draft_execution_requires_its_owning_team_key_even_for_multiteam_members(workflow_gateway: Gateway) -> None:
    with workflow_gateway.scenario() as scenario, wire_server(_completion) as wire:
        model: Final = scenario.model(model="openai/gpt-6-sol", api_base=wire.url + "/v1")
        author: Final = scenario.user(user_role="internal_user")
        members: Final = [{"role": "admin", "user_id": author}]
        owning_team: Final = scenario.team(models=[model], members_with_roles=members)
        other_team: Final = scenario.team(models=[model], members_with_roles=members)
        owning_key: Final = scenario.key(team_id=owning_team, user_id=author, models=[model])
        other_key: Final = scenario.key(team_id=other_team, user_id=author, models=[model])
        actors: Final = Actors(owning_key, owning_key, owning_key)
        draft: Final = _register(workflow_gateway, actors, model)
        refused: Final = workflow_gateway.request(
            "POST",
            f"/v1/workflows/{draft.workflow_id}/test",
            {"messages": _request()["messages"], "authorize_billing": True},
            key=other_key,
            headers={"If-Match": draft.etag},
        )
        assert refused.status_code == 403, refused.text
        assert (
            workflow_gateway.rows(
                'SELECT run_id FROM "LiteLLM_WorkflowExecution" WHERE principal=%s',
                (hashlib.sha256(other_key.encode()).hexdigest(),),
            )
            == []
        )
        assert wire.drain() == ()
        accepted: Final = workflow_gateway.request(
            "POST",
            f"/v1/workflows/{draft.workflow_id}/test",
            {"messages": _request()["messages"], "authorize_billing": True},
            key=owning_key,
            headers={"If-Match": draft.etag},
        )
        assert accepted.status_code == 202, accepted.text
        run_id: Final = TypeAdapter(str).validate_python(_OBJECT.validate_json(accepted.content)["run_id"])
        trace: Final = eventually(
            lambda: _trace(workflow_gateway, actors, run_id),
            lambda result: result.state == "completed",
            seconds=70,
        )
        assert tuple(step.state for step in trace.steps) == ("completed",) * 3


def test_etags_and_independent_publication_do_not_execute_drafts(workflow_gateway: Gateway) -> None:
    with workflow_gateway.scenario() as scenario, wire_server(_completion) as wire:
        model: Final = scenario.model(model="openai/gpt-6-sol", api_base=wire.url + "/v1")
        actors: Final = _actors(scenario, model)
        draft: Final = _register(workflow_gateway, actors, model)
        validation: Final = workflow_gateway.request(
            "POST", f"/v1/workflows/{draft.workflow_id}/validate", key=actors.author, headers={"If-Match": draft.etag}
        )
        assert validation.status_code == 200, validation.text
        self_publication: Final = workflow_gateway.request(
            "POST",
            f"/v1/workflows/{draft.workflow_id}/publish",
            {"digest": draft.digest},
            key=actors.author,
            headers={"If-Match": draft.etag},
        )
        assert self_publication.status_code == 403, self_publication.text
        for method, suffix, body in (
            ("PUT", "", {"bundle": dict(_bundle(model))}),
            ("POST", "/publish", {"digest": draft.digest}),
        ):
            stale = workflow_gateway.request(
                method,
                f"/v1/workflows/{draft.workflow_id}{suffix}",
                body,
                key=actors.publisher,
                headers={"If-Match": '"stale"'},
            )
            assert stale.status_code == 412, stale.text
        listed: Final = workflow_gateway.request("GET", "/v1/workflows", key=actors.author)
        assert listed.status_code == 200, listed.text
        assert draft.workflow_id in listed.text
        assert wire.drain() == ()
        _publish(workflow_gateway, actors, draft)
        invoked: Final = workflow_gateway.request("POST", "/v1/chat/completions", _request(), key=actors.caller)
        assert invoked.status_code == 200, invoked.text
        assert len(wire.drain()) == 2


@pytest.mark.parametrize("structured", [False, True])
def test_typed_checkpoints_idempotency_bounded_delegation_and_single_child_billing(
    workflow_gateway: Gateway,
    structured: bool,
) -> None:
    with workflow_gateway.scenario() as scenario, wire_server(_completion) as wire:
        model: Final = scenario.model(
            model="openai/gpt-6-sol", api_base=wire.url + "/v1", input_cost_per_token=0.001, output_cost_per_token=0.002
        )
        actors: Final = _actors(scenario, model)
        draft: Final = _register(workflow_gateway, actors, model, structured=structured)
        _publish(workflow_gateway, actors, draft)
        denied: Final = workflow_gateway.request(
            "POST", "/v1/chat/completions", {**_request(), "model": model}, key=actors.caller
        )
        assert denied.status_code in (401, 403), denied.text
        response: Final = workflow_gateway.request(
            "POST",
            "/v1/chat/completions",
            _request(),
            key=actors.caller,
            headers={"Idempotency-Key": "controlled-retry"},
        )
        assert response.status_code == 200, response.text
        run_id: Final = response.headers["x-litellm-workflow-run-id"]
        retried: Final = workflow_gateway.request(
            "POST",
            "/v1/chat/completions",
            _request(),
            key=actors.caller,
            headers={"Idempotency-Key": "controlled-retry"},
        )
        assert retried.status_code == 200, retried.text
        assert retried.headers["x-litellm-workflow-run-id"] == run_id
        assert retried.json() == response.json()
        trace: Final = _trace(workflow_gateway, actors, run_id)
        assert trace.state == "completed"
        assert tuple(step.state for step in trace.steps) == ("completed",) * 3
        assert trace.steps[1].input == {"text": "synthetic draft"}
        assert trace.steps[1].output == [{"role": "user", "content": "Review this draft: synthetic draft"}]
        assert trace.steps[2].output == ({"approved": True} if structured else "synthetic draft")
        assert tuple(step.attempts for step in trace.steps) == (1, 1, 1)
        assert trace.steps[0].response_cost == pytest.approx(2 * 0.001 + 3 * 0.002)
        checkpoints: Final = workflow_gateway.rows(
            'SELECT checkpoints FROM "LiteLLM_WorkflowExecution" WHERE run_id=%s', (run_id,)
        )
        assert len(TypeAdapter(tuple[object, ...]).validate_python(checkpoints[0]["checkpoints"])) == 3
        assert len(wire.drain()) == 2
        charges: Final = eventually(
            lambda: workflow_gateway.rows(
                'SELECT spend, model FROM "LiteLLM_SpendLogs" WHERE session_id=%s', (run_id,)
            ),
            lambda rows: len(rows) == 2,
            seconds=70,
        )
        assert sum(TypeAdapter(float).validate_python(row["spend"]) for row in charges) == pytest.approx(
            2 * (2 * 0.001 + 3 * 0.002)
        )
        assert all(row["model"] != "workflow/email-reply" for row in charges)
        mutation: Final = workflow_gateway.request("PATCH", f"/v1/workflows/runs/{run_id}", {"status": "failed"})
        assert mutation.status_code == 409, mutation.text


def test_cancellation_stops_the_next_node_without_cancelling_disconnected_work(workflow_gateway: Gateway) -> None:
    entered: Final = threading.Event()
    release: Final = threading.Event()

    def blocked(request: Request) -> Reply:
        entered.set()
        assert release.wait(30), "test did not release the controlled upstream"
        return _completion(request)

    with workflow_gateway.scenario() as scenario, wire_server(blocked) as wire:
        model: Final = scenario.model(model="openai/gpt-6-sol", api_base=wire.url + "/v1")
        actors: Final = _actors(scenario, model)
        draft: Final = _register(workflow_gateway, actors, model)
        _publish(workflow_gateway, actors, draft)
        try:
            with pytest.raises(httpx.ReadTimeout):
                workflow_gateway.client.request(
                    "POST",
                    "/v1/chat/completions",
                    json=_request(),
                    headers={"Idempotency-Key": "detached-run", "Authorization": f"Bearer {actors.caller}"},
                    timeout=0.5,
                )
            assert entered.wait(15), "detached invocation never reached the controlled upstream"
            executions: Final = workflow_gateway.rows(
                'SELECT run_id, state FROM "LiteLLM_WorkflowExecution" WHERE name=%s AND state=%s',
                ("email-reply", "running"),
            )
            assert len(executions) == 1, executions
            run_id: Final = str(executions[0]["run_id"])
            cancelled: Final = workflow_gateway.request(
                "POST", f"/v1/workflows/runs/{run_id}/cancel", key=actors.caller
            )
            assert cancelled.status_code == 202, cancelled.text
            final: Final = eventually(
                lambda: _trace(workflow_gateway, actors, run_id), lambda trace: trace.state == "cancelled", seconds=15
            )
            assert len(final.steps) == 1
        finally:
            release.set()
        assert len(wire.drain()) == 1

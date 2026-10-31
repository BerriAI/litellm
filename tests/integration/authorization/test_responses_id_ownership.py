from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, object_value
from integration._support.process import owned_proxy
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.worker_readiness import wait_for_worker_readiness
from openai.types.responses import ResponseCompletedEvent, ResponseInputParam
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

_MODEL: Final = "gpt-5"
_API_KEY: Final = "synthetic-responses-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROXY_WORKERS: Final = max(1, int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1")))
_ALIASES: Final = ("/v1", "", "/openai/v1")
_DISABLE_HINT: Final = (
    "To disable this security feature, set general_settings::disable_responses_id_security to True "
    "in the config.yaml file."
)
_OTHER_USER_DETAIL: Final = (
    f"Forbidden. The response id is not associated with the user, who this key belongs to. {_DISABLE_HINT}"
)
_OTHER_TEAM_DETAIL: Final = (
    f"Forbidden. The response id is not associated with the team, who this key belongs to. {_DISABLE_HINT}"
)
_UNMANAGED_DETAIL: Final = (
    "Forbidden. This response id was not issued by this proxy, so the proxy cannot tell who owns it. "
    "To let keys address responses this proxy did not issue, set "
    "general_settings::allow_unmanaged_response_ids to True in the config.yaml file."
)


class _DeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["response"]
    deleted: bool


def _deployments_in_group(gateway: Gateway, group: str) -> int:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    return sum(object_value(entry).get("model_name") == group for entry in entries)


def _wait_for_workers(gateway: Gateway, group: str, deployments: int) -> None:
    wait_for_worker_readiness(gateway, group, deployments, _PROXY_WORKERS)


def _model_discovery_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {"object": "list", "data": [{"id": _MODEL, "object": "model", "created": 1, "owned_by": "openai"}]}
        ).encode()
    )


def _contract_requests(wire: Wire) -> tuple[Request, ...]:
    requests: Final = wire.drain()
    discovery: Final = tuple(request for request in requests if request.target == "/v1/models")
    assert all(request.method == "GET" and request.body == b"" for request in discovery), requests
    return tuple(request for request in requests if request.target != "/v1/models")


def _response(identity: str, status: str = "completed") -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "response",
            "created_at": 1,
            "status": status,
            "model": _MODEL,
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    ).encode()


def _deployment(scenario: Scenario, wire: Wire) -> str:
    model: Final = scenario.model(
        model=f"openai/{_MODEL}",
        api_key=_API_KEY,
        api_base=f"{wire.url}/v1",
    )
    _wait_for_workers(scenario.gateway, model, 1)
    return model


def _key(scenario: Scenario, team_id: str, user_id: str, model: str) -> str:
    return scenario.key(team_id=team_id, user_id=user_id, models=[model])


def _client(gateway: Gateway, key: str, prefix: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{gateway.client.base_url}{prefix}", api_key=key, max_retries=0)


def _error(response: httpx.Response, detail: str) -> None:
    assert response.status_code == 403, response.text
    assert response.json() == {
        "error": {
            "message": detail,
            "type": "permission_error",
            "param": None,
            "code": "403",
        }
    }, response.text


def _refused(call: Callable[[], object], detail: str) -> None:
    with pytest.raises(openai.PermissionDeniedError) as refusal:
        call()
    _error(refusal.value.response, detail)


def _every_id_route(client: openai.OpenAI, model: str, response_id: str) -> tuple[Callable[[], object], ...]:
    return (
        lambda: client.responses.retrieve(response_id),
        lambda: client.responses.delete(response_id),
        lambda: client.responses.cancel(response_id),
        lambda: client.responses.input_items.list(response_id),
        lambda: client.responses.create(model=model, input="cross-tenant follow-up", previous_response_id=response_id),
    )


def _comparison_body(response_id: str, nested: bool) -> Mapping[str, object]:
    options: Final = {"comparison_response_id": response_id}
    if nested:
        return {
            "prompt_cache_options": {"mode": "explicit", "ttl": "12h"},
            "extra_body": {"prompt_cache_options": options},
        }
    return {"prompt_cache_options": options}


@pytest.mark.parametrize(
    ("prefix", "nested", "stream"),
    (("/v1", False, False), ("", True, False), ("/openai/v1", False, True), ("/v1", True, True)),
)
def test_regression_comparison_response_id_owner_repeated_requests_reach_raw_provider_id(
    gateway: Gateway, prefix: str, nested: bool, stream: bool
) -> None:
    raw_id: Final = f"resp_comparison_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.target == "/v1/models":
            return _model_discovery_reply()
        assert (request.method, request.target) == ("POST", "/v1/responses")
        body: Final = _JSON_OBJECT.validate_json(request.body)
        if body["input"] == "create comparison response":
            assert body == {"model": _MODEL, "input": "create comparison response"}, request.body
            return Reply(body=_response(raw_id))
        assert body == {
            "model": _MODEL,
            "input": "compare owned response",
            "stream": stream,
            "prompt_cache_options": {"comparison_response_id": raw_id},
        }, request.body
        if stream:
            return Reply(
                content_type="text/event-stream",
                chunks=(
                    b"data: "
                    + json.dumps({"type": "response.completed", "response": json.loads(_response(raw_id))}).encode()
                    + b"\n\n",
                    b"data: [DONE]\n\n",
                ),
            )
        return Reply(body=_response(raw_id))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        with _client(gateway, _key(scenario, team, user, model), prefix) as client:
            created: Final = client.responses.create(model=model, input="create comparison response")
            assert created.id != raw_id, created.model_dump_json()
            assert len(_contract_requests(wire)) == 1

            def compare_again() -> None:
                result: Final = client.responses.create(
                    model=model,
                    input="compare owned response",
                    stream=stream,
                    extra_body=_comparison_body(created.id, nested),
                )
                if isinstance(result, openai.Stream):
                    events: Final = tuple(result)
                    assert isinstance(events[-1], ResponseCompletedEvent), events
                    assert events[-1].response.status == "completed", events
                else:
                    assert result.status == "completed", result.model_dump_json()

            for _ in range(2):
                compare_again()
            assert len(_contract_requests(wire)) == 2


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("stranger", ("user", "team"))
@pytest.mark.parametrize("retained", (False, True), ids=("ordinary", "client-internal-field"))
def test_regression_comparison_response_id_other_identity_is_denied_before_dispatch(
    gateway: Gateway, nested: bool, stranger: str, retained: bool
) -> None:
    raw_id: Final = f"resp_comparison_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.target == "/v1/models":
            return _model_discovery_reply()
        return Reply(body=_response(raw_id))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        user_a: Final = scenario.member(team_a)
        user_b: Final = scenario.member(team_a)
        gateway.post("/team/member_add", {"team_id": team_b, "member": {"role": "user", "user_id": user_a}})
        with (
            _client(gateway, _key(scenario, team_a, user_a, model), "/v1") as owner,
            _client(
                gateway,
                _key(
                    scenario, team_a if stranger == "user" else team_b, user_b if stranger == "user" else user_a, model
                ),
                "/v1",
            ) as other,
        ):
            created: Final = owner.responses.create(model=model, input="create private comparison")
            assert created.id != raw_id
            assert len(_contract_requests(wire)) == 1
            _refused(
                lambda: other.responses.create(
                    model=model,
                    input="compare another identity",
                    extra_body={
                        **_comparison_body(created.id, nested),
                        **(
                            {"_litellm_addressed_comparison_response_id": "resp_unmanaged_comparison"}
                            if retained
                            else {}
                        ),
                    },
                ),
                _OTHER_USER_DETAIL if stranger == "user" else _OTHER_TEAM_DETAIL,
            )
            assert _contract_requests(wire) == (), "A cross-identity comparison reached the provider"


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("comparison_id", ("resp_unmanaged_comparison", "resp_!!!not-base64", "cmF3X3ZlbmRvcl9pZA=="))
def test_guard_unmanaged_comparison_response_ids_pass_through_without_opt_in(
    gateway: Gateway, nested: bool, comparison_id: str
) -> None:
    def respond(request: Request) -> Reply:
        if request.target == "/v1/models":
            return _model_discovery_reply()
        assert (request.method, request.target) == ("POST", "/v1/responses")
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": _MODEL,
            "input": "unmanaged comparison",
            "prompt_cache_options": {"comparison_response_id": comparison_id},
        }, request.body
        return Reply(body=_response(f"resp_result_{uuid.uuid4().hex}"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        with _client(gateway, _key(scenario, team, user, model), "/v1") as client:
            compared: Final = client.responses.create(
                model=model, input="unmanaged comparison", extra_body=_comparison_body(comparison_id, nested)
            )
            assert compared.status == "completed", compared.model_dump_json()
            assert len(_contract_requests(wire)) == 1


@pytest.mark.parametrize("effective_owner", (False, True), ids=("guard-raw", "regression-owned"))
def test_comparison_extra_body_ignores_foreign_top_level(gateway: Gateway, effective_owner: bool) -> None:
    foreign_raw_id: Final = f"resp_ignored_{uuid.uuid4().hex}"
    owner_raw_id: Final = f"resp_effective_{uuid.uuid4().hex}"
    unmanaged_id: Final = "resp_raw_effective_comparison"

    def respond(request: Request) -> Reply:
        if request.target == "/v1/models":
            return _model_discovery_reply()
        assert (request.method, request.target) == ("POST", "/v1/responses")
        body: Final = _JSON_OBJECT.validate_json(request.body)
        if body["input"] in ("seed ignored comparison", "seed effective comparison"):
            assert body == {"model": _MODEL, "input": body["input"]}, request.body
            return Reply(body=_response(foreign_raw_id if body["input"] == "seed ignored comparison" else owner_raw_id))
        assert body == {
            "model": _MODEL,
            "input": "compare effective options",
            "prompt_cache_options": {"comparison_response_id": owner_raw_id if effective_owner else unmanaged_id},
        }, request.body
        return Reply(body=_response(owner_raw_id))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team: Final = scenario.team(models=[model])
        owner_user: Final = scenario.member(team)
        foreign_user: Final = scenario.member(team)
        with (
            _client(gateway, _key(scenario, team, owner_user, model), "/v1") as owner,
            _client(gateway, _key(scenario, team, foreign_user, model), "/v1") as foreign,
        ):
            foreign_response: Final = foreign.responses.create(model=model, input="seed ignored comparison")
            assert foreign_response.id != foreign_raw_id
            owner_response: Final = owner.responses.create(model=model, input="seed effective comparison")
            assert owner_response.id != owner_raw_id
            assert len(_contract_requests(wire)) == 2
            compared: Final = owner.responses.create(
                model=model,
                input="compare effective options",
                extra_body={
                    "prompt_cache_options": {
                        "comparison_response_id": foreign_response.id,
                        "mode": "explicit",
                        "ttl": "12h",
                    },
                    "extra_body": {
                        "prompt_cache_options": {
                            "comparison_response_id": owner_response.id if effective_owner else unmanaged_id
                        }
                    },
                },
            )
            assert compared.status == "completed", compared.model_dump_json()
            assert len(_contract_requests(wire)) == 1


@pytest.mark.parametrize("prefix", _ALIASES)
def test_response_ids_are_scoped_to_the_issuing_user_and_team(gateway: Gateway, prefix: str) -> None:
    raw_id: Final = f"resp_owned_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return _model_discovery_reply()
        body: Final = _JSON_OBJECT.validate_json(request.body) if request.body else {}
        if request.method == "POST" and request.target == "/v1/responses":
            if body.get("input") == "create owned response":
                return Reply(body=_response(raw_id))
            return Reply(body=_response(raw_id))
        if request.method == "GET" and urlsplit(request.target).path.endswith("/input_items"):
            return Reply(body=b'{"data":[],"has_more":false,"object":"list"}')
        if request.method == "GET":
            return Reply(body=_response(raw_id))
        if request.method == "DELETE":
            return Reply(body=json.dumps({"id": raw_id, "object": "response", "deleted": True}).encode())
        if request.method == "POST" and request.target.endswith("/cancel"):
            return Reply(body=_response(raw_id, "cancelled"))
        raise AssertionError(f"Unexpected upstream request: {request.method} {request.target}")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        user_a: Final = scenario.member(team_a)
        user_b: Final = scenario.member(team_b)
        gateway.post("/team/member_add", {"team_id": team_b, "member": {"role": "user", "user_id": user_a}})
        client_a: Final = _client(gateway, _key(scenario, team_a, user_a, model), prefix)
        other_user_client: Final = _client(gateway, _key(scenario, team_b, user_b, model), prefix)
        other_team_client: Final = _client(gateway, _key(scenario, team_b, user_a, model), prefix)
        created: Final = client_a.responses.create(model=model, input="create owned response")
        client_id: Final = created.id
        assert client_id.startswith("resp_") and client_id != raw_id, created.model_dump_json()
        created_requests: Final = _contract_requests(wire)
        assert [(request.method, request.target, request.body) for request in created_requests] == [
            ("POST", "/v1/responses", created_requests[0].body)
        ], created_requests
        assert _JSON_OBJECT.validate_json(created_requests[0].body) == {
            "model": _MODEL,
            "input": "create owned response",
        }, created_requests[0].body

        for call in _every_id_route(other_user_client, model, client_id):
            _refused(call, _OTHER_USER_DETAIL)
        for call in _every_id_route(other_team_client, model, client_id):
            _refused(call, _OTHER_TEAM_DETAIL)
        assert _contract_requests(wire) == ()

        retrieved_response: Final = client_a.responses.with_raw_response.retrieve(client_id)
        retrieved: Final = retrieved_response.parse()
        assert retrieved.id != raw_id and same_response(retrieved.id, client_id), retrieved_response.text
        assert retrieved.model_dump(mode="json", exclude_none=True) == {
            "id": retrieved.id,
            "object": "response",
            "created_at": 1.0,
            "status": "completed",
            "model": _MODEL,
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }, retrieved_response.text
        deleted: Final = client_a.responses.with_raw_response.delete(client_id)
        assert deleted.status_code == 200, deleted.text
        deleted_response: Final = _DeleteResponse.model_validate_json(deleted.content)
        assert deleted_response.model_dump(exclude={"id"}) == {"object": "response", "deleted": True}, deleted.text
        assert same_response(deleted_response.id, client_id), deleted.text
        cancelled_response: Final = client_a.responses.with_raw_response.cancel(client_id)
        cancelled: Final = cancelled_response.parse()
        assert cancelled.status == "cancelled", cancelled_response.text
        assert cancelled.id != raw_id and same_response(cancelled.id, client_id), cancelled_response.text
        assert cancelled.model_dump(mode="json", exclude_none=True) == {
            "id": cancelled.id,
            "object": "response",
            "created_at": 1.0,
            "status": "cancelled",
            "model": _MODEL,
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }, cancelled_response.text
        page_response: Final = client_a.responses.input_items.with_raw_response.list(client_id)
        page: Final = page_response.parse()
        assert page.model_dump(mode="json", exclude_none=True) == {
            "data": [],
            "has_more": False,
            "object": "list",
        }, page_response.text
        followed: Final = client_a.responses.create(
            model=model,
            input="owned follow-up",
            previous_response_id=client_id,
        )
        assert followed.id != raw_id and same_response(followed.id, client_id), followed.model_dump_json()
        requests: Final = _contract_requests(wire)
        assert [(request.method, urlsplit(request.target).path, request.body) for request in requests] == [
            ("GET", f"/v1/responses/{raw_id}", b""),
            ("DELETE", f"/v1/responses/{raw_id}", b""),
            ("POST", f"/v1/responses/{raw_id}/cancel", b"{}"),
            ("GET", f"/v1/responses/{raw_id}/input_items", b""),
            ("POST", "/v1/responses", requests[-1].body),
        ], requests
        assert _JSON_OBJECT.validate_json(requests[-1].body) == {
            "input": "owned follow-up",
            "model": _MODEL,
            "previous_response_id": raw_id,
        }, requests[-1].body

        unmanaged_id: Final = f"resp_vendor_{uuid.uuid4().hex}"
        for call in _every_id_route(client_a, model, unmanaged_id):
            _refused(call, _UNMANAGED_DETAIL)
        assert _contract_requests(wire) == ()


@pytest.mark.parametrize("prefix", _ALIASES)
def test_compact_previous_response_id_is_scoped_to_the_issuing_user_and_team(
    gateway: Gateway,
    prefix: str,
) -> None:
    raw_id: Final = f"resp_compact_owned_{uuid.uuid4().hex}"
    compact_input: Final[ResponseInputParam] = [
        {"role": "user", "content": [{"type": "input_text", "text": "compact the owned response"}]}
    ]

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return _model_discovery_reply()
        if request.method == "POST" and request.target == "/v1/responses":
            return Reply(body=_response(raw_id))
        return Reply(body=_response(raw_id))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        user_a: Final = scenario.member(team_a)
        user_b: Final = scenario.member(team_b)
        gateway.post("/team/member_add", {"team_id": team_b, "member": {"role": "user", "user_id": user_a}})
        client_a: Final = _client(gateway, _key(scenario, team_a, user_a, model), prefix)
        other_user_client: Final = _client(gateway, _key(scenario, team_b, user_b, model), prefix)
        other_team_client: Final = _client(gateway, _key(scenario, team_b, user_a, model), prefix)
        created: Final = client_a.responses.create(model=model, input="create compact response")
        client_id: Final = created.id
        assert client_id != raw_id, created.model_dump_json()
        assert len(_contract_requests(wire)) == 1
        _refused(
            lambda: other_user_client.responses.compact(
                model=model,
                previous_response_id=client_id,
                input=[
                    {"role": "user", "content": [{"type": "input_text", "text": "compact someone else's response"}]}
                ],
            ),
            _OTHER_USER_DETAIL,
        )
        _refused(
            lambda: other_team_client.responses.compact(
                model=model,
                previous_response_id=client_id,
                input=[
                    {"role": "user", "content": [{"type": "input_text", "text": "compact someone else's response"}]}
                ],
            ),
            _OTHER_TEAM_DETAIL,
        )
        unmanaged_id: Final = f"resp_vendor_{uuid.uuid4().hex}"
        _refused(
            lambda: client_a.responses.compact(
                model=model,
                previous_response_id=unmanaged_id,
                input=[
                    {"role": "user", "content": [{"type": "input_text", "text": "compact someone else's response"}]}
                ],
            ),
            _UNMANAGED_DETAIL,
        )
        assert _contract_requests(wire) == ()

        client_a.responses.compact(model=model, previous_response_id=client_id, input=compact_input)
        compacted_requests: Final = _contract_requests(wire)
        assert [(request.method, request.target) for request in compacted_requests] == [
            ("POST", "/v1/responses/compact")
        ], compacted_requests
        assert _JSON_OBJECT.validate_json(compacted_requests[0].body) == {
            "model": _MODEL,
            "input": compact_input,
            "previous_response_id": raw_id,
        }, compacted_requests[0].body


def test_allow_unmanaged_response_ids_forward_the_vendor_id_on_create(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    raw_id: Final = f"resp_vendor_{uuid.uuid4().hex}"
    raw_followup_id: Final = f"resp_followup_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return _model_discovery_reply()
        return Reply(body=_response(raw_followup_id))

    config: Final = tmp_path / "allow_unmanaged_responses.yaml"
    config.write_text("general_settings:\n  allow_unmanaged_response_ids: true\n")
    with wire_server(respond) as wire, wire_server(respond) as decoy, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = _key(scenario, team, user, model)
        with owned_proxy(
            gateway,
            tmp_path,
            {
                "INTEGRATION_PROXY_WORKERS": "1",
                "OPENAI_BASE_URL": f"{decoy.url}/v1",
                "OPENAI_API_KEY": "synthetic-env-fallback-key",
            },
            config=config,
        ) as allowed_gateway:
            client: Final = _client(allowed_gateway, key, "/v1")
            followed: Final = client.responses.create(
                model=model,
                input="allowed vendor follow-up",
                previous_response_id=raw_id,
            )
            assert followed.model_dump(mode="json", exclude_none=True) == {
                "id": followed.id,
                "object": "response",
                "created_at": 1.0,
                "status": "completed",
                "model": model,
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }, followed.model_dump_json()
            retrieved_response: Final = client.responses.with_raw_response.retrieve(followed.id)
            retrieved: Final = retrieved_response.parse()
            assert same_response(retrieved.id, followed.id), retrieved_response.text
            assert retrieved.model_dump(mode="json", exclude_none=True) == {
                "id": retrieved.id,
                "object": "response",
                "created_at": 1.0,
                "status": "completed",
                "model": _MODEL,
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }, retrieved_response.text
            requests: Final = _contract_requests(wire)
        assert decoy.drain() == (), "The environment fallback deployment received a request"
        assert [(request.method, request.target, request.body) for request in requests] == [
            ("POST", "/v1/responses", requests[0].body),
            ("GET", f"/v1/responses/{raw_followup_id}", b""),
        ], requests
        assert all(request.headers["authorization"] == f"Bearer {_API_KEY}" for request in requests), requests
        assert _JSON_OBJECT.validate_json(requests[0].body) == {
            "model": _MODEL,
            "input": "allowed vendor follow-up",
            "previous_response_id": raw_id,
        }, requests[0].body


@pytest.mark.parametrize("route", ("retrieve", "delete", "cancel", "input_items"))
def test_allow_unmanaged_response_id_lifecycle_route_reaches_provider(
    gateway: Gateway,
    tmp_path: Path,
    route: Literal["retrieve", "delete", "cancel", "input_items"],
) -> None:
    pytest.skip(
        "BUG: unmanaged response-id lifecycle routes select model=None and return 400 instead of reaching OpenAI"
    )
    raw_id: Final = f"resp_vendor_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return _model_discovery_reply()
        if request.method == "GET" and request.target == f"/v1/responses/{raw_id}/input_items":
            return Reply(body=b'{"data":[],"has_more":false,"object":"list"}')
        if request.method == "DELETE":
            return Reply(body=json.dumps({"id": raw_id, "object": "response", "deleted": True}).encode())
        if request.method == "POST" and request.target.endswith("/cancel"):
            return Reply(body=_response(raw_id, "cancelled"))
        return Reply(body=_response(raw_id))

    config: Final = tmp_path / "allow_unmanaged_responses.yaml"
    config.write_text("general_settings:\n  allow_unmanaged_response_ids: true\n")
    with wire_server(respond) as wire, wire_server(respond) as decoy, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = _key(scenario, team, user, model)
        with owned_proxy(
            gateway,
            tmp_path,
            {
                "INTEGRATION_PROXY_WORKERS": "1",
                "OPENAI_BASE_URL": f"{decoy.url}/v1",
                "OPENAI_API_KEY": "synthetic-env-fallback-key",
            },
            config=config,
        ) as allowed_gateway:
            client: Final = _client(allowed_gateway, key, "/v1")
            expected_request: Final = {
                "retrieve": ("GET", f"/v1/responses/{raw_id}", b""),
                "delete": ("DELETE", f"/v1/responses/{raw_id}", b""),
                "cancel": ("POST", f"/v1/responses/{raw_id}/cancel", b"{}"),
                "input_items": ("GET", f"/v1/responses/{raw_id}/input_items", b""),
            }[route]
            match route:
                case "retrieve":
                    retrieved: Final = client.responses.retrieve(raw_id)
                    assert retrieved.model_dump(mode="json", exclude_none=True) == {
                        "id": raw_id,
                        "object": "response",
                        "created_at": 1.0,
                        "status": "completed",
                        "model": _MODEL,
                        "output": [],
                        "parallel_tool_calls": False,
                        "tool_choice": "auto",
                        "tools": [],
                    }, retrieved.model_dump_json()
                case "delete":
                    deleted: Final = client.responses.delete(raw_id)
                    assert deleted.model_dump(mode="json", exclude_none=True) == {
                        "id": raw_id,
                        "object": "response",
                        "deleted": True,
                    }, deleted.model_dump_json()
                case "cancel":
                    cancelled: Final = client.responses.cancel(raw_id)
                    assert cancelled.model_dump(mode="json", exclude_none=True) == {
                        "id": raw_id,
                        "object": "response",
                        "created_at": 1.0,
                        "status": "cancelled",
                        "model": _MODEL,
                        "output": [],
                        "parallel_tool_calls": False,
                        "tool_choice": "auto",
                        "tools": [],
                    }, cancelled.model_dump_json()
                case "input_items":
                    page: Final = client.responses.input_items.list(raw_id)
                    assert page.model_dump(mode="json", exclude_none=True) == {
                        "data": [],
                        "has_more": False,
                        "object": "list",
                    }, page.model_dump_json()
            requests: Final = _contract_requests(wire)
        assert tuple((request.method, request.target, request.body) for request in requests) == (expected_request,), (
            requests
        )
        assert all(request.headers["authorization"] == f"Bearer {_API_KEY}" for request in requests), requests
        assert decoy.drain() == (), "The environment fallback deployment received a request"

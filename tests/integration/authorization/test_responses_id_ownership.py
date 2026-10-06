from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.process import owned_proxy
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.responses import Response as ResponsesAPIResponse
from pydantic import JsonValue, TypeAdapter

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS

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


def _deployments_in_group(gateway: Gateway, group: str) -> int:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    return sum(object_value(entry).get("model_name") == group for entry in entries)


def _wait_for_workers(gateway: Gateway, group: str, deployments: int) -> None:
    ready_after: Final = time.monotonic() + (0 if _PROXY_WORKERS == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS + 1)
    eventually(
        lambda: (_deployments_in_group(gateway, group), time.monotonic()),
        lambda observation: observation[0] == deployments and observation[1] >= ready_after,
        seconds=30 if _PROXY_WORKERS == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS * 2 + 30,
    )


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


@pytest.mark.parametrize("prefix", _ALIASES)
def test_response_ids_are_scoped_to_the_issuing_user_and_team(gateway: Gateway, prefix: str) -> None:
    raw_id: Final = f"resp_owned_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return _model_discovery_reply()
        body: Final = _JSON_OBJECT.validate_json(request.body) if request.body else {}
        if request.method == "POST" and request.target == "/v1/responses":
            if body.get("input") == "create owned response":
                assert body == {"model": _MODEL, "input": "create owned response"}, request.body
                return Reply(body=_response(raw_id))
            assert body == {
                "model": _MODEL,
                "input": "owned follow-up",
                "previous_response_id": raw_id,
            }, request.body
            return Reply(body=_response(raw_id))
        if request.method == "GET" and urlsplit(request.target).path.endswith("/input_items"):
            assert urlsplit(request.target).path == f"/v1/responses/{raw_id}/input_items", request.target
            return Reply(body=b'{"data":[],"has_more":false,"object":"list"}')
        if request.method == "GET":
            assert request.target == f"/v1/responses/{raw_id}", request.target
            return Reply(body=_response(raw_id))
        if request.method == "DELETE":
            assert request.target == f"/v1/responses/{raw_id}", request.target
            return Reply(body=json.dumps({"id": raw_id, "object": "response", "deleted": True}).encode())
        if request.method == "POST" and request.target.endswith("/cancel"):
            assert request.target == f"/v1/responses/{raw_id}/cancel", request.target
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
        assert [(request.method, request.target) for request in _contract_requests(wire)] == [("POST", "/v1/responses")]

        for call in _every_id_route(other_user_client, model, client_id):
            _refused(call, _OTHER_USER_DETAIL)
        for call in _every_id_route(other_team_client, model, client_id):
            _refused(call, _OTHER_TEAM_DETAIL)
        assert _contract_requests(wire) == ()

        retrieved: Final = client_a.responses.retrieve(client_id)
        assert retrieved.id != raw_id and same_response(retrieved.id, client_id), retrieved.model_dump_json()
        deleted: Final = client_a.responses.with_raw_response.delete(client_id)
        assert deleted.status_code == 200, deleted.text
        cancelled: Final = client_a.responses.cancel(client_id)
        assert cancelled.status == "cancelled", cancelled.model_dump_json()
        page: Final = client_a.responses.input_items.list(client_id)
        assert page.data == [], page.model_dump_json()
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
        _refused(lambda: client_a.responses.retrieve(unmanaged_id), _UNMANAGED_DETAIL)
        _refused(
            lambda: client_a.responses.create(
                model=model, input="unmanaged follow-up", previous_response_id=unmanaged_id
            ),
            _UNMANAGED_DETAIL,
        )
        assert _contract_requests(wire) == ()


def test_allow_unmanaged_response_ids_forwards_previous_id_unchanged(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    raw_id: Final = f"resp_vendor_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return _model_discovery_reply()
        assert request.method == "POST"
        assert request.target == "/v1/responses"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": _MODEL,
            "input": "allowed vendor follow-up",
            "previous_response_id": raw_id,
        }, request.body
        return Reply(body=_response(f"resp_followup_{uuid.uuid4().hex}"))

    config: Final = tmp_path / "allow_unmanaged_responses.yaml"
    config.write_text("general_settings:\n  allow_unmanaged_response_ids: true\n")
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = _key(scenario, team, user, model)
        with owned_proxy(
            gateway,
            tmp_path,
            {"INTEGRATION_PROXY_WORKERS": "1"},
            config=config,
        ) as allowed_gateway:
            response: Final = allowed_gateway.request(
                "POST",
                "/v1/responses",
                {
                    "model": model,
                    "input": "allowed vendor follow-up",
                    "previous_response_id": raw_id,
                },
                key=key,
            )
            assert response.status_code == 200, response.text
            parsed_response: Final = ResponsesAPIResponse.model_validate_json(response.content)
            assert parsed_response.id.startswith("resp_"), response.text
        requests: Final = _contract_requests(wire)
        assert [(request.method, request.target) for request in requests] == [("POST", "/v1/responses")]

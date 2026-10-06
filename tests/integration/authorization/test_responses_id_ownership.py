from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import openai
from integration._support.client import Gateway, Scenario
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.responses_test_support import (
    drain_contract_requests,
    model_discovery_reply,
    wait_for_model_group_workers,
)
from openai.types.responses import Response as ResponsesAPIResponse
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "gpt-5"
_API_KEY: Final = "synthetic-responses-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_UNMANAGED_DETAIL: Final = (
    "Forbidden. This response id was not issued by this proxy, so the proxy cannot tell who owns it. "
    "To let keys address responses this proxy did not issue, set "
    "general_settings::allow_unmanaged_response_ids to True in the config.yaml file."
)


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
    return scenario.model(
        model=f"openai/{_MODEL}",
        api_key=_API_KEY,
        api_base=f"{wire.url}/v1",
    )


def _key(scenario: Scenario, team_id: str, user_id: str, model: str) -> str:
    return scenario.key(team_id=team_id, user_id=user_id, models=[model])


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


def test_response_ids_are_scoped_to_the_issuing_user_and_team(gateway: Gateway) -> None:
    raw_id: Final = f"resp_owned_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
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
            parsed_target: Final = urlsplit(request.target)
            assert parsed_target.path == f"/v1/responses/{raw_id}/input_items", request.target
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
        wait_for_model_group_workers(gateway, model)
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        user_a: Final = scenario.member(team_a)
        user_b: Final = scenario.member(team_b)
        key_a: Final = _key(scenario, team_a, user_a, model)
        key_b: Final = _key(scenario, team_b, user_b, model)
        client_a: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key_a, max_retries=0)
        created: Final = client_a.responses.create(model=model, input="create owned response")
        client_id: Final = created.id
        assert client_id.startswith("resp_") and client_id != raw_id, created.model_dump_json()
        assert [(request.method, request.target) for request in drain_contract_requests(wire)] == [
            ("POST", "/v1/responses")
        ]

        denied: Final = (
            gateway.request("GET", f"/v1/responses/{client_id}", key=key_b),
            gateway.request("DELETE", f"/v1/responses/{client_id}", key=key_b),
            gateway.request("POST", f"/v1/responses/{client_id}/cancel", {}, key=key_b),
            gateway.request(
                "GET",
                f"/v1/responses/{client_id}/input_items",
                key=key_b,
            ),
            gateway.request(
                "POST",
                "/v1/responses",
                {"model": model, "input": "cross-team follow-up", "previous_response_id": client_id},
                key=key_b,
            ),
        )
        for response in denied:
            _error(
                response,
                "Forbidden. The response id is not associated with the user, who this key belongs to. "
                "To disable this security feature, set general_settings::disable_responses_id_security to True "
                "in the config.yaml file.",
            )
        assert drain_contract_requests(wire) == ()

        retrieved: Final = client_a.responses.retrieve(client_id)
        assert retrieved.object == "response", retrieved.model_dump_json()
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
        assert followed.id.startswith("resp_"), followed.model_dump_json()
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, urlsplit(request.target).path) for request in requests] == [
            ("GET", f"/v1/responses/{raw_id}"),
            ("DELETE", f"/v1/responses/{raw_id}"),
            ("POST", f"/v1/responses/{raw_id}/cancel"),
            ("GET", f"/v1/responses/{raw_id}/input_items"),
            ("POST", "/v1/responses"),
        ], requests
        assert requests[0].body == b"", requests[0]
        assert requests[1].body == b"", requests[1]
        assert requests[2].body == b"{}", requests[2]
        assert requests[3].body == b"", requests[3]
        assert _JSON_OBJECT.validate_json(requests[4].body) == {
            "input": "owned follow-up",
            "model": _MODEL,
            "previous_response_id": raw_id,
        }, requests[4].body

        unmanaged_id: Final = f"resp_vendor_{uuid.uuid4().hex}"
        _error(
            gateway.request("GET", f"/v1/responses/{unmanaged_id}", key=key_a),
            _UNMANAGED_DETAIL,
        )
        _error(
            gateway.request(
                "POST",
                "/v1/responses",
                {"model": model, "input": "unmanaged follow-up", "previous_response_id": unmanaged_id},
                key=key_a,
            ),
            _UNMANAGED_DETAIL,
        )
        assert drain_contract_requests(wire) == ()


def test_allow_unmanaged_response_ids_forwards_previous_id_unchanged(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    raw_id: Final = f"resp_vendor_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert request.method == "POST"
        assert request.target == "/v1/responses"
        assert body["previous_response_id"] == raw_id, request.body
        assert body == {"model": _MODEL, "input": "allowed vendor follow-up", "previous_response_id": raw_id}, (
            request.body
        )
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
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target) for request in requests] == [("POST", "/v1/responses")]

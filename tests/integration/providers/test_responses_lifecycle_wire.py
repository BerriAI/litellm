from __future__ import annotations

import json
import uuid
from typing import Final, Literal
from urllib.parse import parse_qsl, urlsplit

import openai
import pytest
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.responses_test_support import (
    drain_contract_requests,
    model_discovery_reply,
    wait_for_model_group_workers,
)
from pydantic import BaseModel, JsonValue, TypeAdapter

_MODEL: Final = "gpt-5"
_API_KEY: Final = "synthetic-responses-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ALIASES: Final = ("/v1", "", "/openai/v1")


class _DeleteResponse(BaseModel):
    id: str
    object: Literal["response"]
    deleted: bool


def _response(identity: str, status: str = "completed") -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "response",
            "created_at": 1,
            "status": status,
            "model": _MODEL,
            "output": [],
        }
    ).encode()


def _stream(identity: str) -> tuple[bytes, ...]:
    return tuple(
        f"data: {json.dumps(event)}\n\n".encode()
        for event in (
            {"type": "response.created", "response": json.loads(_response(identity, "in_progress"))},
            {"type": "response.completed", "response": json.loads(_response(identity))},
        )
    ) + (b"data: [DONE]\n\n",)


def _deployment(gateway: Gateway, scenario: Scenario, wire: Wire) -> str:
    model: Final = scenario.model(
        model=f"openai/{_MODEL}",
        api_key=_API_KEY,
        api_base=f"{wire.url}/v1",
    )
    wait_for_model_group_workers(gateway, model)
    return model


def _register_in_group(gateway: Gateway, scenario: Scenario, wire: Wire, group: str) -> None:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": group,
            "litellm_params": {
                "model": f"openai/{_MODEL}",
                "api_key": _API_KEY,
                "api_base": f"{wire.url}/v1",
            },
            "model_info": {},
        },
    )
    deployment_id: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, deployment_id)
    wait_for_model_group_workers(gateway, group)


def _client(gateway: Gateway, key: str, prefix: str) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}{prefix}",
        api_key=key,
        max_retries=0,
    )


@pytest.mark.parametrize("prefix", _ALIASES)
def test_retrieve_routes_to_the_deployment_that_created_the_response(
    gateway: Gateway,
    prefix: str,
) -> None:
    raw_id: Final = f"resp_affinity_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        if request.method == "POST":
            assert request.target == "/v1/responses"
            assert _JSON_OBJECT.validate_json(request.body) == {"model": _MODEL, "input": "create for affinity"}, (
                request.body
            )
            return Reply(body=_response(raw_id))
        assert request.method == "GET"
        assert request.body == b""
        return Reply(body=_response(raw_id))

    with wire_server(respond) as wire_b, wire_server(respond) as wire_a, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire_b)
        client: Final = _client(gateway, gateway.key, prefix)
        created: Final = client.responses.create(model=model, input="create for affinity")
        client_id: Final = created.id
        assert client_id.startswith("resp_") and client_id != raw_id, created.model_dump_json()
        _register_in_group(gateway, scenario, wire_a, model)
        retrieved: Final = client.responses.retrieve(client_id)
        assert retrieved.id != raw_id, retrieved.model_dump_json()
        assert same_response(retrieved.id, client_id), retrieved.model_dump_json()
        requests: Final = drain_contract_requests(wire_b)
        assert [(request.method, request.target, request.body) for request in requests] == [
            ("POST", "/v1/responses", requests[0].body),
            ("GET", f"/v1/responses/{raw_id}", b""),
        ], requests
        assert _JSON_OBJECT.validate_json(requests[0].body) == {
            "model": _MODEL,
            "input": "create for affinity",
        }, requests[0].body
        assert requests[1].headers["authorization"] == f"Bearer {_API_KEY}", dict(requests[1].headers)
        assert drain_contract_requests(wire_a) == (), (
            f"{prefix or '/responses'} did not preserve response deployment affinity"
        )


@pytest.mark.parametrize("prefix", _ALIASES)
def test_retrieve_returns_the_id_the_client_holds(gateway: Gateway, prefix: str) -> None:
    pytest.skip(
        "BUG: GET /v1/responses/{id} returns a re-encrypted id that differs byte-for-byte from the id the client sent"
    )
    raw_id: Final = f"resp_retrieve_id_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        if request.method == "POST":
            assert request.target == "/v1/responses"
            assert _JSON_OBJECT.validate_json(request.body) == {"model": _MODEL, "input": "retrieve client id"}, (
                request.body
            )
            return Reply(body=_response(raw_id))
        assert request.method == "GET"
        assert request.target == f"/v1/responses/{raw_id}", request.target
        assert request.body == b""
        return Reply(body=_response(raw_id))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire)
        client: Final = _client(gateway, gateway.key, prefix)
        created: Final = client.responses.create(model=model, input="retrieve client id")
        client_id: Final = created.id
        retrieved: Final = client.responses.retrieve(client_id)
        assert retrieved.id == client_id, retrieved.model_dump_json()
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target, request.body) for request in requests] == [
            ("POST", "/v1/responses", requests[0].body),
            ("GET", f"/v1/responses/{raw_id}", b""),
        ], requests
        assert _JSON_OBJECT.validate_json(requests[0].body) == {
            "model": _MODEL,
            "input": "retrieve client id",
        }, requests[0].body


@pytest.mark.parametrize("prefix", _ALIASES)
def test_streamed_response_ids_are_stable_and_retrievable_on_each_alias(
    gateway: Gateway,
    prefix: str,
) -> None:
    raw_id: Final = f"resp_stream_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        if request.method == "POST":
            assert request.target == "/v1/responses"
            assert _JSON_OBJECT.validate_json(request.body) == {
                "model": _MODEL,
                "input": "stream lifecycle",
                "stream": True,
            }, request.body
            return Reply(content_type="text/event-stream", chunks=_stream(raw_id))
        assert request.method == "GET"
        assert request.target == f"/v1/responses/{raw_id}"
        assert request.body == b""
        return Reply(body=_response(raw_id))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        client: Final = _client(gateway, key, prefix)
        stream: Final = client.responses.create(model=model, input="stream lifecycle", stream=True)
        events: Final = list(stream)
        response_ids: Final = tuple(event.response.id for event in events if hasattr(event, "response"))
        assert response_ids and all(response_id == response_ids[0] for response_id in response_ids), events
        client_id: Final = response_ids[0]
        assert client_id.startswith("resp_") and client_id != raw_id, events
        retrieved: Final = client.responses.retrieve(client_id)
        assert retrieved.object == "response", retrieved.model_dump_json()
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/v1/responses"),
            ("GET", f"/v1/responses/{raw_id}"),
        ], requests
        assert all(request.headers["authorization"] == f"Bearer {_API_KEY}" for request in requests), requests


@pytest.mark.parametrize("prefix", _ALIASES)
def test_delete_forwards_provider_id_and_returns_client_id(
    gateway: Gateway,
    prefix: str,
) -> None:
    raw_id: Final = f"resp_delete_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        if request.method == "POST":
            assert request.target == "/v1/responses"
            assert _JSON_OBJECT.validate_json(request.body) == {"model": _MODEL, "input": "delete lifecycle"}, (
                request.body
            )
            return Reply(body=_response(raw_id))
        assert request.method == "DELETE"
        assert request.body == b""
        return Reply(body=json.dumps({"id": raw_id, "object": "response", "deleted": True}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire)
        client: Final = _client(gateway, gateway.key, prefix)
        created: Final = client.responses.create(model=model, input="delete lifecycle")
        deleted: Final = client.responses.with_raw_response.delete(created.id)
        assert deleted.status_code == 200, deleted.text
        parsed: Final = _DeleteResponse.model_validate_json(deleted.content)
        assert parsed.model_dump(mode="json", exclude={"id"}) == {
            "object": "response",
            "deleted": True,
        }, deleted.text
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target, request.body) for request in requests] == [
            ("POST", "/v1/responses", requests[0].body),
            ("DELETE", f"/v1/responses/{raw_id}", b""),
        ], requests


@pytest.mark.parametrize("prefix", _ALIASES)
def test_delete_returns_the_client_held_id_not_the_provider_id(
    gateway: Gateway,
    prefix: str,
) -> None:
    pytest.skip("BUG: DELETE /v1/responses/{id} returns the raw provider id instead of the proxy-issued id")
    raw_id: Final = f"resp_delete_id_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        if request.method == "POST":
            assert request.target == "/v1/responses"
            assert _JSON_OBJECT.validate_json(request.body) == {"model": _MODEL, "input": "delete client id"}, (
                request.body
            )
            return Reply(body=_response(raw_id))
        assert request.method == "DELETE"
        assert request.target == f"/v1/responses/{raw_id}", request.target
        assert request.body == b""
        return Reply(body=json.dumps({"id": raw_id, "object": "response", "deleted": True}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire)
        client: Final = _client(gateway, gateway.key, prefix)
        created: Final = client.responses.create(model=model, input="delete client id")
        deleted: Final = client.responses.with_raw_response.delete(created.id)
        assert deleted.status_code == 200, deleted.text
        parsed: Final = _DeleteResponse.model_validate_json(deleted.content)
        assert parsed.id != raw_id, deleted.text
        assert parsed.model_dump(mode="json") == {
            "id": created.id,
            "object": "response",
            "deleted": True,
        }, deleted.text
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target, request.body) for request in requests] == [
            ("POST", "/v1/responses", requests[0].body),
            ("DELETE", f"/v1/responses/{raw_id}", b""),
        ], requests
        assert _JSON_OBJECT.validate_json(requests[0].body) == {
            "model": _MODEL,
            "input": "delete client id",
        }, requests[0].body


@pytest.mark.parametrize("prefix", _ALIASES)
def test_cancel_forwards_provider_id_and_returns_cancelled_response(
    gateway: Gateway,
    prefix: str,
) -> None:
    raw_id: Final = f"resp_cancel_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        if request.method == "POST" and request.target == "/v1/responses":
            assert _JSON_OBJECT.validate_json(request.body) == {
                "model": _MODEL,
                "input": "cancel lifecycle",
                "background": True,
            }, request.body
            return Reply(body=_response(raw_id, "queued"))
        assert request.body in (b"", b"{}")
        return Reply(body=_response(raw_id, "cancelled"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire)
        client: Final = _client(gateway, gateway.key, prefix)
        created: Final = client.responses.create(model=model, input="cancel lifecycle", background=True)
        cancelled: Final = client.responses.cancel(created.id)
        assert cancelled.status == "cancelled", cancelled.model_dump_json()
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/v1/responses"),
            ("POST", f"/v1/responses/{raw_id}/cancel"),
        ], requests
        assert requests[1].body == b"{}", requests[1]


def test_input_items_pagination_query_reaches_the_provider(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: GET /v1/responses/{id}/input_items drops limit, order and after query params; "
        "upstream receives limit=20&order=desc"
    )
    raw_id: Final = f"resp_input_items_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        if request.method == "POST":
            assert request.target == "/v1/responses"
            assert _JSON_OBJECT.validate_json(request.body) == {
                "model": _MODEL,
                "input": "list response input items",
            }, request.body
            return Reply(body=_response(raw_id))
        assert request.method == "GET"
        parsed_target: Final = urlsplit(request.target)
        assert parsed_target.path == f"/v1/responses/{raw_id}/input_items", request.target
        assert dict(parse_qsl(parsed_target.query)) == {
            "after": "item_after",
            "limit": "2",
            "order": "asc",
        }, request.target
        assert request.body == b""
        return Reply(body=b'{"data":[],"has_more":false,"object":"list"}')

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire)
        client: Final = _client(gateway, gateway.key, "/v1")
        created: Final = client.responses.create(model=model, input="list response input items")
        page: Final = client.responses.input_items.list(
            created.id,
            limit=2,
            order="asc",
            after="item_after",
        )
        assert page.data == [], page.model_dump_json()
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/v1/responses"),
            (
                "GET",
                f"/v1/responses/{raw_id}/input_items?limit=2&order=asc&after=item_after",
            ),
        ], requests

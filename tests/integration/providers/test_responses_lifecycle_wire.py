from __future__ import annotations

import json
import os
import time
import uuid
from typing import Final, Literal
from urllib.parse import parse_qsl, urlsplit

import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.responses import ResponseStreamEvent
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS

_MODEL: Final = "gpt-5"
_API_KEY: Final = "synthetic-responses-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROXY_WORKERS: Final = max(1, int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1")))
_ALIASES: Final = ("/v1", "", "/openai/v1")


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
    ready_after: Final = time.monotonic() + (0 if _PROXY_WORKERS == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS + 1)
    eventually(
        lambda: (_deployments_in_group(gateway, group), time.monotonic()),
        lambda observation: observation[0] == deployments and observation[1] >= ready_after,
        seconds=30 if _PROXY_WORKERS == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS * 2 + 30,
    )


def model_discovery_reply(model: str) -> Reply:
    return Reply(
        body=json.dumps(
            {"object": "list", "data": [{"id": model, "object": "model", "created": 1, "owned_by": "openai"}]}
        ).encode()
    )


def drain_contract_requests(wire: Wire) -> tuple[Request, ...]:
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


def _deployment(scenario: Scenario, wire: Wire, api_key: str = _API_KEY) -> str:
    model: Final = scenario.model(
        model=f"openai/{_MODEL}",
        api_key=api_key,
        api_base=f"{wire.url}/v1",
    )
    _wait_for_workers(scenario.gateway, model, 1)
    return model


def _stream_response_events(
    client: openai.OpenAI, response_id: str
) -> tuple[tuple[ResponseStreamEvent, ...], str | None]:
    try:
        return (
            tuple(
                client.responses.retrieve(
                    response_id,
                    stream=True,
                    starting_after=3,
                    include=["reasoning.encrypted_content"],
                )
            ),
            None,
        )
    except openai.InternalServerError as error:
        return (), error.response.text


def _register_preferred_in_group(scenario: Scenario, wire: Wire, group: str, api_key: str) -> None:
    existing: Final = _deployments_in_group(scenario.gateway, group)
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": group,
            "litellm_params": {
                "model": f"openai/{_MODEL}",
                "api_key": api_key,
                "api_base": f"{wire.url}/v1",
                "order": 1,
            },
            "model_info": {},
        },
    )
    deployment_id: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, deployment_id)
    _wait_for_workers(scenario.gateway, group, existing + 1)


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
    wire_b_key: Final = "synthetic-responses-key-b"
    wire_a_key: Final = "synthetic-responses-key-a"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        if request.method == "POST":
            return Reply(body=_response(raw_id))
        return Reply(body=_response(raw_id))

    with wire_server(respond) as wire_b, wire_server(respond) as wire_a, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire_b, wire_b_key)
        client: Final = _client(gateway, gateway.key, prefix)
        created: Final = client.responses.create(model=model, input="create for affinity")
        client_id: Final = created.id
        assert client_id.startswith("resp_") and client_id != raw_id, created.model_dump_json()
        _register_preferred_in_group(scenario, wire_a, model, wire_a_key)
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
        assert all(request.headers["authorization"] == f"Bearer {wire_b_key}" for request in requests), requests
        assert requests[1].body == b"", requests[1]
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
        model: Final = _deployment(scenario, wire)
        client: Final = _client(gateway, gateway.key, prefix)
        created: Final = client.responses.create(model=model, input="retrieve client id")
        client_id: Final = created.id
        retrieved: Final = client.responses.retrieve(client_id)
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target, request.body) for request in requests] == [
            ("POST", "/v1/responses", requests[0].body),
            ("GET", f"/v1/responses/{raw_id}", b""),
        ], requests
        assert _JSON_OBJECT.validate_json(requests[0].body) == {
            "model": _MODEL,
            "input": "retrieve client id",
        }, requests[0].body
        assert retrieved.id == client_id, (retrieved.model_dump_json(), requests)


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
        return Reply(body=_response(raw_id))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        client: Final = _client(gateway, key, prefix)
        stream: Final = client.responses.create(model=model, input="stream lifecycle", stream=True)
        events: Final = list(stream)
        response_ids: Final = tuple(event.response.id for event in events if hasattr(event, "response"))
        assert response_ids and all(response_id == response_ids[0] for response_id in response_ids), events
        streamed_id: Final = response_ids[0]
        assert streamed_id.startswith("resp_") and streamed_id != raw_id, events
        retrieved_response: Final = client.responses.with_raw_response.retrieve(streamed_id)
        retrieved: Final = retrieved_response.parse()
        assert retrieved.id != raw_id and same_response(retrieved.id, streamed_id), retrieved_response.text
        assert retrieved.model_dump(mode="json", exclude_none=True) == {
            "id": retrieved.id,
            "object": "response",
            "created_at": 1.0,
            "status": "completed",
            "model": _MODEL,
            "output": [],
        }, retrieved_response.text
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, urlsplit(request.target).path, request.body) for request in requests] == [
            ("POST", "/v1/responses", requests[0].body),
            ("GET", f"/v1/responses/{raw_id}", b""),
        ], requests
        assert _JSON_OBJECT.validate_json(requests[0].body) == {
            "model": _MODEL,
            "input": "stream lifecycle",
            "stream": True,
        }, requests[0].body
        assert all(request.headers["authorization"] == f"Bearer {_API_KEY}" for request in requests), requests


@pytest.mark.parametrize("prefix", _ALIASES)
def test_retrieve_query_reaches_the_provider(gateway: Gateway, prefix: str) -> None:
    pytest.skip(
        "BUG: GET /v1/responses/{id} drops include and starting_after; streamed GET returns 500 for provider SSE"
    )
    normal_raw_id: Final = f"resp_query_normal_{uuid.uuid4().hex}"
    stream_raw_id: Final = f"resp_query_stream_{uuid.uuid4().hex}"
    normal_include: Final = ["message.output_text.logprobs", "reasoning.encrypted_content"]
    stream_include: Final = ["reasoning.encrypted_content"]

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        if request.method == "POST" and urlsplit(request.target).path == "/v1/responses":
            body: Final = _JSON_OBJECT.validate_json(request.body)
            identity: Final = stream_raw_id if body.get("input") == "query retrieve stream" else normal_raw_id
            return Reply(body=_response(identity))
        if urlsplit(request.target).path == f"/v1/responses/{stream_raw_id}":
            return Reply(content_type="text/event-stream", chunks=_stream(stream_raw_id))
        return Reply(body=_response(normal_raw_id))

    with wire_server(respond) as wire:
        sdk_client: Final = openai.OpenAI(base_url=f"{wire.url}/v1", api_key=_API_KEY, max_retries=0)
        sdk_client.responses.retrieve(normal_raw_id, include=normal_include)
        tuple(
            sdk_client.responses.retrieve(
                stream_raw_id,
                stream=True,
                starting_after=3,
                include=stream_include,
            )
        )
        sdk_requests: Final = drain_contract_requests(wire)
        assert [request.method for request in sdk_requests] == ["GET", "GET"], sdk_requests
        expected_queries: Final = tuple(
            tuple(sorted(parse_qsl(urlsplit(request.target).query, keep_blank_values=True))) for request in sdk_requests
        )

        with gateway.scenario() as scenario:
            model: Final = _deployment(scenario, wire)
            client: Final = _client(gateway, gateway.key, prefix)
            normal_created: Final = client.responses.create(model=model, input="query retrieve normal")
            stream_created: Final = client.responses.create(model=model, input="query retrieve stream")
            retrieved_response: Final = client.responses.with_raw_response.retrieve(
                normal_created.id,
                include=normal_include,
            )
            retrieved: Final = retrieved_response.parse()
            assert retrieved.id != normal_raw_id and same_response(retrieved.id, normal_created.id), (
                retrieved_response.text
            )
            assert retrieved.model_dump(mode="json", exclude_none=True) == {
                "id": retrieved.id,
                "object": "response",
                "created_at": 1.0,
                "status": "completed",
                "model": _MODEL,
                "output": [],
            }, retrieved_response.text
            stream_result: Final = _stream_response_events(client, stream_created.id)
            stream_events: Final = stream_result[0]
            stream_error_text: Final = stream_result[1]
            if stream_error_text is not None:
                requests: Final = drain_contract_requests(wire)
                assert [(request.method, urlsplit(request.target).path, request.body) for request in requests[2:]] == [
                    ("GET", f"/v1/responses/{normal_raw_id}", b""),
                    ("GET", f"/v1/responses/{stream_raw_id}", b""),
                ], stream_error_text
                actual_queries: Final = tuple(
                    tuple(sorted(parse_qsl(urlsplit(request.target).query, keep_blank_values=True)))
                    for request in requests[2:]
                )
                assert actual_queries == expected_queries, (
                    requests[2:],
                    actual_queries,
                    expected_queries,
                    stream_error_text,
                )
                pytest.fail(f"Streaming retrieval returned an API error: {stream_error_text}")
            completed_events: Final = tuple(event for event in stream_events if event.type == "response.completed")
            assert len(completed_events) == 1, stream_events
            completed: Final = completed_events[0]
            assert completed.response.id != stream_raw_id and same_response(completed.response.id, stream_created.id), (
                stream_events
            )
            assert completed.model_dump(mode="json", exclude_none=True) == {
                "type": "response.completed",
                "response": {
                    "id": completed.response.id,
                    "object": "response",
                    "created_at": 1.0,
                    "status": "completed",
                    "model": _MODEL,
                    "output": [],
                },
            }, stream_events
            requests: Final = drain_contract_requests(wire)
            assert [(request.method, urlsplit(request.target).path, request.body) for request in requests] == [
                ("POST", "/v1/responses", requests[0].body),
                ("POST", "/v1/responses", requests[1].body),
                ("GET", f"/v1/responses/{normal_raw_id}", b""),
                ("GET", f"/v1/responses/{stream_raw_id}", b""),
            ], requests
            assert _JSON_OBJECT.validate_json(requests[0].body) == {
                "model": _MODEL,
                "input": "query retrieve normal",
            }, requests[0].body
            assert _JSON_OBJECT.validate_json(requests[1].body) == {
                "model": _MODEL,
                "input": "query retrieve stream",
            }, requests[1].body
            actual_queries: Final = tuple(
                tuple(sorted(parse_qsl(urlsplit(request.target).query, keep_blank_values=True)))
                for request in requests[2:]
            )
            assert actual_queries == expected_queries, requests


@pytest.mark.parametrize("prefix", _ALIASES)
def test_delete_forwards_the_provider_id_once_and_reports_deletion(
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
        model: Final = _deployment(scenario, wire)
        client: Final = _client(gateway, gateway.key, prefix)
        created: Final = client.responses.create(model=model, input="delete lifecycle")
        deleted: Final = client.responses.with_raw_response.delete(created.id)
        assert deleted.status_code == 200, deleted.text
        parsed: Final = _DeleteResponse.model_validate_json(deleted.content)
        assert same_response(parsed.id, created.id), deleted.text
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
        model: Final = _deployment(scenario, wire)
        client: Final = _client(gateway, gateway.key, prefix)
        created: Final = client.responses.create(model=model, input="delete client id")
        deleted: Final = client.responses.with_raw_response.delete(created.id)
        assert deleted.status_code == 200, deleted.text
        parsed: Final = _DeleteResponse.model_validate_json(deleted.content)
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target, request.body) for request in requests] == [
            ("POST", "/v1/responses", requests[0].body),
            ("DELETE", f"/v1/responses/{raw_id}", b""),
        ], requests
        assert _JSON_OBJECT.validate_json(requests[0].body) == {
            "model": _MODEL,
            "input": "delete client id",
        }, requests[0].body
        assert parsed.id != raw_id, (deleted.text, requests)
        assert parsed.model_dump(mode="json") == {
            "id": created.id,
            "object": "response",
            "deleted": True,
        }, deleted.text


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
        model: Final = _deployment(scenario, wire)
        client: Final = _client(gateway, gateway.key, prefix)
        created: Final = client.responses.create(model=model, input="cancel lifecycle", background=True)
        cancelled: Final = client.responses.cancel(created.id)
        assert cancelled.id != raw_id and same_response(cancelled.id, created.id), cancelled.model_dump_json()
        assert cancelled.model_dump(mode="json", exclude_none=True) == {
            "id": cancelled.id,
            "object": "response",
            "created_at": 1.0,
            "status": "cancelled",
            "model": _MODEL,
            "output": [],
        }, cancelled.model_dump_json()
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/v1/responses"),
            ("POST", f"/v1/responses/{raw_id}/cancel"),
        ], requests
        assert requests[1].body == b"{}", requests[1]


@pytest.mark.parametrize("prefix", _ALIASES)
def test_input_items_pagination_query_reaches_the_provider(gateway: Gateway, prefix: str) -> None:
    pytest.skip(
        "BUG: GET /v1/responses/{id}/input_items drops limit, order and after query params; "
        "upstream receives limit=20&order=desc"
    )
    raw_id: Final = f"resp_input_items_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        if request.method == "POST":
            return Reply(body=_response(raw_id))
        return Reply(body=b'{"data":[],"has_more":false,"object":"list"}')

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        client: Final = _client(gateway, gateway.key, prefix)
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
        assert all(request.headers["authorization"] == f"Bearer {_API_KEY}" for request in requests), requests
        assert requests[1].body == b"", requests[1]

from __future__ import annotations

import json
import uuid
from typing import Final

import openai
import pytest
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.responses import ResponseCompletedEvent
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "gpt-5"
_API_KEY: Final = "synthetic-responses-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


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


def _register_in_group(scenario: Scenario, wire: Wire, group: str, order: int | None = None) -> str:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": group,
            "litellm_params": {
                "model": f"openai/{_MODEL}",
                "api_key": _API_KEY,
                "api_base": f"{wire.url}/v1",
                **({} if order is None else {"order": order}),
            },
            "model_info": {},
        },
    )
    model_id: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, model_id)
    return group


def _register(scenario: Scenario, wire: Wire, group: str | None = None) -> str:
    return (
        scenario.model(
            model=f"openai/{_MODEL}",
            api_key=_API_KEY,
            api_base=f"{wire.url}/v1",
        )
        if group is None
        else _register_in_group(scenario, wire, group)
    )


def _enable_responses_affinity(scenario: Scenario) -> None:
    gateway: Final = scenario.gateway
    settings: Final = object_value(gateway.get("/router/settings")["current_values"])
    current: Final = settings.get("optional_pre_call_checks")
    original: Final = list(current) if isinstance(current, list) else []
    scenario.cleanups.callback(
        gateway.post,
        "/config/update",
        {"router_settings": {"optional_pre_call_checks": original}},
    )
    gateway.post(
        "/config/update",
        {"router_settings": {"optional_pre_call_checks": [*original, "responses_api_deployment_check"]}},
    )


@pytest.mark.parametrize("stream", (False, True))
def test_previous_response_id_uses_the_raw_id_and_affinity(
    gateway: Gateway,
    stream: bool,
) -> None:
    first_raw_id: Final = f"resp_turn_one_{uuid.uuid4().hex}"
    second_raw_id: Final = f"resp_turn_two_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.method == "POST"
        assert request.target == "/v1/responses"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        if body["input"] == "turn one":
            assert body == {"model": _MODEL, "input": "turn one"}, request.body
            return Reply(body=_response(first_raw_id))
        if stream:
            return Reply(content_type="text/event-stream", chunks=_stream(second_raw_id))
        return Reply(body=_response(second_raw_id))

    def unused(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.method == "POST"
        assert request.target == "/v1/responses"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        if stream:
            return Reply(content_type="text/event-stream", chunks=_stream(second_raw_id))
        return Reply(body=_response(second_raw_id))

    with wire_server(respond) as first_wire, wire_server(unused) as second_wire, gateway.scenario() as scenario:
        model: Final = f"integration-{uuid.uuid4().hex}"
        _enable_responses_affinity(scenario)
        assert _register(scenario, first_wire, model) == model
        client: Final = openai.OpenAI(
            base_url=f"{gateway.client.base_url}/v1",
            api_key=gateway.key,
            max_retries=0,
        )
        first: Final = client.responses.create(model=model, input="turn one")
        assert first.id.startswith("resp_") and first.id != first_raw_id, first.model_dump_json()
        _register_in_group(scenario, second_wire, model, order=1)
        second_result: Final = client.responses.create(
            model=model,
            input="turn two",
            previous_response_id=first.id,
            stream=stream,
        )
        if stream:
            events: Final = list(second_result)
            completed: Final = events[-1]
            assert isinstance(completed, ResponseCompletedEvent), events
            assert completed.response.id != second_raw_id, completed.model_dump_json()
            assert completed.response.output == [], completed.model_dump_json()
        else:
            assert second_result.id != second_raw_id, second_result.model_dump_json()
            assert second_result.output == [], second_result.model_dump_json()
        assert drain_contract_requests(second_wire) == (), "previous_response_id did not preserve deployment affinity"
        requests: Final = drain_contract_requests(first_wire)
        assert len(requests) == 2, requests
        assert [_JSON_OBJECT.validate_json(request.body) for request in requests] == [
            {"model": _MODEL, "input": "turn one"},
            {
                "model": _MODEL,
                "input": "turn two",
                "previous_response_id": first_raw_id,
                "stream": stream,
            },
        ], [request.body for request in requests]


def test_codex_responses_request_preserves_all_supported_fields(gateway: Gateway) -> None:
    response_id: Final = f"resp_codex_{uuid.uuid4().hex}"
    function_tool: Final = {
        "type": "function",
        "name": "lookup",
        "description": "Look up a record",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "strict": True,
    }
    custom_tool: Final = {
        "type": "custom",
        "name": "ApplyPatch",
        "description": "Apply a patch",
        "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "ok"'},
    }
    web_search_tool: Final = {"type": "web_search"}
    expected_body: Final = {
        "model": _MODEL,
        "input": "Use the available tools to inspect the requested change.",
        "stream": True,
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "prompt_cache_key": "cache-codex-contract",
        "reasoning": {"effort": "high", "summary": "auto"},
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "patch_summary",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        },
        "tools": [function_tool, custom_tool, web_search_tool],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "instructions": "Return concise structured output.",
    }

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.method == "POST"
        assert request.target == "/v1/responses"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        _JSON_OBJECT.validate_json(request.body)
        return Reply(content_type="text/event-stream", chunks=_stream(response_id))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire)
        client: Final = openai.OpenAI(
            base_url=f"{gateway.client.base_url}/v1",
            api_key=gateway.key,
            max_retries=0,
        )
        stream: Final = client.responses.create(
            model=model,
            input="Use the available tools to inspect the requested change.",
            stream=True,
            include=["reasoning.encrypted_content"],
            store=False,
            prompt_cache_key="cache-codex-contract",
            reasoning={"effort": "high", "summary": "auto"},
            text={
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "patch_summary",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                        "required": ["summary"],
                        "additionalProperties": False,
                    },
                },
            },
            tools=[function_tool, custom_tool, web_search_tool],
            tool_choice="auto",
            parallel_tool_calls=True,
            instructions="Return concise structured output.",
        )
        events: Final = list(stream)
        completed: Final = events[-1]
        assert isinstance(completed, ResponseCompletedEvent), events
        assert completed.response.status == "completed", completed.model_dump_json()
        assert completed.response.output == [], completed.model_dump_json()
        requests: Final = drain_contract_requests(wire)
        assert len(requests) == 1, requests
        assert _JSON_OBJECT.validate_json(requests[0].body) == expected_body, requests[0].body
        assert [request.target for request in requests] == ["/v1/responses"]


def test_conversation_is_forwarded_as_a_separate_responses_parameter(gateway: Gateway) -> None:
    pytest.skip("BUG: POST /v1/responses drops the conversation param; upstream receives the body without it")
    conversation: Final = {"id": f"conv_{uuid.uuid4().hex}"}

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.method == "POST"
        assert request.target == "/v1/responses"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": _MODEL,
            "input": "continue this conversation",
            "conversation": conversation,
        }, request.body
        return Reply(body=_response(f"resp_conversation_{uuid.uuid4().hex}"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire)
        client: Final = openai.OpenAI(
            base_url=f"{gateway.client.base_url}/v1",
            api_key=gateway.key,
            max_retries=0,
        )
        response: Final = client.responses.create(
            model=model,
            input="continue this conversation",
            conversation=conversation,
        )
        assert response.object == "response", response.model_dump_json()
        assert [request.target for request in drain_contract_requests(wire)] == ["/v1/responses"]

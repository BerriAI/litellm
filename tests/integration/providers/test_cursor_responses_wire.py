from __future__ import annotations

import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "gpt-5"
_API_KEY: Final = "synthetic-cursor-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROMPT: Final = "Inspect the requested change."
_INPUT: Final = [{"role": "user", "content": [{"type": "input_text", "text": _PROMPT}]}]
_CUSTOM_TOOL: Final = {
    "type": "custom",
    "name": "ApplyPatch",
    "description": "Apply a patch",
    "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "ok"'},
}
_FUNCTION_TOOL: Final = {
    "type": "function",
    "name": "lookup",
    "description": "Look up a record",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    "strict": True,
}


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


def _response(identity: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": _MODEL,
        "output": [
            {
                "id": "msg_cursor",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Patch inspected.", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
    }


def _stream(identity: str) -> tuple[bytes, ...]:
    response: Final = _response(identity)
    return tuple(
        f"data: {json.dumps(event)}\n\n".encode()
        for event in (
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
            {
                "type": "response.output_text.delta",
                "item_id": "msg_cursor",
                "output_index": 0,
                "content_index": 0,
                "delta": "Patch inspected.",
            },
            {"type": "response.completed", "response": response},
        )
    ) + (b"data: [DONE]\n\n",)


def _chat_completion(stream: bool) -> bytes | tuple[bytes, ...]:
    if stream:
        return (
            b'data: {"id":"chat_cursor_1","object":"chat.completion.chunk","created":1,"model":"gpt-5","choices":[{"index":0,"delta":{"content":"Patch inspected."},"finish_reason":null}]}\n\n',
            b'data: {"id":"chat_cursor_1","object":"chat.completion.chunk","created":1,"model":"gpt-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        )
    return json.dumps(
        {
            "id": "chat_cursor_1",
            "object": "chat.completion",
            "created": 1,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Patch inspected."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }
    ).encode()


def _deployment(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"openai/{_MODEL}",
        api_key=_API_KEY,
        api_base=f"{wire.url}/v1",
    )


@pytest.mark.parametrize(
    ("suffix", "reasoning"),
    (
        ("-thinking-high", {"reasoning": {"effort": "high"}}),
        ("-fast", {}),
        ("-thinking-high-fast", {"reasoning": {"effort": "high"}}),
    ),
)
@pytest.mark.parametrize("stream", (False, True))
def test_cursor_responses_input_uses_responses_wire_and_chat_output(
    gateway: Gateway,
    stream: bool,
    suffix: str,
    reasoning: dict[str, JsonValue],
) -> None:
    raw_id: Final = f"resp_cursor_{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        team_alias: Final = f"integration-{uuid.uuid4().hex}"
        team: Final = scenario.team(team_alias=team_alias)
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user)

        def respond(request: Request) -> Reply:
            if request.method == "GET" and request.target == "/v1/models":
                return model_discovery_reply(_MODEL)
            assert request.method == "POST"
            assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
            if request.target == "/v1/chat/completions":
                completion: Final = _chat_completion(stream)
                return Reply(
                    content_type="text/event-stream" if stream else "application/json",
                    chunks=completion if isinstance(completion, tuple) else None,
                    body=completion if isinstance(completion, bytes) else b"",
                )
            assert request.target == "/v1/responses", request.target
            body: Final = _JSON_OBJECT.validate_json(request.body)
            assert {key: value for key, value in body.items() if key != "metadata"} == {
                "model": _MODEL,
                "input": _INPUT,
                **reasoning,
                "tools": [_CUSTOM_TOOL],
                "stream": stream,
            }, request.body
            if stream:
                return Reply(content_type="text/event-stream", chunks=_stream(raw_id))
            return Reply(body=json.dumps(_response(raw_id)).encode())

        with wire_server(respond) as wire:
            model: Final = _deployment(scenario, wire)
            response: Final = gateway.request(
                "POST",
                "/cursor/chat/completions",
                {
                    "model": f"{model}{suffix}",
                    "input": _INPUT,
                    "messages": [],
                    "tools": [_CUSTOM_TOOL],
                    "stream_options": {"include_usage": True, "include_obfuscation": False},
                    "stream": stream,
                },
                key=key,
            )
            assert response.status_code == 200, response.text
            if stream:
                events: Final = tuple(
                    _JSON_OBJECT.validate_json(line.removeprefix("data: "))
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                )
                assert events, response.text
                assert all(event.get("object") == "chat.completion.chunk" for event in events), response.text
                assert all(not str(event.get("type", "")).startswith("response.") for event in events), response.text
                assert [{k: v for k, v in event.items() if k not in ("id", "created")} for event in events] == [
                    {
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Patch inspected."}}],
                    },
                    {
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                ], response.text
            else:
                payload: Final = _JSON_OBJECT.validate_json(response.content)
                identity: Final = payload.get("id")
                created: Final = payload.get("created")
                assert isinstance(identity, str) and same_response(identity, raw_id), response.text
                assert isinstance(created, int) and created > 0, response.text
                assert payload == {
                    "id": identity,
                    "created": created,
                    "model": model,
                    "object": "chat.completion",
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "index": 0,
                            "message": {
                                "content": "Patch inspected.",
                                "role": "assistant",
                                "tool_calls": None,
                                "function_call": None,
                                "provider_specific_fields": None,
                            },
                        }
                    ],
                    "usage": {
                        "completion_tokens": 3,
                        "prompt_tokens": 4,
                        "total_tokens": 7,
                        "completion_tokens_details": None,
                        "prompt_tokens_details": None,
                    },
                }, response.text
            assert [request.target for request in drain_contract_requests(wire)] == ["/v1/responses"]


def test_cursor_responses_body_carries_no_proxy_metadata(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /cursor/chat/completions forwards proxy-internal metadata (key hash, team, user, client ip, user agent) "
        "to the provider in the Responses body"
    )
    raw_id: Final = f"resp_cursor_metadata_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.method == "POST" and request.target == "/v1/responses", request.target
        return Reply(body=json.dumps(_response(raw_id)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/cursor/chat/completions",
            {
                "model": f"{model}-thinking-high",
                "input": _INPUT,
                "messages": [],
                "tools": [_CUSTOM_TOOL],
            },
        )
        assert response.status_code == 200, response.text
        requests: Final = drain_contract_requests(wire)
        assert len(requests) == 1, requests
        body: Final = _JSON_OBJECT.validate_json(requests[0].body)
        assert body == {
            "model": _MODEL,
            "input": _INPUT,
            "reasoning": {"effort": "high"},
            "tools": [_CUSTOM_TOOL],
        }, requests[0].body


def test_cursor_stream_reports_usage_when_the_client_asks_for_it(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /cursor/chat/completions with stream_options.include_usage streams no usage chunk; "
        "the Responses usage from response.completed never reaches the chat client"
    )
    raw_id: Final = f"resp_cursor_usage_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert (request.method, request.target) == ("POST", "/v1/responses"), request.target
        return Reply(content_type="text/event-stream", chunks=_stream(raw_id))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/cursor/chat/completions",
            {
                "model": model,
                "input": _INPUT,
                "stream_options": {"include_usage": True},
                "stream": True,
            },
        )
        assert response.status_code == 200, response.text
        events: Final = tuple(
            _JSON_OBJECT.validate_json(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
        assert [event["usage"] for event in events if event.get("usage")] == [
            {"completion_tokens": 3, "prompt_tokens": 4, "total_tokens": 7}
        ], response.text
        assert [request.target for request in drain_contract_requests(wire)] == ["/v1/responses"]


def test_cursor_messages_body_uses_chat_completions_wire(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        return Reply(
            body=json.dumps(
                {
                    "id": "chat_cursor_1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": _MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Chat response."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        response: Final = gateway.request(
            "POST",
            "/cursor/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "genuine chat request"}],
                "tools": [_FUNCTION_TOOL, _CUSTOM_TOOL],
                "tool_choice": {"type": "function", "name": "lookup"},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["object"] == "chat.completion", response.text
        assert payload["choices"] == [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Chat response.",
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ], response.text
        requests: Final = drain_contract_requests(wire)
        assert [(request.method, request.target) for request in requests] == [("POST", "/v1/chat/completions")], (
            requests
        )
        assert _JSON_OBJECT.validate_json(requests[0].body) == {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "genuine chat request"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a record",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        "strict": True,
                    },
                },
                {
                    "type": "custom",
                    "custom": {
                        "name": "ApplyPatch",
                        "description": "Apply a patch",
                        "format": {
                            "type": "grammar",
                            "grammar": {"syntax": "lark", "definition": 'start: "ok"'},
                        },
                    },
                },
            ],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }, requests[0].body

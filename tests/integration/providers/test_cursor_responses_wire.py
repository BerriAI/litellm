from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from hashlib import sha256
from importlib.resources import files
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, list_value, object_value
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.worker_readiness import wait_for_worker_readiness
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "gpt-5"
_API_KEY: Final = "synthetic-cursor-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROXY_WORKERS: Final = max(1, int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1")))
_COST_MAP: Final = TypeAdapter(dict[str, dict[str, JsonValue]])
_MODEL_ROW: Final = _COST_MAP.validate_json(
    (files("litellm") / "model_prices_and_context_window_backup.json").read_bytes()
)[_MODEL]
_THINKING_LEVELS: Final = tuple(
    level
    for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    if _MODEL_ROW.get(f"supports_{level}_reasoning_effort") is not False
)
_CHAT_THINKING_LEVELS: Final = tuple(level for level in _THINKING_LEVELS if level != "xhigh")
_RESPONSES_SUFFIXES: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = tuple(
    (f"-thinking-{level}", {"reasoning": {"effort": level}}) for level in _THINKING_LEVELS
) + (
    ("-fast", {}),
    ("-thinking-low-fast", {"reasoning": {"effort": "low"}}),
    ("-thinking-high-fast", {"reasoning": {"effort": "high"}}),
)
_CHAT_SUFFIXES: Final[tuple[tuple[str, str], ...]] = (("", ""),) + tuple(
    (f"-thinking-{level}", level) for level in _CHAT_THINKING_LEVELS
)
_PROMPT: Final = "Inspect the requested change."
_REQUEST_TAG: Final = "cursor-integration"
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


def _chat_completion(stream: bool, identity: str) -> bytes | tuple[bytes, ...]:
    if stream:
        return tuple(
            f"data: {json.dumps(chunk)}\n\n".encode()
            for chunk in (
                {
                    "id": identity,
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": _MODEL,
                    "choices": [{"index": 0, "delta": {"content": "Patch inspected."}, "finish_reason": None}],
                },
                {
                    "id": identity,
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": _MODEL,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            )
        ) + (b"data: [DONE]\n\n",)
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _MODEL,
            "system_fingerprint": "fp_cursor",
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


def _deployment(scenario: Scenario, wire: Wire, **pricing: JsonValue) -> str:
    model: Final = scenario.model(
        model=f"openai/{_MODEL}",
        api_key=_API_KEY,
        api_base=f"{wire.url}/v1",
        **pricing,
    )
    wait_for_worker_readiness(scenario.gateway, model, 1, _PROXY_WORKERS)
    return model


@pytest.mark.parametrize(
    ("suffix", "reasoning"),
    _RESPONSES_SUFFIXES,
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
                completion: Final = _chat_completion(stream, raw_id)
                return Reply(
                    content_type="text/event-stream" if stream else "application/json",
                    chunks=completion if isinstance(completion, tuple) else None,
                    body=completion if isinstance(completion, bytes) else b"",
                )
            assert request.target == "/v1/responses", request.target
            body: Final = _JSON_OBJECT.validate_json(request.body)
            assert body == {
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


@pytest.mark.parametrize(
    "client_metadata", ({}, {"cursor_conversation": "integration"}), ids=("no_client_metadata", "client_metadata")
)
@pytest.mark.parametrize("stream", (False, True))
def test_cursor_responses_body_sends_only_client_metadata_and_bills_the_key(
    gateway: Gateway,
    stream: bool,
    client_metadata: Mapping[str, JsonValue],
) -> None:
    raw_id: Final = f"resp_cursor_metadata_{uuid.uuid4().hex}"
    alias: Final = f"integration-cursor-{uuid.uuid4().hex}"
    sent_metadata: Final = {"metadata": client_metadata} if client_metadata else {}

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.method == "POST" and request.target == "/v1/responses", request.target
        if stream:
            return Reply(content_type="text/event-stream", chunks=_stream(raw_id))
        return Reply(body=json.dumps(_response(raw_id)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        key: Final = scenario.key(key_alias=alias)
        hashed_key: Final = sha256(key.encode()).hexdigest()
        response: Final = gateway.request(
            "POST",
            "/cursor/chat/completions",
            {
                "model": model,
                "input": _INPUT,
                "messages": [],
                "tools": [_CUSTOM_TOOL],
                "stream": stream,
                **sent_metadata,
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        requests: Final = drain_contract_requests(wire)
        assert len(requests) == 1, requests
        assert _JSON_OBJECT.validate_json(requests[0].body) == {
            "model": _MODEL,
            "input": _INPUT,
            "tools": [_CUSTOM_TOOL],
            "stream": stream,
            **sent_metadata,
        }, requests[0].body
        (row,) = eventually(
            lambda: gateway.rows(
                'SELECT call_type, spend, metadata FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed_key,)
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert row["call_type"] == "aresponses", row
        assert object_value(row["metadata"])["user_api_key_alias"] == alias, row
        assert isinstance(row["spend"], float) and row["spend"] > 0, row
        eventually(
            lambda: gateway.rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed_key,)),
            lambda tokens: len(tokens) == 1 and tokens[0]["spend"] == pytest.approx(row["spend"]),
            seconds=70,
        )


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


@pytest.mark.parametrize(("suffix", "reasoning_effort"), _CHAT_SUFFIXES)
@pytest.mark.parametrize("stream", (False, True))
def test_cursor_messages_body_uses_chat_completions_wire(
    gateway: Gateway,
    stream: bool,
    suffix: str,
    reasoning_effort: str,
) -> None:
    chat_id: Final = f"chat_cursor_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        completion: Final = _chat_completion(stream, chat_id)
        return Reply(
            content_type="text/event-stream" if stream else "application/json",
            chunks=completion if isinstance(completion, tuple) else None,
            body=completion if isinstance(completion, bytes) else b"",
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
                "model": f"{model}{suffix}",
                "messages": [{"role": "user", "content": "genuine chat request"}],
                "tools": [_FUNCTION_TOOL, _CUSTOM_TOOL],
                "tool_choice": {"type": "function", "name": "lookup"},
                "stream": stream,
            },
            key=key,
            headers={"x-litellm-tags": _REQUEST_TAG},
        )
        assert response.status_code == 200, response.text
        if stream:
            events: Final = tuple(
                _JSON_OBJECT.validate_json(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
            assert [
                {key: value for key, value in event.items() if key not in ("id", "created")} for event in events
            ] == [
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
            assert isinstance(identity, str) and same_response(identity, chat_id), response.text
            assert isinstance(created, int) and created > 0, response.text
            assert payload == {
                "id": identity,
                "created": created,
                "model": model,
                "object": "chat.completion",
                "system_fingerprint": "fp_cursor",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": "Patch inspected.",
                            "role": "assistant",
                        },
                        "provider_specific_fields": {},
                    }
                ],
                "usage": {
                    "completion_tokens": 3,
                    "prompt_tokens": 4,
                    "total_tokens": 7,
                },
            }, response.text
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
            **({"stream": True} if stream else {}),
            **({"stream_options": {"include_usage": True}} if stream else {}),
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
        }, requests[0].body
        (row,) = eventually(
            lambda: gateway.rows(
                'SELECT call_type, team_id, request_tags FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert (row["call_type"], row["team_id"]) == ("acompletion", team), row
        assert _REQUEST_TAG in list_value(row["request_tags"]), row


@pytest.mark.parametrize(
    ("body_shape", "metadata_as_json_string"), (("messages", False), ("input", False), ("input", True))
)
def test_cursor_body_keeps_client_and_key_tags_and_enforces_the_client_tag_budget(
    gateway: Gateway, body_shape: str, metadata_as_json_string: bool
) -> None:
    chat_id: Final = f"chat_cursor_{uuid.uuid4().hex}"
    response_id: Final = f"resp_cursor_tags_{uuid.uuid4().hex}"
    budget_tag: Final = f"cursor-tag-budget-{uuid.uuid4().hex}"
    key_tag: Final = f"cursor-key-tag-{uuid.uuid4().hex}"
    client_metadata: Final = {"tags": [budget_tag]}

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        if request.target == "/v1/responses":
            return Reply(body=json.dumps(_response(response_id)).encode())
        completion: Final = _chat_completion(False, chat_id)
        return Reply(body=completion if isinstance(completion, bytes) else b"")

    def delete_tag() -> None:
        gateway.post("/tag/delete", {"name": budget_tag})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, input_cost_per_token=0.01, output_cost_per_token=0.01)
        key: Final = scenario.key(models=[model], metadata={"tags": [key_tag]})
        gateway.post("/tag/new", {"name": budget_tag, "max_budget": 0.0001})
        scenario.cleanups.callback(delete_tag)

        def tagged_chat(text: str) -> httpx.Response:
            return gateway.request(
                "POST",
                "/cursor/chat/completions",
                {
                    "model": model,
                    body_shape: [{"role": "user", "content": text}],
                    "metadata": json.dumps(client_metadata) if metadata_as_json_string else client_metadata,
                },
                key=key,
            )

        first: Final = tagged_chat("first tagged chat")
        assert first.status_code == 200, first.text
        (row,) = eventually(
            lambda: gateway.rows(
                'SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert {budget_tag, key_tag} <= set(list_value(row["request_tags"])), row
        blocked: Final = eventually(
            lambda: tagged_chat("tag budget probe"), lambda response: response.status_code != 200, seconds=70
        )
        assert blocked.status_code == 422, blocked.text
        assert "budget" in blocked.text.lower(), blocked.text

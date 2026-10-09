import itertools
import json
import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_BACKEND: Final = "databricks-glm-5-2"
_API_KEY: Final = "synthetic-databricks-key"
_PROMPT: Final = "Summarise the cached briefing in one sentence."
_JSON_SCHEMA_BACKEND: Final = "databricks-qwen35-122b-a10b"
_CLAUDE_BACKEND: Final = "databricks-claude-haiku-4-5"
_JSON_SCHEMA_PROMPT: Final = "Return the requested JSON."
_JSON_CONTENT: Final = '{"p":{"name":"Ada"}}'
_PROVIDER_USAGE: Final[Mapping[str, JsonValue]] = {
    "prompt_tokens": 12011,
    "completion_tokens": 8,
    "total_tokens": 12019,
    "cache_read_input_tokens": 12002,
    "cache_creation_input_tokens": 0,
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PERSON_SCHEMA: Final[Mapping[str, JsonValue]] = {
    "$defs": {
        "Person": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    },
    "type": "object",
    "properties": {"p": {"$ref": "#/$defs/Person"}},
    "required": ["p"],
}
_JSON_SCHEMA_RESPONSE_FORMAT: Final[Mapping[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {"name": "P", "strict": True, "schema": _PERSON_SCHEMA},
}
_MESSAGES_PERSON_SCHEMA: Final[Mapping[str, JsonValue]] = {
    "$defs": {
        "Person": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        }
    },
    "type": "object",
    "properties": {"p": {"$ref": "#/$defs/Person"}},
    "required": ["p"],
    "additionalProperties": False,
}
_MESSAGES_RESPONSE_FORMAT: Final[Mapping[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {
        "name": "structured_output",
        "schema": _MESSAGES_PERSON_SCHEMA,
        "strict": True,
    },
}
_JSON_TOOL_CALL_MESSAGE: Final[dict[str, JsonValue]] = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "json-tool-call",
            "type": "function",
            "function": {"name": "json_tool_call", "arguments": _JSON_CONTENT},
        }
    ],
}


def _chat_completion_response(model: str, message: dict[str, JsonValue], finish_reason: str) -> bytes:
    response: Final = {
        "id": "databricks-json-schema-response",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return json.dumps(response).encode()


class _PromptTokensDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cached_tokens: int | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: _PromptTokensDetails | None = None


class _Delta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    content: str | None = None


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    delta: _Delta


class _Chunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    choices: tuple[_Choice, ...]
    usage: _Usage | None = None


def _frame(
    identity: str,
    choices: list[Mapping[str, object]],
    usage: Mapping[str, JsonValue] | None = None,
    model: str = _BACKEND,
) -> bytes:
    value: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": choices,
        **({} if usage is None else {"usage": usage}),
    }
    return b"data: " + json.dumps(value).encode() + b"\n\n"


@pytest.mark.covers("other.provider_wire.databricks.stream_usage_and_cache_reads_reach_client_and_spend_log")
def test_databricks_stream_final_usage_chunk_reaches_client_and_spend_log(gateway: Gateway) -> None:
    identity: Final = f"databricks-stream-{uuid.uuid4().hex}"
    frames: Final = (
        _frame(
            identity, [{"index": 0, "delta": {"role": "assistant", "content": "The briefing "}, "finish_reason": None}]
        ),
        _frame(identity, [{"index": 0, "delta": {"content": "is short."}, "finish_reason": None}]),
        _frame(identity, [{"index": 0, "delta": {}, "finish_reason": "stop"}]),
        _frame(identity, [], usage=_PROVIDER_USAGE),
        b"data: [DONE]\n\n",
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["messages"] == [{"role": "user", "content": _PROMPT}]
        assert body["stream"] is True
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": _PROMPT}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read()
            lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
        assert lines[-1] == "data: [DONE]", lines
        chunks: Final = tuple(_Chunk.model_validate_json(line.removeprefix("data: ")) for line in lines[:-1])
        assert {chunk.id for chunk in chunks} == {identity}
        assert (
            "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
            == "The briefing is short."
        )
        usages: Final = tuple(chunk.usage for chunk in chunks if chunk.usage is not None)
        assert len(usages) == 1, lines
        assert (
            usages[0].prompt_tokens,
            usages[0].completion_tokens,
            usages[0].total_tokens,
            usages[0].prompt_tokens_details.cached_tokens if usages[0].prompt_tokens_details is not None else None,
        ) == (12011, 8, 12019, 12002), lines
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT prompt_tokens, completion_tokens, total_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"], rows[0]["total_tokens"]) == (12011, 8, 12019)


def test_databricks_json_schema_refs_are_preserved_in_chat_completions(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _JSON_SCHEMA_BACKEND
        assert body["messages"] == [{"role": "user", "content": _JSON_SCHEMA_PROMPT}]
        assert body["stream"] is False
        assert body["response_format"] == _JSON_SCHEMA_RESPONSE_FORMAT
        return Reply(
            body=_chat_completion_response(
                _JSON_SCHEMA_BACKEND,
                {"role": "assistant", "content": _JSON_CONTENT},
                "stop",
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{_JSON_SCHEMA_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": _JSON_SCHEMA_PROMPT}],
                "response_format": _JSON_SCHEMA_RESPONSE_FORMAT,
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        assert body["choices"] == [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _JSON_CONTENT},
                "finish_reason": "stop",
            }
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


def test_databricks_streaming_json_schema_refs_are_preserved(gateway: Gateway) -> None:
    identity: Final = f"databricks-json-schema-stream-{uuid.uuid4().hex}"
    frames: Final = (
        _frame(
            identity,
            [{"index": 0, "delta": {"role": "assistant", "content": '{"p":'}, "finish_reason": None}],
            model=_JSON_SCHEMA_BACKEND,
        ),
        _frame(
            identity,
            [{"index": 0, "delta": {"content": '{"name":"Ada"}}'}, "finish_reason": None}],
            model=_JSON_SCHEMA_BACKEND,
        ),
        _frame(identity, [{"index": 0, "delta": {}, "finish_reason": "stop"}], model=_JSON_SCHEMA_BACKEND),
        b"data: [DONE]\n\n",
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _JSON_SCHEMA_BACKEND
        assert body["messages"] == [{"role": "user", "content": _JSON_SCHEMA_PROMPT}]
        assert body["stream"] is True
        assert body["response_format"] == _JSON_SCHEMA_RESPONSE_FORMAT
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{_JSON_SCHEMA_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": _JSON_SCHEMA_PROMPT}],
                "response_format": _JSON_SCHEMA_RESPONSE_FORMAT,
                "stream": True,
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read()
            lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
        assert lines[-1] == "data: [DONE]", lines
        chunks: Final = tuple(_Chunk.model_validate_json(line.removeprefix("data: ")) for line in lines[:-1])
        choices: Final = tuple(itertools.chain.from_iterable(chunk.choices for chunk in chunks))
        assert "".join(choice.delta.content or "" for choice in choices) == _JSON_CONTENT
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


def test_databricks_claude_json_schema_refs_are_preserved_in_tool_parameters(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _CLAUDE_BACKEND
        assert body["messages"] == [{"role": "user", "content": _JSON_SCHEMA_PROMPT}]
        assert "response_format" not in body
        assert body["tools"] == [
            {
                "type": "function",
                "function": {"name": "json_tool_call", "parameters": _PERSON_SCHEMA},
            }
        ]
        assert body["tool_choice"] == {"type": "function", "function": {"name": "json_tool_call"}}
        return Reply(body=_chat_completion_response(_CLAUDE_BACKEND, _JSON_TOOL_CALL_MESSAGE, "tool_calls"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{_CLAUDE_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": _JSON_SCHEMA_PROMPT}],
                "response_format": _JSON_SCHEMA_RESPONSE_FORMAT,
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        assert body["choices"] == [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _JSON_CONTENT},
                "finish_reason": "stop",
            }
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


def test_databricks_responses_json_schema_refs_are_preserved_in_chat_bridge(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _JSON_SCHEMA_BACKEND
        assert body["messages"] == [{"role": "user", "content": _JSON_SCHEMA_PROMPT}]
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "P", "schema": _PERSON_SCHEMA, "strict": True},
        }
        return Reply(
            body=_chat_completion_response(
                _JSON_SCHEMA_BACKEND,
                {"role": "assistant", "content": _JSON_CONTENT},
                "stop",
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{_JSON_SCHEMA_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": _JSON_SCHEMA_PROMPT,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "P",
                        "strict": True,
                        "schema": _PERSON_SCHEMA,
                    }
                },
            },
        )
        assert response.status_code == 200, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


def test_databricks_messages_json_schema_refs_are_preserved_for_qwen(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _JSON_SCHEMA_BACKEND
        assert body["response_format"] == _MESSAGES_RESPONSE_FORMAT
        return Reply(
            body=_chat_completion_response(
                _JSON_SCHEMA_BACKEND,
                {"role": "assistant", "content": _JSON_CONTENT},
                "stop",
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{_JSON_SCHEMA_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _JSON_SCHEMA_PROMPT}],
                "output_format": {"type": "json_schema", "schema": _PERSON_SCHEMA},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        assert body["content"] == [{"type": "text", "text": _JSON_CONTENT}]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


def test_databricks_messages_claude_json_schema_refs_are_preserved_in_tool_parameters(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _CLAUDE_BACKEND
        assert "response_format" not in body
        assert body["tools"] == [
            {
                "type": "function",
                "function": {"name": "json_tool_call", "parameters": _MESSAGES_PERSON_SCHEMA},
            }
        ]
        assert body["tool_choice"] == {"type": "function", "function": {"name": "json_tool_call"}}
        return Reply(body=_chat_completion_response(_CLAUDE_BACKEND, _JSON_TOOL_CALL_MESSAGE, "tool_calls"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{_CLAUDE_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _JSON_SCHEMA_PROMPT}],
                "output_format": {"type": "json_schema", "schema": _PERSON_SCHEMA},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        assert body["content"] == [{"type": "text", "text": _JSON_CONTENT}]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]

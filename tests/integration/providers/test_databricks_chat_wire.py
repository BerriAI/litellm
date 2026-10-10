import itertools
import json
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Final, Literal, TypeAlias

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


def _shape_reply(identity: str, stream: bool) -> Reply:
    if stream:
        chunks: Final = (
            _frame(
                identity,
                [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
            ),
            _frame(identity, [{"index": 0, "delta": {}, "finish_reason": "stop"}]),
            b"data: [DONE]\n\n",
        )
        return Reply(content_type="text/event-stream", chunks=chunks)

    body: Final = {
        "id": identity,
        "object": "chat.completion",
        "created": 1,
        "model": _BACKEND,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return Reply(body=json.dumps(body).encode())


@pytest.mark.parametrize(
    "case,endpoint,provider_model,request_body,expected_messages,responses_mode,upstream_path",
    [
        pytest.param(
            "chat_nonstream",
            "/v1/chat/completions",
            "databricks-meta-llama-3-3-70b-instruct",
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply in JSON"}]}],
                "response_format": {"type": "json_object"},
            },
            [{"role": "user", "content": "Reply in JSON"}],
            False,
            "/chat/completions",
            id="A1-chat-nonstream",
        ),
        pytest.param(
            "chat_stream",
            "/v1/chat/completions",
            "databricks-meta-llama-3-3-70b-instruct",
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply in JSON"}]}],
                "response_format": {"type": "json_object"},
                "stream": True,
            },
            [{"role": "user", "content": "Reply in JSON"}],
            False,
            "/chat/completions",
            id="A2-chat-stream",
        ),
        pytest.param(
            "responses-mode-bridge",
            "/v1/chat/completions",
            "databricks-meta-llama-3-1-8b-instruct",
            {
                "messages": [{"role": "user", "content": "Reply in JSON"}],
                "response_format": {"type": "json_object"},
            },
            [{"role": "user", "content": "Reply in JSON"}],
            True,
            "/chat/completions",
            id="A3-responses-mode",
        ),
        pytest.param(
            "responses-mode-bridge-unity-catalog",
            "/v1/chat/completions",
            "system.ai.deepseek-v4-1-flash",
            {
                "messages": [{"role": "user", "content": "Reply in JSON with key a. hi"}],
                "response_format": {"type": "json_object"},
            },
            [{"role": "user", "content": "Reply in JSON with key a. hi"}],
            True,
            "/ai-gateway/mlflow/v1/chat/completions",
            id="A3b-responses-mode-unity-catalog-model",
        ),
        pytest.param(
            "messages-endpoint",
            "/v1/messages",
            "databricks-meta-llama-3-3-70b-instruct",
            {
                "max_tokens": 32,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply in JSON"}]}],
            },
            [{"role": "user", "content": "Reply in JSON"}],
            False,
            "/chat/completions",
            id="A4-messages",
        ),
        pytest.param(
            "responses-endpoint",
            "/v1/responses",
            "databricks-meta-llama-3-3-70b-instruct",
            {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Reply in JSON"}]}]},
            [{"role": "user", "content": "Reply in JSON"}],
            False,
            "/chat/completions",
            id="A5-responses-list-input",
        ),
    ],
)
def test_databricks_non_claude_single_text_block_reaches_upstream_as_string(
    gateway: Gateway,
    case: str,
    endpoint: str,
    provider_model: str,
    request_body: dict[str, JsonValue],
    expected_messages: list[JsonValue],
    responses_mode: bool,
    upstream_path: str,
) -> None:
    identity: Final = f"databricks-single-text-{case}-{uuid.uuid4().hex}"
    streaming: Final = request_body.get("stream") is True

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["messages"] == expected_messages, body["messages"]
        return _shape_reply(identity, streaming)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"databricks/{provider_model}",
            api_base=wire.url,
            api_key=_API_KEY,
            model_info={"mode": "responses"} if responses_mode else None,
        )
        headers: Final = {"anthropic-version": "2023-06-01"} if endpoint == "/v1/messages" else {}
        body: Final = {"model": model, **request_body}
        if streaming:
            with gateway.client.stream(
                "POST", endpoint, json=body, headers={"Authorization": f"Bearer {gateway.key}", **headers}
            ) as response:
                assert response.status_code == 200, response.read()
                lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
            assert lines[-1] == "data: [DONE]", lines
            assert any('"content":"ok"' in line for line in lines), lines
        else:
            response: Final = gateway.request("POST", endpoint, body, headers=headers)
            assert response.status_code == 200, response.text
            assert _JSON_OBJECT.validate_json(response.content)["id"]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", upstream_path)]


@pytest.mark.parametrize(
    "case,provider_model,content,expected_content",
    [
        pytest.param(
            "cache-control",
            "databricks-meta-llama-3-3-70b-instruct",
            [{"type": "text", "text": "Reply in JSON", "cache_control": {"type": "ephemeral"}}],
            [{"type": "text", "text": "Reply in JSON", "cache_control": {"type": "ephemeral"}}],
            id="B1-cache-control-stays-list",
        ),
        pytest.param(
            "two-text-blocks",
            "databricks-meta-llama-3-3-70b-instruct",
            [{"type": "text", "text": "Reply"}, {"type": "text", "text": " in JSON"}],
            [{"type": "text", "text": "Reply"}, {"type": "text", "text": " in JSON"}],
            id="B2-two-text-blocks-stays-list",
        ),
        pytest.param(
            "text-and-image",
            "databricks-meta-llama-3-3-70b-instruct",
            [
                {"type": "text", "text": "Reply in JSON"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
            [
                {"type": "text", "text": "Reply in JSON"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
            id="B3-text-and-image-stays-list",
        ),
        pytest.param(
            "claude",
            "databricks-claude-opus-5-5",
            [{"type": "text", "text": "Reply in JSON"}],
            [{"type": "text", "text": "Reply in JSON"}],
            id="B4-claude-stays-list",
        ),
    ],
)
def test_databricks_list_content_stays_list(
    gateway: Gateway,
    case: str,
    provider_model: str,
    content: list[JsonValue],
    expected_content: list[JsonValue],
) -> None:
    identity: Final = f"databricks-list-content-{case}-{uuid.uuid4().hex}"
    expected_messages: Final = [{"role": "user", "content": expected_content}]

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["messages"] == expected_messages, body["messages"]
        return _shape_reply(identity, False)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{provider_model}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": content}]},
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["id"]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


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


_Endpoint: TypeAlias = Literal["chat", "responses", "messages"]
_PATHS: Final[Mapping[_Endpoint, str]] = {
    "chat": "/v1/chat/completions",
    "responses": "/v1/responses",
    "messages": "/v1/messages",
}
_OUTAGE: Final = "upstream-outage"
_OUTAGE_MESSAGE: Final = "The service is temporarily unavailable"
_EXPECTED_JSON: Final[JsonValue] = _JSON_OBJECT.validate_json(_JSON_CONTENT)
_CONTENT_DELTAS: Final[tuple[Mapping[str, JsonValue], ...]] = (
    {"role": "assistant", "content": '{"p":'},
    {"content": '{"name":"Ada"}}'},
)
_TOOL_CALL_DELTAS: Final[tuple[Mapping[str, JsonValue], ...]] = (
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "index": 0,
                "id": "json-tool-call",
                "type": "function",
                "function": {"name": "json_tool_call", "arguments": ""},
            }
        ],
    },
    {"tool_calls": [{"index": 0, "function": {"arguments": _JSON_CONTENT}}]},
)


class _ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    content: str | None = None


class _ChatChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    message: _ChatMessage


class _ChatCompletion(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    choices: tuple[_ChatChoice, ...]


class _OutputPart(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    type: str
    text: str | None = None


class _OutputItem(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    type: str
    content: tuple[_OutputPart, ...] = ()
    name: str | None = None
    arguments: str | None = None


class _ResponsesBody(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    output: tuple[_OutputItem, ...]


class _ResponsesEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    type: str
    response: _ResponsesBody | None = None


class _MessagesBlock(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    type: str
    text: str | None = None


class _MessagesBody(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    content: tuple[_MessagesBlock, ...]


class _MessagesDelta(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    type: str | None = None
    text: str | None = None
    partial_json: str | None = None


class _MessagesEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    type: str
    delta: _MessagesDelta | None = None


class _FunctionDelta(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    arguments: str | None = None


class _ToolCallDelta(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    function: _FunctionDelta | None = None


class _StreamDelta(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    content: str | None = None
    tool_calls: tuple[_ToolCallDelta, ...] | None = None


class _StreamChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    delta: _StreamDelta


class _StreamChunk(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    choices: tuple[_StreamChoice, ...]


def _json_schema_format(schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "json_schema", "json_schema": {"name": "P", "strict": True, "schema": dict(schema)}}


def _client_body(
    endpoint: _Endpoint,
    model: str,
    stream: bool,
    prompt: str = _JSON_SCHEMA_PROMPT,
) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": _json_schema_format(_PERSON_SCHEMA),
                "stream": stream,
            }
        case "responses":
            return {
                "model": model,
                "input": prompt,
                "text": {
                    "format": {"type": "json_schema", "name": "P", "strict": True, "schema": dict(_PERSON_SCHEMA)}
                },
                "stream": stream,
            }
        case "messages":
            return {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": prompt}],
                "output_format": {"type": "json_schema", "schema": dict(_PERSON_SCHEMA)},
                "stream": stream,
            }


def _expected_outbound(endpoint: _Endpoint, backend: str) -> dict[str, JsonValue]:
    schema: Final = dict(_MESSAGES_PERSON_SCHEMA if endpoint == "messages" else _PERSON_SCHEMA)
    if "claude" in backend:
        return _JSON_OBJECT.validate_python(
            {
                "tools": [{"type": "function", "function": {"name": "json_tool_call", "parameters": schema}}],
                "tool_choice": {"type": "function", "function": {"name": "json_tool_call"}},
            }
        )
    name: Final = "structured_output" if endpoint == "messages" else "P"
    return _JSON_OBJECT.validate_python(
        {"response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}}
    )


def _assert_outbound(request: Request, endpoint: _Endpoint, backend: str, stream: bool) -> None:
    assert (request.method, request.target) == ("POST", "/chat/completions")
    body: Final = _JSON_OBJECT.validate_json(request.body)
    expected: Final = _expected_outbound(endpoint, backend)
    assert {key: body.get(key) for key in expected} == expected
    assert (body["model"], body.get("stream") is True) == (backend, stream)
    assert "claude" not in backend or "response_format" not in body


def _upstream_reply(backend: str, stream: bool) -> Reply:
    claude: Final = "claude" in backend
    finish_reason: Final = "tool_calls" if claude else "stop"
    if not stream:
        message: Final[dict[str, JsonValue]] = (
            _JSON_TOOL_CALL_MESSAGE if claude else {"role": "assistant", "content": _JSON_CONTENT}
        )
        return Reply(body=_chat_completion_response(backend, message, finish_reason))
    identity: Final = f"databricks-json-schema-{uuid.uuid4().hex}"
    deltas: Final = _TOOL_CALL_DELTAS if claude else _CONTENT_DELTAS
    frames: Final = (
        *(_frame(identity, [{"index": 0, "delta": delta, "finish_reason": None}], model=backend) for delta in deltas),
        _frame(identity, [{"index": 0, "delta": {}, "finish_reason": finish_reason}], model=backend),
        b"data: [DONE]\n\n",
    )
    return Reply(content_type="text/event-stream", chunks=frames)


def _outage_or_reply(request: Request) -> Reply:
    if _OUTAGE.encode() in request.body:
        error: Final = {"error_code": "TEMPORARILY_UNAVAILABLE", "message": _OUTAGE_MESSAGE}
        return Reply(status=503, body=json.dumps(error).encode())
    body: Final = _JSON_OBJECT.validate_json(request.body)
    return _upstream_reply(str(body["model"]), body.get("stream") is True)


def _item_text(item: _OutputItem) -> str:
    match item.type:
        case "message":
            return "".join(part.text or "" for part in item.content)
        case "function_call" if item.name == "json_tool_call":
            return item.arguments or ""
        case _:
            return ""


def _responses_text(body: _ResponsesBody) -> str:
    return "".join(_item_text(item) for item in body.output)


def _chat_delta_text(delta: _StreamDelta) -> str:
    calls: Final = delta.tool_calls or ()
    return (delta.content or "") + "".join(call.function.arguments or "" for call in calls if call.function is not None)


def _messages_delta_text(delta: _MessagesDelta) -> str:
    match delta.type:
        case "text_delta":
            return delta.text or ""
        case "input_json_delta":
            return delta.partial_json or ""
        case _:
            return ""


def _final_text(endpoint: _Endpoint, content: bytes) -> str:
    match endpoint:
        case "chat":
            return _ChatCompletion.model_validate_json(content).choices[0].message.content or ""
        case "responses":
            return _responses_text(_ResponsesBody.model_validate_json(content))
        case "messages":
            return "".join(block.text or "" for block in _MessagesBody.model_validate_json(content).content)


def _streamed_text(endpoint: _Endpoint, data: tuple[str, ...]) -> str:
    match endpoint:
        case "chat":
            assert data[-1] == "[DONE]", data
            chunks: Final = tuple(_StreamChunk.model_validate_json(line) for line in data[:-1])
            choices: Final = tuple(itertools.chain.from_iterable(chunk.choices for chunk in chunks))
            return "".join(_chat_delta_text(choice.delta) for choice in choices)
        case "responses":
            events: Final = tuple(_ResponsesEvent.model_validate_json(line) for line in data if line != "[DONE]")
            completed: Final = tuple(event.response for event in events if event.type == "response.completed")
            assert len(completed) == 1 and completed[0] is not None, data
            return _responses_text(completed[0])
        case "messages":
            message_events: Final = tuple(_MessagesEvent.model_validate_json(line) for line in data)
            assert message_events[-1].type == "message_stop", data
            deltas: Final = tuple(event.delta for event in message_events if event.delta is not None)
            return "".join(_messages_delta_text(delta) for delta in deltas)


def _outcome(status: int, text: str) -> tuple[int, JsonValue]:
    if status == 200:
        return status, _JSON_OBJECT.validate_json(text)
    return status, _OUTAGE_MESSAGE if _OUTAGE_MESSAGE in text else text


def _call(gateway: Gateway, endpoint: _Endpoint, body: Mapping[str, JsonValue]) -> tuple[int, str]:
    if body["stream"] is not True:
        response: Final = gateway.request("POST", _PATHS[endpoint], body)
        return response.status_code, _final_text(endpoint, response.content) if response.is_success else response.text
    with gateway.client.stream(
        "POST", _PATHS[endpoint], json=dict(body), headers={"Authorization": f"Bearer {gateway.key}"}
    ) as streamed:
        if not streamed.is_success:
            return streamed.status_code, streamed.read().decode()
        data: Final = tuple(line.removeprefix("data: ") for line in streamed.iter_lines() if line.startswith("data: "))
    return 200, _streamed_text(endpoint, data)


@pytest.mark.parametrize(
    ("endpoint", "backend", "stream"),
    [
        pytest.param("chat", _CLAUDE_BACKEND, True, id="chat-stream-claude"),
        pytest.param("responses", _CLAUDE_BACKEND, False, id="responses-claude"),
        pytest.param("responses", _JSON_SCHEMA_BACKEND, True, id="responses-stream-qwen"),
        pytest.param("responses", _CLAUDE_BACKEND, True, id="responses-stream-claude"),
        pytest.param("messages", _JSON_SCHEMA_BACKEND, True, id="messages-stream-qwen"),
        pytest.param("messages", _CLAUDE_BACKEND, True, id="messages-stream-claude"),
    ],
)
def test_databricks_json_schema_refs_are_preserved_on_every_endpoint(
    gateway: Gateway, endpoint: _Endpoint, backend: str, stream: bool
) -> None:
    with wire_server(lambda _request: _upstream_reply(backend, stream)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{backend}", api_base=wire.url, api_key=_API_KEY)
        status, text = _call(gateway, endpoint, _client_body(endpoint, model, stream))
        received: Final = wire.drain()
    assert len(received) == 1, received
    _assert_outbound(received[0], endpoint, backend, stream)
    assert _outcome(status, text) == (200, _EXPECTED_JSON), text


@pytest.mark.parametrize(
    "response_format",
    [
        pytest.param(
            _json_schema_format(
                {
                    "definitions": {
                        "Person": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
                    },
                    "type": "object",
                    "properties": {"p": {"$ref": "#/definitions/Person"}},
                    "required": ["p"],
                }
            ),
            id="definitions-ref",
        ),
        pytest.param(
            _json_schema_format(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "children": {"type": "array", "items": {"$ref": "#"}},
                    },
                    "required": ["name", "children"],
                }
            ),
            id="recursive-root-ref",
        ),
        pytest.param(
            _json_schema_format({"type": "object", "properties": {"$ref": {"type": "string"}}, "required": ["$ref"]}),
            id="property-named-ref",
        ),
        pytest.param(
            _json_schema_format({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
            id="flat-schema",
        ),
        pytest.param({"type": "json_object"}, id="json-object"),
    ],
)
def test_databricks_response_format_reaches_upstream_as_sent(
    gateway: Gateway, response_format: dict[str, JsonValue]
) -> None:
    with (
        wire_server(lambda _request: _upstream_reply(_JSON_SCHEMA_BACKEND, False)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"databricks/{_JSON_SCHEMA_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": _JSON_SCHEMA_PROMPT}],
                "response_format": response_format,
            },
        )
        received: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert _final_text("chat", response.content) == _JSON_CONTENT
    assert [(request.method, request.target) for request in received] == [("POST", "/chat/completions")]
    assert _JSON_OBJECT.validate_json(received[0].body)["response_format"] == response_format


def test_databricks_json_schema_without_schema_returns_the_upstream_error(gateway: Gateway) -> None:
    response_format: Final[dict[str, JsonValue]] = {"type": "json_schema", "json_schema": {"name": "P", "strict": True}}
    upstream_error: Final = {"error_code": "INVALID_PARAMETER_VALUE", "message": "json_schema.schema is required"}

    def respond(_request: Request) -> Reply:
        return Reply(status=400, body=json.dumps(upstream_error).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{_JSON_SCHEMA_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": _JSON_SCHEMA_PROMPT}],
                "response_format": response_format,
            },
        )
        received: Final = wire.drain()
    assert (response.status_code, "json_schema.schema is required" in response.text) == (400, True), response.text
    assert [(request.method, request.target) for request in received] == [("POST", "/chat/completions")]
    assert _JSON_OBJECT.validate_json(received[0].body)["response_format"] == response_format


def _assert_burst_outbound(request: Request) -> None:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    endpoint: Final[_Endpoint] = "messages" if body["response_format"] == _MESSAGES_RESPONSE_FORMAT else "chat"
    _assert_outbound(request, endpoint, _JSON_SCHEMA_BACKEND, body.get("stream") is True)


def _burst_request(index: int) -> tuple[_Endpoint, bool, bool]:
    endpoints: Final[tuple[_Endpoint, ...]] = ("chat", "responses", "messages")
    return endpoints[index % 3], index % 2 == 1, index % 5 == 2


def test_databricks_json_schema_burst_through_an_upstream_outage_keeps_every_ref(gateway: Gateway) -> None:
    plan: Final = tuple(_burst_request(index) for index in range(24))
    recovery: Final[tuple[_Endpoint, ...]] = ("chat", "responses", "messages")
    with wire_server(_outage_or_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{_JSON_SCHEMA_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        endpoints: Final = tuple(endpoint for endpoint, _stream, _down in plan)
        bodies: Final = tuple(
            _client_body(endpoint, model, stream, f"{_JSON_SCHEMA_PROMPT} {index} {_OUTAGE if down else ''}")
            for index, (endpoint, stream, down) in enumerate(plan)
        )
        with ThreadPoolExecutor(max_workers=12) as pool:
            results: Final = tuple(pool.map(_call, itertools.repeat(gateway), endpoints, bodies))
        recovered: Final = tuple(
            _call(gateway, endpoint, _client_body(endpoint, model, False)) for endpoint in recovery
        )
        received: Final = wire.drain()
    assert len(received) == len(plan) + len(recovery), received
    for request in received:
        _assert_burst_outbound(request)
    expected: Final = tuple(
        (503, _OUTAGE_MESSAGE) if down else (200, _EXPECTED_JSON) for _endpoint, _stream, down in plan
    )
    assert tuple(_outcome(*result) for result in results) == expected
    assert tuple(_outcome(*result) for result in recovered) == ((200, _EXPECTED_JSON),) * len(recovery)

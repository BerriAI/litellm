import json
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Final, Literal
from urllib.parse import unquote

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_KIMI: Final = "global.moonshotai.kimi-k3"
_GROK: Final = "us.xai.grok-4.7"
_NOVA: Final = "us.amazon.nova-lite-v1:0"
_CLAUDE: Final = "global.anthropic.claude-opus-4-8"
_PROFILE_ARN: Final = "arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/lookaround0"
_AWS: Final[dict[str, JsonValue]] = {
    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
    "aws_secret_access_key": "scripted-secret",
    "aws_region_name": "us-east-1",
}
_NO_CACHE: Final[dict[str, JsonValue]] = {"cache": {"no-cache": True}, "num_retries": 0}
_LOOKAHEAD: Final = r"^(?!\.\.?(?:\/|$))[A-Za-z0-9_\-.~:@+]{1,200}$"
_NEGATIVE_LOOKBEHIND: Final = r"^(?<!tmp_)[a-z]+$"
_POSITIVE_LOOKBEHIND: Final = r"(?<=v)[0-9]+"
_POSITIVE_LOOKAHEAD_KEY: Final = r"^x_(?=[a-z])"
_PLAIN: Final = r"^[a-z][a-z0-9_]*$"
_TOOL: Final = "ArtifactData"
_PLAIN_TOOL: Final = "ListNotes"
_PROMPT: Final = "Read the notes document from the notes collection."
_ANSWER: Final = "lookaround regex control answer"
_TOOL_INPUT: Final[dict[str, JsonValue]] = {"collection": "notes", "doc_id": "notes"}
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_BEDROCK_REJECTION: Final = "structured output schema uses unsupported regex negative look-ahead"
_USAGE: Final[dict[str, JsonValue]] = {"inputTokens": 21, "outputTokens": 7, "totalTokens": 28}
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_LIST: Final = TypeAdapter(list[JsonValue])
_SCHEMA_AS_SENT: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {
        "collection": {"type": "string", "description": "Collection name", "pattern": _LOOKAHEAD},
        "doc_id": {"type": "string", "description": "Document id", "pattern": _PLAIN},
        "filters": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string", "pattern": _NEGATIVE_LOOKBEHIND},
                    {"type": "string", "pattern": _POSITIVE_LOOKBEHIND},
                ]
            },
        },
        "labels": {
            "type": "object",
            "patternProperties": {_POSITIVE_LOOKAHEAD_KEY: {"type": "string"}, "^v_": {"type": "integer"}},
            "additionalProperties": False,
        },
        "meta": {"type": "object", "default": {"pattern": _LOOKAHEAD}},
    },
    "required": ["collection"],
    "additionalProperties": False,
}
_SCHEMA_LOOKAROUND_FREE: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {
        "collection": {"type": "string", "description": "Collection name"},
        "doc_id": {"type": "string", "description": "Document id", "pattern": _PLAIN},
        "filters": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "string"}]}},
        "labels": {
            "type": "object",
            "patternProperties": {"^v_": {"type": "integer"}},
            "additionalProperties": {"type": "string"},
        },
        "meta": {"type": "object", "default": {"pattern": _LOOKAHEAD}},
    },
    "required": ["collection"],
    "additionalProperties": False,
}
_PLAIN_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"limit": {"type": "integer", "minimum": 1}, "prefix": {"type": "string", "pattern": _PLAIN}},
    "required": ["limit"],
    "additionalProperties": False,
}


def _converse_root(schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "type": schema["type"],
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


_WIRE_AS_SENT: Final = _converse_root(_SCHEMA_AS_SENT)
_WIRE_LOOKAROUND_FREE: Final = _converse_root(_SCHEMA_LOOKAROUND_FREE)
_WIRE_PLAIN: Final = _converse_root(_PLAIN_SCHEMA)

Endpoint = Literal["chat", "messages", "responses"]
_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")


def _frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


_TOOL_USE_RESPONSE: Final = json.dumps(
    {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "tooluse_lookaround_1", "name": _TOOL, "input": _TOOL_INPUT}}],
            }
        },
        "stopReason": "tool_use",
        "usage": _USAGE,
        "metrics": {"latencyMs": 1},
    }
).encode()
_STREAM_FRAMES: Final = b"".join(
    (
        _frame("messageStart", {"role": "assistant"}),
        _frame("contentBlockDelta", {"delta": {"text": _ANSWER}, "contentBlockIndex": 0}),
        _frame("contentBlockStop", {"contentBlockIndex": 0}),
        _frame("messageStop", {"stopReason": "end_turn"}),
        _frame("metadata", {"usage": _USAGE}),
    )
)


def _bedrock_peer(request: Request) -> Reply:
    if unquote(request.target).endswith("/converse-stream"):
        return Reply(body=_STREAM_FRAMES, content_type=_EVENT_STREAM)
    return Reply(body=_TOOL_USE_RESPONSE)


def _rejecting_peer(request: Request) -> Reply:
    return Reply(status=400, body=json.dumps({"message": _BEDROCK_REJECTION}).encode())


def _openai_tool(name: str, schema: Mapping[str, JsonValue], **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {"name": name, "description": f"{name} tool", "parameters": dict(schema), **extra},
    }


def _anthropic_tool(name: str, schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"name": name, "description": f"{name} tool", "input_schema": dict(schema)}


def _responses_tool(name: str, schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "function", "name": name, "description": f"{name} tool", "parameters": dict(schema)}


def _tool_for(endpoint: Endpoint, name: str, schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return _openai_tool(name, schema)
        case "messages":
            return _anthropic_tool(name, schema)
        case "responses":
            return _responses_tool(name, schema)


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(
    endpoint: Endpoint,
    model: str,
    tools: Sequence[Mapping[str, JsonValue]],
    *,
    stream: bool = False,
    **extra: JsonValue,
) -> dict[str, JsonValue]:
    tool_list: Final[list[JsonValue]] = [dict(tool) for tool in tools]
    match endpoint:
        case "chat":
            return {
                "model": model,
                "messages": [{"role": "user", "content": _PROMPT}],
                "max_tokens": 64,
                "stream": stream,
                "tools": tool_list,
                **_NO_CACHE,
                **extra,
            }
        case "messages":
            return {
                "model": model,
                "messages": [{"role": "user", "content": _PROMPT}],
                "max_tokens": 64,
                "stream": stream,
                "tools": tool_list,
                **_NO_CACHE,
                **extra,
            }
        case "responses":
            return {
                "model": model,
                "input": _PROMPT,
                "max_output_tokens": 64,
                "stream": stream,
                "tools": tool_list,
                **_NO_CACHE,
                **extra,
            }


def _deployment(
    scenario: Scenario,
    wire: Wire,
    model: str,
    *,
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> str:
    return scenario.model(model=model, api_base=wire.url, **_AWS, **params, model_info=model_info)


def _received_specs(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    received: Final = wire.drain()
    assert len(received) == 1, [request.target for request in received]
    body: Final = _JSON.validate_json(received[0].body)
    tools: Final = _LIST.validate_python(object_value(body["toolConfig"])["tools"])
    return tuple(object_value(object_value(tool)["toolSpec"]) for tool in tools)


def _schema_of(spec: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(object_value(spec["inputSchema"])["json"])


def _only_schema(wire: Wire) -> dict[str, JsonValue]:
    (spec,) = _received_specs(wire)
    assert spec["name"] == _TOOL, spec
    return _schema_of(spec)


def _assert_tool_call_relayed(endpoint: Endpoint, response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    body: Final = _JSON.validate_json(response.content)
    match endpoint:
        case "chat":
            message: Final = object_value(object_value(_LIST.validate_python(body["choices"])[0])["message"])
            (call,) = _LIST.validate_python(message["tool_calls"])
            function: Final = object_value(object_value(call)["function"])
            assert function["name"] == _TOOL and json.loads(string_value(function["arguments"])) == _TOOL_INPUT, (
                response.text
            )
        case "messages":
            blocks: Final = tuple(object_value(block) for block in _LIST.validate_python(body["content"]))
            (tool_use,) = tuple(block for block in blocks if block.get("type") == "tool_use")
            assert tool_use["name"] == _TOOL and tool_use["input"] == _TOOL_INPUT, response.text
        case "responses":
            items: Final = tuple(object_value(item) for item in _LIST.validate_python(body["output"]))
            (call_item,) = tuple(item for item in items if item.get("type") == "function_call")
            assert call_item["name"] == _TOOL and json.loads(string_value(call_item["arguments"])) == _TOOL_INPUT, (
                response.text
            )


def _stream_text(gateway: Gateway, endpoint: Endpoint, body: Mapping[str, JsonValue]) -> str:
    headers: Final = {"Authorization": f"Bearer {gateway.key}"}
    with gateway.client.stream("POST", _path(endpoint), json=body, headers=headers) as response:
        lines: Final = tuple(line for line in response.iter_lines() if line)
    assert response.status_code == 200, "\n".join(lines)
    return "\n".join(lines)


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _schema_sent_through(
    gateway: Gateway, wire: Wire, endpoint: Endpoint, model: str, tool: Mapping[str, JsonValue], **extra: JsonValue
) -> dict[str, JsonValue]:
    response: Final = gateway.request("POST", _path(endpoint), _body(endpoint, model, (tool,), **extra))
    _assert_tool_call_relayed(endpoint, response)
    return _only_schema(wire)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_flagged_model_receives_a_lookaround_free_schema_and_the_tool_call_comes_back(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        tool: Final = _tool_for(endpoint, _TOOL, _SCHEMA_AS_SENT)
        assert _schema_sent_through(gateway, wire, endpoint, model, tool) == _WIRE_LOOKAROUND_FREE


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_flagged_model_streams_after_the_schema_lost_its_lookarounds(gateway: Gateway, endpoint: Endpoint) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        tool: Final = _tool_for(endpoint, _TOOL, _SCHEMA_AS_SENT)
        streamed: Final = _stream_text(gateway, endpoint, _body(endpoint, model, (tool,), stream=True))
        assert _ANSWER in streamed, streamed
        received: Final = wire.drain()
        assert len(received) == 1 and unquote(received[0].target).endswith("/converse-stream"), received
        (tool_block,) = _LIST.validate_python(
            object_value(_JSON.validate_json(received[0].body)["toolConfig"])["tools"]
        )
        assert _schema_of(object_value(object_value(tool_block)["toolSpec"])) == _WIRE_LOOKAROUND_FREE


def test_openai_sdk_sync_chat_sends_a_lookaround_free_schema(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        client: Final = _openai_client(gateway)
        completion: Final = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _PROMPT}],
            tools=[_openai_tool(_TOOL, _SCHEMA_AS_SENT)],
            max_tokens=64,
            extra_body=_NO_CACHE,
        )
        (call,) = completion.choices[0].message.tool_calls or ()
        assert call.function.name == _TOOL and json.loads(call.function.arguments) == _TOOL_INPUT
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE
        chunks: Final = tuple(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _PROMPT}],
                tools=[_openai_tool(_TOOL, _SCHEMA_AS_SENT)],
                max_tokens=64,
                stream=True,
                extra_body=_NO_CACHE,
            )
        )
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == _ANSWER
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE


async def test_openai_sdk_async_chat_sends_a_lookaround_free_schema(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        client: Final = _async_openai_client(gateway)
        completion: Final = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _PROMPT}],
            tools=[_openai_tool(_TOOL, _SCHEMA_AS_SENT)],
            max_tokens=64,
            extra_body=_NO_CACHE,
        )
        (call,) = completion.choices[0].message.tool_calls or ()
        assert call.function.name == _TOOL and json.loads(call.function.arguments) == _TOOL_INPUT
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE
        stream: Final = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _PROMPT}],
            tools=[_openai_tool(_TOOL, _SCHEMA_AS_SENT)],
            max_tokens=64,
            stream=True,
            extra_body=_NO_CACHE,
        )
        text: Final = "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])
        assert text == _ANSWER
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE


def test_anthropic_sdk_sync_messages_send_a_lookaround_free_schema(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        message: Final = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": _PROMPT}],
            tools=[_anthropic_tool(_TOOL, _SCHEMA_AS_SENT)],
            extra_body=_NO_CACHE,
        )
        (tool_use,) = tuple(block for block in message.content if block.type == "tool_use")
        assert tool_use.name == _TOOL and tool_use.input == _TOOL_INPUT
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE
        with client.messages.stream(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": _PROMPT}],
            tools=[_anthropic_tool(_TOOL, _SCHEMA_AS_SENT)],
            extra_body=_NO_CACHE,
        ) as stream:
            text: Final = "".join(stream.text_stream)
        assert text == _ANSWER
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE


async def test_anthropic_sdk_async_messages_send_a_lookaround_free_schema(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
        )
        message: Final = await client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": _PROMPT}],
            tools=[_anthropic_tool(_TOOL, _SCHEMA_AS_SENT)],
            extra_body=_NO_CACHE,
        )
        (tool_use,) = tuple(block for block in message.content if block.type == "tool_use")
        assert tool_use.name == _TOOL and tool_use.input == _TOOL_INPUT
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE
        async with client.messages.stream(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": _PROMPT}],
            tools=[_anthropic_tool(_TOOL, _SCHEMA_AS_SENT)],
            extra_body=_NO_CACHE,
        ) as stream:
            text: Final = "".join([piece async for piece in stream.text_stream])
        assert text == _ANSWER
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE


def test_openai_sdk_sync_responses_send_a_lookaround_free_schema(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        client: Final = _openai_client(gateway)
        response: Final = client.responses.create(
            model=model,
            input=_PROMPT,
            tools=[_responses_tool(_TOOL, _SCHEMA_AS_SENT)],
            max_output_tokens=64,
            extra_body=_NO_CACHE,
        )
        (call,) = tuple(item for item in response.output if item.type == "function_call")
        assert call.name == _TOOL and json.loads(call.arguments) == _TOOL_INPUT
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE
        events: Final = tuple(
            client.responses.create(
                model=model,
                input=_PROMPT,
                tools=[_responses_tool(_TOOL, _SCHEMA_AS_SENT)],
                max_output_tokens=64,
                stream=True,
                extra_body=_NO_CACHE,
            )
        )
        deltas: Final = "".join(event.delta for event in events if event.type == "response.output_text.delta")
        assert deltas == _ANSWER
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE


async def test_openai_sdk_async_responses_send_a_lookaround_free_schema(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        client: Final = _async_openai_client(gateway)
        response: Final = await client.responses.create(
            model=model,
            input=_PROMPT,
            tools=[_responses_tool(_TOOL, _SCHEMA_AS_SENT)],
            max_output_tokens=64,
            extra_body=_NO_CACHE,
        )
        (call,) = tuple(item for item in response.output if item.type == "function_call")
        assert call.name == _TOOL and json.loads(call.arguments) == _TOOL_INPUT
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE
        stream: Final = await client.responses.create(
            model=model,
            input=_PROMPT,
            tools=[_responses_tool(_TOOL, _SCHEMA_AS_SENT)],
            max_output_tokens=64,
            stream=True,
            extra_body=_NO_CACHE,
        )
        deltas: Final = "".join([event.delta async for event in stream if event.type == "response.output_text.delta"])
        assert deltas == _ANSWER
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE


def test_grok_on_the_explicit_converse_route_is_flagged_too(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/converse/{_GROK}")
        tool: Final = _openai_tool(_TOOL, _SCHEMA_AS_SENT)
        assert _schema_sent_through(gateway, wire, "chat", model, tool) == _WIRE_LOOKAROUND_FREE


def test_a_tool_without_lookarounds_beside_a_cleaned_one_is_forwarded_untouched(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        tools: Final = (_openai_tool(_TOOL, _SCHEMA_AS_SENT), _openai_tool(_PLAIN_TOOL, _PLAIN_SCHEMA))
        response: Final = gateway.request("POST", _path("chat"), _body("chat", model, tools))
        _assert_tool_call_relayed("chat", response)
        cleaned, plain = _received_specs(wire)
        assert (cleaned["name"], _schema_of(cleaned)) == (_TOOL, _WIRE_LOOKAROUND_FREE)
        assert plain == {
            "name": _PLAIN_TOOL,
            "description": f"{_PLAIN_TOOL} tool",
            "inputSchema": {"json": _WIRE_PLAIN},
        }, plain


@pytest.mark.parametrize("model_id", (_NOVA, _CLAUDE))
def test_models_without_the_flag_keep_their_schema_as_sent(gateway: Gateway, model_id: str) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/converse/{model_id}")
        tool: Final = _openai_tool(_TOOL, _SCHEMA_AS_SENT)
        assert _schema_sent_through(gateway, wire, "chat", model, tool) == _WIRE_AS_SENT


@pytest.mark.parametrize(
    ("model_id", "model_info", "params", "expected"),
    (
        (_KIMI, {"supports_regex_lookaround": True}, {}, _WIRE_AS_SENT),
        (_NOVA, {"supports_regex_lookaround": False}, {}, _WIRE_LOOKAROUND_FREE),
        (_PROFILE_ARN, None, {"base_model": f"bedrock/{_KIMI}"}, _WIRE_LOOKAROUND_FREE),
        (_PROFILE_ARN, None, {}, _WIRE_AS_SENT),
        (_KIMI, {"supports_regex_lookaround": None}, {}, _WIRE_LOOKAROUND_FREE),
        (_NOVA, {"supports_regex_lookaround": "false"}, {}, _WIRE_AS_SENT),
        (_PROFILE_ARN, {"supports_regex_lookaround": True}, {"base_model": f"bedrock/{_KIMI}"}, _WIRE_AS_SENT),
        (_KIMI, None, {"base_model": ""}, _WIRE_LOOKAROUND_FREE),
    ),
    ids=(
        "deployment-true-wins-over-map",
        "deployment-false-flags-an-unflagged-model",
        "base-model-flags-a-profile-arn",
        "bare-profile-arn-keeps-the-schema",
        "null-falls-back-to-the-map",
        "string-false-is-not-a-flag",
        "deployment-true-wins-over-base-model",
        "empty-base-model-falls-back-to-the-model",
    ),
)
def test_deployment_settings_decide_before_the_cost_map(
    gateway: Gateway,
    model_id: str,
    model_info: Mapping[str, JsonValue] | None,
    params: Mapping[str, JsonValue],
    expected: Mapping[str, JsonValue],
) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{model_id}", model_info=model_info, **params)
        tool: Final = _openai_tool(_TOOL, _SCHEMA_AS_SENT)
        assert _schema_sent_through(gateway, wire, "chat", model, tool) == expected


@pytest.mark.parametrize(
    ("model_id", "flag", "expected_for_the_bare_sibling"),
    ((_KIMI, True, _WIRE_LOOKAROUND_FREE), (_NOVA, False, _WIRE_AS_SENT)),
    ids=("kimi-sibling-keeps-the-map-false", "nova-sibling-keeps-the-map-absence"),
)
@pytest.mark.parametrize("flagged_first", (True, False), ids=("flagged-registered-first", "bare-registered-first"))
def test_a_deployment_flag_never_reaches_its_sibling_on_the_same_model(
    gateway: Gateway,
    model_id: str,
    flag: bool,
    expected_for_the_bare_sibling: Mapping[str, JsonValue],
    flagged_first: bool,
) -> None:
    flag_info: Final[dict[str, JsonValue]] = {"supports_regex_lookaround": flag}
    first_info, second_info = (flag_info, None) if flagged_first else (None, flag_info)
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        first: Final = _deployment(scenario, wire, f"bedrock/{model_id}", model_info=first_info)
        second: Final = _deployment(scenario, wire, f"bedrock/{model_id}", model_info=second_info)
        bare: Final = second if flagged_first else first
        tool: Final = _openai_tool(_TOOL, _SCHEMA_AS_SENT)
        assert _schema_sent_through(gateway, wire, "chat", bare, tool) == expected_for_the_bare_sibling


@pytest.mark.parametrize(
    ("model_id", "body_base_model", "expected"),
    ((_NOVA, f"bedrock/{_KIMI}", _WIRE_LOOKAROUND_FREE), (_KIMI, f"bedrock/{_NOVA}", _WIRE_LOOKAROUND_FREE)),
    ids=("client-base-model-can-loosen-an-unflagged-deployment", "client-base-model-cannot-restore-a-flagged-one"),
)
def test_a_base_model_in_the_request_body_only_ever_loosens(
    gateway: Gateway, model_id: str, body_base_model: str, expected: Mapping[str, JsonValue]
) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{model_id}")
        tool: Final = _openai_tool(_TOOL, _SCHEMA_AS_SENT)
        assert _schema_sent_through(gateway, wire, "chat", model, tool, base_model=body_base_model) == expected


@pytest.mark.parametrize(
    ("subschema", "expected"),
    (
        (
            {
                "type": "object",
                "patternProperties": {_POSITIVE_LOOKAHEAD_KEY: {"type": "string"}, r"^y_(?!z)": {"type": "integer"}},
                "additionalProperties": False,
            },
            {
                "type": "object",
                "patternProperties": {},
                "additionalProperties": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
        ),
        (
            {"type": "object", "patternProperties": {_POSITIVE_LOOKAHEAD_KEY: {"type": "string"}}},
            {"type": "object", "patternProperties": {}},
        ),
        (
            {"type": "object", "properties": {"name": {"type": "string", "pattern": r"\(?=x"}}},
            {"type": "object", "properties": {"name": {"type": "string"}}},
        ),
        (
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "dependencies": {"name": {"properties": {"alias": {"type": "string", "pattern": _LOOKAHEAD}}}},
            },
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "dependencies": {"name": {"properties": {"alias": {"type": "string", "pattern": _LOOKAHEAD}}}},
            },
        ),
    ),
    ids=(
        "two-dropped-pattern-properties-become-an-anyof",
        "an-open-object-just-loses-the-key",
        "an-escaped-literal-spelling-an-opener-is-dropped-too",
        "draft-07-dependencies-are-not-walked",
    ),
)
def test_schema_shapes_at_the_edges_of_the_walk(
    gateway: Gateway, subschema: Mapping[str, JsonValue], expected: Mapping[str, JsonValue]
) -> None:
    schema: Final[dict[str, JsonValue]] = {"type": "object", "properties": {"labels": dict(subschema)}}
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        assert _schema_sent_through(gateway, wire, "chat", model, _openai_tool(_TOOL, schema)) == {
            "type": "object",
            "properties": {"labels": dict(expected)},
            "required": [],
        }


def test_strict_is_still_withheld_from_a_flagged_non_anthropic_model(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        tool: Final = _openai_tool(_TOOL, _SCHEMA_AS_SENT, strict=True)
        response: Final = gateway.request("POST", _path("chat"), _body("chat", model, (tool,)))
        _assert_tool_call_relayed("chat", response)
        (spec,) = _received_specs(wire)
        assert spec == {"name": _TOOL, "description": f"{_TOOL} tool", "inputSchema": {"json": _WIRE_LOOKAROUND_FREE}}


def test_a_json_schema_response_format_rides_the_same_tool_path(gateway: Gateway) -> None:
    schema: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {"collection": {"type": "string", "pattern": _LOOKAHEAD}},
        "required": ["collection"],
    }
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        response: Final = gateway.request(
            "POST",
            _path("chat"),
            {
                "model": model,
                "messages": [{"role": "user", "content": _PROMPT}],
                "max_tokens": 64,
                "response_format": {"type": "json_schema", "json_schema": {"name": "document", "schema": schema}},
                **_NO_CACHE,
            },
        )
        assert response.status_code == 200, response.text
        (spec,) = _received_specs(wire)
        assert spec["name"] == "json_tool_call", spec
        assert _schema_of(spec) == {
            "type": "object",
            "properties": {"collection": {"type": "string"}},
            "required": ["collection"],
        }, spec


@pytest.mark.parametrize(
    ("pattern", "expected_property"),
    (
        (5, {"type": "string", "pattern": 5}),
        ([_LOOKAHEAD], {"type": "string", "pattern": [_LOOKAHEAD]}),
        ("", {"type": "string", "pattern": ""}),
        ("a" * 5120, {"type": "string", "pattern": "a" * 5120}),
        ("a" * 5120 + "(?=b)", {"type": "string"}),
    ),
    ids=("int", "list", "empty", "5kb-plain", "5kb-ending-in-a-lookahead"),
)
def test_odd_pattern_values_are_forwarded_unless_they_are_a_lookaround_string(
    gateway: Gateway, pattern: JsonValue, expected_property: Mapping[str, JsonValue]
) -> None:
    schema: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {
            "collection": {"type": "string", "pattern": pattern},
            "doc_id": {"type": "string", "pattern": pattern},
        },
    }
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        assert _schema_sent_through(gateway, wire, "chat", model, _openai_tool(_TOOL, schema)) == {
            "type": "object",
            "properties": {"collection": dict(expected_property), "doc_id": dict(expected_property)},
            "required": [],
        }


@pytest.mark.parametrize(
    "parameters",
    (None, {"type": "object", "properties": [{"name": "collection", "pattern": _LOOKAHEAD}]}),
    ids=("null-parameters", "properties-as-a-list"),
)
def test_malformed_tool_parameters_never_take_the_proxy_down(gateway: Gateway, parameters: JsonValue) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        tool: Final[dict[str, JsonValue]] = {
            "type": "function",
            "function": {"name": _TOOL, "description": f"{_TOOL} tool", "parameters": parameters},
        }
        response: Final = gateway.request("POST", _path("chat"), _body("chat", model, (tool,)))
        assert response.status_code in (200, 400), response.text
        if response.status_code == 400:
            assert "error" in _JSON.validate_json(response.content), response.text
        wire.drain()
        control: Final = gateway.request(
            "POST", _path("chat"), _body("chat", model, (_openai_tool(_TOOL, _SCHEMA_AS_SENT),))
        )
        _assert_tool_call_relayed("chat", control)
        assert _only_schema(wire) == _WIRE_LOOKAROUND_FREE


def test_an_unauthenticated_request_never_reaches_the_peer(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        response: Final = gateway.request(
            "POST", _path("chat"), _body("chat", model, (_openai_tool(_TOOL, _SCHEMA_AS_SENT),)), key="sk-not-a-key"
        )
        assert response.status_code == 401, response.text
        assert wire.drain() == ()


def test_a_bedrock_rejection_of_an_unflagged_model_reaches_the_caller(gateway: Gateway) -> None:
    with wire_server(_rejecting_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/converse/{_CLAUDE}")
        response: Final = gateway.request(
            "POST", _path("chat"), _body("chat", model, (_openai_tool(_TOOL, _SCHEMA_AS_SENT),))
        )
        assert response.status_code == 400, response.text
        assert _BEDROCK_REJECTION in response.text, response.text
        assert _only_schema(wire) == _WIRE_AS_SENT


@pytest.mark.timeout(120)
def test_the_worst_case_lookaround_input_scans_in_linear_time(gateway: Gateway) -> None:
    pattern: Final = "(?<" * (2 * 1024 * 1024 // 3)
    schema: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {"collection": {"type": "string", "pattern": pattern}},
    }
    liveliness: Final[list[tuple[float, int]]] = []
    stop: Final = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            liveliness.append(_timed_liveliness(gateway))

    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}")
        poller: Final = threading.Thread(target=poll)
        poller.start()
        started: Final = time.perf_counter()
        response: Final = gateway.request("POST", _path("chat"), _body("chat", model, (_openai_tool(_TOOL, schema),)))
        elapsed: Final = time.perf_counter() - started
        stop.set()
        poller.join()
        _assert_tool_call_relayed("chat", response)
        assert elapsed < 30, elapsed
        assert liveliness and max(latency for latency, _ in liveliness) < 5, liveliness
        assert {status for _, status in liveliness} == {200}, liveliness
        assert len(wire.drain()) == 1


def _timed_liveliness(gateway: Gateway) -> tuple[float, int]:
    started: Final = time.perf_counter()
    probe: Final = gateway.client.get("/health/liveliness")
    return time.perf_counter() - started, probe.status_code


def _model_id(gateway: Gateway, name: str) -> str:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    (identity,) = (
        string_value(object_value(object_value(entry)["model_info"])["id"])
        for entry in entries
        if object_value(entry)["model_name"] == name
    )
    return identity


def _settled_schema(gateway: Gateway, wire: Wire, model: str, expected: Mapping[str, JsonValue]) -> None:
    tool: Final = _openai_tool(_TOOL, _SCHEMA_AS_SENT)
    eventually(
        lambda: tuple(_schema_sent_through(gateway, wire, "chat", model, tool) for _ in range(8)),
        lambda schemas: all(schema == expected for schema in schemas),
        seconds=90,
    )


def _patch_flag(gateway: Gateway, identity: str, flag: bool) -> None:
    patched: Final = gateway.request(
        "PATCH", f"/model/{identity}/update", {"model_info": {"supports_regex_lookaround": flag}}
    )
    assert patched.status_code == 200, patched.text


@pytest.mark.timeout(300)
def test_updating_the_flag_on_a_live_deployment_takes_effect_without_a_restart(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/{_KIMI}", model_info={"supports_regex_lookaround": True})
        _settled_schema(gateway, wire, model, _WIRE_AS_SENT)
        identity: Final = _model_id(gateway, model)
        _patch_flag(gateway, identity, False)
        _settled_schema(gateway, wire, model, _WIRE_LOOKAROUND_FREE)
        _patch_flag(gateway, identity, True)
        _settled_schema(gateway, wire, model, _WIRE_AS_SENT)

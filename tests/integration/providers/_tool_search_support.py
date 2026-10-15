import json
import re
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from itertools import chain
from typing import Final, Literal
from urllib.parse import unquote, urlsplit

import httpx
from integration._support.anthropic_thinking import (
    JSON_LIST,
    JSON_OBJECT,
    NO_CACHE,
    chunks_of,
    deltas_of,
    stream_reply,
    streams,
)
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request
from pydantic import JsonValue

Event = dict[str, JsonValue]
Provider = Literal[
    "anthropic", "bedrock_invoke", "claude_platform", "bedrock_mantle", "vertex_ai", "azure_ai", "snowflake"
]
Responder = Callable[[Request], Reply]

MODEL: Final = "claude-opus-5-5"
BEDROCK_MODEL: Final = "anthropic.claude-opus-5-5"
SEARCH_ID: Final = "srvtoolu_01ScriptedSearch"
WEATHER_ID: Final = "toolu_01ScriptedWeather"
SEARCH_TOOL: Final = "tool_search_tool_bm25"
SEARCH_INPUT: Final[Event] = {"query": "weather"}
WEATHER_INPUT: Final[Event] = {"city": "Paris"}
WEATHER_ANSWER: Final = "18C and cloudy"
VERTEX_PROJECT: Final = "scripted-project"
VERTEX_LOCATION: Final = "us-east5"
VERTEX_MODEL_PATH: Final = (
    f"/v1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/publishers/anthropic/models/{MODEL}"
)
TOOLS: Final[list[JsonValue]] = [
    {"type": "tool_search_tool_bm25_20251119", "name": SEARCH_TOOL},
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
        "defer_loading": True,
    },
]
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")


def search_result(tool_use_id: str = SEARCH_ID) -> Event:
    return {
        "type": "tool_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": {
            "type": "tool_search_tool_search_result",
            "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
        },
    }


SEARCH_RESULT: Final = search_result()
SEARCH_USE: Final[Event] = {"type": "server_tool_use", "id": SEARCH_ID, "name": SEARCH_TOOL, "input": SEARCH_INPUT}
WEATHER_USE: Final[Event] = {"type": "tool_use", "id": WEATHER_ID, "name": "get_weather", "input": WEATHER_INPUT}
SEARCH_TURN: Final[tuple[Event, ...]] = (SEARCH_USE, SEARCH_RESULT, WEATHER_USE)


def question(marker: str) -> str:
    return f"What is the weather in Paris? marker-{marker}"


def answer(marker: str) -> str:
    return f"{WEATHER_ANSWER} for marker-{marker}"


def marker_of(request: Request) -> str:
    found: Final = _MARKER.findall(request.body.decode())
    assert found, request.body
    return found[0]


def anthropic_message(identity: str, content: Sequence[Event], stop_reason: str | None) -> Event:
    return {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": list(content),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 40, "output_tokens": 12},
    }


def _block_events(index: int, block: Event) -> tuple[Event, ...]:
    stop: Final[Event] = {"type": "content_block_stop", "index": index}
    match block["type"]:
        case "server_tool_use" | "tool_use":
            return (
                {"type": "content_block_start", "index": index, "content_block": {**block, "input": {}}},
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
                },
                stop,
            )
        case "text":
            return (
                {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}},
                stop,
            )
        case _:
            return ({"type": "content_block_start", "index": index, "content_block": block}, stop)


def message_events(identity: str, blocks: Sequence[Event], stop_reason: str) -> tuple[Event, ...]:
    return (
        {"type": "message_start", "message": anthropic_message(identity, (), None)},
        *chain.from_iterable(_block_events(index, block) for index, block in enumerate(blocks)),
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 12},
        },
        {"type": "message_stop"},
    )


def turn_reply(
    request: Request, identity: str, blocks: Sequence[Event], stop_reason: str, *, abort_after: int | None = None
) -> Reply:
    if streams(request):
        return stream_reply(request, message_events(identity, blocks, stop_reason), abort_after=abort_after)
    return Reply(body=json.dumps(anthropic_message(identity, blocks, stop_reason)).encode())


def request_messages(request: Request) -> tuple[Event, ...]:
    messages: Final = JSON_LIST.validate_python(JSON_OBJECT.validate_json(request.body)["messages"])
    return tuple(JSON_OBJECT.validate_python(message) for message in messages)


def answer_reply(request: Request) -> Reply:
    marker: Final = marker_of(request)
    return turn_reply(request, f"msg_{marker}_answer", ({"type": "text", "text": answer(marker)},), "end_turn")


def scripted_peer(first_turn: Sequence[Event] = SEARCH_TURN) -> Responder:
    def respond(request: Request) -> Reply:
        if len(request_messages(request)) > 1:
            return answer_reply(request)
        return turn_reply(request, f"msg_{marker_of(request)}_search", first_turn, "tool_use")

    return respond


def deployment(scenario: Scenario, provider: Provider, wire_url: str, upstream_url: str) -> str:
    match provider:
        case "anthropic":
            return scenario.model(model=f"anthropic/{MODEL}", api_base=wire_url, api_key="scripted-anthropic-key")
        case "bedrock_invoke":
            return scenario.model(
                model=f"bedrock/invoke/{BEDROCK_MODEL}",
                api_base=wire_url,
                aws_access_key_id="AKIASCRIPTEDPROVIDER",
                aws_secret_access_key="scripted-secret",
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire_url,
            )
        case "claude_platform":
            return scenario.model(
                model=f"bedrock/claude_platform/{MODEL}",
                api_base=wire_url,
                api_key="scripted-platform-key",
                aws_region_name="us-east-1",
                workspace_id="scripted-workspace",
            )
        case "bedrock_mantle":
            return scenario.model(
                model=f"bedrock_mantle/{BEDROCK_MODEL}", api_base=wire_url, api_key="scripted-mantle-key"
            )
        case "vertex_ai":
            return scenario.model(
                model=f"vertex_ai/{MODEL}",
                api_base=f"{wire_url}{VERTEX_MODEL_PATH}",
                api_key=None,
                vertex_project=VERTEX_PROJECT,
                vertex_location=VERTEX_LOCATION,
                vertex_credentials=service_account_json(VERTEX_PROJECT, upstream_url.rstrip("/")),
            )
        case "azure_ai":
            return scenario.model(model=f"azure_ai/{MODEL}", api_base=wire_url, api_key="scripted-azure-key")
        case "snowflake":
            return scenario.model(model=f"snowflake/{MODEL}", api_base=wire_url, api_key="scripted-snowflake-key")


def target(provider: Provider, stream: bool) -> str:
    match provider:
        case "anthropic" | "claude_platform":
            return "/v1/messages"
        case "bedrock_invoke":
            return f"/model/{BEDROCK_MODEL}/invoke-with-response-stream" if stream else f"/model/{BEDROCK_MODEL}/invoke"
        case "bedrock_mantle" | "azure_ai":
            return "/anthropic/v1/messages"
        case "vertex_ai":
            return f"{VERTEX_MODEL_PATH}:streamRawPredict" if stream else f"{VERTEX_MODEL_PATH}:rawPredict"
        case "snowflake":
            return "/api/v2/cortex/v1/messages"


def target_path(request: Request) -> str:
    return urlsplit(unquote(request.target)).path


def chat_body(model: str, messages: Sequence[JsonValue], stream: bool, **fields: JsonValue) -> Event:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": list(messages),
        "tools": TOOLS,
        "stream": stream,
        **NO_CACHE,
        **fields,
    }


def chat(gateway: Gateway, body: Mapping[str, JsonValue], *, key: str | None = None) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", body, key=key)


def first_messages(marker: str) -> list[JsonValue]:
    return [{"role": "user", "content": question(marker)}]


def tool_answer(tool_call_id: JsonValue, content: str = WEATHER_ANSWER) -> Event:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def replay_messages(
    marker: str, assistant: Event, answers: Sequence[JsonValue] = (tool_answer(WEATHER_ID),)
) -> list[JsonValue]:
    return [{"role": "user", "content": question(marker)}, assistant, *answers]


def _stream_tool_call_parts(deltas: Sequence[Event]) -> tuple[Event, ...]:
    calls: Final = chain.from_iterable(JSON_LIST.validate_python(delta.get("tool_calls") or []) for delta in deltas)
    return tuple(JSON_OBJECT.validate_python(call) for call in calls)


def _folded_tool_call(parts: Sequence[Event], index: JsonValue) -> Event:
    mine: Final = tuple(part for part in parts if part["index"] == index)
    functions: Final = tuple(JSON_OBJECT.validate_python(part.get("function") or {}) for part in mine)
    return {
        "id": next(part["id"] for part in mine if part.get("id")),
        "type": "function",
        "function": {
            "name": "".join(str(function.get("name") or "") for function in functions),
            "arguments": "".join(str(function.get("arguments") or "") for function in functions),
        },
    }


def _merged_fields(merged: Event, delta: Event) -> Event:
    return {**merged, **JSON_OBJECT.validate_python(delta.get("provider_specific_fields") or {})}


def folded_assistant(deltas: Sequence[Event]) -> Event:
    parts: Final = _stream_tool_call_parts(deltas)
    indexes: Final = tuple(dict.fromkeys(part["index"] for part in parts))
    content: Final = "".join(str(delta.get("content") or "") for delta in deltas)
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [_folded_tool_call(parts, index) for index in indexes],
        "provider_specific_fields": reduce(_merged_fields, deltas, {}),
    }


def assistant_from(response: httpx.Response, stream: bool) -> Event:
    if stream:
        return folded_assistant(deltas_of(chunks_of(response.text)))
    choices: Final = JSON_LIST.validate_python(JSON_OBJECT.validate_json(response.content)["choices"])
    return JSON_OBJECT.validate_python(JSON_OBJECT.validate_python(choices[0])["message"])


def response_id(response: httpx.Response, stream: bool) -> str:
    if stream:
        return str(chunks_of(response.text)[0]["id"])
    return str(JSON_OBJECT.validate_json(response.content)["id"])


def answer_text(response: httpx.Response, stream: bool) -> str:
    if stream:
        return "".join(str(delta.get("content") or "") for delta in deltas_of(chunks_of(response.text)))
    return str(assistant_from(response, stream=False)["content"])


def kept_results(assistant: Event) -> JsonValue:
    return JSON_OBJECT.validate_python(assistant.get("provider_specific_fields") or {}).get("tool_results")


def provider_fields(assistant: Event) -> Event:
    return JSON_OBJECT.validate_python(assistant.get("provider_specific_fields") or {})


def tool_call_ids(assistant: Event) -> list[JsonValue]:
    calls: Final = JSON_LIST.validate_python(assistant.get("tool_calls") or [])
    return [JSON_OBJECT.validate_python(call)["id"] for call in calls]


def wire_content(message: Event) -> list[JsonValue]:
    content: Final = message["content"]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return JSON_LIST.validate_python(content)


def wire_assistant(request: Request, position: int = 1) -> list[JsonValue]:
    messages: Final = request_messages(request)
    assert messages[position]["role"] == "assistant", request.body
    return wire_content(messages[position])


def answered_ids(request: Request) -> list[JsonValue]:
    blocks: Final = chain.from_iterable(
        wire_content(message) for message in request_messages(request) if message["role"] == "user"
    )
    objects: Final = tuple(JSON_OBJECT.validate_python(block) for block in blocks)
    return [block["tool_use_id"] for block in objects if block["type"] == "tool_result"]


def user_texts(request: Request) -> list[JsonValue]:
    blocks: Final = chain.from_iterable(
        wire_content(message) for message in request_messages(request) if message["role"] == "user"
    )
    objects: Final = tuple(JSON_OBJECT.validate_python(block) for block in blocks)
    return [block["text"] for block in objects if block["type"] == "text"]


def spend_row(request_id: str) -> Event:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status, model_group FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def hand_built_assistant(
    tool_calls: Sequence[tuple[str, str, Event]], fields: JsonValue = None, **extra: JsonValue
) -> Event:
    calls: Final[list[JsonValue]] = [
        {"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        for identity, name, arguments in tool_calls
    ]
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": calls,
        **({} if fields is None else {"provider_specific_fields": fields}),
        **extra,
    }


SEARCH_CALL: Final[tuple[str, str, Event]] = (SEARCH_ID, SEARCH_TOOL, SEARCH_INPUT)
WEATHER_CALL: Final[tuple[str, str, Event]] = (WEATHER_ID, "get_weather", WEATHER_INPUT)
KEPT_FIELDS: Final[Event] = {"tool_results": [SEARCH_RESULT]}

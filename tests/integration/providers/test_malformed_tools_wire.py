from __future__ import annotations

import asyncio
import http.client
import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import openai
import pytest
from google import genai
from google.genai import types
from integration._support.anthropic_sse import error_body, message_json, message_stream
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    held,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply, openai_error, responses_reply
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.worker_readiness import wait_for_worker_readiness
from integration.authorization._guardrail_opt_out import upstream_observations
from pydantic import JsonValue, TypeAdapter

from tests.integration.caching.response_cache_case import await_entry, redis_store

_OPENAI_BACKEND: Final = "gpt-4o-mini"
_CLAUDE_BACKEND: Final = "claude-haiku-4-5"
_GEMINI_BACKEND: Final = "gemini-2.5-flash"
_PROVIDER_KEY: Final = "scripted-malformed-tools-key"
_COUNT: Final = 37
_CONFIG_STORE: Final = "vs_integration_config_store"
_OTHER_STORE: Final = "vs_malformed_tools_other_store"
_CLAUDE_CODE_AGENT: Final = "claude-cli/2.1.0 (external, cli)"
_CODEX_AGENT: Final = "codex_cli_rs/0.50.0 (Linux 6.8; x86_64) xterm-256color"
_REJECT_MARK: Final = "upstream-401"
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_PROXY_WORKERS: Final = max(1, int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1")))
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SPEND_ROWS: Final = (
    'SELECT call_type, status, cache_hit FROM "LiteLLM_SpendLogs" WHERE request_id = %s OR litellm_call_id = %s'
)
_NOT_A_LIST: Final = "tools: Input should be a valid list"
_NOT_AN_OBJECT: Final = "tools.0: Input should be a valid dictionary"
_TOOL_FIELDS: Final = ("tools", "tool_choice", "parallel_tool_calls", "toolConfig", "tool_config")
_WEATHER_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def _marker() -> str:
    return f"marker-{uuid.uuid4().hex}"


def _tag(text: str) -> str:
    found: Final = _MARKER.search(text)
    return found.group(1) if found else "none"


def _answer(tag: str) -> str:
    return f"answer {tag}"


def _gemini_body(tag: str) -> dict[str, JsonValue]:
    return {
        "responseId": f"gemini-{tag}-{uuid.uuid4().hex}",
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": _answer(tag)}]}, "index": 0, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2, "totalTokenCount": 11},
        "modelVersion": _GEMINI_BACKEND,
    }


def _interaction_body(tag: str) -> dict[str, JsonValue]:
    return {
        "id": f"interaction-{tag}",
        "object": "interaction",
        "model": _GEMINI_BACKEND,
        "status": "completed",
        "created": "2026-10-15T00:00:00Z",
        "steps": [{"type": "content", "content": {"type": "text", "text": _answer(tag)}}],
        "usage": {"total_tokens": 11},
    }


def _rejected_by_provider(path: str) -> Reply:
    if path.startswith("/v1/messages"):
        return Reply(status=401, body=error_body(401, "scripted 401"))
    if path == "/v1beta/interactions":
        return Reply(
            status=401,
            body=json.dumps({"error": {"code": 401, "message": "scripted 401", "status": "UNAUTHENTICATED"}}).encode(),
        )
    return openai_error(401)


def _respond(request: Request) -> Reply:
    text: Final = request.body.decode()
    tag: Final = _tag(text)
    path: Final = request.target.split("?")[0]
    if _REJECT_MARK in text:
        return _rejected_by_provider(path)
    stream: Final = bool(text) and _JSON_OBJECT.validate_json(request.body).get("stream") is True
    if path == "/v1/chat/completions":
        return chat_reply(f"chatcmpl-{tag}", _OPENAI_BACKEND, _answer(tag), stream=stream)
    if path == "/v1/responses":
        return responses_reply(f"resp_{tag}", _OPENAI_BACKEND, _answer(tag), stream=stream)
    if path == "/v1/messages/count_tokens":
        return Reply(body=json.dumps({"input_tokens": _COUNT}).encode())
    if path == "/v1/messages" and stream:
        return Reply(
            content_type="text/event-stream", chunks=message_stream(f"msg_{tag}", _CLAUDE_BACKEND, _answer(tag))
        )
    if path == "/v1/messages":
        return Reply(body=message_json(f"msg_{tag}", _CLAUDE_BACKEND, _answer(tag)))
    if path.endswith(":streamGenerateContent"):
        return Reply(
            content_type="text/event-stream", chunks=(f"data: {json.dumps(_gemini_body(tag))}\r\n\r\n".encode(),)
        )
    if path.endswith(":generateContent"):
        return Reply(body=json.dumps(_gemini_body(tag)).encode())
    if path == "/v1beta/interactions":
        return Reply(body=json.dumps(_interaction_body(tag)).encode())
    return Reply(status=404, body=json.dumps({"error": f"unscripted {request.target}"}).encode())


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    wire: Wire
    openai: str
    anthropic: str
    gemini: str
    drop_group: str
    gemini_drop_group: str
    mixed_group: str
    field_group: str


def _same_name_deployments(scenario: Scenario, api_base: str, variants: tuple[Mapping[str, JsonValue], ...]) -> str:
    name: Final = f"integration-mixed-{uuid.uuid4().hex}"
    created: Final = tuple(
        scenario.gateway.post(
            "/model/new",
            {
                "model_name": name,
                "litellm_params": {
                    "model": f"openai/{_OPENAI_BACKEND}",
                    "api_key": _PROVIDER_KEY,
                    "api_base": api_base,
                    **variant,
                },
                "model_info": {},
            },
        )
        for variant in variants
    )
    for deployment in created:
        scenario.cleanups.callback(scenario.delete_model, string_value(object_value(deployment["model_info"])["id"]))
    return name


@pytest.fixture(scope="module")
def rig() -> Iterator[_Rig]:
    with gateway_from_environment() as gateway, wire_server(_respond) as wire, gateway.scenario() as scenario:
        openai_base: Final = f"{wire.url}/v1"
        built: Final = _Rig(
            gateway=gateway,
            wire=wire,
            openai=scenario.model(api_base=openai_base, api_key=_PROVIDER_KEY),
            anthropic=scenario.model(model=f"anthropic/{_CLAUDE_BACKEND}", api_base=wire.url, api_key=_PROVIDER_KEY),
            gemini=scenario.model(model=f"gemini/{_GEMINI_BACKEND}", api_base=wire.url, api_key=_PROVIDER_KEY),
            drop_group=scenario.model(api_base=openai_base, api_key=_PROVIDER_KEY, drop_params=True),
            gemini_drop_group=scenario.model(
                model=f"gemini/{_GEMINI_BACKEND}", api_base=wire.url, api_key=_PROVIDER_KEY, drop_params=True
            ),
            mixed_group=_same_name_deployments(scenario, openai_base, ({"drop_params": True}, {})),
            field_group=scenario.model(api_base=openai_base, api_key=_PROVIDER_KEY, additional_drop_params=["tools"]),
        )
        for group, deployments in (
            (built.openai, 1),
            (built.anthropic, 1),
            (built.gemini, 1),
            (built.drop_group, 1),
            (built.gemini_drop_group, 1),
            (built.mixed_group, 2),
            (built.field_group, 1),
        ):
            wait_for_worker_readiness(gateway, group, deployments, _PROXY_WORKERS)
        yield built


@pytest.fixture(autouse=True)
def _drained(rig: _Rig) -> None:
    rig.wire.drain()


@dataclass(frozen=True, slots=True)
class _Family:
    name: str
    tool: dict[str, JsonValue]
    model: Callable[[_Rig], str]
    path: Callable[[str, bool], str]
    body: Callable[[str, str, bool], dict[str, JsonValue]]
    target: Callable[[bool], str]
    call_type: Callable[[bool], str]
    upstream_id: Callable[[str], str | None]
    tool_choice: tuple[str, JsonValue]
    exact_tools: bool


def _chat_body(model: str, marker: str, stream: bool) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": marker}], "stream": stream}


def _messages_body(model: str, marker: str, stream: bool) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}], "stream": stream}


def _responses_body(model: str, marker: str, stream: bool) -> dict[str, JsonValue]:
    return {"model": model, "input": marker, "stream": stream}


def _generate_body(model: str, marker: str, stream: bool) -> dict[str, JsonValue]:
    return {"contents": [{"role": "user", "parts": [{"text": marker}]}]}


def _generate_path(model: str, stream: bool) -> str:
    return (
        f"/v1beta/models/{model}:streamGenerateContent?alt=sse" if stream else f"/v1beta/models/{model}:generateContent"
    )


CHAT: Final = _Family(
    name="chat",
    tool={
        "type": "function",
        "function": {"name": "get_weather", "description": "Look up the weather", "parameters": _WEATHER_SCHEMA},
    },
    model=lambda rig: rig.openai,
    path=lambda model, stream: "/v1/chat/completions",
    body=_chat_body,
    target=lambda stream: "/v1/chat/completions",
    call_type=lambda stream: "acompletion",
    upstream_id=lambda tag: f"chatcmpl-{tag}",
    tool_choice=("tool_choice", {"type": "function", "function": {"name": "get_weather"}}),
    exact_tools=True,
)
MESSAGES: Final = _Family(
    name="messages",
    tool={"name": "get_weather", "description": "Look up the weather", "input_schema": _WEATHER_SCHEMA},
    model=lambda rig: rig.anthropic,
    path=lambda model, stream: "/v1/messages",
    body=_messages_body,
    target=lambda stream: "/v1/messages",
    call_type=lambda stream: "anthropic_messages",
    upstream_id=lambda tag: f"msg_{tag}",
    tool_choice=("tool_choice", {"type": "tool", "name": "get_weather"}),
    exact_tools=True,
)
RESPONSES: Final = _Family(
    name="responses",
    tool={
        "type": "function",
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": _WEATHER_SCHEMA,
    },
    model=lambda rig: rig.openai,
    path=lambda model, stream: "/v1/responses",
    body=_responses_body,
    target=lambda stream: "/v1/responses",
    call_type=lambda stream: "aresponses",
    upstream_id=lambda tag: None,
    tool_choice=("tool_choice", {"type": "function", "name": "get_weather"}),
    exact_tools=True,
)
GENERATE: Final = _Family(
    name="generate_content",
    tool={
        "functionDeclarations": [
            {"name": "get_weather", "description": "Look up the weather", "parameters": _WEATHER_SCHEMA}
        ]
    },
    model=lambda rig: rig.gemini,
    path=_generate_path,
    body=_generate_body,
    target=lambda stream: ":streamGenerateContent" if stream else ":generateContent",
    call_type=lambda stream: "agenerate_content_stream" if stream else "agenerate_content",
    upstream_id=lambda tag: None,
    tool_choice=("toolConfig", {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}}),
    exact_tools=False,
)
_FAMILIES: Final = (CHAT, MESSAGES, RESPONSES, GENERATE)
_FAMILY_IDS: Final = tuple(family.name for family in _FAMILIES)
_STREAMS: Final = pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
_EACH_FAMILY: Final = pytest.mark.parametrize("family", _FAMILIES, ids=_FAMILY_IDS)


@dataclass(frozen=True, slots=True)
class _Route:
    path: Callable[[str], str]
    body: Callable[[str, str], dict[str, JsonValue]]
    model: Callable[[_Rig], str]


_ROUTES: Final[Mapping[str, _Route]] = MappingProxyType(
    {
        "chat": _Route(lambda model: "/v1/chat/completions", lambda m, k: _chat_body(m, k, False), CHAT.model),
        "chat_unversioned": _Route(lambda model: "/chat/completions", lambda m, k: _chat_body(m, k, False), CHAT.model),
        "chat_azure_alias": _Route(
            lambda model: f"/openai/deployments/{model}/chat/completions",
            lambda m, k: _chat_body(m, k, False),
            CHAT.model,
        ),
        "chat_stream": _Route(lambda model: "/v1/chat/completions", lambda m, k: _chat_body(m, k, True), CHAT.model),
        "messages": _Route(lambda model: "/v1/messages", lambda m, k: _messages_body(m, k, False), MESSAGES.model),
        "messages_stream": _Route(
            lambda model: "/v1/messages", lambda m, k: _messages_body(m, k, True), MESSAGES.model
        ),
        "responses": _Route(lambda model: "/v1/responses", lambda m, k: _responses_body(m, k, False), RESPONSES.model),
        "responses_stream": _Route(
            lambda model: "/v1/responses", lambda m, k: _responses_body(m, k, True), RESPONSES.model
        ),
        "responses_compact": _Route(
            lambda model: "/v1/responses/compact", lambda m, k: {"model": m, "input": k}, RESPONSES.model
        ),
        "generate_content": _Route(
            lambda model: _generate_path(model, False), lambda m, k: _generate_body(m, k, False), GENERATE.model
        ),
        "stream_generate_content": _Route(
            lambda model: _generate_path(model, True), lambda m, k: _generate_body(m, k, True), GENERATE.model
        ),
        "interactions": _Route(
            lambda model: "/v1beta/interactions", lambda m, k: {"model": m, "input": k}, lambda rig: rig.gemini
        ),
    }
)
_MALFORMED_TOOLS: Final[Mapping[str, tuple[JsonValue, str]]] = MappingProxyType(
    {
        "int": (5, _NOT_A_LIST),
        "empty_string": ("", _NOT_A_LIST),
        "five_kb_string": ("x" * 5120, _NOT_A_LIST),
        "list_of_names": (["get_weather"], _NOT_AN_OBJECT),
        "empty_object": ({}, _NOT_A_LIST),
        "false": (False, _NOT_A_LIST),
        "zero": (0, _NOT_A_LIST),
    }
)
_MALFORMED_ENTRIES: Final[Mapping[str, tuple[JsonValue, str]]] = MappingProxyType(
    {
        "int_entry": (5, _NOT_AN_OBJECT),
        "string_function": (
            {"type": "function", "function": "get_weather"},
            "tools.0.function: Input should be a valid dictionary",
        ),
        "string_function_parameters": (
            {"type": "function", "function": {"name": "get_weather", "parameters": "bad"}},
            "tools.0.function.parameters: Input should be a valid dictionary",
        ),
        "list_input_schema": (
            {"name": "get_weather", "input_schema": [1]},
            "tools.0.input_schema: Input should be a valid dictionary",
        ),
        "int_parameters": (
            {"type": "function", "name": "get_weather", "parameters": 5},
            "tools.0.parameters: Input should be a valid dictionary",
        ),
        "string_vector_store_ids": (
            {"type": "file_search", "vector_store_ids": _OTHER_STORE},
            "tools.0.vector_store_ids: Input should be a valid list",
        ),
    }
)


def _send(
    rig: _Rig,
    family: _Family,
    marker: str,
    stream: bool,
    *,
    model: str | None = None,
    key: str | None = None,
    call_id: str | None = None,
    user_agent: str | None = None,
    **fields: JsonValue,
) -> httpx.Response:
    chosen: Final = model or family.model(rig)
    headers: Final = {
        **({"x-litellm-call-id": call_id} if call_id else {}),
        **({"user-agent": user_agent} if user_agent else {}),
    }
    return rig.gateway.request(
        "POST", family.path(chosen, stream), {**family.body(chosen, marker, stream), **fields}, key=key, headers=headers
    )


def _delivered(rig: _Rig, marker: str) -> tuple[Request, ...]:
    return tuple(request for request in rig.wire.drain() if marker in request.body.decode())


def _one_delivery(rig: _Rig, marker: str, target: str) -> dict[str, JsonValue]:
    (sent,) = _delivered(rig, marker)
    assert sent.method == "POST" and target in sent.target, sent.target
    return _JSON_OBJECT.validate_json(sent.body)


def _tool_names(family: _Family, body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    tools: Final = body.get("tools")
    if not isinstance(tools, list):
        return ()
    if family is GENERATE:
        declarations: Final = chain.from_iterable(_declarations(object_value(tool)) for tool in tools)
        return tuple(string_value(object_value(declaration)["name"]) for declaration in declarations)
    if family is CHAT:
        return tuple(string_value(object_value(object_value(tool)["function"])["name"]) for tool in tools)
    return tuple(string_value(object_value(tool)["name"]) for tool in tools)


def _declarations(tool: Mapping[str, JsonValue]) -> list[JsonValue]:
    declarations: Final = tool.get("functionDeclarations", tool.get("function_declarations"))
    assert isinstance(declarations, list), tool
    return declarations


def _assert_answered(family: _Family, response: httpx.Response, marker: str) -> None:
    assert response.status_code == 200, response.text
    tag: Final = _tag(marker)
    assert _answer(tag) in response.text, response.text
    upstream_id: Final = family.upstream_id(tag)
    if upstream_id is not None:
        assert upstream_id in response.text, response.text
    if family is RESPONSES and not response.headers.get("content-type", "").startswith("text/event-stream"):
        assert same_response(string_value(_JSON_OBJECT.validate_json(response.content)["id"]), f"resp_{tag}")


def _rejection(response: httpx.Response) -> tuple[str, str, JsonValue]:
    body: Final = _JSON_OBJECT.validate_json(response.content)
    error: Final = object_value(body["error"])
    if body.get("type") == "error":
        return string_value(error["type"]), string_value(error["message"]), None
    return string_value(error["type"]), string_value(error["message"]), error.get("param")


def _assert_rejected(response: httpx.Response, message: str, param: str = "tools") -> None:
    assert response.status_code == 400, response.text
    kind, said, named = _rejection(response)
    assert kind == "invalid_request_error", response.text
    assert said == message, response.text
    assert named in (param, None), response.text


def _spend_rows(call_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(_SPEND_ROWS, (call_id, call_id))


def _logged_once(call_id: str) -> dict[str, JsonValue]:
    (row,) = held(lambda: _spend_rows(call_id), lambda rows: len(rows) == 1, holding=3, seconds=70)
    return row


def _tool_values(sent: Mapping[str, JsonValue]) -> frozenset[str]:
    return frozenset(name for name in _TOOL_FIELDS if sent.get(name) is not None)


def _call_id() -> str:
    return f"malformed-tools-{uuid.uuid4().hex}"


@_STREAMS
@_EACH_FAMILY
def test_well_formed_tools_reach_the_provider_and_log_one_success_row(rig: _Rig, family: _Family, stream: bool) -> None:
    marker: Final = _marker()
    call_id: Final = _call_id()
    response: Final = _send(rig, family, marker, stream, call_id=call_id, tools=[family.tool])
    _assert_answered(family, response, marker)
    sent: Final = _one_delivery(rig, marker, family.target(stream))
    assert _tool_names(family, sent) == ("get_weather",), sent
    if family.exact_tools:
        assert sent["tools"] == [family.tool], sent
    row: Final = _logged_once(call_id)
    assert (row["call_type"], row["status"]) == (family.call_type(stream), "success"), row


def test_openai_sdk_chat_with_tools_answers_with_the_provider_reply(rig: _Rig) -> None:
    marker: Final = _marker()
    client: Final = openai.OpenAI(
        base_url=f"{rig.gateway.client.base_url}/v1",
        api_key=rig.gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )
    completion: Final = client.chat.completions.create(
        model=rig.openai,
        messages=[{"role": "user", "content": marker}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather",
                    "parameters": _WEATHER_SCHEMA,
                },
            }
        ],
    )
    assert completion.id == f"chatcmpl-{_tag(marker)}", completion
    assert completion.choices[0].message.content == _answer(_tag(marker)), completion
    assert _one_delivery(rig, marker, "/v1/chat/completions")["tools"] == [CHAT.tool]


def test_openai_sdk_rejects_malformed_tools_with_a_400(rig: _Rig) -> None:
    marker: Final = _marker()
    client: Final = openai.OpenAI(
        base_url=f"{rig.gateway.client.base_url}/v1",
        api_key=rig.gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )
    with pytest.raises(openai.BadRequestError) as raised:
        client.chat.completions.create(
            model=rig.openai, messages=[{"role": "user", "content": marker}], extra_body={"tools": 5}
        )
    assert raised.value.status_code == 400, raised.value
    assert object_value(raised.value.body)["param"] == "tools", raised.value.body
    assert _delivered(rig, marker) == ()


def test_async_openai_sdk_streams_responses_with_tools(rig: _Rig) -> None:
    marker: Final = _marker()

    async def stream_text() -> str:
        client: Final = openai.AsyncOpenAI(
            base_url=f"{rig.gateway.client.base_url}/v1",
            api_key=rig.gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False),
        )
        events: Final = await client.responses.create(
            model=rig.openai,
            input=marker,
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Look up the weather",
                    "parameters": _WEATHER_SCHEMA,
                    "strict": False,
                }
            ],
        )
        completed: Final = [event async for event in events if event.type == "response.completed"]
        assert len(completed) == 1, completed
        return completed[0].response.output_text

    assert asyncio.run(stream_text()) == _answer(_tag(marker))
    assert _tool_names(RESPONSES, _one_delivery(rig, marker, "/v1/responses")) == ("get_weather",)


def test_anthropic_sdk_streams_messages_with_tools(rig: _Rig) -> None:
    marker: Final = _marker()
    client: Final = anthropic.Anthropic(
        base_url=str(rig.gateway.client.base_url),
        api_key=rig.gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )
    with client.messages.stream(
        model=rig.anthropic,
        max_tokens=16,
        messages=[{"role": "user", "content": marker}],
        tools=[{"name": "get_weather", "description": "Look up the weather", "input_schema": _WEATHER_SCHEMA}],
    ) as stream:
        final: Final = stream.get_final_message()
    assert final.id == f"msg_{_tag(marker)}", final
    assert _one_delivery(rig, marker, "/v1/messages")["tools"] == [MESSAGES.tool]


def test_google_genai_sdk_generates_with_tools(rig: _Rig) -> None:
    marker: Final = _marker()
    client: Final = genai.Client(api_key=rig.gateway.key, http_options={"base_url": str(rig.gateway.client.base_url)})
    response: Final = client.models.generate_content(
        model=rig.gemini,
        contents=marker,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="get_weather",
                            description="Look up the weather",
                            parameters_json_schema=_WEATHER_SCHEMA,
                        )
                    ]
                )
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    assert response.text == _answer(_tag(marker)), response
    assert _tool_names(GENERATE, _one_delivery(rig, marker, ":generateContent")) == ("get_weather",)


@pytest.mark.parametrize("value", _MALFORMED_TOOLS.values(), ids=_MALFORMED_TOOLS.keys())
@pytest.mark.parametrize("route", _ROUTES.values(), ids=_ROUTES.keys())
def test_malformed_tools_answer_400_and_never_reach_the_provider(
    rig: _Rig, route: _Route, value: tuple[JsonValue, str]
) -> None:
    marker: Final = _marker()
    model: Final = route.model(rig)
    tools, message = value
    response: Final = rig.gateway.request("POST", route.path(model), {**route.body(model, marker), "tools": tools})
    _assert_rejected(response, message)
    assert _delivered(rig, marker) == ()


@pytest.mark.parametrize("entry", _MALFORMED_ENTRIES.values(), ids=_MALFORMED_ENTRIES.keys())
@pytest.mark.parametrize("family", (CHAT, MESSAGES), ids=("chat", "messages"))
def test_a_malformed_tool_entry_answers_400_naming_its_index(
    rig: _Rig, family: _Family, entry: tuple[JsonValue, str]
) -> None:
    marker: Final = _marker()
    tool, message = entry
    _assert_rejected(_send(rig, family, marker, False, tools=[tool]), message)
    assert _delivered(rig, marker) == ()


@pytest.mark.parametrize("value", [5, "x", []], ids=["int", "string", "list"])
def test_malformed_web_search_options_answer_400(rig: _Rig, value: JsonValue) -> None:
    marker: Final = _marker()
    response: Final = _send(rig, CHAT, marker, False, web_search_options=value)
    expected: Final = "web_search_options: Input should be a valid dictionary"
    _assert_rejected(response, expected, param="web_search_options")
    assert _delivered(rig, marker) == ()


@pytest.mark.parametrize("value", [5, "x", []], ids=["int", "string", "list"])
def test_malformed_web_search_options_drop_under_drop_params(rig: _Rig, value: JsonValue) -> None:
    marker: Final = _marker()
    response: Final = _send(rig, CHAT, marker, False, web_search_options=value, drop_params=True)
    _assert_answered(CHAT, response, marker)
    assert "web_search_options" not in _one_delivery(rig, marker, "/v1/chat/completions")


@_EACH_FAMILY
def test_wrong_key_with_malformed_tools_answers_401(rig: _Rig, family: _Family) -> None:
    marker: Final = _marker()
    response: Final = _send(rig, family, marker, False, key=f"sk-wrong-{uuid.uuid4().hex}", tools=5)
    assert response.status_code == 401, response.text
    assert _delivered(rig, marker) == ()


def _count_body(rig: _Rig, marker: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": rig.anthropic, "messages": [{"role": "user", "content": marker}], **fields}


@pytest.mark.parametrize(
    "fields",
    [
        {"tools": 5},
        {"tools": ""},
        {"tools": "x" * 5120},
        {"tools": ["get_weather"]},
        {"messages": "x"},
    ],
    ids=["int_tools", "empty_string_tools", "five_kb_tools", "list_of_names", "string_messages"],
)
def test_count_tokens_answers_an_anthropic_400_and_keeps_the_connection(
    rig: _Rig, fields: dict[str, JsonValue]
) -> None:
    marker: Final = _marker()
    port: Final = rig.gateway.client.base_url.port
    assert port is not None
    connection: Final = http.client.HTTPConnection(rig.gateway.client.base_url.host, port, timeout=30)
    headers: Final = {"Authorization": f"Bearer {rig.gateway.key}", "Content-Type": "application/json"}
    try:
        connection.request(
            "POST", "/v1/messages/count_tokens", body=json.dumps(_count_body(rig, marker, **fields)), headers=headers
        )
        counted: Final = connection.getresponse()
        counted_body: Final = _JSON_OBJECT.validate_json(counted.read())
        socket_after_count: Final = connection.sock
        connection.request(
            "POST", "/v1/messages", body=json.dumps(_messages_body(rig.anthropic, marker, False)), headers=headers
        )
        followed: Final = connection.getresponse()
        followed_body: Final = followed.read().decode()
        socket_after_follow_up: Final = connection.sock
    finally:
        connection.close()
    assert counted.status == 400, counted_body
    assert counted_body["type"] == "error", counted_body
    error: Final = object_value(counted_body["error"])
    assert error["type"] == "invalid_request_error", counted_body
    assert string_value(error["message"]).startswith(next(iter(fields))), counted_body
    assert followed.status == 200, followed_body
    assert f"msg_{_tag(marker)}" in followed_body, followed_body
    assert socket_after_count is not None and socket_after_follow_up is socket_after_count
    (sent,) = _delivered(rig, marker)
    assert sent.target == "/v1/messages", sent.target


def test_anthropic_sdk_count_tokens_rejects_malformed_tools(rig: _Rig) -> None:
    marker: Final = _marker()
    client: Final = anthropic.Anthropic(
        base_url=str(rig.gateway.client.base_url),
        api_key=rig.gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )
    with pytest.raises(anthropic.BadRequestError) as raised:
        client.messages.count_tokens(
            model=rig.anthropic, messages=[{"role": "user", "content": marker}], extra_body={"tools": 5}
        )
    assert raised.value.status_code == 400, raised.value
    assert object_value(object_value(raised.value.body)["error"])["message"] == _NOT_A_LIST, raised.value.body
    assert _delivered(rig, marker) == ()


def test_count_tokens_forwards_well_formed_tools(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST", "/v1/messages/count_tokens", _count_body(rig, marker, tools=[MESSAGES.tool])
    )
    assert response.status_code == 200, response.text
    assert _JSON_OBJECT.validate_json(response.content) == {"input_tokens": _COUNT}, response.text
    assert _one_delivery(rig, marker, "/v1/messages/count_tokens")["tools"] == [MESSAGES.tool]


@pytest.mark.parametrize(
    ("headers", "fields"),
    [({}, {"drop_params": True}), ({"user-agent": _CLAUDE_CODE_AGENT}, {})],
    ids=["drop_params", "claude_code_agent"],
)
def test_count_tokens_drops_only_the_bad_entry(
    rig: _Rig, headers: dict[str, str], fields: dict[str, JsonValue]
) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/messages/count_tokens",
        _count_body(rig, marker, tools=[MESSAGES.tool, 5], **fields),
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert _one_delivery(rig, marker, "/v1/messages/count_tokens")["tools"] == [MESSAGES.tool]


def test_responses_input_tokens_counts_well_formed_tools(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST", "/v1/responses/input_tokens", {"model": rig.anthropic, "input": marker, "tools": [RESPONSES.tool]}
    )
    assert response.status_code == 200, response.text
    assert _JSON_OBJECT.validate_json(response.content) == {
        "object": "response.input_tokens",
        "input_tokens": _COUNT,
    }, response.text
    sent: Final = _one_delivery(rig, marker, "/v1/messages/count_tokens")
    assert "get_weather" in json.dumps(sent["tools"]), sent


def test_responses_input_tokens_rejects_malformed_tools(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST", "/v1/responses/input_tokens", {"model": rig.anthropic, "input": marker, "tools": 5}
    )
    _assert_rejected(response, _NOT_A_LIST)
    assert _delivered(rig, marker) == ()


def test_responses_input_tokens_drops_only_the_bad_entry(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/responses/input_tokens",
        {"model": rig.anthropic, "input": marker, "tools": [RESPONSES.tool, 5], "drop_params": True},
    )
    assert response.status_code == 200, response.text
    sent: Final = _one_delivery(rig, marker, "/v1/messages/count_tokens")
    tools: Final = sent["tools"]
    assert isinstance(tools, list) and len(tools) == 1 and "get_weather" in json.dumps(tools), sent


def _allowlisted_key(scenario: Scenario) -> str:
    return scenario.key(metadata={"allowed_tools": ["get_weather"]})


@_EACH_FAMILY
def test_allowlisted_key_keeps_its_allowlist_with_malformed_tools(rig: _Rig, family: _Family) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = _allowlisted_key(scenario)
        allowed: Final = _marker()
        _assert_answered(family, _send(rig, family, allowed, False, key=key, tools=[family.tool]), allowed)
        assert len(_delivered(rig, allowed)) == 1
        malformed: Final = _marker()
        _assert_rejected(_send(rig, family, malformed, False, key=key, tools=5), _NOT_A_LIST)
        denied_tool: Final = json.loads(json.dumps(family.tool).replace("get_weather", "delete_files"))
        denied: Final = _marker()
        refused: Final = _send(rig, family, denied, False, key=key, tools=[denied_tool])
        assert refused.status_code == 403, refused.text
        assert "delete_files" in refused.text, refused.text
        assert _delivered(rig, malformed) + _delivered(rig, denied) == ()


def _file_search(store_ids: JsonValue) -> dict[str, JsonValue]:
    return {"type": "file_search", "vector_store_ids": store_ids}


def test_store_limited_key_searches_its_store_from_a_file_search_tool(rig: _Rig) -> None:
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(object_permission={"vector_stores": [_CONFIG_STORE]})
        response: Final = _send(rig, CHAT, marker, False, key=key, tools=[_file_search([_CONFIG_STORE])])
    _assert_answered(CHAT, response, marker)
    sent: Final = _one_delivery(rig, marker, "/v1/chat/completions")
    assert "file_search" not in json.dumps(sent), sent
    searches: Final = eventually(
        lambda: tuple(
            observation
            for observation in upstream_observations(rig.gateway)
            if observation["path"] == f"/vector_stores/{_CONFIG_STORE}/search" and marker in json.dumps(observation)
        ),
        lambda found: len(found) == 1,
    )
    assert len(searches) == 1, searches


@pytest.mark.parametrize(
    ("store_ids", "status", "named"),
    [
        ([_OTHER_STORE], 401, _OTHER_STORE),
        (_CONFIG_STORE, 400, "tools.0.vector_store_ids: Input should be a valid list"),
        (_OTHER_STORE, 401, _OTHER_STORE),
        ([5, _OTHER_STORE], 401, _OTHER_STORE),
        ([5, _CONFIG_STORE], 400, "tools.0.vector_store_ids.0: Input should be a valid string"),
    ],
    ids=["other_store", "allowed_store_as_string", "other_store_as_string", "int_beside_other", "int_beside_allowed"],
)
def test_store_limited_key_reads_every_string_store_id(
    rig: _Rig, store_ids: JsonValue, status: int, named: str
) -> None:
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(object_permission={"vector_stores": [_CONFIG_STORE]})
        response: Final = _send(rig, CHAT, marker, False, key=key, tools=[_file_search(store_ids)])
    assert response.status_code == status, response.text
    assert named in response.text, response.text
    assert _delivered(rig, marker) == ()


def test_store_limited_key_treats_null_store_ids_as_none(rig: _Rig) -> None:
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(object_permission={"vector_stores": [_CONFIG_STORE]})
        response: Final = _send(rig, CHAT, marker, False, key=key, tools=[_file_search(None)])
    _assert_answered(CHAT, response, marker)
    assert _one_delivery(rig, marker, "/v1/chat/completions")["tools"] == [_file_search(None)]


@pytest.mark.parametrize(
    ("first", "last", "status"),
    [("5", "WELL_FORMED", 200), ("WELL_FORMED", "5", 400)],
    ids=["well_formed_last", "malformed_last"],
)
def test_duplicate_tools_keys_use_the_last_one(rig: _Rig, first: str, last: str, status: int) -> None:
    marker: Final = _marker()
    well_formed: Final = json.dumps([CHAT.tool])
    prefix: Final = json.dumps({"model": rig.openai, "messages": [{"role": "user", "content": marker}]})[:-1]
    raw: Final = (
        f'{prefix}, "tools": {first.replace("WELL_FORMED", well_formed)}, '
        f'"tools": {last.replace("WELL_FORMED", well_formed)}}}'
    )
    response: Final = rig.gateway.client.post(
        "/v1/chat/completions",
        content=raw.encode(),
        headers={"Authorization": f"Bearer {rig.gateway.key}", "Content-Type": "application/json"},
    )
    assert response.status_code == status, response.text
    delivered: Final = _delivered(rig, marker)
    if status == 400:
        _assert_rejected(response, _NOT_A_LIST)
        assert delivered == ()
        return
    (sent,) = delivered
    assert _JSON_OBJECT.validate_json(sent.body)["tools"] == [CHAT.tool]


@pytest.mark.parametrize(
    ("family", "route", "expected"),
    [
        (CHAT, None, "acompletion"),
        (MESSAGES, None, "anthropic_messages"),
        (RESPONSES, None, "aresponses"),
        (GENERATE, None, "agenerate_content"),
        (None, "interactions", "acreate_interaction"),
    ],
    ids=["chat", "messages", "responses", "generate_content", "interactions"],
)
def test_a_malformed_tools_rejection_logs_one_failure_row_with_its_call_type(
    rig: _Rig, family: _Family | None, route: str | None, expected: str
) -> None:
    marker: Final = _marker()
    call_id: Final = _call_id()
    response: Final = (
        _send(rig, family, marker, False, call_id=call_id, tools=5)
        if family is not None
        else rig.gateway.request(
            "POST",
            "/v1beta/interactions",
            {"model": rig.gemini, "input": [{"type": "text", "text": marker}], "tools": 5},
            headers={"x-litellm-call-id": call_id},
        )
    )
    _assert_rejected(response, _NOT_A_LIST)
    row: Final = _logged_once(call_id)
    assert (row["call_type"], row["status"]) == (expected, "failure"), row


@pytest.mark.parametrize(
    ("path", "body", "expected"),
    [
        ("/v1/chat/completions", "chat", "acompletion"),
        ("/v1/messages", "messages", "anthropic_messages"),
        ("/v1beta/interactions", "interactions", "acreate_interaction"),
    ],
    ids=["chat", "messages", "interactions_list_input"],
)
def test_a_provider_401_with_well_formed_tools_logs_its_endpoint_call_type(
    rig: _Rig, path: str, body: str, expected: str
) -> None:
    marker: Final = f"{_marker()} {_REJECT_MARK}"
    call_id: Final = _call_id()
    payloads: Final[Mapping[str, dict[str, JsonValue]]] = {
        "chat": {**_chat_body(rig.openai, marker, False), "tools": [CHAT.tool]},
        "messages": {**_messages_body(rig.anthropic, marker, False), "tools": [MESSAGES.tool]},
        "interactions": {
            "model": rig.gemini,
            "input": [{"type": "text", "text": marker}],
            "tools": [{"type": "function", "name": "get_weather", "parameters": _WEATHER_SCHEMA}],
        },
    }
    response: Final = rig.gateway.request("POST", path, payloads[body], headers={"x-litellm-call-id": call_id})
    assert response.status_code == 401, response.text
    assert len(_delivered(rig, marker)) == 1
    row: Final = _logged_once(call_id)
    assert (row["call_type"], row["status"]) == (expected, "failure"), row


@pytest.mark.parametrize("drop", [True, "true"], ids=["bool", "string"])
@_EACH_FAMILY
def test_request_drop_params_drops_malformed_tools_and_their_companions(
    rig: _Rig, family: _Family, drop: JsonValue
) -> None:
    marker: Final = _marker()
    companion, value = family.tool_choice
    response: Final = _send(rig, family, marker, False, tools=5, drop_params=drop, **{companion: value})
    _assert_answered(family, response, marker)
    sent: Final = _one_delivery(rig, marker, family.target(False))
    assert _tool_values(sent) == frozenset(), sent


@_EACH_FAMILY
def test_drop_params_keeps_the_good_entries_and_the_tool_choice(rig: _Rig, family: _Family) -> None:
    marker: Final = _marker()
    companion, value = family.tool_choice
    response: Final = _send(rig, family, marker, False, tools=[family.tool, 5], drop_params=True, **{companion: value})
    _assert_answered(family, response, marker)
    sent: Final = _one_delivery(rig, marker, family.target(False))
    assert _tool_names(family, sent) == ("get_weather",), sent
    if family.exact_tools:
        assert sent["tools"] == [family.tool], sent
    assert companion in sent, sent


def test_additional_drop_params_naming_tools_drops_them(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = _send(rig, CHAT, marker, False, tools=5, additional_drop_params=["tools"])
    _assert_answered(CHAT, response, marker)
    assert "tools" not in _one_delivery(rig, marker, "/v1/chat/completions")


def test_additional_drop_params_naming_another_field_still_rejects_tools(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = _send(rig, CHAT, marker, False, tools=5, additional_drop_params=["web_search_options"])
    _assert_rejected(response, _NOT_A_LIST)
    assert _delivered(rig, marker) == ()


@pytest.mark.parametrize("family", (CHAT, GENERATE), ids=("chat", "generate_content"))
def test_a_drop_params_deployment_group_drops_malformed_tools(rig: _Rig, family: _Family) -> None:
    marker: Final = _marker()
    model: Final = rig.drop_group if family is CHAT else rig.gemini_drop_group
    response: Final = _send(rig, family, marker, False, model=model, tools=5)
    _assert_answered(family, response, marker)
    assert _tool_values(_one_delivery(rig, marker, family.target(False))) == frozenset()


@pytest.mark.parametrize("drop", [False, None], ids=["false", "null"])
def test_an_explicit_request_drop_params_overrides_the_deployment_group(rig: _Rig, drop: JsonValue) -> None:
    marker: Final = _marker()
    response: Final = _send(rig, CHAT, marker, False, model=rig.drop_group, tools=5, drop_params=drop)
    _assert_rejected(response, _NOT_A_LIST)
    assert _delivered(rig, marker) == ()


def test_a_group_where_only_some_deployments_drop_rejects_malformed_tools(rig: _Rig) -> None:
    rejected: Final = _marker()
    _assert_rejected(_send(rig, CHAT, rejected, False, model=rig.mixed_group, tools=5), _NOT_A_LIST)
    dropped: Final = _marker()
    response: Final = _send(rig, CHAT, dropped, False, model=rig.mixed_group, tools=5, drop_params=True)
    _assert_answered(CHAT, response, dropped)
    assert "tools" not in _one_delivery(rig, dropped, "/v1/chat/completions")
    assert _delivered(rig, rejected) == ()


def test_a_deployment_dropping_tools_drops_every_tool(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = _send(rig, CHAT, marker, False, model=rig.field_group, tools=[CHAT.tool, 5])
    _assert_answered(CHAT, response, marker)
    assert "tools" not in _one_delivery(rig, marker, "/v1/chat/completions")


def test_a_request_additional_drop_params_list_keeps_tools_a_deployment_list_drops(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = _send(
        rig,
        CHAT,
        marker,
        False,
        model=rig.field_group,
        tools=[CHAT.tool],
        additional_drop_params=["web_search_options"],
    )
    _assert_answered(CHAT, response, marker)
    assert _tool_names(CHAT, _one_delivery(rig, marker, "/v1/chat/completions")) == ("get_weather",)


def test_a_request_additional_drop_params_list_replaces_the_deployment_list(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = _send(
        rig,
        CHAT,
        marker,
        False,
        model=rig.field_group,
        tools=[CHAT.tool, 5],
        additional_drop_params=["web_search_options"],
    )
    _assert_rejected(response, "tools.1: Input should be a valid dictionary")
    assert _delivered(rig, marker) == ()


@pytest.mark.parametrize(
    ("family", "agent"),
    [(MESSAGES, _CLAUDE_CODE_AGENT), (RESPONSES, _CODEX_AGENT), (CHAT, _CLAUDE_CODE_AGENT)],
    ids=["claude_code_messages", "codex_responses", "claude_code_chat"],
)
def test_an_agentic_cli_drops_only_the_bad_entry(rig: _Rig, family: _Family, agent: str) -> None:
    marker: Final = _marker()
    response: Final = _send(rig, family, marker, False, user_agent=agent, tools=[family.tool, 5])
    _assert_answered(family, response, marker)
    assert _one_delivery(rig, marker, family.target(False))["tools"] == [family.tool]


def test_an_agentic_cli_with_explicit_drop_params_false_rejects(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = _send(
        rig, MESSAGES, marker, False, user_agent=_CLAUDE_CODE_AGENT, tools=[MESSAGES.tool, 5], drop_params=False
    )
    _assert_rejected(response, "tools.1: Input should be a valid dictionary")
    assert _delivered(rig, marker) == ()


@pytest.mark.parametrize(
    "fields", [{"tools": None}, {"tools": []}, {}], ids=["null_tools", "empty_list_tools", "missing_tools"]
)
def test_empty_null_and_missing_tools_still_answer(rig: _Rig, fields: dict[str, JsonValue]) -> None:
    marker: Final = _marker()
    response: Final = _send(rig, CHAT, marker, False, **fields)
    _assert_answered(CHAT, response, marker)
    assert not _one_delivery(rig, marker, "/v1/chat/completions").get("tools")


def test_null_web_search_options_answers_and_logs_success(rig: _Rig) -> None:
    marker: Final = _marker()
    call_id: Final = _call_id()
    response: Final = _send(rig, CHAT, marker, False, call_id=call_id, web_search_options=None)
    _assert_answered(CHAT, response, marker)
    assert _one_delivery(rig, marker, "/v1/chat/completions").get("web_search_options") is None
    row: Final = _logged_once(call_id)
    assert (row["call_type"], row["status"]) == ("acompletion", "success"), row


@_EACH_FAMILY
def test_request_drop_params_with_well_formed_tools_logs_one_success_row(rig: _Rig, family: _Family) -> None:
    marker: Final = _marker()
    call_id: Final = _call_id()
    response: Final = _send(rig, family, marker, False, call_id=call_id, tools=[family.tool], drop_params=True)
    _assert_answered(family, response, marker)
    assert _tool_names(family, _one_delivery(rig, marker, family.target(False))) == ("get_weather",)
    row: Final = _logged_once(call_id)
    assert (row["call_type"], row["status"]) == (family.call_type(False), "success"), row


def test_repeated_identical_dropped_requests_each_log_once(rig: _Rig) -> None:
    marker: Final = _marker()
    call_ids: Final = tuple(_call_id() for _ in range(3))
    for call_id in call_ids:
        response = _send(rig, GENERATE, marker, False, call_id=call_id, tools=[GENERATE.tool, 5], drop_params=True)
        _assert_answered(GENERATE, response, marker)
    delivered: Final = _delivered(rig, marker)
    assert (
        tuple(_tool_names(GENERATE, _JSON_OBJECT.validate_json(sent.body)) for sent in delivered)
        == (("get_weather",),) * 3
    )
    rows: Final = tuple(_logged_once(call_id) for call_id in call_ids)
    assert tuple((row["call_type"], row["status"]) for row in rows) == (("agenerate_content", "success"),) * 3


@pytest.mark.parametrize("family", (CHAT, MESSAGES, RESPONSES), ids=("chat", "messages", "responses"))
def test_a_cache_hit_twin_answers_from_the_cache_and_logs_once_each(rig: _Rig, family: _Family) -> None:
    marker: Final = _marker()
    first_id: Final = _call_id()
    second_id: Final = _call_id()
    first: Final = _send(rig, family, marker, False, call_id=first_id, tools=[family.tool])
    _assert_answered(family, first, marker)
    await_entry(redis_store(), _answer(_tag(marker)))
    second: Final = _send(rig, family, marker, False, call_id=second_id, tools=[family.tool])
    _assert_answered(family, second, marker)
    assert len(_delivered(rig, marker)) == 1
    first_row: Final = _logged_once(first_id)
    second_row: Final = _logged_once(second_id)
    assert (first_row["call_type"], second_row["call_type"]) == (family.call_type(False),) * 2, (first_row, second_row)
    assert second_row["cache_hit"] == "True", second_row


def test_a_dropped_request_shares_the_cache_entry_of_its_clean_twin(rig: _Rig) -> None:
    marker: Final = _marker()
    dropped: Final = _send(rig, CHAT, marker, False, tools=[CHAT.tool, 5], drop_params=True)
    _assert_answered(CHAT, dropped, marker)
    await_entry(redis_store(), _answer(_tag(marker)))
    clean: Final = _send(rig, CHAT, marker, False, tools=[CHAT.tool], drop_params=True)
    _assert_answered(CHAT, clean, marker)
    (sent,) = _delivered(rig, marker)
    assert _JSON_OBJECT.validate_json(sent.body)["tools"] == [CHAT.tool]

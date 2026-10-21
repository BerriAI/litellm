import base64
import itertools
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote

import anthropic
import httpx
import openai
import pytest
from _pytest.mark.structures import ParameterSet
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.timing import assert_within_factor_of_control, timed
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.responses import ResponseCompletedEvent
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with

_BACKEND: Final = "gemini-2.5-flash"
_ARGON: Final = "gemini-4-argon"
_API_KEY: Final = "synthetic-gemini-key"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_BUCKET: Final = "scripted-bucket"
_UPLOAD_PREFIX: Final = f"/upload/storage/v1/b/{_BUCKET}/o?uploadType=media&name="
_SALT: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")
_QUESTION: Final = "Which type does the value property of Config use?"
_FOLLOW_UP: Final = "Answer in one word."
_ANSWER: Final = "A humanScalar string."
_TOOL: Final = "inspect_schema"
_ARGS: Final[Mapping[str, JsonValue]] = {"name": "Config"}
_PARAMETERS: Final[Mapping[str, JsonValue]] = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}
_FUNCTION: Final[Mapping[str, JsonValue]] = {
    "name": _TOOL,
    "description": "Return the JSON Schema of a named config",
    "parameters": _PARAMETERS,
}
_DECLARATIONS: Final[Mapping[str, JsonValue]] = {"tools": [{"function_declarations": [dict(_FUNCTION)]}]}
_SCHEMA_TEXT: Final = (
    '{"$defs":{"humanScalar":{"type":"string"}},"type":"object","properties":{"value":{"$ref":"#/$defs/humanScalar"}}}'
)
_PLAIN_TEXT: Final = '{"type":"object","properties":{"value":{"type":"string"}}}'
_PNG: Final = "iVBORw0KGgo="
_SKIP_SIGNATURE: Final = base64.b64encode(b"skip_thought_signature_validator").decode()
_CACHE_BUST: Final[Mapping[str, JsonValue]] = {"cache": {"no-cache": True}}
_USAGE: Final = {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_LIST: Final = TypeAdapter(list[JsonValue])

Provider: TypeAlias = Literal["gemini", "vertex_ai"]
Endpoint: TypeAlias = Literal["chat", "messages", "responses"]
Client: TypeAlias = Literal["sdk_sync", "sdk_async", "httpx"]
_PROVIDERS: Final[tuple[Provider, ...]] = ("gemini", "vertex_ai")
_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
_CLIENTS: Final[tuple[Client, ...]] = ("sdk_sync", "sdk_async", "httpx")


@dataclass(frozen=True, slots=True)
class _Observed:
    response_id: str
    answer: str


def _model_path(backend: str) -> str:
    return f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{backend}"


def _children(node: JsonValue) -> Sequence[JsonValue]:
    if isinstance(node, dict):
        return tuple(node.values())
    if isinstance(node, list):
        return tuple(child for child in node if not isinstance(child, list))
    return ()


def _references(node: JsonValue) -> Iterator[str]:
    if isinstance(node, dict):
        reference: Final = node.get("$ref")
        if isinstance(reference, str) and reference:
            yield reference
    for child in _children(node):
        yield from _references(child)


def _object(value: JsonValue) -> Mapping[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _items(value: JsonValue) -> Sequence[JsonValue]:
    return value if isinstance(value, list) else ()


def _display_name(part: JsonValue) -> JsonValue:
    blob: Final = _object(_object(part).get("inline_data") or _object(part).get("inlineData"))
    return blob.get("display_name") or blob.get("displayName")


def _function_responses(body: Mapping[str, JsonValue]) -> Iterator[Mapping[str, JsonValue]]:
    for content in _items(body.get("contents")):
        for part in _items(_object(content).get("parts")):
            found = _object(_object(part).get("function_response") or _object(part).get("functionResponse"))
            if found:
                yield found


def _unresolved_reference(raw: bytes) -> str | None:
    for function_response in _function_responses(json.loads(raw)):
        names = frozenset(str(_display_name(part)) for part in _items(function_response.get("parts")))
        for reference in _references(function_response.get("response")):
            if reference not in names:
                return reference
    return None


def _rejection(reference: str) -> Reply:
    message: Final = (
        f"The referenced name `{reference}` in function_response.response does not match to a display_name in the "
        "function_response.parts."
    )
    return Reply(
        status=400,
        body=json.dumps({"error": {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}}).encode(),
    )


def _frame(response_id: str, text: str, finished: bool) -> Mapping[str, JsonValue]:
    return {
        "responseId": response_id,
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "index": 0,
                **({"finishReason": "STOP"} if finished else {}),
            }
        ],
        "usageMetadata": dict(_USAGE),
        "modelVersion": _BACKEND,
    }


def _reply(response_id: str, stream: bool) -> Reply:
    if not stream:
        return Reply(body=json.dumps(_frame(response_id, _ANSWER, finished=True)).encode())
    frames: Final = (_frame(response_id, "A humanScalar ", finished=False), _frame(response_id, "string.", True))
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames),
    )


def _vendor(response_ids: Sequence[str], stream: bool) -> Callable[[Request], Reply]:
    served: Final = itertools.count()

    def respond(request: Request) -> Reply:
        unresolved: Final = _unresolved_reference(request.body)
        if unresolved is not None:
            return _rejection(unresolved)
        return _reply(response_ids[next(served)], stream)

    return respond


def _register(gateway: Gateway, scenario: Scenario, provider: Provider, wire: Wire, backend: str = _BACKEND) -> str:
    if provider == "gemini":
        return scenario.model(model=f"gemini/{backend}", api_base=wire.url, api_key=_API_KEY)
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_base=f"{wire.url}{_model_path(backend)}",
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
    )


def _target(provider: Provider, stream: bool, backend: str = _BACKEND) -> str:
    prefix: Final = f"/models/{backend}" if provider == "gemini" else _model_path(backend)
    return f"{prefix}:streamGenerateContent?alt=sse" if stream else f"{prefix}:generateContent"


def _expected_auth(provider: Provider) -> tuple[str, str]:
    return ("x-goog-api-key", _API_KEY) if provider == "gemini" else ("authorization", "Bearer scripted-token")


def _args(index: int) -> Mapping[str, JsonValue]:
    return _ARGS if index == 0 else {"name": f"Config{index}"}


def _chat_messages(tool_contents: Sequence[JsonValue]) -> list[JsonValue]:
    calls: Final = [
        {"id": f"call_{index}", "type": "function", "function": {"name": _TOOL, "arguments": json.dumps(_args(index))}}
        for index in range(len(tool_contents))
    ]
    results: Final = [
        {"role": "tool", "tool_call_id": f"call_{index}", "content": content}
        for index, content in enumerate(tool_contents)
    ]
    return [
        {"role": "user", "content": _QUESTION},
        {"role": "assistant", "content": None, "tool_calls": calls},
        *results,
    ]


def _messages_history(tool_content: JsonValue) -> list[JsonValue]:
    return [
        {"role": "user", "content": _QUESTION},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_0", "name": _TOOL, "input": dict(_ARGS)}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_0", "content": tool_content}]},
    ]


def _responses_input(tool_output: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": _QUESTION},
        {"type": "function_call", "call_id": "call_0", "name": _TOOL, "arguments": json.dumps(_ARGS)},
        {"type": "function_call_output", "call_id": "call_0", "output": tool_output},
    ]


def _extra(endpoint: Endpoint) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return {"tools": [{"type": "function", "function": dict(_FUNCTION)}], **_CACHE_BUST}
        case "messages":
            return {
                "tools": [{"name": _TOOL, "description": _FUNCTION["description"], "input_schema": dict(_PARAMETERS)}],
                **_CACHE_BUST,
            }
        case "responses":
            return {"tools": [{"type": "function", **_FUNCTION}], **_CACHE_BUST}


def _http_body(endpoint: Endpoint, model: str, tool_text: str, stream: bool) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return {"model": model, "messages": _chat_messages((tool_text,)), "stream": stream, **_extra(endpoint)}
        case "messages":
            return {
                "model": model,
                "max_tokens": 64,
                "messages": _messages_history(tool_text),
                "stream": stream,
                **_extra(endpoint),
            }
        case "responses":
            return {"model": model, "input": _responses_input(tool_text), "stream": stream, **_extra(endpoint)}


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _openai(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _async_openai(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def _anthropic(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _async_anthropic(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def _single(values: frozenset[str]) -> str:
    assert len(values) == 1, values
    return next(iter(values))


def _chat_sync(gateway: Gateway, model: str, stream: bool, text: str) -> _Observed:
    client: Final = _openai(gateway)
    messages: Final = _chat_messages((text,))
    if not stream:
        reply: Final = client.chat.completions.create(model=model, messages=messages, extra_body=_extra("chat"))
        return _Observed(reply.id, reply.choices[0].message.content or "")
    chunks: Final = list(
        client.chat.completions.create(model=model, messages=messages, stream=True, extra_body=_extra("chat"))
    )
    return _Observed(
        _single(frozenset(chunk.id for chunk in chunks)),
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices),
    )


async def _chat_async(gateway: Gateway, model: str, stream: bool, text: str) -> _Observed:
    client: Final = _async_openai(gateway)
    messages: Final = _chat_messages((text,))
    if not stream:
        reply: Final = await client.chat.completions.create(model=model, messages=messages, extra_body=_extra("chat"))
        return _Observed(reply.id, reply.choices[0].message.content or "")
    chunks: Final = [
        chunk
        async for chunk in await client.chat.completions.create(
            model=model, messages=messages, stream=True, extra_body=_extra("chat")
        )
    ]
    return _Observed(
        _single(frozenset(chunk.id for chunk in chunks)),
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices),
    )


def _messages_sync(gateway: Gateway, model: str, stream: bool, text: str) -> _Observed:
    client: Final = _anthropic(gateway)
    history: Final = _messages_history(text)
    if not stream:
        reply: Final = client.messages.create(
            model=model, max_tokens=64, messages=history, extra_body=_extra("messages")
        )
        return _Observed(reply.id, "".join(block.text for block in reply.content if block.type == "text"))
    with client.messages.stream(
        model=model, max_tokens=64, messages=history, extra_body=_extra("messages")
    ) as streamed:
        final: Final = streamed.get_final_message()
    return _Observed(final.id, "".join(block.text for block in final.content if block.type == "text"))


async def _messages_async(gateway: Gateway, model: str, stream: bool, text: str) -> _Observed:
    client: Final = _async_anthropic(gateway)
    history: Final = _messages_history(text)
    if not stream:
        reply: Final = await client.messages.create(
            model=model, max_tokens=64, messages=history, extra_body=_extra("messages")
        )
        return _Observed(reply.id, "".join(block.text for block in reply.content if block.type == "text"))
    async with client.messages.stream(
        model=model, max_tokens=64, messages=history, extra_body=_extra("messages")
    ) as streamed:
        final: Final = await streamed.get_final_message()
    return _Observed(final.id, "".join(block.text for block in final.content if block.type == "text"))


def _responses_sync(gateway: Gateway, model: str, stream: bool, text: str) -> _Observed:
    client: Final = _openai(gateway)
    items: Final = _responses_input(text)
    if not stream:
        reply: Final = client.responses.create(model=model, input=items, extra_body=_extra("responses"))
        return _Observed(reply.id, reply.output_text)
    events: Final = list(client.responses.create(model=model, input=items, stream=True, extra_body=_extra("responses")))
    completed: Final = events[-1]
    assert isinstance(completed, ResponseCompletedEvent), events
    return _Observed(completed.response.id, completed.response.output_text)


async def _responses_async(gateway: Gateway, model: str, stream: bool, text: str) -> _Observed:
    client: Final = _async_openai(gateway)
    items: Final = _responses_input(text)
    if not stream:
        reply: Final = await client.responses.create(model=model, input=items, extra_body=_extra("responses"))
        return _Observed(reply.id, reply.output_text)
    events: Final = [
        event
        async for event in await client.responses.create(
            model=model, input=items, stream=True, extra_body=_extra("responses")
        )
    ]
    completed: Final = events[-1]
    assert isinstance(completed, ResponseCompletedEvent), events
    return _Observed(completed.response.id, completed.response.output_text)


def _sse_payloads(text: str) -> tuple[Mapping[str, JsonValue], ...]:
    lines: Final = tuple(line for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]")
    return tuple(_JSON_OBJECT.validate_json(line.removeprefix("data: ").encode()) for line in lines)


def _output_text(payload: Mapping[str, JsonValue]) -> str:
    messages: Final = tuple(item for item in _items(payload["output"]) if _object(item).get("type") == "message")
    parts: Final = tuple(itertools.chain.from_iterable(_items(_object(item)["content"]) for item in messages))
    return "".join(str(_object(part)["text"]) for part in parts if _object(part).get("type") == "output_text")


def _chat_answer(payload: Mapping[str, JsonValue]) -> str:
    return str(_object(_object(_items(payload["choices"])[0])["message"])["content"])


def _http_observed(endpoint: Endpoint, stream: bool, response: httpx.Response) -> _Observed:
    assert response.status_code == 200, response.text
    if not stream:
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        match endpoint:
            case "chat":
                return _Observed(str(payload["id"]), _chat_answer(payload))
            case "messages":
                blocks: Final = tuple(_object(block) for block in _items(payload["content"]))
                return _Observed(str(payload["id"]), "".join(str(block["text"]) for block in blocks))
            case "responses":
                return _Observed(str(payload["id"]), _output_text(payload))
    events: Final = _sse_payloads(response.text)
    match endpoint:
        case "chat":
            assert response.text.rstrip().endswith("data: [DONE]"), response.text
            deltas: Final = tuple(_object(_object(_items(event["choices"])[0])["delta"]) for event in events)
            return _Observed(
                _single(frozenset(str(event["id"]) for event in events)),
                "".join(str(delta["content"]) for delta in deltas if delta.get("content")),
            )
        case "messages":
            assert events[-1]["type"] == "message_stop", events
            start: Final = next(event for event in events if event["type"] == "message_start")
            texts: Final = tuple(_object(event["delta"]) for event in events if event["type"] == "content_block_delta")
            return _Observed(
                str(_object(start["message"])["id"]),
                "".join(str(delta["text"]) for delta in texts if delta["type"] == "text_delta"),
            )
        case "responses":
            assert events[-1]["type"] == "response.completed", events
            completed: Final = _object(events[-1]["response"])
            return _Observed(str(completed["id"]), _output_text(completed))


def _httpx(gateway: Gateway, endpoint: Endpoint, model: str, stream: bool, text: str) -> _Observed:
    response: Final = gateway.request(
        "POST", _path(endpoint), _http_body(endpoint, model, text, stream), headers={"anthropic-version": "2023-06-01"}
    )
    return _http_observed(endpoint, stream, response)


async def _observe(
    gateway: Gateway, endpoint: Endpoint, client: Client, model: str, stream: bool, text: str
) -> _Observed:
    match (endpoint, client):
        case ("chat", "sdk_sync"):
            return _chat_sync(gateway, model, stream, text)
        case ("chat", "sdk_async"):
            return await _chat_async(gateway, model, stream, text)
        case ("messages", "sdk_sync"):
            return _messages_sync(gateway, model, stream, text)
        case ("messages", "sdk_async"):
            return await _messages_async(gateway, model, stream, text)
        case ("responses", "sdk_sync"):
            return _responses_sync(gateway, model, stream, text)
        case ("responses", "sdk_async"):
            return await _responses_async(gateway, model, stream, text)
        case (_, "httpx"):
            return _httpx(gateway, endpoint, model, stream, text)
    raise AssertionError((endpoint, client))


def _upstream_response_id(endpoint: Endpoint, caller_id: str) -> str:
    if endpoint != "responses":
        return caller_id
    decrypted: Final = decrypt_if_encrypted_with(caller_id.removeprefix("resp_"), _SALT)
    assert decrypted is not None, caller_id
    issued: Final = decrypted.split(";")[0].split("response_id:")[-1]
    return base64.b64decode(issued.removeprefix("resp_")).decode().split(";")[-1].removeprefix("response_id:")


def _assert_caller_id(endpoint: Endpoint, stream: bool, caller_id: str, response_id: str) -> None:
    if endpoint == "messages" and stream:
        assert caller_id.startswith("msg_"), caller_id
        return
    assert _upstream_response_id(endpoint, caller_id) == response_id, caller_id


def _call_type(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "acompletion"
        case "messages":
            return "anthropic_messages"
        case "responses":
            return "aresponses"


def _spend_rows(request_ids: Sequence[str], expected: int) -> tuple[Mapping[str, JsonValue], ...]:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id, model_group, call_type, status, prompt_tokens, completion_tokens "
            'FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)',
            (list(request_ids),),  # pyright: ignore[reportArgumentType]  # psycopg adapts the list to a text array
        ),
        lambda found: len(found) >= expected,
        seconds=70,
    )
    assert len(rows) == expected, rows
    return tuple(rows)


def _spend_row(*request_ids: str) -> Mapping[str, JsonValue]:
    row: Final = dict(_spend_rows(request_ids, 1)[0])
    return {key: value for key, value in row.items() if key != "request_id"}


def _user(text: str) -> Mapping[str, JsonValue]:
    return {"role": "user", "parts": [{"text": text}]}


def _call_part(call_id: str | None = None, index: int = 0) -> Mapping[str, JsonValue]:
    call: Final = {"name": _TOOL, "args": dict(_args(index)), **({"id": call_id} if call_id else {})}
    return {"function_call": call, **({"thoughtSignature": _SKIP_SIGNATURE} if call_id else {})}


def _result_part(response: JsonValue, call_id: str | None = None, parts: JsonValue = None) -> Mapping[str, JsonValue]:
    result: Final = {
        "name": _TOOL,
        "response": response,
        **({"id": call_id} if call_id else {}),
        **({"parts": parts} if parts is not None else {}),
    }
    return {"function_response": result}


def _expected_body(endpoint: Endpoint, *responses: JsonValue) -> Mapping[str, JsonValue]:
    return {
        "contents": [
            _user(_QUESTION),
            {"role": "model", "parts": [_call_part(index=index) for index in range(len(responses))]},
            {"role": "user", "parts": [_result_part(response) for response in responses]},
        ],
        **_DECLARATIONS,
        **({"generationConfig": {"max_output_tokens": 64}} if endpoint == "messages" else {}),
    }


def _only_request(wire: Wire, provider: Provider, stream: bool, backend: str = _BACKEND) -> Mapping[str, JsonValue]:
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [
        ("POST", _target(provider, stream, backend))
    ], received
    header, value = _expected_auth(provider)
    assert received[0].headers[header] == value, dict(received[0].headers)
    return _JSON_OBJECT.validate_json(received[0].body)


def _matrix() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(
            provider, endpoint, stream, client, id=f"{provider}-{endpoint}-{'stream' if stream else 'sync'}-{client}"
        )
        for provider, endpoint, stream, client in itertools.product(_PROVIDERS, _ENDPOINTS, (False, True), _CLIENTS)
    )


@pytest.mark.parametrize(("provider", "endpoint", "stream", "client"), _matrix())
async def test_a_tool_result_holding_a_string_ref_reaches_gemini_as_its_raw_text(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool, client: Client
) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(_vendor((response_id,), stream)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, provider, wire)
        observed: Final = await _observe(gateway, endpoint, client, model, stream, _SCHEMA_TEXT)
        assert observed.answer == _ANSWER, observed
        _assert_caller_id(endpoint, stream, observed.response_id, response_id)
        assert _only_request(wire, provider, stream) == _expected_body(endpoint, {"content": _SCHEMA_TEXT})
        assert _spend_row(observed.response_id, response_id) == {
            "model_group": model,
            "call_type": _call_type(endpoint),
            "status": "success",
            "prompt_tokens": 20,
            "completion_tokens": 5,
        }


_LIST_OF_LISTS_TEXT: Final = '{"rows":[[{"$ref":"#/users/1"},"alice"]]}'


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    tuple(pytest.param(p, e, id=f"{p}-{e}") for p, e in itertools.product(_PROVIDERS, _ENDPOINTS)),
)
@pytest.mark.parametrize(
    "text",
    (pytest.param(_PLAIN_TEXT, id="no-ref"), pytest.param(_LIST_OF_LISTS_TEXT, id="ref-under-a-list-of-lists")),
)
def test_a_tool_result_without_a_part_reference_still_reaches_gemini_structured(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, text: str
) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(_vendor((response_id,), False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, provider, wire)
        observed: Final = _httpx(gateway, endpoint, model, False, text)
        assert observed.answer == _ANSWER, observed
        _assert_caller_id(endpoint, False, observed.response_id, response_id)
        assert _only_request(wire, provider, False) == _expected_body(endpoint, json.loads(text))
        assert _spend_row(observed.response_id, response_id)["status"] == "success"


_CHART_TEXT: Final = '{"chart":{"$ref":"#/$defs/Chart"},"$defs":{"Chart":{"type":"string"}}}'
_INLINE_PNG: Final[list[JsonValue]] = [{"inline_data": {"data": _PNG, "mime_type": "image/png"}}]


def _multimodal_body(endpoint: Endpoint, model: str) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return {
                "model": model,
                "messages": _chat_messages(
                    (
                        [
                            {"type": "text", "text": _CHART_TEXT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG}"}},
                        ],
                    )
                ),
                **_extra(endpoint),
            }
        case "messages":
            return {
                "model": model,
                "max_tokens": 64,
                "messages": _messages_history(
                    [
                        {"type": "text", "text": _CHART_TEXT},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _PNG}},
                    ]
                ),
                **_extra(endpoint),
            }
        case "responses":
            raise AssertionError(endpoint)


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    (
        pytest.param("gemini", "chat", id="gemini-chat"),
        pytest.param("vertex_ai", "chat", id="vertex_ai-chat"),
        pytest.param("gemini", "messages", id="gemini-messages"),
    ),
)
def test_a_multimodal_tool_result_with_a_ref_keeps_its_image_parts_beside_the_raw_text(
    gateway: Gateway, provider: Provider, endpoint: Endpoint
) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(_vendor((response_id,), False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, provider, wire)
        response: Final = gateway.request(
            "POST", _path(endpoint), _multimodal_body(endpoint, model), headers={"anthropic-version": "2023-06-01"}
        )
        observed: Final = _http_observed(endpoint, False, response)
        assert observed == _Observed(response_id, _ANSWER), observed
        expected: Final = _expected_body(endpoint, {"content": _CHART_TEXT})
        assert _only_request(wire, provider, False) == {
            **expected,
            "contents": [
                *_items(expected["contents"])[:2],
                {"role": "user", "parts": [_result_part({"content": _CHART_TEXT}, parts=_INLINE_PNG)]},
            ],
        }
        assert _spend_row(response_id)["status"] == "success"


_CACHE_NAME: Final = "cachedContents/synthetic-cache"
_CACHED_POLICY: Final = " ".join(f"policy clause {index} applies" for index in range(600))
_EPHEMERAL: Final[Mapping[str, JsonValue]] = {"type": "ephemeral"}


def _cached_chat_history(tool_text: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": [{"type": "text", "text": _CACHED_POLICY, "cache_control": dict(_EPHEMERAL)}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": _TOOL, "arguments": json.dumps(_ARGS)}}
            ],
            "cache_control": dict(_EPHEMERAL),
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [{"type": "text", "text": tool_text, "cache_control": dict(_EPHEMERAL)}],
        },
        {"role": "user", "content": [{"type": "text", "text": "Acknowledged.", "cache_control": dict(_EPHEMERAL)}]},
        {"role": "user", "content": _FOLLOW_UP},
    ]


def _cached_messages_history(tool_text: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": [{"type": "text", "text": _CACHED_POLICY, "cache_control": dict(_EPHEMERAL)}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_0",
                    "name": _TOOL,
                    "input": dict(_ARGS),
                    "cache_control": dict(_EPHEMERAL),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_0",
                    "content": [{"type": "text", "text": tool_text, "cache_control": dict(_EPHEMERAL)}],
                },
                {"type": "text", "text": "Acknowledged.", "cache_control": dict(_EPHEMERAL)},
            ],
        },
        {"role": "user", "content": _FOLLOW_UP},
    ]


def _cache_split(endpoint: Endpoint) -> tuple[list[JsonValue], list[JsonValue]]:
    tool_turns: Final[list[JsonValue]] = [
        {"role": "model", "parts": [_call_part()]},
        {"role": "user", "parts": [_result_part({"content": _SCHEMA_TEXT})]},
    ]
    if endpoint == "chat":
        return [_user(_CACHED_POLICY), *tool_turns, _user("Acknowledged.")], [_user(_FOLLOW_UP)]
    return [_user(_CACHED_POLICY)], [
        *tool_turns,
        {"role": "user", "parts": [{"text": "Acknowledged."}, {"text": _FOLLOW_UP}]},
    ]


@pytest.mark.parametrize("endpoint", ("chat", "messages"))
def test_a_context_cached_tool_result_with_a_ref_is_stored_as_its_raw_text(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b"{}")
        unresolved: Final = _unresolved_reference(request.body)
        if unresolved is not None:
            return _rejection(unresolved)
        if request.target.endswith("cachedContents"):
            return Reply(body=json.dumps({"name": _CACHE_NAME, "model": f"models/{_BACKEND}"}).encode())
        return _reply(response_id, False)

    body: Final = (
        {"messages": _cached_chat_history(_SCHEMA_TEXT)}
        if endpoint == "chat"
        else {"max_tokens": 64, "messages": _cached_messages_history(_SCHEMA_TEXT)}
    )
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        response: Final = gateway.request(
            "POST",
            _path(endpoint),
            {"model": model, **body, **_CACHE_BUST},
            headers={"anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["id"] == response_id, response.text
        received: Final = wire.drain()
        assert [(request.method, request.target.endswith("cachedContents")) for request in received] == [
            ("GET", True),
            ("POST", True),
            ("POST", False),
        ], received
        stored: Final = _JSON_OBJECT.validate_json(received[1].body)
        assert isinstance(stored["displayName"], str) and stored["displayName"], stored
        cached_turns, live_turns = _cache_split(endpoint)
        assert stored == {
            "contents": cached_turns,
            "model": f"models/{_BACKEND}",
            "displayName": stored["displayName"],
            "tools": None,
        }, stored
        assert _JSON_OBJECT.validate_json(received[2].body) == {
            "contents": live_turns,
            "cachedContent": _CACHE_NAME,
            **({"generationConfig": {"max_output_tokens": 64}} if endpoint == "messages" else {}),
        }
        assert _spend_row(response_id)["status"] == "success"


def _gcs_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target.startswith(_UPLOAD_PREFIX), request.target
    assert request.headers["authorization"] == "Bearer scripted-token"
    name: Final = unquote(request.target.removeprefix(_UPLOAD_PREFIX))
    stored: Final = {
        "kind": "storage#object",
        "id": f"{_BUCKET}/{name}/1759950000000000",
        "name": name,
        "bucket": _BUCKET,
        "size": str(len(request.body)),
        "timeCreated": "2026-10-15T08:00:00.000Z",
        "contentType": "application/octet-stream",
    }
    return Reply(body=json.dumps(stored).encode())


def _batch_line(url: str, model: str, tool_text: str) -> Mapping[str, JsonValue]:
    body: Final = (
        {"model": model, "messages": _chat_messages((tool_text,)), "tools": _extra("chat")["tools"]}
        if url == "/v1/chat/completions"
        else {"model": model, "input": _responses_input(tool_text), "tools": _extra("responses")["tools"]}
    )
    return {"custom_id": "schema-row", "method": "POST", "url": url, "body": body}


@pytest.mark.parametrize("url", ("/v1/chat/completions", "/v1/responses"))
def test_a_vertex_batch_line_with_a_ref_tool_result_is_stored_as_its_raw_text(gateway: Gateway, url: str) -> None:
    with wire_server(_gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"vertex_ai/{_BACKEND}",
            api_key=None,
            api_base=wire.url,
            vertex_project=_PROJECT,
            vertex_location=_LOCATION,
            vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url),
            gcs_bucket_name=_BUCKET,
        )
        lines: Final = (_batch_line(url, model, _SCHEMA_TEXT), _batch_line(url, model, _PLAIN_TEXT))
        response: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {
                "file": (
                    "in.jsonl",
                    ("\n".join(json.dumps(line) for line in lines) + "\n").encode(),
                    "application/jsonl",
                )
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["purpose"] == "batch", response.text
        uploads: Final = wire.drain()
        assert len(uploads) == 1, [upload.target for upload in uploads]
        stored: Final = tuple(
            _object(_JSON_OBJECT.validate_json(line)["request"])
            for line in uploads[0].body.splitlines()
            if line.strip()
        )
        assert [request["contents"] for request in stored] == [
            _expected_body("chat", {"content": _SCHEMA_TEXT})["contents"],
            _expected_body("chat", json.loads(_PLAIN_TEXT))["contents"],
        ]
        assert [request["tools"] for request in stored] == [_DECLARATIONS["tools"]] * 2
        assert [sorted(request) for request in stored] == [["contents", "labels", "tools"]] * 2


_NATIVE_CONTENTS: Final[list[JsonValue]] = [
    {"role": "user", "parts": [{"text": "Describe the chart."}]},
    {"role": "model", "parts": [{"functionCall": {"name": "render_chart", "args": {}}}]},
    {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "render_chart",
                    "response": {"image": {"$ref": "chart.png"}},
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": _PNG, "displayName": "chart.png"}}],
                }
            }
        ],
    },
]


def test_a_native_generate_content_part_reference_is_forwarded_verbatim(gateway: Gateway) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(_vendor((response_id,), False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        response: Final = gateway.request(
            "POST", f"/v1beta/models/{model}:generateContent", {"contents": _NATIVE_CONTENTS}
        )
        assert response.status_code == 200, response.text
        candidate: Final = _object(_items(_JSON_OBJECT.validate_json(response.content)["candidates"])[0])
        assert _object(candidate["content"])["parts"] == [{"text": _ANSWER}], response.text
        assert _only_request(wire, "gemini", False) == {
            "model": _BACKEND,
            "contents": _NATIVE_CONTENTS,
            "tools": None,
            "generationConfig": {},
        }


def test_a_gemini_3_tool_result_with_a_ref_keeps_its_call_id_beside_the_raw_text(gateway: Gateway) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(_vendor((response_id,), False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire, _ARGON)
        observed: Final = _httpx(gateway, "chat", model, False, _SCHEMA_TEXT)
        assert observed == _Observed(response_id, _ANSWER), observed
        assert _only_request(wire, "gemini", False, _ARGON) == {
            "contents": [
                _user(_QUESTION),
                {"role": "model", "parts": [_call_part("call_0")]},
                {"role": "user", "parts": [_result_part({"content": _SCHEMA_TEXT}, call_id="call_0")]},
            ],
            **_DECLARATIONS,
        }
        assert _spend_row(response_id)["status"] == "success"


def _send_chat(gateway: Gateway, model: str, tool_contents: Sequence[JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(tool_contents), **_extra("chat")}
    )


def _assert_answered(response: httpx.Response, response_id: str) -> None:
    assert response.status_code == 200, response.text
    assert _http_observed("chat", False, response) == _Observed(response_id, _ANSWER), response.text


_FIVE_KB_TEXT: Final = json.dumps(
    {"padding": "p" * 5000, "properties": {"value": {"$ref": "#/$defs/humanScalar"}}}, separators=(",", ":")
)
_SPLIT_PARTS: Final[list[JsonValue]] = [
    {"type": "text", "text": '{"value":{"$r'},
    {"type": "text", "text": 'ef":"#/$defs/humanScalar"}}'},
]


@pytest.mark.parametrize(
    ("tool_contents", "expected"),
    (
        pytest.param((5,), ({"content": ""},), id="content-int"),
        pytest.param(("",), ({"content": ""},), id="content-empty-string"),
        pytest.param(
            (_SPLIT_PARTS,),
            ({"content": '{"value":{"$ref":"#/$defs/humanScalar"}}'},),
            id="ref-key-split-across-text-parts",
        ),
        pytest.param((_FIVE_KB_TEXT,), ({"content": _FIVE_KB_TEXT},), id="5kb-ref-at-the-end"),
        pytest.param(
            (_SCHEMA_TEXT, _PLAIN_TEXT),
            ({"content": _SCHEMA_TEXT}, json.loads(_PLAIN_TEXT)),
            id="two-results-one-ref",
        ),
        pytest.param(
            (_SCHEMA_TEXT, _SCHEMA_TEXT),
            ({"content": _SCHEMA_TEXT}, {"content": _SCHEMA_TEXT}),
            id="two-results-both-ref",
        ),
        pytest.param(('{"v":{"$ref":"#/a","$ref":null}}',), ({"v": {"$ref": None}},), id="ref-twice-string-then-null"),
        pytest.param(
            ('{"v":{"$ref":null,"$ref":"#/a"}}',),
            ({"content": '{"v":{"$ref":null,"$ref":"#/a"}}'},),
            id="ref-twice-null-then-string",
        ),
        pytest.param(('{"v":{"$ref":"#/a"',), ({"content": '{"v":{"$ref":"#/a"'},), id="malformed-json"),
        pytest.param(('[{"$ref":"#/a"}]',), ({"content": '[{"$ref":"#/a"}]'},), id="array-root"),
    ),
)
def test_sad_tool_result_shapes_reach_gemini_in_the_form_it_accepts(
    gateway: Gateway, tool_contents: tuple[JsonValue, ...], expected: tuple[JsonValue, ...]
) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(_vendor((response_id,), False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        _assert_answered(_send_chat(gateway, model, tool_contents), response_id)
        assert _only_request(wire, "gemini", False) == _expected_body("chat", *expected)
        assert _spend_row(response_id)["status"] == "success"


_UPSTREAM_ERRORS: Final = (
    pytest.param(401, "UNAUTHENTICATED", "Request had invalid authentication credentials.", id="401"),
    pytest.param(404, "NOT_FOUND", f"models/{_BACKEND} is not found for API version v1beta", id="404"),
    pytest.param(
        400,
        "INVALID_ARGUMENT",
        "The input token count (2000000) exceeds the maximum number of tokens allowed (1048576).",
        id="400-oversized",
    ),
)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize(("status", "reason", "message"), _UPSTREAM_ERRORS)
def test_an_upstream_error_on_a_ref_tool_result_reaches_the_caller_and_the_next_request_succeeds(
    gateway: Gateway, endpoint: Endpoint, status: int, reason: str, message: str
) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    served: Final = itertools.count()
    vendor: Final = _vendor((response_id,), False)

    def respond(request: Request) -> Reply:
        if next(served) == 0:
            error: Final = {"error": {"code": status, "message": message, "status": reason}}
            return Reply(status=status, body=json.dumps(error).encode())
        return vendor(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        failed: Final = gateway.request(
            "POST",
            _path(endpoint),
            _http_body(endpoint, model, _SCHEMA_TEXT, False),
            headers={"anthropic-version": "2023-06-01"},
        )
        assert failed.status_code == status, failed.text
        assert message in failed.text, failed.text
        assert _only_request(wire, "gemini", False) == _expected_body(endpoint, {"content": _SCHEMA_TEXT})
        recovered: Final = _httpx(gateway, endpoint, model, False, _SCHEMA_TEXT)
        assert recovered.answer == _ANSWER, recovered
        _assert_caller_id(endpoint, False, recovered.response_id, response_id)
        assert _only_request(wire, "gemini", False) == _expected_body(endpoint, {"content": _SCHEMA_TEXT})


def test_a_ref_tool_result_without_a_matching_call_fails_before_reaching_gemini(gateway: Gateway) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(_vendor((response_id,), False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        orphan: Final = [
            *_chat_messages((_SCHEMA_TEXT,))[:2],
            {"role": "tool", "tool_call_id": "call_unknown", "content": _SCHEMA_TEXT},
        ]
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": orphan, **_extra("chat")}
        )
        assert response.status_code == 500, response.text
        assert "Missing corresponding tool call for tool response message" in response.text, response.text
        assert wire.drain() == ()
        _assert_answered(_send_chat(gateway, model, (_SCHEMA_TEXT,)), response_id)


def _escaped(key: str, index: int) -> str:
    return key[:index] + f"\\u{ord(key[index]):04x}" + key[index + 1 :]


@pytest.mark.parametrize(
    "text",
    (
        pytest.param('{"v":{"$ref":null}}', id="ref-null"),
        pytest.param('{"v":{"$ref":""}}', id="ref-empty-string"),
        pytest.param('{"v":{"$ref":5}}', id="ref-int"),
        pytest.param('{"v":{"$ref":true}}', id="ref-bool"),
        pytest.param('{"v":{"$ref":{"a":1}}}', id="ref-object"),
        pytest.param('{"v":{"$ref":["#/a"]}}', id="ref-list"),
        pytest.param('{"note":"$ref","tags":["$ref"]}', id="ref-as-a-value"),
        pytest.param(json.dumps({"text": json.dumps({"$ref": "#/a"})}), id="ref-inside-a-json-string"),
        pytest.param('{"price":"\\u0024 5"}', id="escaped-dollar-in-a-value"),
        pytest.param("{}", id="empty-object"),
        pytest.param(_LIST_OF_LISTS_TEXT, id="ref-under-a-list-of-lists"),
        pytest.param('{"rows":[[{"cell":{"$ref":"#/users/1"}}]]}', id="ref-in-an-object-under-a-list-of-lists"),
        pytest.param('{"rows":[[{"cell":[{"$ref":"#/users/1"}]}]]}', id="ref-in-a-list-under-a-list-of-lists"),
        pytest.param('{"grid":[[[{"$ref":"#/users/1"}]]]}', id="ref-under-three-lists"),
        pytest.param('{"a":[{"b":[[{"$ref":"#/users/1"}]]}]}', id="list-of-lists-under-an-object-in-a-list"),
    ),
)
def test_edge_tool_results_gemini_accepts_stay_structured(gateway: Gateway, text: str) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(_vendor((response_id,), False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        _assert_answered(_send_chat(gateway, model, (text,)), response_id)
        assert _only_request(wire, "gemini", False) == _expected_body("chat", json.loads(text))


def _deep(levels: int, leaf: JsonValue) -> str:
    return '{"n":' * levels + json.dumps(leaf, separators=(",", ":")) + "}" * levels


@pytest.mark.parametrize(
    "text",
    (
        *(
            pytest.param('{"v":{"' + _escaped("$ref", index) + '":"#/a"}}', id=f"escaped-key-{index}")
            for index in range(4)
        ),
        pytest.param('{"$ref":"#/$defs/Config","$defs":{"Config":{"type":"object"}}}', id="ref-on-the-root"),
        pytest.param(_deep(200, {"$ref": "#/deep"}), id="ref-200-levels-deep"),
        pytest.param('{"items":[{"a":1},{"b":[{"$ref":"#/b"}]}]}', id="ref-inside-a-list"),
        pytest.param('{"rows":[{"$ref":"#/users/1"},"alice"]}', id="ref-in-a-one-level-list"),
        pytest.param(
            '{"rows":[[{"$ref":"#/users/1"}]],"owner":{"$ref":"#/users/2"}}',
            id="list-of-lists-ref-beside-a-reachable-ref",
        ),
    ),
)
def test_edge_tool_results_with_a_string_ref_reach_gemini_as_raw_text(gateway: Gateway, text: str) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(_vendor((response_id,), False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        _assert_answered(_send_chat(gateway, model, (text,)), response_id)
        assert _only_request(wire, "gemini", False) == _expected_body("chat", {"content": text})


def test_the_same_ref_tool_result_sent_twice_is_wrapped_and_logged_once_each(gateway: Gateway) -> None:
    response_ids: Final = (f"gemini-reply-{uuid.uuid4().hex}", f"gemini-reply-{uuid.uuid4().hex}")
    with wire_server(_vendor(response_ids, False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        answers: Final = tuple(_send_chat(gateway, model, (_SCHEMA_TEXT,)) for _ in response_ids)
        for answer, response_id in zip(answers, response_ids, strict=True):
            _assert_answered(answer, response_id)
        received: Final = wire.drain()
        assert [_JSON_OBJECT.validate_json(request.body) for request in received] == [
            _expected_body("chat", {"content": _SCHEMA_TEXT})
        ] * 2
        rows: Final = _spend_rows(response_ids, 2)
        assert sorted(str(row["request_id"]) for row in rows) == sorted(response_ids)
        assert [row["status"] for row in rows] == ["success", "success"]


_BODY_BYTES: Final = 8 * 1024 * 1024


def _escape_heavy(with_reference: bool) -> str:
    count: Final = _BODY_BYTES // 6
    tail: Final = '"\\u0024ref":"#/a"' if with_reference else '"tail":"#/a"'
    filler: Final = "\\u0000" * count if with_reference else "a" * (6 * count)
    return '{"blob":"' + filler + '","end":{' + tail + "}}"


def _node_heavy(with_reference: bool) -> str:
    count: Final = _BODY_BYTES // 3
    tail: Final = '{"$ref":"#/a"}' if with_reference else '{"tail":"#/a"}'
    return '{"items":[' + "{}," * count + tail + "]}"


def _list_heavy(with_reference: bool) -> str:
    count: Final = _BODY_BYTES // 3
    tail: Final = '{"$ref":"#/a"}' if with_reference else '{"tail":"#/a"}'
    return '{"items":[' + "[]," * count + tail + "]}"


def _send_large_chat(gateway: Gateway, model: str, tool_text: str) -> httpx.Response:
    return gateway.client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": _chat_messages((tool_text,)), **_extra("chat")},
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=120,
    )


def _timed_liveliness(gateway: Gateway) -> tuple[float, int]:
    started: Final = time.perf_counter()
    probe: Final = gateway.client.get("/health/liveliness")
    return time.perf_counter() - started, probe.status_code


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "build",
    (
        pytest.param(_escape_heavy, id="escapes"),
        pytest.param(_node_heavy, id="nodes"),
        pytest.param(_list_heavy, id="lists"),
    ),
)
def test_the_worst_case_ref_scan_stays_within_ten_times_a_same_size_control(
    gateway: Gateway, build: Callable[[bool], str], record_property: Callable[[str, object], None]
) -> None:
    response_ids: Final = (f"gemini-reply-{uuid.uuid4().hex}", f"gemini-reply-{uuid.uuid4().hex}")
    worst_text: Final = build(True)
    samples: Final[SimpleQueue[tuple[float, int]]] = SimpleQueue()
    stop: Final = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            samples.put(_timed_liveliness(gateway))

    with wire_server(_vendor(response_ids, False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        control: Final = timed(lambda: _send_large_chat(gateway, model, build(False)))
        poller: Final = threading.Thread(target=poll)
        poller.start()
        worst: Final = timed(lambda: _send_large_chat(gateway, model, worst_text))
        stop.set()
        poller.join()
        liveliness: Final = tuple(samples.get_nowait() for _ in range(samples.qsize()))
        record_property("body_bytes", len(worst_text))
        record_property("control_seconds", round(control.seconds, 3))
        record_property("worst_seconds", round(worst.seconds, 3))
        record_property("liveliness_max_seconds", round(max(latency for latency, _ in liveliness), 3))
        _assert_answered(control.value, response_ids[0])
        _assert_answered(worst.value, response_ids[1])
        assert_within_factor_of_control(worst.seconds, control=control.seconds, factor=10)
        assert {status for _, status in liveliness} == {200}, liveliness
        assert_within_factor_of_control(max(latency for latency, _ in liveliness), control=control.seconds, factor=10)
        received: Final = wire.drain()
        assert len(received) == 2, [request.target for request in received]
        assert _function_response_of(received[1]) == {"content": worst_text}


def _function_response_of(request: Request) -> JsonValue:
    responses: Final = tuple(_function_responses(_JSON_OBJECT.validate_json(request.body)))
    assert len(responses) == 1, len(responses)
    return responses[0]["response"]

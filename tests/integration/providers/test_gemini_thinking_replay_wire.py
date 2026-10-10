import base64
import itertools
import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import anthropic
import httpx
import openai
import pytest
import yaml
from _pytest.mark.structures import ParameterSet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.responses import ResponseCompletedEvent
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with

_BACKEND: Final = "gemini-2.5-flash"
_API_KEY: Final = "synthetic-gemini-key"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_SALT: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")
_CLAUDE_SIGNATURE: Final = "CAQSyAsKEAgSGAI4AUIIdGhpbmtpbmcSDAlY-synthetic-claude-signature"
_GEMINI_SIGNATURE: Final = "synthetic-gemini-thought-signature"
_REASONING: Final = "private thought about fruit"
_QUESTION: Final = "How many apples are left?"
_PRIOR_ANSWER: Final = "Two apples."
_FOLLOW_UP: Final = "And after eating one?"
_ANSWER: Final = "One left."
_CACHE_BUST: Final[Mapping[str, JsonValue]] = {"cache": {"no-cache": True}}
_USAGE: Final = {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25}
_FUNCTION: Final = {
    "name": "count_fruit",
    "description": "Count fruit",
    "parameters": {"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]},
}
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


def _thinking_block(signature: JsonValue, thinking: str = _REASONING) -> Mapping[str, JsonValue]:
    return {"type": "thinking", "thinking": thinking, "signature": signature}


def _user(text: str) -> Mapping[str, JsonValue]:
    return {"role": "user", "parts": [{"text": text}]}


def _model_turn(*parts: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {"role": "model", "parts": list(parts)}


def _thought(text: str = _REASONING) -> Mapping[str, JsonValue]:
    return {"thought": True, "text": text}


_REPLAYED_TURN: Final = _model_turn(_thought(), {"text": _PRIOR_ANSWER})


def _chat_history(assistant_fields: Mapping[str, JsonValue]) -> Sequence[Mapping[str, JsonValue]]:
    return [
        {"role": "user", "content": _QUESTION},
        {"role": "assistant", "content": _PRIOR_ANSWER, **assistant_fields},
        {"role": "user", "content": _FOLLOW_UP},
    ]


_CHAT_HISTORY: Final = _chat_history(
    {"reasoning_content": _REASONING, "thinking_blocks": [_thinking_block(_CLAUDE_SIGNATURE)]}
)
_CONTROL_HISTORY: Final = _chat_history({})


def _messages_history(*assistant_blocks: Mapping[str, JsonValue]) -> Sequence[Mapping[str, JsonValue]]:
    return [
        {"role": "user", "content": _QUESTION},
        {"role": "assistant", "content": [*assistant_blocks, {"type": "text", "text": _PRIOR_ANSWER}]},
        {"role": "user", "content": _FOLLOW_UP},
    ]


_MESSAGES_HISTORY: Final = _messages_history(_thinking_block(_CLAUDE_SIGNATURE))


def _responses_input(*blocks: Mapping[str, JsonValue]) -> Sequence[Mapping[str, JsonValue]]:
    return [
        {"role": "user", "content": _QUESTION},
        {
            "type": "reasoning",
            "id": "rs_prior",
            "summary": [{"type": "summary_text", "text": _REASONING}],
            "encrypted_content": json.dumps(list(blocks)),
        },
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_prior",
            "status": "completed",
            "content": [{"type": "output_text", "text": _PRIOR_ANSWER, "annotations": []}],
        },
        {"role": "user", "content": _FOLLOW_UP},
    ]


_RESPONSES_INPUT: Final = _responses_input(_thinking_block(_CLAUDE_SIGNATURE))


def _service_account_json(token_url: str) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": _PROJECT,
            "private_key_id": "scripted",
            "private_key": private_key,
            "client_email": f"scripted@{_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": f"{token_url}/_oauth/token",
        }
    )


def _register(gateway: Gateway, scenario: Scenario, provider: Provider, wire: Wire) -> str:
    if provider == "gemini":
        return scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
    return scenario.model(
        model=f"vertex_ai/{_BACKEND}",
        api_base=f"{wire.url}{_MODEL_PATH}",
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=_service_account_json(gateway.upstream_url.rstrip("/")),
    )


def _target(provider: Provider, stream: bool) -> str:
    prefix: Final = f"/models/{_BACKEND}" if provider == "gemini" else _MODEL_PATH
    return f"{prefix}:streamGenerateContent?alt=sse" if stream else f"{prefix}:generateContent"


def _expected_auth(provider: Provider) -> tuple[str, str]:
    return ("x-goog-api-key", _API_KEY) if provider == "gemini" else ("authorization", "Bearer scripted-token")


def _frame(response_id: str, parts: Sequence[Mapping[str, JsonValue]], finished: bool) -> Mapping[str, JsonValue]:
    return {
        "responseId": response_id,
        "candidates": [
            {
                "content": {"role": "model", "parts": list(parts)},
                "index": 0,
                **({"finishReason": "STOP"} if finished else {}),
            }
        ],
        "usageMetadata": dict(_USAGE),
        "modelVersion": _BACKEND,
    }


def _reply(response_id: str, stream: bool, parts: Sequence[Mapping[str, JsonValue]] = ({"text": _ANSWER},)) -> Reply:
    if not stream:
        return Reply(body=json.dumps(_frame(response_id, parts, finished=True)).encode())
    frames: Final = (_frame(response_id, parts[:-1], finished=False), _frame(response_id, parts[-1:], finished=True))
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames if frame["candidates"]),
    )


def _single(values: AbstractSet[str]) -> str:
    assert len(values) == 1, values
    return next(iter(values))


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


def _chat_sync(gateway: Gateway, model: str, stream: bool) -> _Observed:
    client: Final = _openai(gateway)
    if not stream:
        reply: Final = client.chat.completions.create(model=model, messages=_CHAT_HISTORY, extra_body=dict(_CACHE_BUST))
        return _Observed(reply.id, reply.choices[0].message.content or "")
    chunks: Final = list(
        client.chat.completions.create(model=model, messages=_CHAT_HISTORY, stream=True, extra_body=dict(_CACHE_BUST))
    )
    return _Observed(
        _single({chunk.id for chunk in chunks}),
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices),
    )


async def _chat_async(gateway: Gateway, model: str, stream: bool) -> _Observed:
    client: Final = _async_openai(gateway)
    if not stream:
        reply: Final = await client.chat.completions.create(
            model=model, messages=_CHAT_HISTORY, extra_body=dict(_CACHE_BUST)
        )
        return _Observed(reply.id, reply.choices[0].message.content or "")
    chunks: Final = [
        chunk
        async for chunk in await client.chat.completions.create(
            model=model, messages=_CHAT_HISTORY, stream=True, extra_body=dict(_CACHE_BUST)
        )
    ]
    return _Observed(
        _single({chunk.id for chunk in chunks}),
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices),
    )


def _messages_sync(gateway: Gateway, model: str, stream: bool) -> _Observed:
    client: Final = _anthropic(gateway)
    if not stream:
        reply: Final = client.messages.create(
            model=model, max_tokens=64, messages=_MESSAGES_HISTORY, extra_body=dict(_CACHE_BUST)
        )
        return _Observed(reply.id, "".join(block.text for block in reply.content if block.type == "text"))
    with client.messages.stream(
        model=model, max_tokens=64, messages=_MESSAGES_HISTORY, extra_body=dict(_CACHE_BUST)
    ) as streamed:
        final: Final = streamed.get_final_message()
    return _Observed(final.id, "".join(block.text for block in final.content if block.type == "text"))


async def _messages_async(gateway: Gateway, model: str, stream: bool) -> _Observed:
    client: Final = _async_anthropic(gateway)
    if not stream:
        reply: Final = await client.messages.create(
            model=model, max_tokens=64, messages=_MESSAGES_HISTORY, extra_body=dict(_CACHE_BUST)
        )
        return _Observed(reply.id, "".join(block.text for block in reply.content if block.type == "text"))
    async with client.messages.stream(
        model=model, max_tokens=64, messages=_MESSAGES_HISTORY, extra_body=dict(_CACHE_BUST)
    ) as streamed:
        final: Final = await streamed.get_final_message()
    return _Observed(final.id, "".join(block.text for block in final.content if block.type == "text"))


def _responses_sync(gateway: Gateway, model: str, stream: bool) -> _Observed:
    client: Final = _openai(gateway)
    if not stream:
        reply: Final = client.responses.create(model=model, input=_RESPONSES_INPUT, extra_body=dict(_CACHE_BUST))
        return _Observed(reply.id, reply.output_text)
    events: Final = list(
        client.responses.create(model=model, input=_RESPONSES_INPUT, stream=True, extra_body=dict(_CACHE_BUST))
    )
    completed: Final = events[-1]
    assert isinstance(completed, ResponseCompletedEvent), events
    return _Observed(completed.response.id, completed.response.output_text)


async def _responses_async(gateway: Gateway, model: str, stream: bool) -> _Observed:
    client: Final = _async_openai(gateway)
    if not stream:
        reply: Final = await client.responses.create(model=model, input=_RESPONSES_INPUT, extra_body=dict(_CACHE_BUST))
        return _Observed(reply.id, reply.output_text)
    events: Final = [
        event
        async for event in await client.responses.create(
            model=model, input=_RESPONSES_INPUT, stream=True, extra_body=dict(_CACHE_BUST)
        )
    ]
    completed: Final = events[-1]
    assert isinstance(completed, ResponseCompletedEvent), events
    return _Observed(completed.response.id, completed.response.output_text)


def _sse_payloads(text: str) -> tuple[Mapping[str, JsonValue], ...]:
    lines: Final = tuple(line for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]")
    return tuple(_JSON_OBJECT.validate_json(line.removeprefix("data: ").encode()) for line in lines)


def _httpx_chat(gateway: Gateway, model: str, stream: bool) -> _Observed:
    response: Final = gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": _CHAT_HISTORY, "stream": stream, **_CACHE_BUST}
    )
    assert response.status_code == 200, response.text
    if not stream:
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        choice: Final = _JSON_OBJECT.validate_python(_JSON_LIST.validate_python(payload["choices"])[0])
        return _Observed(str(payload["id"]), str(_JSON_OBJECT.validate_python(choice["message"])["content"]))
    assert response.text.rstrip().endswith("data: [DONE]"), response.text
    chunks: Final = _sse_payloads(response.text)
    deltas: Final = tuple(
        _JSON_OBJECT.validate_python(
            _JSON_OBJECT.validate_python(_JSON_LIST.validate_python(chunk["choices"])[0])["delta"]
        )
        for chunk in chunks
    )
    return _Observed(
        _single({str(chunk["id"]) for chunk in chunks}),
        "".join(str(delta["content"]) for delta in deltas if delta.get("content")),
    )


def _httpx_messages(gateway: Gateway, model: str, stream: bool) -> _Observed:
    response: Final = gateway.request(
        "POST",
        "/v1/messages",
        {"model": model, "max_tokens": 64, "messages": _MESSAGES_HISTORY, "stream": stream, **_CACHE_BUST},
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200, response.text
    if not stream:
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        blocks: Final = tuple(
            _JSON_OBJECT.validate_python(block) for block in _JSON_LIST.validate_python(payload["content"])
        )
        return _Observed(str(payload["id"]), "".join(str(block["text"]) for block in blocks if block["type"] == "text"))
    events: Final = _sse_payloads(response.text)
    starts: Final = tuple(event for event in events if event["type"] == "message_start")
    assert events[-1]["type"] == "message_stop", events
    deltas: Final = tuple(
        _JSON_OBJECT.validate_python(event["delta"]) for event in events if event["type"] == "content_block_delta"
    )
    return _Observed(
        str(_JSON_OBJECT.validate_python(starts[0]["message"])["id"]),
        "".join(str(delta["text"]) for delta in deltas if delta["type"] == "text_delta"),
    )


def _httpx_responses(gateway: Gateway, model: str, stream: bool) -> _Observed:
    response: Final = gateway.request(
        "POST", "/v1/responses", {"model": model, "input": _RESPONSES_INPUT, "stream": stream, **_CACHE_BUST}
    )
    assert response.status_code == 200, response.text
    if not stream:
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        return _Observed(str(payload["id"]), _output_text(payload))
    events: Final = _sse_payloads(response.text)
    assert events[-1]["type"] == "response.completed", events
    completed: Final = _JSON_OBJECT.validate_python(events[-1]["response"])
    return _Observed(str(completed["id"]), _output_text(completed))


def _content_parts(item: Mapping[str, JsonValue]) -> Iterator[Mapping[str, JsonValue]]:
    return (_JSON_OBJECT.validate_python(part) for part in _JSON_LIST.validate_python(item["content"]))


def _output_text(payload: Mapping[str, JsonValue]) -> str:
    items: Final = tuple(_JSON_OBJECT.validate_python(item) for item in _JSON_LIST.validate_python(payload["output"]))
    messages: Final = tuple(item for item in items if item["type"] == "message")
    parts: Final = tuple(itertools.chain.from_iterable(_content_parts(item) for item in messages))
    return "".join(str(part["text"]) for part in parts if part["type"] == "output_text")


async def _observe(gateway: Gateway, endpoint: Endpoint, client: Client, model: str, stream: bool) -> _Observed:
    match (endpoint, client):
        case ("chat", "sdk_sync"):
            return _chat_sync(gateway, model, stream)
        case ("chat", "sdk_async"):
            return await _chat_async(gateway, model, stream)
        case ("chat", "httpx"):
            return _httpx_chat(gateway, model, stream)
        case ("messages", "sdk_sync"):
            return _messages_sync(gateway, model, stream)
        case ("messages", "sdk_async"):
            return await _messages_async(gateway, model, stream)
        case ("messages", "httpx"):
            return _httpx_messages(gateway, model, stream)
        case ("responses", "sdk_sync"):
            return _responses_sync(gateway, model, stream)
        case ("responses", "sdk_async"):
            return await _responses_async(gateway, model, stream)
        case ("responses", "httpx"):
            return _httpx_responses(gateway, model, stream)
    raise AssertionError((endpoint, client))


def _upstream_response_id(endpoint: Endpoint, caller_id: str) -> str:
    if endpoint != "responses":
        return caller_id
    decrypted: Final = decrypt_if_encrypted_with(caller_id.removeprefix("resp_"), _SALT)
    assert decrypted is not None, caller_id
    issued: Final = decrypted.split(";")[0].split("response_id:")[-1]
    return base64.b64decode(issued.removeprefix("resp_")).decode().split(";")[-1].removeprefix("response_id:")


def _call_type(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "acompletion"
        case "messages":
            return "anthropic_messages"
        case "responses":
            return "aresponses"


def _spend_row(*request_ids: str) -> Mapping[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model_group, call_type, status, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
            "WHERE request_id = ANY(%s)",
            (list(request_ids),),  # pyright: ignore[reportArgumentType]  # psycopg adapts the list to a text array
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _spend_proxy_server_request(request_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT proxy_server_request FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (request_id,),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return _JSON_OBJECT.validate_python(rows[0]["proxy_server_request"])


def _config_storing_prompts(directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"]["store_prompts_in_spend_logs"] = True
    path: Final = directory / "store-prompts.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _assert_caller_id(endpoint: Endpoint, stream: bool, caller_id: str, response_id: str) -> None:
    if endpoint == "messages" and stream:
        assert caller_id.startswith("msg_"), caller_id
        return
    assert _upstream_response_id(endpoint, caller_id) == response_id, caller_id


def _expected_body(endpoint: Endpoint, *contents: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {
        "contents": list(contents),
        **({"generationConfig": {"max_output_tokens": 64}} if endpoint == "messages" else {}),
    }


def _only_request(wire: Wire, provider: Provider, stream: bool) -> Mapping[str, JsonValue]:
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [("POST", _target(provider, stream))], received
    header, value = _expected_auth(provider)
    assert received[0].headers[header] == value, dict(received[0].headers)
    assert "thoughtSignature" not in received[0].body.decode(), received[0].body
    return _JSON_OBJECT.validate_json(received[0].body)


def _happy_cells() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(
            provider, endpoint, stream, client, id=f"{provider}-{endpoint}-{'stream' if stream else 'sync'}-{client}"
        )
        for provider, endpoint, stream, client in itertools.product(_PROVIDERS, _ENDPOINTS, (False, True), _CLIENTS)
    )


@pytest.mark.parametrize(("provider", "endpoint", "stream", "client"), _happy_cells())
async def test_claude_thinking_replay_reaches_gemini_as_a_thought_part_without_its_signature(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool, client: Client
) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return _reply(response_id, stream)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, provider, wire)
        observed: Final = await _observe(gateway, endpoint, client, model, stream)
        assert observed.answer == _ANSWER, observed
        _assert_caller_id(endpoint, stream, observed.response_id, response_id)
        assert _only_request(wire, provider, stream) == _expected_body(
            endpoint, _user(_QUESTION), _REPLAYED_TURN, _user(_FOLLOW_UP)
        )
        assert _spend_row(observed.response_id, response_id) == {
            "model_group": model,
            "call_type": _call_type(endpoint),
            "status": "success",
            "prompt_tokens": 20,
            "completion_tokens": 5,
        }


_GEMINI_TOOL_REPLY_PARTS: Final = (
    _thought("I should count."),
    {"functionCall": {"name": "count_fruit", "args": {"kind": "apple"}}, "thoughtSignature": _GEMINI_SIGNATURE},
)
_GEMINI_TOOL_REPLAYED_TURN: Final = _model_turn(
    _thought("I should count."),
    {"function_call": {"name": "count_fruit", "args": {"kind": "apple"}}, "thoughtSignature": _GEMINI_SIGNATURE},
)
_FUNCTION_RESPONSE_TURN: Final = {
    "role": "user",
    "parts": [{"function_response": {"name": "count_fruit", "response": {"content": "12"}}}],
}
_DECLARATIONS: Final = {"tools": [{"function_declarations": [_FUNCTION]}]}


def _tool_round_one(endpoint: Endpoint, model: str) -> Mapping[str, JsonValue]:
    match endpoint:
        case "chat":
            return {
                "model": model,
                "tools": [{"type": "function", "function": _FUNCTION}],
                "messages": [{"role": "user", "content": "Count the apples."}],
                **_CACHE_BUST,
            }
        case "messages":
            return {
                "model": model,
                "max_tokens": 64,
                "tools": [
                    {"name": "count_fruit", "description": "Count fruit", "input_schema": _FUNCTION["parameters"]}
                ],
                "messages": [{"role": "user", "content": "Count the apples."}],
                **_CACHE_BUST,
            }
        case "responses":
            return {
                "model": model,
                "tools": [{"type": "function", **_FUNCTION}],
                "input": [{"role": "user", "content": "Count the apples."}],
                **_CACHE_BUST,
            }


def _tool_round_two(
    endpoint: Endpoint, first: Mapping[str, JsonValue], payload: Mapping[str, JsonValue]
) -> Mapping[str, JsonValue]:
    match endpoint:
        case "chat":
            choice: Final = _JSON_OBJECT.validate_python(_JSON_LIST.validate_python(payload["choices"])[0])
            message: Final = _JSON_OBJECT.validate_python(choice["message"])
            call: Final = _JSON_OBJECT.validate_python(_JSON_LIST.validate_python(message["tool_calls"])[0])
            return {
                **first,
                "messages": [
                    *_JSON_LIST.validate_python(first["messages"]),
                    message,
                    {"role": "tool", "tool_call_id": call["id"], "content": "12"},
                ],
            }
        case "messages":
            blocks: Final = tuple(
                _JSON_OBJECT.validate_python(block) for block in _JSON_LIST.validate_python(payload["content"])
            )
            tool_use: Final = next(block for block in blocks if block["type"] == "tool_use")
            return {
                **first,
                "messages": [
                    *_JSON_LIST.validate_python(first["messages"]),
                    {"role": "assistant", "content": list(blocks)},
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "12"}],
                    },
                ],
            }
        case "responses":
            items: Final = tuple(
                _JSON_OBJECT.validate_python(item) for item in _JSON_LIST.validate_python(payload["output"])
            )
            function_call: Final = next(item for item in items if item["type"] == "function_call")
            return {
                **first,
                "input": [
                    *_JSON_LIST.validate_python(first["input"]),
                    *items,
                    {"type": "function_call_output", "call_id": function_call["call_id"], "output": "12"},
                ],
            }


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_gemini_own_tool_call_signature_is_still_replayed_on_the_function_call_part(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    rounds: Final = (f"gemini-reply-{uuid.uuid4().hex}", f"gemini-reply-{uuid.uuid4().hex}")
    calls: Final = itertools.count()

    def respond(request: Request) -> Reply:
        if next(calls) == 0:
            return _reply(rounds[0], stream=False, parts=_GEMINI_TOOL_REPLY_PARTS)
        return _reply(rounds[1], stream=False, parts=({"text": "Twelve apples."},))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        first: Final = _tool_round_one(endpoint, model)
        headers: Final = {"anthropic-version": "2023-06-01"}
        opened: Final = gateway.request("POST", _path(endpoint), first, headers=headers)
        assert opened.status_code == 200, opened.text
        assert _GEMINI_SIGNATURE in opened.text, opened.text
        second: Final = _tool_round_two(endpoint, first, _JSON_OBJECT.validate_json(opened.content))
        closed: Final = gateway.request("POST", _path(endpoint), second, headers=headers)
        assert closed.status_code == 200, closed.text
        assert "Twelve apples." in closed.text, closed.text
        received: Final = wire.drain()
        assert [(request.method, request.target) for request in received] == [("POST", _target("gemini", False))] * 2
        replay: Final = _JSON_OBJECT.validate_json(received[1].body)
        assert replay == {
            **_expected_body(endpoint, _user("Count the apples."), _GEMINI_TOOL_REPLAYED_TURN, _FUNCTION_RESPONSE_TURN),
            **_DECLARATIONS,
        }, replay
        assert _spend_row(rounds[1])["status"] == "success"


def test_responses_previous_response_id_replays_gemini_session_history(
    gateway: Gateway, tmp_path: Path
) -> None:
    first_prompt: Final = f"gemini-session-first-{uuid.uuid4().hex}"
    second_prompt: Final = f"gemini-session-second-{uuid.uuid4().hex}"
    first_answer: Final = f"gemini-answer-first-{uuid.uuid4().hex}"
    second_answer: Final = f"gemini-answer-second-{uuid.uuid4().hex}"
    first_provider_id: Final = f"gemini-response-first-{uuid.uuid4().hex}"
    second_provider_id: Final = f"gemini-response-second-{uuid.uuid4().hex}"
    calls: Final = itertools.count()

    def respond(request: Request) -> Reply:
        if next(calls) == 0:
            return _reply(first_provider_id, stream=False, parts=({"text": first_answer},))
        return _reply(second_provider_id, stream=False, parts=({"text": second_answer},))

    with (
        wire_server(respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_config_storing_prompts(tmp_path), workers=1) as prompt_gateway,
        prompt_gateway.scenario() as scenario,
    ):
        model: Final = _register(prompt_gateway, scenario, "gemini", wire)
        first_response: Final = prompt_gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": first_prompt, **_CACHE_BUST},
        )
        assert first_response.status_code == 200, first_response.text
        first_body: Final = _JSON_OBJECT.validate_json(first_response.content)
        first_response_id: Final = str(first_body["id"])
        assert _output_text(first_body) == first_answer
        first_spend_id: Final = _upstream_response_id("responses", first_response_id)
        assert _spend_row(first_spend_id)["status"] == "success"
        first_proxy_request: Final = _spend_proxy_server_request(first_spend_id)
        assert first_proxy_request["input"] == first_prompt, first_proxy_request

        second_response: Final = prompt_gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": second_prompt,
                "previous_response_id": first_response_id,
                **_CACHE_BUST,
            },
        )
        assert second_response.status_code == 200, second_response.text
        second_body: Final = _JSON_OBJECT.validate_json(second_response.content)
        second_response_id: Final = str(second_body["id"])
        assert _output_text(second_body) == second_answer
        assert _spend_row(_upstream_response_id("responses", second_response_id))["status"] == "success"

        requests: Final = wire.drain()
        assert [(request.method, request.target) for request in requests] == [
            ("POST", _target("gemini", False)),
            ("POST", _target("gemini", False)),
        ]
        first_provider_body: Final = _JSON_OBJECT.validate_json(requests[0].body)
        second_provider_body: Final = _JSON_OBJECT.validate_json(requests[1].body)
        assert first_provider_body == _expected_body("responses", _user(first_prompt))
        assert second_provider_body == _expected_body(
            "responses",
            _user(first_prompt),
            _model_turn({"text": first_answer}),
            _user(second_prompt),
        )


_FIVE_KB: Final = "s" * 5000
_ASSISTANT_SHAPES: Final = (
    pytest.param({"reasoning_content": _REASONING, "thinking_blocks": 5}, _REPLAYED_TURN, id="blocks-int"),
    pytest.param(
        {"reasoning_content": _REASONING, "thinking_blocks": _FIVE_KB}, _REPLAYED_TURN, id="blocks-5kb-string"
    ),
    pytest.param({"reasoning_content": _REASONING, "thinking_blocks": ""}, _REPLAYED_TURN, id="blocks-empty-string"),
    pytest.param(
        {"reasoning_content": _REASONING, "thinking_blocks": [{"type": "thinking"}]},
        _REPLAYED_TURN,
        id="block-without-thinking",
    ),
    pytest.param(
        {"reasoning_content": _REASONING, "thinking_blocks": [{"signature": "s"}]},
        _REPLAYED_TURN,
        id="block-without-type",
    ),
    pytest.param(
        {"reasoning_content": _REASONING, "thinking_blocks": [_thinking_block(5)]}, _REPLAYED_TURN, id="signature-int"
    ),
    pytest.param({"reasoning_content": _REASONING, "thinking_blocks": ["str"]}, _REPLAYED_TURN, id="block-string"),
    pytest.param(
        {"reasoning_content": _REASONING, "thinking_blocks": [_thinking_block(_FIVE_KB)]},
        _REPLAYED_TURN,
        id="signature-5kb",
    ),
    pytest.param(
        {"reasoning_content": "", "thinking_blocks": [_thinking_block(_CLAUDE_SIGNATURE)]},
        _model_turn(_thought(""), {"text": _PRIOR_ANSWER}),
        id="reasoning-empty-with-blocks",
    ),
    pytest.param(
        {"reasoning_content": None, "thinking_blocks": None}, _model_turn({"text": _PRIOR_ANSWER}), id="both-null"
    ),
    pytest.param({"reasoning_content": _REASONING, "thinking_blocks": []}, _REPLAYED_TURN, id="blocks-empty-list"),
    pytest.param({}, _model_turn({"text": _PRIOR_ANSWER}), id="both-missing"),
    pytest.param(
        {"thinking_blocks": [_thinking_block(_CLAUDE_SIGNATURE)]},
        _model_turn({"text": _PRIOR_ANSWER}),
        id="blocks-without-reasoning-content",
    ),
    pytest.param(
        {
            "reasoning_content": _REASONING,
            "thinking_blocks": [_thinking_block(_CLAUDE_SIGNATURE, json.dumps(_thought()))],
        },
        _REPLAYED_TURN,
        id="json-era-block",
    ),
)


@pytest.mark.parametrize(("assistant_fields", "expected_turn"), _ASSISTANT_SHAPES)
def test_chat_assistant_thinking_shapes_never_crash_or_leak_a_signature(
    gateway: Gateway, assistant_fields: Mapping[str, JsonValue], expected_turn: Mapping[str, JsonValue]
) -> None:
    response_ids: Final = (f"gemini-reply-{uuid.uuid4().hex}", f"gemini-reply-{uuid.uuid4().hex}")
    calls: Final = itertools.count()

    def respond(request: Request) -> Reply:
        return _reply(response_ids[next(calls)], stream=False)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        shaped: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": _chat_history(assistant_fields), **_CACHE_BUST},
        )
        assert shaped.status_code == 200, shaped.text
        assert _JSON_OBJECT.validate_json(shaped.content)["id"] == response_ids[0], shaped.text
        control: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _CONTROL_HISTORY, **_CACHE_BUST}
        )
        assert control.status_code == 200, control.text
        assert _JSON_OBJECT.validate_json(control.content)["id"] == response_ids[1], control.text
        received: Final = wire.drain()
        assert [(request.method, request.target) for request in received] == [("POST", _target("gemini", False))] * 2
        assert "thoughtSignature" not in received[0].body.decode(), received[0].body
        assert _JSON_OBJECT.validate_json(received[0].body) == _expected_body(
            "chat", _user(_QUESTION), expected_turn, _user(_FOLLOW_UP)
        )
        assert _JSON_OBJECT.validate_json(received[1].body) == _expected_body(
            "chat", _user(_QUESTION), _model_turn({"text": _PRIOR_ANSWER}), _user(_FOLLOW_UP)
        )
        assert [_spend_row(response_id)["status"] for response_id in response_ids] == ["success", "success"]


def test_chat_same_thinking_replay_sent_twice_lands_one_spend_row_per_request(gateway: Gateway) -> None:
    response_ids: Final = (f"gemini-reply-{uuid.uuid4().hex}", f"gemini-reply-{uuid.uuid4().hex}")
    calls: Final = itertools.count()

    def respond(request: Request) -> Reply:
        return _reply(response_ids[next(calls)], stream=False)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        body: Final = {"model": model, "messages": _CHAT_HISTORY, **_CACHE_BUST}
        answers: Final = tuple(gateway.request("POST", "/v1/chat/completions", body) for _ in response_ids)
        assert [answer.status_code for answer in answers] == [200, 200], [answer.text for answer in answers]
        assert [_JSON_OBJECT.validate_json(answer.content)["id"] for answer in answers] == list(response_ids)
        received: Final = wire.drain()
        assert [_JSON_OBJECT.validate_json(request.body) for request in received] == [
            _expected_body("chat", _user(_QUESTION), _REPLAYED_TURN, _user(_FOLLOW_UP))
        ] * 2
        assert [_spend_row(response_id)["status"] for response_id in response_ids] == ["success", "success"]


def test_chat_two_consecutive_thinking_turns_merge_into_one_signature_free_model_turn(gateway: Gateway) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(lambda request: _reply(response_id, stream=False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": _QUESTION},
                    {
                        "role": "assistant",
                        "content": "Two.",
                        "reasoning_content": "first thought",
                        "thinking_blocks": [_thinking_block(_CLAUDE_SIGNATURE, "first thought")],
                    },
                    {
                        "role": "assistant",
                        "content": "Wait, three.",
                        "reasoning_content": "second thought",
                        "thinking_blocks": [_thinking_block(_CLAUDE_SIGNATURE, "second thought")],
                    },
                    {"role": "user", "content": _FOLLOW_UP},
                ],
                **_CACHE_BUST,
            },
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["id"] == response_id, response.text
        assert _only_request(wire, "gemini", False) == _expected_body(
            "chat",
            _user(_QUESTION),
            _model_turn(
                _thought("first thought"), {"text": "Two."}, _thought("second thought"), {"text": "Wait, three."}
            ),
            _user(_FOLLOW_UP),
        )
        assert _spend_row(response_id)["status"] == "success"


def test_chat_claude_thinking_beside_a_tool_call_replays_the_thought_and_the_call_only(gateway: Gateway) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(lambda request: _reply(response_id, stream=False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "tools": [{"type": "function", "function": _FUNCTION}],
                "messages": [
                    {"role": "user", "content": "Count the apples."},
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": _REASONING,
                        "thinking_blocks": [_thinking_block(_CLAUDE_SIGNATURE)],
                        "tool_calls": [
                            {
                                "id": "call_prior",
                                "type": "function",
                                "function": {"name": "count_fruit", "arguments": json.dumps({"kind": "apple"})},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_prior", "content": "12"},
                ],
                **_CACHE_BUST,
            },
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["id"] == response_id, response.text
        assert _only_request(wire, "gemini", False) == {
            **_expected_body(
                "chat",
                _user("Count the apples."),
                _model_turn(_thought(), {"function_call": {"name": "count_fruit", "args": {"kind": "apple"}}}),
                _FUNCTION_RESPONSE_TURN,
            ),
            **_DECLARATIONS,
        }
        assert _spend_row(response_id)["status"] == "success"


_MESSAGES_SHAPES: Final = (
    pytest.param(_thinking_block(5), _REPLAYED_TURN, id="signature-int"),
    pytest.param(_thinking_block(""), _REPLAYED_TURN, id="signature-empty"),
    pytest.param(
        {"type": "redacted_thinking", "data": "synthetic-redacted-data"},
        _model_turn({"text": _PRIOR_ANSWER}),
        id="redacted-thinking",
    ),
)


@pytest.mark.parametrize(("assistant_block", "expected_turn"), _MESSAGES_SHAPES)
def test_messages_assistant_thinking_shapes_never_leak_a_signature(
    gateway: Gateway, assistant_block: Mapping[str, JsonValue], expected_turn: Mapping[str, JsonValue]
) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(lambda request: _reply(response_id, stream=False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 64, "messages": _messages_history(assistant_block), **_CACHE_BUST},
            headers={"anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["id"] == response_id, response.text
        assert _only_request(wire, "gemini", False) == _expected_body(
            "messages", _user(_QUESTION), expected_turn, _user(_FOLLOW_UP)
        )
        assert _spend_row(response_id)["status"] == "success"


def test_responses_encrypted_block_with_an_int_signature_still_replays_the_summary_only(gateway: Gateway) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"
    with wire_server(lambda request: _reply(response_id, stream=False)) as wire, gateway.scenario() as scenario:
        model: Final = _register(gateway, scenario, "gemini", wire)
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": _responses_input(_thinking_block(5)), **_CACHE_BUST}
        )
        assert response.status_code == 200, response.text
        caller_id: Final = str(_JSON_OBJECT.validate_json(response.content)["id"])
        assert _upstream_response_id("responses", caller_id) == response_id, caller_id
        assert _only_request(wire, "gemini", False) == _expected_body(
            "responses", _user(_QUESTION), _REPLAYED_TURN, _user(_FOLLOW_UP)
        )
        assert _spend_row(response_id)["status"] == "success"


_CACHE_NAME: Final = "cachedContents/synthetic-cache"
_CACHED_POLICY: Final = " ".join(f"policy clause {index} applies" for index in range(600))
_EPHEMERAL: Final = {"type": "ephemeral"}


def _cached_reply(response_id: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                **_frame(response_id, ({"text": _ANSWER},), finished=True),
                "usageMetadata": {**_USAGE, "promptTokenCount": 1300, "cachedContentTokenCount": 1290},
            }
        ).encode()
    )


def _cached_chat_history() -> Sequence[Mapping[str, JsonValue]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": _CACHED_POLICY, "cache_control": _EPHEMERAL}]},
        {
            "role": "assistant",
            "content": _PRIOR_ANSWER,
            "reasoning_content": _REASONING,
            "thinking_blocks": [_thinking_block(_CLAUDE_SIGNATURE)],
            "cache_control": _EPHEMERAL,
        },
        {"role": "user", "content": [{"type": "text", "text": "Acknowledged.", "cache_control": _EPHEMERAL}]},
        {"role": "user", "content": _FOLLOW_UP},
    ]


def _cached_messages_history() -> Sequence[Mapping[str, JsonValue]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": _CACHED_POLICY, "cache_control": _EPHEMERAL}]},
        {
            "role": "assistant",
            "content": [
                _thinking_block(_CLAUDE_SIGNATURE),
                {"type": "text", "text": _PRIOR_ANSWER, "cache_control": _EPHEMERAL},
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "Acknowledged.", "cache_control": _EPHEMERAL}]},
        {"role": "user", "content": _FOLLOW_UP},
    ]


@pytest.mark.parametrize("endpoint", ("chat", "messages"))
def test_context_cached_thinking_turn_is_stored_without_its_signature(gateway: Gateway, endpoint: Endpoint) -> None:
    response_id: Final = f"gemini-reply-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b"{}")
        if request.target.endswith("cachedContents"):
            return Reply(body=json.dumps({"name": _CACHE_NAME, "model": f"models/{_BACKEND}"}).encode())
        return _cached_reply(response_id)

    body: Final = (
        {"messages": _cached_chat_history()}
        if endpoint == "chat"
        else {"max_tokens": 64, "messages": _cached_messages_history()}
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
        assert _ANSWER in response.text, response.text
        received: Final = wire.drain()
        assert [(request.method, request.target.endswith("cachedContents")) for request in received] == [
            ("GET", True),
            ("POST", True),
            ("POST", False),
        ], received
        assert received[2].target == f"/models/{_BACKEND}:generateContent", received
        assert all("thoughtSignature" not in request.body.decode() for request in received[1:]), received
        stored: Final = _JSON_OBJECT.validate_json(received[1].body)
        assert isinstance(stored["displayName"], str) and stored["displayName"], stored
        assert stored == {
            "contents": [_user(_CACHED_POLICY), _REPLAYED_TURN, _user("Acknowledged.")],
            "model": f"models/{_BACKEND}",
            "displayName": stored["displayName"],
            "tools": None,
        }, stored
        assert _JSON_OBJECT.validate_json(received[2].body) == {
            **_expected_body(endpoint, _user(_FOLLOW_UP)),
            "cachedContent": _CACHE_NAME,
        }
        assert _spend_row(response_id)["status"] == "success"

import asyncio
import base64
import json
import os
import re
import signal
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import unquote, urlsplit

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with

_KIMI_US: Final = "us.moonshotai.kimi-k3"
_KIMI_GLOBAL: Final = "global.moonshotai.kimi-k3"
_KIMI_BASE: Final = "moonshotai.kimi-k3"
_CLAUDE: Final = "us.anthropic.claude-sonnet-5"
_GPT_OSS: Final = "openai.gpt-oss-120b-1:0"
_LLAMA: Final = "us.meta.llama4-maverick-17b-instruct-v1:0"
_PROFILE_ARN_PREFIX: Final = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/"
_YAML_FLAGGED_ARN: Final = f"{_PROFILE_ARN_PREFIX}yaml-flagged-kimi"
_YAML_PLAIN_ARN: Final = f"{_PROFILE_ARN_PREFIX}yaml-plain-kimi"
_KNOWN_MODEL_IDS: Final = frozenset({_KIMI_US, _KIMI_GLOBAL, _KIMI_BASE, _CLAUDE, _GPT_OSS, _LLAMA})
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_ANSWER: Final = "kimi cache point control"
_REJECTION: Final = "This model doesn't support the cachePoint field. Remove cachePoint and try again."
_UNKNOWN_MODEL: Final = "The provided model identifier is invalid."
_CACHE_USAGE_MARKER: Final = "[peer:cache-usage]"
_REJECT_MARKER: Final = "[peer:reject]"
_SLOW_MARKER: Final = "[peer:slow]"
_HOLD_MARKER: Final = "[peer:hold]"
_TARGET: Final = re.compile(
    r"^/model/(?P<model>.+)/(?P<action>converse|converse-stream|invoke|invoke-with-response-stream)$"
)
_CONVERSE_LIKE: Final = "converse_like"
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_SIGNING_KEY: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_EPHEMERAL: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
_AWS: Final[dict[str, JsonValue]] = {
    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
    "aws_secret_access_key": "scripted-secret",
    "aws_region_name": "us-east-1",
}
_EXTRA: Final[dict[str, JsonValue]] = {"num_retries": 0, "cache": {"no-cache": True}}
_SYSTEM_TEXT: Final = "You are terse."
_TOOL_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

_NAMES: Final = MappingProxyType(
    {
        "kimi-us": f"bedrock/{_KIMI_US}",
        "kimi-global-converse": f"bedrock/converse/{_KIMI_GLOBAL}",
        "kimi-base": f"bedrock/{_KIMI_BASE}",
        "kimi-regional": f"bedrock/us-east-1/{_KIMI_US}",
        "claude-control": f"bedrock/{_CLAUDE}",
        "gpt-oss-control": f"bedrock/{_GPT_OSS}",
        "llama-control": f"bedrock/{_LLAMA}",
        "arn-yaml-flagged": f"bedrock/{_YAML_FLAGGED_ARN}",
        "arn-yaml-plain": f"bedrock/{_YAML_PLAIN_ARN}",
        "kimi-inject-message": f"bedrock/{_KIMI_US}",
        "kimi-inject-tool": f"bedrock/{_KIMI_US}",
        "claude-inject-message": f"bedrock/{_CLAUDE}",
        "bedrock/*": "bedrock/*",
    }
)
_MODEL_INFO: Final = MappingProxyType({"arn-yaml-flagged": {"supports_prompt_cache_breakpoint": False}})
_INJECTION: Final = MappingProxyType(
    {
        "kimi-inject-message": [{"location": "message", "role": "system"}],
        "claude-inject-message": [{"location": "message", "role": "system"}],
        "kimi-inject-tool": [{"location": "tool_config"}],
    }
)

Endpoint = Literal["chat", "messages", "responses"]
Marker = Literal["system", "user", "tool"]
_ALL_MARKERS: Final = frozenset[Marker]({"system", "user", "tool"})
_USER_ONLY: Final = frozenset[Marker]({"user"})
_SYSTEM_AND_USER: Final = frozenset[Marker]({"system", "user"})
_NO_MARKERS: Final = frozenset[Marker]()


def _frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


def _usage(cache_read: int = 0, cache_write: int = 0) -> dict[str, JsonValue]:
    cached: Final[dict[str, JsonValue]] = {
        **({"cacheReadInputTokens": cache_read} if cache_read else {}),
        **({"cacheWriteInputTokens": cache_write} if cache_write else {}),
    }
    return {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15 + cache_read + cache_write, **cached}


def _converse_reply(usage: Mapping[str, JsonValue]) -> bytes:
    return json.dumps(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": _ANSWER}]}},
            "stopReason": "end_turn",
            "usage": dict(usage),
            "metrics": {"latencyMs": 1},
        }
    ).encode()


def _text_parts(pieces: int) -> tuple[str, ...]:
    words: Final = _ANSWER.split(" ")
    assert pieces in (1, len(words)), pieces
    if pieces == 1:
        return (_ANSWER,)
    return tuple(word if index == len(words) - 1 else f"{word} " for index, word in enumerate(words))


def _stream_frames(usage: Mapping[str, JsonValue], pieces: int = 1) -> tuple[bytes, ...]:
    deltas: Final = tuple(
        _frame("contentBlockDelta", {"delta": {"text": part}, "contentBlockIndex": 0}) for part in _text_parts(pieces)
    )
    return (
        _frame("messageStart", {"role": "assistant"}),
        *deltas,
        _frame("contentBlockStop", {"contentBlockIndex": 0}),
        _frame("messageStop", {"stopReason": "end_turn"}),
        _frame("metadata", {"usage": dict(usage)}),
    )


def _known(model: str) -> bool:
    return model in _KNOWN_MODEL_IDS or "application-inference-profile" in model


def _error(message: str) -> bytes:
    return json.dumps({"message": message}).encode()


def _anthropic_message(usage: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": _CLAUDE,
        "content": [{"type": "text", "text": _ANSWER}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"]},
    }


def _invoke_chunk(event: Mapping[str, JsonValue]) -> bytes:
    encoded: Final = base64.b64encode(json.dumps(event, separators=(",", ":")).encode()).decode()
    return _frame("chunk", {"bytes": encoded})


def _invoke_stream(usage: Mapping[str, JsonValue]) -> bytes:
    message: Final = {
        **_anthropic_message(usage),
        "content": [],
        "usage": {"input_tokens": usage["inputTokens"], "output_tokens": 1},
    }
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "message_start", "message": message},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _ANSWER}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": usage["outputTokens"]},
        },
        {"type": "message_stop"},
    )
    return b"".join(_invoke_chunk(event) for event in events)


@dataclass(frozen=True, slots=True)
class _Target:
    model: str
    action: str


def _target(raw: str) -> _Target:
    path: Final = unquote(raw)
    if path == "/":
        return _Target(_CONVERSE_LIKE, "converse")
    found: Final = _TARGET.match(path)
    assert found is not None, raw
    return _Target(found["model"], found["action"])


def _peer(hold: threading.Event | None = None, held: SimpleQueue[str] | None = None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        target: Final = _target(request.target)
        streaming: Final = target.action in ("converse-stream", "invoke-with-response-stream")
        text: Final = request.body.decode(errors="replace")
        if target.model != _CONVERSE_LIKE and not _known(target.model):
            return Reply(status=400, body=_error(_UNKNOWN_MODEL))
        if _REJECT_MARKER in text and not streaming:
            return Reply(status=400, body=_error(_REJECTION))
        if _REJECT_MARKER in text:
            return Reply(body=_frame("validationException", {"message": _REJECTION}), content_type=_EVENT_STREAM)
        usage: Final = _usage(100, 50) if _CACHE_USAGE_MARKER in text else _usage()
        if target.action == "invoke":
            return Reply(body=json.dumps(_anthropic_message(usage)).encode())
        if target.action == "invoke-with-response-stream":
            return Reply(body=_invoke_stream(usage), content_type=_EVENT_STREAM)
        if not streaming:
            return Reply(body=_converse_reply(usage))
        if _HOLD_MARKER in text and hold is not None:
            if held is not None:
                held.put(target.model)
            return Reply(content_type=_EVENT_STREAM, chunks=_stream_frames(usage), gate_after_first=hold)
        if _SLOW_MARKER in text:
            return Reply(content_type=_EVENT_STREAM, chunks=_stream_frames(usage, 4), pause_between_chunks=0.2)
        return Reply(body=b"".join(_stream_frames(usage)), content_type=_EVENT_STREAM)

    return respond


@dataclass(frozen=True, slots=True)
class _Received:
    model: str
    action: str
    streaming: bool
    body: dict[str, JsonValue]
    cache_points: int
    cache_controls: int


def _message_blocks(messages: JsonValue) -> Iterator[JsonValue]:
    if not isinstance(messages, list):
        return
    for message in messages:
        content: Final = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            yield from content


def _tool_blocks(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    tool_config: Final = body.get("toolConfig")
    tools: Final = tool_config.get("tools") if isinstance(tool_config, dict) else None
    return tuple(tools) if isinstance(tools, list) else ()


def _cache_points(body: Mapping[str, JsonValue]) -> int:
    system: Final = body.get("system")
    blocks: Final = (
        *(system if isinstance(system, list) else ()),
        *_message_blocks(body.get("messages")),
        *_tool_blocks(body),
    )
    return sum(1 for block in blocks if isinstance(block, dict) and "cachePoint" in block)


def _cache_controls(value: JsonValue) -> int:
    if isinstance(value, dict):
        return sum(_cache_controls(item) for item in value.values()) + (1 if "cache_control" in value else 0)
    if isinstance(value, list):
        return sum(_cache_controls(item) for item in value)
    return 0


def _parse(request: Request) -> _Received:
    target: Final = _target(request.target)
    body: Final = _JSON.validate_python(json.loads(request.body))
    streaming: Final = target.action in ("converse-stream", "invoke-with-response-stream")
    return _Received(target.model, target.action, streaming, body, _cache_points(body), _cache_controls(body))


def _received(wire: Wire) -> tuple[_Received, ...]:
    return tuple(_parse(request) for request in wire.drain())


def _only_received(wire: Wire) -> _Received:
    (received,) = _received(wire)
    return received


def _prompt(*markers: str) -> str:
    return " ".join((f"Reply with the control sentence {uuid.uuid4().hex}", *markers))


def _text_block(text: str, cached: bool, kind: str = "text") -> dict[str, JsonValue]:
    return {"type": kind, "text": text, **({"cache_control": dict(_EPHEMERAL)} if cached else {})}


def _chat_tools(cached: bool) -> dict[str, JsonValue]:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather for a city",
                    "parameters": _TOOL_PARAMETERS,
                },
                **({"cache_control": dict(_EPHEMERAL)} if cached else {}),
            }
        ],
    }


def _anthropic_tools(cached: bool) -> dict[str, JsonValue]:
    return {
        "tools": [
            {
                "name": "get_weather",
                "description": "Weather for a city",
                "input_schema": _TOOL_PARAMETERS,
                **({"cache_control": dict(_EPHEMERAL)} if cached else {}),
            }
        ]
    }


def _chat_body(
    model: str, prompt: str, markers: frozenset[Marker], *, stream: bool = False, with_tools: bool = False
) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 16,
        "stream": stream,
        "messages": [
            {"role": "system", "content": [_text_block(_SYSTEM_TEXT, "system" in markers)]},
            {"role": "user", "content": [_text_block(prompt, "user" in markers)]},
        ],
        **(_chat_tools("tool" in markers) if with_tools or "tool" in markers else {}),
        **_EXTRA,
    }


def _messages_body(
    model: str, prompt: str, markers: frozenset[Marker], *, stream: bool = False, with_tools: bool = False
) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 16,
        "stream": stream,
        "system": [_text_block(_SYSTEM_TEXT, "system" in markers)],
        "messages": [{"role": "user", "content": [_text_block(prompt, "user" in markers)]}],
        **(_anthropic_tools("tool" in markers) if with_tools or "tool" in markers else {}),
        **_EXTRA,
    }


def _responses_body(
    model: str, prompt: str, markers: frozenset[Marker], *, stream: bool = False
) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_output_tokens": 16,
        "stream": stream,
        "input": [{"role": "user", "content": [_text_block(prompt, "user" in markers, "input_text")]}],
        **_EXTRA,
    }


def _body(
    endpoint: Endpoint, model: str, prompt: str, markers: frozenset[Marker], *, stream: bool = False
) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return _chat_body(model, prompt, markers, stream=stream)
        case "messages":
            return _messages_body(model, prompt, markers, stream=stream)
        case "responses":
            return _responses_body(model, prompt, markers, stream=stream)


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


@dataclass(frozen=True, slots=True)
class _Outcome:
    status: int
    call_id: str
    response_id: str
    text: str
    headers: Mapping[str, str]
    raw: str


def _sse_payloads(lines: Iterable[str]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(json.loads(line[6:]) for line in lines if line.startswith("data: ") and line != "data: [DONE]")


def _first_choice(chunk: Mapping[str, JsonValue]) -> dict[str, JsonValue] | None:
    choices: Final = chunk.get("choices")
    return _JSON.validate_python(choices[0]) if isinstance(choices, list) and choices else None


def _chat_stream_text(chunks: Iterable[dict[str, JsonValue]]) -> str:
    choices: Final = tuple(choice for choice in map(_first_choice, chunks) if choice is not None)
    return "".join(str(_JSON.validate_python(choice["delta"]).get("content") or "") for choice in choices)


def _chat_stream_id(chunks: Iterable[dict[str, JsonValue]]) -> str:
    (identity,) = {str(chunk["id"]) for chunk in chunks if "id" in chunk}
    return identity


def _message_stream_id(payloads: Iterable[dict[str, JsonValue]]) -> str:
    (started,) = tuple(payload for payload in payloads if payload.get("type") == "message_start")
    return str(_JSON.validate_python(started["message"])["id"])


def _messages_stream_text(payloads: Iterable[dict[str, JsonValue]]) -> str:
    deltas: Final = tuple(payload for payload in payloads if payload.get("type") == "content_block_delta")
    return "".join(str(_JSON.validate_python(delta["delta"]).get("text") or "") for delta in deltas)


def _responses_stream_text(events: Iterable[dict[str, JsonValue]]) -> str:
    return "".join(
        str(event.get("delta") or "") for event in events if event.get("type") == "response.output_text.delta"
    )


def _inner_response_id(identity: str) -> str:
    managed: Final = decrypt_if_encrypted_with(identity.removeprefix("resp_"), _SIGNING_KEY)
    assert managed is not None, identity
    issued: Final = managed.split(";", 1)[0].rsplit("response_id:", 1)[1]
    decoded: Final = base64.b64decode(issued.removeprefix("resp_")).decode()
    return decoded.rsplit("response_id:", 1)[1]


def _completed_response_id(events: Iterable[dict[str, JsonValue]]) -> str:
    (completed,) = tuple(event for event in events if event.get("type") == "response.completed")
    return _inner_response_id(str(_JSON.validate_python(completed["response"])["id"]))


def _chat_text(body: Mapping[str, JsonValue]) -> str:
    choices: Final = body.get("choices")
    assert isinstance(choices, list) and choices, body
    return str(_JSON.validate_python(_JSON.validate_python(choices[0])["message"]).get("content") or "")


def _messages_text(body: Mapping[str, JsonValue]) -> str:
    content: Final = body.get("content")
    assert isinstance(content, list), body
    return "".join(str(block.get("text") or "") for block in content if isinstance(block, dict))


def _responses_text(body: Mapping[str, JsonValue]) -> str:
    output: Final = body.get("output")
    assert isinstance(output, list), body
    return "".join(
        str(part.get("text") or "")
        for item in output
        if isinstance(item, dict)
        for part in (
            item.get("content") if isinstance(item.get("content"), list) else ()
        )  # comprehension-ok: nested response items
        if isinstance(part, dict)
    )


def _outcome_of(endpoint: Endpoint, stream: bool, response: httpx.Response, lines: tuple[str, ...]) -> _Outcome:
    call_id: Final = response.headers.get("x-litellm-call-id", "")
    raw: Final = "\n".join(lines)
    if response.status_code != 200:
        return _Outcome(response.status_code, call_id, "", "", response.headers, raw)
    if stream:
        payloads: Final = _sse_payloads(lines)
        match endpoint:
            case "chat":
                return _Outcome(
                    200, call_id, _chat_stream_id(payloads), _chat_stream_text(payloads), response.headers, raw
                )
            case "messages":
                return _Outcome(
                    200, call_id, _message_stream_id(payloads), _messages_stream_text(payloads), response.headers, raw
                )
            case "responses":
                return _Outcome(
                    200,
                    call_id,
                    _completed_response_id(payloads),
                    _responses_stream_text(payloads),
                    response.headers,
                    raw,
                )
    body: Final = _JSON.validate_python(json.loads(raw))
    match endpoint:
        case "chat":
            return _Outcome(200, call_id, str(body["id"]), _chat_text(body), response.headers, raw)
        case "messages":
            return _Outcome(200, call_id, str(body["id"]), _messages_text(body), response.headers, raw)
        case "responses":
            return _Outcome(
                200, call_id, _inner_response_id(str(body["id"])), _responses_text(body), response.headers, raw
            )


def _send(gateway: Gateway, endpoint: Endpoint, body: Mapping[str, JsonValue], *, key: str | None = None) -> _Outcome:
    stream: Final = body.get("stream") is True
    headers: Final = {"Authorization": f"Bearer {gateway.key if key is None else key}"}
    with gateway.client.stream("POST", _path(endpoint), json=body, headers=headers, timeout=60) as response:
        lines: Final = tuple(line for line in response.iter_lines() if line)
    return _outcome_of(endpoint, stream, response, lines)


def _spend_rows(request_ids: frozenset[str], *, expected: int, seconds: float = 90) -> tuple[dict[str, JsonValue], ...]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status, model, spend FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(string_to_array(%s, %s))',
            (",".join(sorted(request_ids)), ","),
        ),
        lambda found: len(found) >= expected,
        seconds=seconds,
    )
    return tuple(rows)


def _success_row(request_id: str) -> dict[str, JsonValue]:
    assert request_id, "No id to look the spend row up by"
    (row,) = _spend_rows(frozenset({request_id}), expected=1)
    assert row["status"] == "success", row
    return row


def _failure_row(call_id: str) -> dict[str, JsonValue]:
    assert call_id, "No call id to look the failure row up by"
    (row,) = _spend_rows(frozenset({call_id}), expected=1)
    assert row["status"] == "failure", row
    return row


def _assert_answered(outcome: _Outcome) -> None:
    assert outcome.status == 200, (outcome.status, outcome.raw)
    assert outcome.text == _ANSWER, outcome.raw


def _assert_cache_points(received: _Received, emits: bool, model: str) -> None:
    assert received.model == model, (received.model, model)
    assert (received.cache_points > 0) == emits, (emits, received.body)


def _head_strips() -> bool:
    return os.environ.get("INTEGRATION_LEG", "head") == "head"


def _owned_config(wire: Wire, directory: Path, *, names: Iterable[str] = tuple(_NAMES)) -> Path:
    base: Final = _JSON.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final[dict[str, JsonValue]] = {
        **base,
        "model_list": [
            {
                "model_name": name,
                "litellm_params": {
                    "model": _NAMES[name],
                    "api_base": wire.url,
                    "num_retries": 0,
                    **_AWS,
                    **({"cache_control_injection_points": _INJECTION[name]} if name in _INJECTION else {}),
                },
                **({"model_info": dict(_MODEL_INFO[name])} if name in _MODEL_INFO else {}),
            }
            for name in names
        ],
        "router_settings": {**_JSON.validate_python(base["router_settings"]), "num_retries": 0},
    }
    path: Final = directory / f"kimi-k3-cache-point-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    wire: Wire
    owned: OwnedProxy


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("kimi-k3-cache-point")
    with gateway_from_environment() as environment, wire_server(_peer()) as wire:
        config: Final = _owned_config(wire, directory)
        with owned_proxy_process(environment, directory, {}, config=config, workers=2) as owned:
            eventually(
                lambda: len(_STARTED_WORKER.findall(owned.log.read_text())), lambda count: count == 2, seconds=60
            )
            wire.drain()
            yield _Rig(owned.gateway, wire, owned)


def _observe(rig: _Rig, endpoint: Endpoint, body: Mapping[str, JsonValue]) -> tuple[_Outcome, _Received]:
    rig.wire.drain()
    outcome: Final = _send(rig.gateway, endpoint, body)
    return outcome, _only_received(rig.wire)


_KIMI_FORMS: Final = ("kimi-us", "kimi-global-converse", "kimi-base", "kimi-regional", "arn-yaml-flagged")
_EMITTING_CONTROLS: Final = ("claude-control", "arn-yaml-plain")
_SILENT_CONTROLS: Final = ("gpt-oss-control", "llama-control")


@pytest.mark.timeout(600)
@pytest.mark.parametrize("name", _KIMI_FORMS, ids=tuple(f"m-{name}" for name in _KIMI_FORMS))
def test_m01_to_m05_every_kimi_k3_form_sends_converse_without_a_cache_point(rig: _Rig, name: str) -> None:
    outcome, received = _observe(rig, "chat", _chat_body(name, _prompt(), _ALL_MARKERS))
    _assert_answered(outcome)
    assert received.cache_points == 0, received.body
    assert received.body.get("toolConfig") is not None, received.body
    _success_row(outcome.response_id)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("name", _EMITTING_CONTROLS, ids=tuple(f"m-{name}" for name in _EMITTING_CONTROLS))
def test_m08_m09_models_that_take_cache_points_still_get_all_three(rig: _Rig, name: str) -> None:
    outcome, received = _observe(rig, "chat", _chat_body(name, _prompt(), _ALL_MARKERS))
    _assert_answered(outcome)
    assert received.cache_points == 3, received.body
    _success_row(outcome.response_id)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("name", _SILENT_CONTROLS, ids=tuple(f"m-{name}" for name in _SILENT_CONTROLS))
def test_m10_m11_models_without_caching_never_got_cache_points(rig: _Rig, name: str) -> None:
    outcome, received = _observe(rig, "chat", _chat_body(name, _prompt(), _ALL_MARKERS))
    _assert_answered(outcome)
    assert received.cache_points == 0, received.body
    _success_row(outcome.response_id)


@pytest.mark.timeout(600)
def test_m14_a_wildcard_deployment_resolves_the_caller_model_before_deciding(rig: _Rig) -> None:
    kimi, kimi_received = _observe(rig, "chat", _chat_body(f"bedrock/{_KIMI_US}", _prompt(), _SYSTEM_AND_USER))
    _assert_answered(kimi)
    assert kimi_received.model == _KIMI_US, kimi_received.model
    assert kimi_received.cache_points == 0, kimi_received.body
    claude, claude_received = _observe(rig, "chat", _chat_body(f"bedrock/{_CLAUDE}", _prompt(), _SYSTEM_AND_USER))
    _assert_answered(claude)
    assert claude_received.model == _CLAUDE, claude_received.model
    assert claude_received.cache_points == 2, claude_received.body
    _success_row(kimi.response_id)
    _success_row(claude.response_id)


_RAW_CELLS: Final = (
    ("chat", False),
    ("chat", True),
    ("messages", False),
    ("messages", True),
    ("responses", False),
    ("responses", True),
)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("endpoint", "stream"),
    _RAW_CELLS,
    ids=tuple(f"e-{endpoint}-{'stream' if stream else 'plain'}" for endpoint, stream in _RAW_CELLS),
)
def test_e01_to_e06_every_endpoint_reaches_kimi_without_a_cache_point(
    rig: _Rig, endpoint: Endpoint, stream: bool
) -> None:
    outcome, received = _observe(rig, endpoint, _body(endpoint, "kimi-us", _prompt(), _SYSTEM_AND_USER, stream=stream))
    _assert_answered(outcome)
    assert received.streaming == stream, received
    assert received.model == _KIMI_US, received.model
    assert received.cache_points == 0, received.body
    _success_row(outcome.response_id)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("endpoint", "stream"),
    _RAW_CELLS,
    ids=tuple(f"e-{endpoint}-{'stream' if stream else 'plain'}" for endpoint, stream in _RAW_CELLS),
)
def test_e07_to_e12_every_endpoint_still_sends_claude_its_cache_points(
    rig: _Rig, endpoint: Endpoint, stream: bool
) -> None:
    outcome, received = _observe(
        rig, endpoint, _body(endpoint, "claude-control", _prompt(), _SYSTEM_AND_USER, stream=stream)
    )
    _assert_answered(outcome)
    assert received.model == _CLAUDE, received.model
    assert received.streaming == stream, received
    if endpoint == "messages":
        assert received.action.startswith("invoke"), received.action
        assert received.cache_controls == 2 and received.cache_points == 0, received.body
    else:
        assert received.action.startswith("converse"), received.action
        expected: Final = 1 if endpoint == "responses" else 2
        assert received.cache_points == expected, (endpoint, received.body)
    _success_row(outcome.response_id)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("endpoint", "stream"),
    (("messages", False), ("messages", True)),
    ids=("e-messages-arn-plain", "e-messages-arn-stream"),
)
def test_e13_e14_messages_keeps_cache_control_for_an_arn_and_the_flag_decides(
    rig: _Rig, endpoint: Endpoint, stream: bool
) -> None:
    flagged, flagged_received = _observe(
        rig, endpoint, _body(endpoint, "arn-yaml-flagged", _prompt(), _SYSTEM_AND_USER, stream=stream)
    )
    _assert_answered(flagged)
    assert flagged_received.model == _YAML_FLAGGED_ARN, flagged_received.model
    assert flagged_received.cache_points == 0, flagged_received.body
    plain, plain_received = _observe(
        rig, endpoint, _body(endpoint, "arn-yaml-plain", _prompt(), _SYSTEM_AND_USER, stream=stream)
    )
    _assert_answered(plain)
    assert plain_received.model == _YAML_PLAIN_ARN, plain_received.model
    assert plain_received.cache_points == 2, plain_received.body
    _success_row(flagged.response_id)
    _success_row(plain.response_id)


def _openai_client(rig: _Rig) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{_proxy_url(rig.gateway)}/v1", api_key=rig.gateway.key, max_retries=0)


def _async_openai_client(rig: _Rig) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=f"{_proxy_url(rig.gateway)}/v1", api_key=rig.gateway.key, max_retries=0)


def _anthropic_client(rig: _Rig) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=_proxy_url(rig.gateway), api_key=rig.gateway.key, max_retries=0)


def _async_anthropic_client(rig: _Rig) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(base_url=_proxy_url(rig.gateway), api_key=rig.gateway.key, max_retries=0)


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _sdk_messages(prompt: str) -> list[dict[str, JsonValue]]:
    return [
        {"role": "system", "content": [_text_block(_SYSTEM_TEXT, True)]},
        {"role": "user", "content": [_text_block(prompt, True)]},
    ]


@pytest.mark.timeout(600)
def test_k01_openai_sdk_sync_chat_reaches_kimi_without_a_cache_point(rig: _Rig) -> None:
    rig.wire.drain()
    completion: Final = _openai_client(rig).chat.completions.create(
        model="kimi-us",
        messages=_sdk_messages(_prompt()),  # pyright: ignore[reportArgumentType]  # cache_control rides as an extra key
        max_tokens=16,
        extra_body=_EXTRA,
    )
    assert completion.choices[0].message.content == _ANSWER, completion
    received: Final = _only_received(rig.wire)
    assert received.model == _KIMI_US and not received.streaming, received
    assert received.cache_points == 0, received.body
    _success_row(completion.id)


@pytest.mark.timeout(600)
async def test_k02_openai_sdk_async_chat_stream_reaches_kimi_without_a_cache_point(rig: _Rig) -> None:
    rig.wire.drain()
    stream: Final = await _async_openai_client(rig).chat.completions.create(
        model="kimi-us",
        messages=_sdk_messages(_prompt()),  # pyright: ignore[reportArgumentType]  # cache_control rides as an extra key
        max_tokens=16,
        stream=True,
        extra_body=_EXTRA,
    )
    chunks: Final = [chunk async for chunk in stream]
    text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert text == _ANSWER, chunks
    (identity,) = {chunk.id for chunk in chunks}
    received: Final = _only_received(rig.wire)
    assert received.model == _KIMI_US and received.streaming, received
    assert received.cache_points == 0, received.body
    _success_row(identity)


_ANTHROPIC_SDK_TARGETS: Final = (("kimi-us", _KIMI_US), ("arn-yaml-flagged", _YAML_FLAGGED_ARN))


@pytest.mark.timeout(600)
@pytest.mark.parametrize(("name", "model_id"), _ANTHROPIC_SDK_TARGETS, ids=("k03-kimi", "k03-flagged-arn"))
def test_k03_anthropic_sdk_sync_messages_reaches_the_model_without_a_cache_point(
    rig: _Rig, name: str, model_id: str
) -> None:
    rig.wire.drain()
    message: Final = _anthropic_client(rig).messages.create(
        model=name,
        max_tokens=16,
        system=[{"type": "text", "text": _SYSTEM_TEXT, "cache_control": {"type": "ephemeral"}}],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": _prompt(), "cache_control": {"type": "ephemeral"}}]}
        ],
        extra_body=_EXTRA,
    )
    assert "".join(block.text for block in message.content if block.type == "text") == _ANSWER, message
    received: Final = _only_received(rig.wire)
    assert received.model == model_id and not received.streaming, received
    assert received.cache_points == 0, received.body
    _success_row(message.id)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(("name", "model_id"), _ANTHROPIC_SDK_TARGETS, ids=("k04-kimi", "k04-flagged-arn"))
async def test_k04_anthropic_sdk_async_stream_reaches_the_model_without_a_cache_point(
    rig: _Rig, name: str, model_id: str
) -> None:
    rig.wire.drain()
    async with _async_anthropic_client(rig).messages.stream(
        model=name,
        max_tokens=16,
        system=[{"type": "text", "text": _SYSTEM_TEXT, "cache_control": {"type": "ephemeral"}}],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": _prompt(), "cache_control": {"type": "ephemeral"}}]}
        ],
        extra_body=_EXTRA,
    ) as stream:
        final: Final = await stream.get_final_message()
    assert "".join(block.text for block in final.content if block.type == "text") == _ANSWER, final
    received: Final = _only_received(rig.wire)
    assert received.model == model_id and received.streaming, received
    assert received.cache_points == 0, received.body
    _success_row(final.id)


@pytest.mark.timeout(600)
def test_k05_openai_sdk_sync_responses_reaches_kimi_without_a_cache_point(rig: _Rig) -> None:
    rig.wire.drain()
    response: Final = _openai_client(rig).responses.create(
        model="kimi-us",
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": _prompt(), "cache_control": {"type": "ephemeral"}}],
            }
        ],  # pyright: ignore[reportArgumentType]  # cache_control rides as an extra key
        max_output_tokens=16,
        extra_body=_EXTRA,
    )
    assert response.output_text == _ANSWER, response
    received: Final = _only_received(rig.wire)
    assert received.model == _KIMI_US and not received.streaming, received
    assert received.cache_points == 0, received.body
    _success_row(_inner_response_id(response.id))


@pytest.mark.timeout(600)
async def test_k06_openai_sdk_async_responses_stream_reaches_kimi_without_a_cache_point(rig: _Rig) -> None:
    rig.wire.drain()
    stream: Final = await _async_openai_client(rig).responses.create(
        model="kimi-us",
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": _prompt(), "cache_control": {"type": "ephemeral"}}],
            }
        ],  # pyright: ignore[reportArgumentType]  # cache_control rides as an extra key
        max_output_tokens=16,
        stream=True,
        extra_body=_EXTRA,
    )
    completed: Final = [event.response async for event in stream if event.type == "response.completed"]
    assert len(completed) == 1 and completed[0].output_text == _ANSWER, completed
    received: Final = _only_received(rig.wire)
    assert received.model == _KIMI_US and received.streaming, received
    assert received.cache_points == 0, received.body
    _success_row(_inner_response_id(completed[0].id))


_LOCATIONS: Final = (
    ("system", frozenset[Marker]({"system"})),
    ("user", frozenset[Marker]({"user"})),
    ("tool", frozenset[Marker]({"tool"})),
)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(("location", "markers"), _LOCATIONS, ids=tuple(f"l-{location}" for location, _ in _LOCATIONS))
def test_l01_to_l03_each_marker_location_is_dropped_for_kimi_and_kept_for_claude(
    rig: _Rig, location: str, markers: frozenset[Marker]
) -> None:
    kimi, kimi_received = _observe(rig, "chat", _chat_body("kimi-us", _prompt(), markers, with_tools=True))
    _assert_answered(kimi)
    assert kimi_received.cache_points == 0, (location, kimi_received.body)
    claude, claude_received = _observe(rig, "chat", _chat_body("claude-control", _prompt(), markers, with_tools=True))
    _assert_answered(claude)
    assert claude_received.cache_points == 1, (location, claude_received.body)
    _success_row(kimi.response_id)
    _success_row(claude.response_id)


@pytest.mark.timeout(600)
def test_l04_l05_gateway_injection_points_are_dropped_for_kimi_and_kept_for_claude(rig: _Rig) -> None:
    kimi_message, kimi_message_received = _observe(
        rig, "chat", _chat_body("kimi-inject-message", _prompt(), _NO_MARKERS, with_tools=True)
    )
    _assert_answered(kimi_message)
    assert kimi_message_received.cache_points == 0, kimi_message_received.body
    kimi_tool, kimi_tool_received = _observe(
        rig, "chat", _chat_body("kimi-inject-tool", _prompt(), _NO_MARKERS, with_tools=True)
    )
    _assert_answered(kimi_tool)
    assert kimi_tool_received.cache_points == 0, kimi_tool_received.body
    claude, claude_received = _observe(
        rig, "chat", _chat_body("claude-inject-message", _prompt(), _NO_MARKERS, with_tools=True)
    )
    _assert_answered(claude)
    assert claude_received.cache_points == 1, claude_received.body
    system: Final = claude_received.body.get("system")
    assert isinstance(system, list) and "cachePoint" in _JSON.validate_python(system[-1]), claude_received.body
    for outcome in (kimi_message, kimi_tool, claude):
        _success_row(outcome.response_id)


@pytest.mark.timeout(600)
def test_l06_a_request_without_markers_was_never_touched(rig: _Rig) -> None:
    for name in ("kimi-us", "claude-control"):
        outcome, received = _observe(rig, "chat", _chat_body(name, _prompt(), _NO_MARKERS, with_tools=True))
        _assert_answered(outcome)
        assert received.cache_points == 0, (name, received.body)
        _success_row(outcome.response_id)


def _rates(gateway: Gateway, name: str) -> dict[str, float]:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    (entry,) = tuple(
        _JSON.validate_python(item) for item in entries if _JSON.validate_python(item)["model_name"] == name
    )
    info: Final = _JSON.validate_python(entry["model_info"])
    return {
        key: float(str(info[key]))
        for key in (
            "input_cost_per_token",
            "output_cost_per_token",
            "cache_read_input_token_cost",
            "cache_creation_input_token_cost",
        )
    }


@pytest.mark.timeout(600)
def test_p01_kimi_cache_usage_is_still_priced_from_its_own_row(rig: _Rig) -> None:
    outcome, received = _observe(rig, "chat", _chat_body("kimi-us", _prompt(_CACHE_USAGE_MARKER), _NO_MARKERS))
    _assert_answered(outcome)
    assert received.cache_points == 0, received.body
    body: Final = _JSON.validate_python(json.loads(outcome.raw))
    usage: Final = _JSON.validate_python(body["usage"])
    assert usage["prompt_tokens"] == 161 and usage["completion_tokens"] == 4, usage
    details: Final = _JSON.validate_python(usage["prompt_tokens_details"])
    assert details["cached_tokens"] == 100, usage
    assert usage["cache_creation_input_tokens"] == 50, usage
    rates: Final = _rates(rig.gateway, "kimi-us")
    expected: Final = (
        11 * rates["input_cost_per_token"]
        + 100 * rates["cache_read_input_token_cost"]
        + 50 * rates["cache_creation_input_token_cost"]
        + 4 * rates["output_cost_per_token"]
    )
    assert abs(float(outcome.headers["x-litellm-response-cost"]) - expected) < 1e-12, (outcome.headers, rates)
    row: Final = _success_row(outcome.response_id)
    assert abs(float(str(row["spend"])) - expected) < 1e-12, (row, rates)


@pytest.mark.timeout(600)
def test_p02_a_plain_kimi_reply_is_priced_from_its_own_row(rig: _Rig) -> None:
    outcome, received = _observe(rig, "chat", _chat_body("kimi-us", _prompt(), _NO_MARKERS))
    _assert_answered(outcome)
    assert received.cache_points == 0, received.body
    rates: Final = _rates(rig.gateway, "kimi-us")
    expected: Final = 11 * rates["input_cost_per_token"] + 4 * rates["output_cost_per_token"]
    assert abs(float(outcome.headers["x-litellm-response-cost"]) - expected) < 1e-12, (outcome.headers, rates)
    row: Final = _success_row(outcome.response_id)
    assert abs(float(str(row["spend"])) - expected) < 1e-12, (row, rates)


@pytest.mark.timeout(600)
def test_s07_a_5kb_caller_model_under_the_wildcard_is_a_bounded_4xx_and_liveliness_stays_up(rig: _Rig) -> None:
    rig.wire.drain()
    started: Final = time.monotonic()
    outcome: Final = _send(rig.gateway, "chat", _chat_body(f"bedrock/{'a' * 5120}", _prompt(), _SYSTEM_AND_USER))
    elapsed: Final = time.monotonic() - started
    liveliness: Final = rig.gateway.client.get("/health/liveliness", timeout=5)
    assert liveliness.status_code == 200, liveliness.text
    assert outcome.status in (400, 404), (outcome.status, outcome.raw[:300])
    error: Final = _JSON.validate_python(_JSON.validate_python(json.loads(outcome.raw))["error"])
    assert "a" * 5120 in str(error["message"]), outcome.raw[:300]
    assert elapsed < 10, elapsed
    assert rig.wire.drain() == ()
    _failure_row(outcome.call_id)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("stream", (False, True), ids=("s08-plain", "s08-stream"))
def test_s08_a_peer_rejection_on_kimi_is_a_400_with_the_message_and_a_failure_row(rig: _Rig, stream: bool) -> None:
    rig.wire.drain()
    outcome: Final = _send(
        rig.gateway, "chat", _chat_body("kimi-us", _prompt(_REJECT_MARKER), _NO_MARKERS, stream=stream)
    )
    received: Final = _only_received(rig.wire)
    assert received.cache_points == 0, received.body
    assert outcome.status == 400, (outcome.status, outcome.raw)
    assert _REJECTION in outcome.raw.replace('\\"', '"'), outcome.raw
    _failure_row(outcome.call_id)


@pytest.mark.timeout(600)
def test_s09_an_unauthenticated_kimi_request_never_reaches_the_peer(rig: _Rig) -> None:
    rig.wire.drain()
    outcome: Final = _send(
        rig.gateway, "chat", _chat_body("kimi-us", _prompt(), _SYSTEM_AND_USER), key="sk-integration-bogus"
    )
    assert outcome.status == 401, outcome.raw
    assert rig.wire.drain() == ()


@pytest.mark.timeout(600)
def test_x02_two_identical_uncached_requests_are_two_peer_calls_and_two_rows(rig: _Rig) -> None:
    body: Final = _chat_body("kimi-us", _prompt(), _NO_MARKERS)
    rig.wire.drain()
    first: Final = _send(rig.gateway, "chat", body)
    second: Final = _send(rig.gateway, "chat", body)
    _assert_answered(first)
    _assert_answered(second)
    assert first.response_id != second.response_id, (first.response_id, second.response_id)
    received: Final = _received(rig.wire)
    assert len(received) == 2 and all(item.cache_points == 0 for item in received), received
    rows: Final = _spend_rows(frozenset({first.response_id, second.response_id}), expected=2)
    assert {str(row["request_id"]) for row in rows} == {first.response_id, second.response_id}, rows


@pytest.mark.timeout(600)
def test_x03_a_response_cache_hit_still_answers_after_fewer_peer_calls_than_sends(rig: _Rig) -> None:
    body: Final = {key: value for key, value in _chat_body("kimi-us", _prompt(), _NO_MARKERS).items() if key != "cache"}
    rig.wire.drain()
    first: Final = _send(rig.gateway, "chat", body)
    _assert_answered(first)
    sends: Final[list[_Outcome]] = []

    def resend() -> _Outcome:
        served: Final = _send(rig.gateway, "chat", body)
        sends.append(served)
        return served

    hit: Final = eventually(
        resend, lambda served: served.status == 200 and served.response_id == first.response_id, seconds=15
    )
    _assert_answered(hit)
    received: Final = _received(rig.wire)
    assert all(item.cache_points == 0 for item in received), received
    assert len(received) == len(sends), (len(received), len(sends))
    _success_row(first.response_id)


@pytest.mark.timeout(600)
def test_x05_an_arn_whose_profile_name_starts_with_openai_dot_is_classified_by_the_pre_existing_family_rule(
    gateway: Gateway,
) -> None:
    arn: Final = f"{_PROFILE_ARN_PREFIX}openai.custom-{uuid.uuid4().hex[:8]}"
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        name: Final = _deployment(gateway, scenario, wire, f"bedrock/{arn}")
        outcome, received = _observe_at(gateway, wire, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER))
        _assert_answered(outcome)
        assert received.model == arn, received.model
        assert received.cache_points == 0, received.body
        _success_row(outcome.response_id)


def _deployment(
    gateway: Gateway,
    scenario: Scenario,
    wire: Wire,
    model: str,
    model_info: Mapping[str, JsonValue] | None = None,
    *,
    cleanup: bool = True,
) -> str:
    name: Final = f"kimi-audit-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": model, "api_base": wire.url, "num_retries": 0, **_AWS},
            "model_info": dict(model_info) if model_info is not None else {},
        },
    )
    identity: Final = str(_JSON.validate_python(created["model_info"])["id"])
    if cleanup:
        scenario.cleanups.callback(gateway.post, "/model/delete", {"id": identity})
    _settled(gateway, name, wire)
    return name


def _settled(gateway: Gateway, name: str, wire: Wire) -> None:
    eventually(
        lambda: tuple(
            gateway.request("POST", "/v1/chat/completions", _chat_body(name, _prompt(), _NO_MARKERS)).status_code
            for _ in range(12)
        ),
        lambda codes: all(code == 200 for code in codes),
        seconds=60,
    )
    wire.drain()


def _observe_at(
    gateway: Gateway, wire: Wire, endpoint: Endpoint, body: Mapping[str, JsonValue]
) -> tuple[_Outcome, _Received]:
    wire.drain()
    outcome: Final = _send(gateway, endpoint, body)
    return outcome, _only_received(wire)


def _arn(label: str) -> str:
    return f"{_PROFILE_ARN_PREFIX}{label}-{uuid.uuid4().hex[:12]}"


_ROUTES: Final = ("", "converse/", "converse_like/")


@pytest.mark.timeout(600)
@pytest.mark.parametrize("route", _ROUTES, ids=("m05-plain-route", "m06-converse-route", "m07-converse-like-route"))
def test_m05_to_m07_a_deployment_flag_false_on_an_arn_strips_cache_points_on_every_route(
    gateway: Gateway, route: str
) -> None:
    arn: Final = _arn("flagged")
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        name: Final = _deployment(
            gateway, scenario, wire, f"bedrock/{route}{arn}", {"supports_prompt_cache_breakpoint": False}
        )
        expected_model: Final = _CONVERSE_LIKE if route == "converse_like/" else arn
        for endpoint in ("chat", "messages", "responses"):
            outcome, received = _observe_at(gateway, wire, endpoint, _body(endpoint, name, _prompt(), _SYSTEM_AND_USER))
            _assert_answered(outcome)
            assert received.model == expected_model, (endpoint, received.model)
            assert received.action == "converse", (endpoint, received.action)
            assert received.cache_points == 0, (endpoint, received.body)
            _success_row(outcome.response_id)


@pytest.mark.timeout(600)
def test_m08_an_arn_without_the_flag_still_gets_cache_points(gateway: Gateway) -> None:
    arn: Final = _arn("plain")
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        name: Final = _deployment(gateway, scenario, wire, f"bedrock/{arn}")
        outcome, received = _observe_at(gateway, wire, "chat", _chat_body(name, _prompt(), _ALL_MARKERS))
        _assert_answered(outcome)
        assert received.cache_points == 3, received.body
        _success_row(outcome.response_id)


@pytest.mark.timeout(600)
def test_m12_a_deployment_flag_true_on_kimi_overrides_the_cost_map_row(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        name: Final = _deployment(
            gateway, scenario, wire, f"bedrock/{_KIMI_BASE}", {"supports_prompt_cache_breakpoint": True}
        )
        outcome, received = _observe_at(gateway, wire, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER))
        _assert_answered(outcome)
        assert received.model == _KIMI_BASE, received.model
        assert received.cache_points == 2, received.body
        _success_row(outcome.response_id)


@pytest.mark.timeout(600)
def test_m13_a_null_deployment_flag_on_kimi_falls_back_to_the_cost_map_row(gateway: Gateway) -> None:
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        name: Final = _deployment(
            gateway, scenario, wire, f"bedrock/{_KIMI_US}", {"supports_prompt_cache_breakpoint": None}
        )
        outcome, received = _observe_at(gateway, wire, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER))
        _assert_answered(outcome)
        assert received.model == _KIMI_US, received.model
        assert received.cache_points == 0, received.body
        _success_row(outcome.response_id)


_ODD_FLAGS: Final[tuple[tuple[str, JsonValue], ...]] = (
    ("s01-string-true", "true"),
    ("s02-int-one", 1),
    ("s03-int-zero", 0),
    ("s04-empty-list", []),
    ("s05-empty-string", ""),
    ("s06-5kb-string", "x" * 5120),
)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(("label", "flag"), _ODD_FLAGS, ids=tuple(label for label, _ in _ODD_FLAGS))
def test_s01_to_s06_an_odd_typed_flag_is_read_as_false_never_as_true(
    gateway: Gateway, label: str, flag: JsonValue
) -> None:
    arn: Final = _arn(label)
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        name: Final = _deployment(gateway, scenario, wire, f"bedrock/{arn}", {"supports_prompt_cache_breakpoint": flag})
        outcome, received = _observe_at(gateway, wire, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER))
        _assert_answered(outcome)
        assert received.cache_points == 0, (label, received.body)
        _success_row(outcome.response_id)


@pytest.mark.timeout(600)
def test_s10_s11_two_deployments_of_one_arn_share_the_flag_and_a_delete_leaves_it_in_place(gateway: Gateway) -> None:
    arn: Final = _arn("shared")
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        flagged: Final = _deployment(
            gateway, scenario, wire, f"bedrock/{arn}", {"supports_prompt_cache_breakpoint": False}, cleanup=False
        )
        plain: Final = _deployment(gateway, scenario, wire, f"bedrock/{arn}")
        for name in (flagged, plain):
            outcome, received = _observe_at(gateway, wire, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER))
            _assert_answered(outcome)
            assert received.cache_points == 0, (name, received.body)
            _success_row(outcome.response_id)
        flagged_identity: Final = _model_id(flagged)
        gateway.post("/model/delete", {"id": flagged_identity})
        eventually(
            lambda: tuple(
                gateway.request("POST", "/v1/chat/completions", _chat_body(flagged, _prompt(), _NO_MARKERS)).status_code
                for _ in range(12)
            ),
            lambda codes: all(code != 200 for code in codes),
            seconds=60,
        )
        wire.drain()
        after, after_received = _observe_at(gateway, wire, "chat", _chat_body(plain, _prompt(), _SYSTEM_AND_USER))
        _assert_answered(after)
        assert after_received.cache_points == 0, after_received.body
        _success_row(after.response_id)


def _model_id(name: str) -> str:
    (row,) = read_rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_name=%s', (name,))
    return str(row["model_id"])


def _stored_flag(identity: str) -> JsonValue:
    (row,) = read_rows('SELECT model_info FROM "LiteLLM_ProxyModelTable" WHERE model_id=%s', (identity,))
    stored: Final = row["model_info"]
    info: Final = _JSON.validate_python(stored if isinstance(stored, dict) else json.loads(str(stored)))
    return info.get("supports_prompt_cache_breakpoint")


def _six_cache_point_counts(gateway: Gateway, wire: Wire, name: str) -> tuple[int, ...]:
    wire.drain()
    outcomes: Final = tuple(_send(gateway, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER)) for _ in range(6))
    assert all(item.status == 200 for item in outcomes), outcomes
    return tuple(item.cache_points for item in _received(wire))


@pytest.mark.timeout(600)
def test_x01_flipping_the_flag_to_true_through_a_patch_update_turns_cache_points_back_on(gateway: Gateway) -> None:
    arn: Final = _arn("flip")
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        name: Final = _deployment(
            gateway, scenario, wire, f"bedrock/{arn}", {"supports_prompt_cache_breakpoint": False}
        )
        before, before_received = _observe_at(gateway, wire, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER))
        _assert_answered(before)
        assert before_received.cache_points == 0, before_received.body
        identity: Final = _model_id(name)
        patched: Final = gateway.request(
            "PATCH", f"/model/{identity}/update", {"model_info": {"supports_prompt_cache_breakpoint": True}}
        )
        assert patched.status_code == 200, patched.text
        assert _stored_flag(identity) is True
        points: Final = eventually(
            lambda: _six_cache_point_counts(gateway, wire, name),
            lambda counts: len(counts) == 6 and all(count == 2 for count in counts),
            seconds=60,
        )
        assert points == (2,) * 6, points
        _success_row(before.response_id)


@pytest.mark.timeout(600)
def test_x06_the_legacy_post_update_answers_200_and_leaves_the_stored_flag_alone(gateway: Gateway) -> None:
    arn: Final = _arn("legacy")
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        name: Final = _deployment(
            gateway, scenario, wire, f"bedrock/{arn}", {"supports_prompt_cache_breakpoint": False}
        )
        before, before_received = _observe_at(gateway, wire, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER))
        _assert_answered(before)
        identity: Final = _model_id(name)
        updated: Final = gateway.request(
            "POST",
            "/model/update",
            {
                "model_name": name,
                "litellm_params": {"model": f"bedrock/{arn}", "api_base": wire.url, "num_retries": 0, **_AWS},
                "model_info": {"id": identity, "supports_prompt_cache_breakpoint": True},
            },
        )
        assert updated.status_code == 200, updated.text
        assert _stored_flag(identity) is False
        _settled(gateway, name, wire)
        after, after_received = _observe_at(gateway, wire, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER))
        _assert_answered(after)
        assert after_received.cache_points == before_received.cache_points, (before_received.body, after_received.body)
        _success_row(before.response_id)
        _success_row(after.response_id)


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    index: int
    stream: bool


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    outcome: _Outcome


def _burst_calls(count: int) -> tuple[_Call, ...]:
    endpoints: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
    return tuple(_Call(endpoints[index % 3], index, index % 2 == 0) for index in range(count))


async def _send_async(client: httpx.AsyncClient, key: str, model: str, call: _Call, prompt: str) -> _Served:
    body: Final = _body(call.endpoint, model, f"{prompt} {call.index}", _SYSTEM_AND_USER, stream=call.stream)
    async with client.stream(
        "POST", _path(call.endpoint), json=body, headers={"Authorization": f"Bearer {key}"}
    ) as response:
        raw: Final = (await response.aread()).decode()
    lines: Final = tuple(line for line in raw.splitlines() if line)
    return _Served(call, _outcome_of(call.endpoint, call.stream, response, lines))


async def _burst(
    base_url: str,
    key: str,
    model: str,
    calls: tuple[_Call, ...],
    prompt: str,
    *,
    tolerate_transport_errors: bool = False,
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send_async(client, key, model, call, prompt) for call in calls),
            return_exceptions=tolerate_transport_errors,
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _assert_rows_once(served: tuple[_Served, ...]) -> None:
    ids: Final = frozenset(item.outcome.response_id for item in served)
    assert len(ids) == len(served), ids
    rows: Final = _spend_rows(ids, expected=len(ids))
    assert sorted(str(row["request_id"]) for row in rows) == sorted(ids), rows
    assert all(row["status"] == "success" for row in rows), rows


def _reserved_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


@pytest.mark.timeout(900)
async def test_c01_a_peer_outage_mid_traffic_fails_cleanly_and_recovery_lands_every_id_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    port: Final = _reserved_port()
    prompt: Final = _prompt()
    with wire_server(_peer(), port=port) as first_peer:
        config: Final = _owned_config(first_peer, tmp_path, names=("kimi-us", "claude-control"))
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate = owned.gateway  # rebind-ok: the same name covers the restarted proxy below
            url = _proxy_url(candidate)  # rebind-ok: the same name covers the restarted proxy below
            served: Final = await _burst(url, candidate.key, "kimi-us", _burst_calls(30), prompt)
            assert len(served) == 30
            for item in served:
                _assert_answered(item.outcome)
            received: Final = _received(first_peer)
            assert len(received) == 30 and all(item.cache_points == 0 for item in received), received
            _assert_rows_once(served)
    with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
        candidate = owned.gateway
        url = _proxy_url(candidate)
        first_peer_down: Final = await _burst(url, candidate.key, "kimi-us", _burst_calls(30), prompt)
        assert len(first_peer_down) == 30
        for item in first_peer_down:
            assert item.outcome.status >= 500, (item.call, item.outcome.status, item.outcome.raw)
        liveliness: Final = candidate.client.get("/health/liveliness", timeout=5)
        assert liveliness.status_code == 200, liveliness.text
        failed_ids: Final = frozenset(
            item.outcome.call_id
            for item in first_peer_down
            if item.call.endpoint != "messages" and item.outcome.call_id
        )
        assert len(failed_ids) == 20, failed_ids
        failure_rows: Final = _spend_rows(failed_ids, expected=20)
        assert all(row["status"] == "failure" for row in failure_rows), failure_rows
        with wire_server(_peer(), port=port) as second_peer:
            recovered: Final = await _burst(url, candidate.key, "kimi-us", _burst_calls(30), prompt)
            assert len(recovered) == 30
            for item in recovered:
                _assert_answered(item.outcome)
            again: Final = _received(second_peer)
            assert len(again) == 30 and all(item.cache_points == 0 for item in again), again
            _assert_rows_once(recovered)
            control: Final = await _burst(url, candidate.key, "claude-control", _burst_calls(6), prompt)
            assert all(item.outcome.status == 200 for item in control), control
            control_received: Final = _received(second_peer)
            assert len(control_received) == 6, control_received
            assert all(item.cache_points + item.cache_controls > 0 for item in control_received), control_received


def _open_peer_connections(pid: int, peer_url: str) -> int:
    port: Final = urlsplit(peer_url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


async def _held_burst(url: str, key: str, count: int, prompt: str) -> tuple[_Served, ...]:
    calls: Final = tuple(_Call("chat", index, True) for index in range(count))
    return await _burst(url, key, "kimi-us", calls, f"{prompt} {_HOLD_MARKER}", tolerate_transport_errors=True)


@pytest.mark.timeout(900)
async def test_c02_a_worker_killed_mid_traffic_leaves_the_survivor_and_the_respawn_stripping(
    gateway: Gateway, tmp_path: Path
) -> None:
    hold: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    prompt: Final = _prompt()
    with wire_server(_peer(hold, held)) as wire:
        config: Final = _owned_config(wire, tmp_path, names=("kimi-us",))
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            url: Final = _proxy_url(candidate)
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=60,
            )
            burst: Final = asyncio.create_task(_held_burst(url, candidate.key, 20, prompt))
            await asyncio.to_thread(eventually, held.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_peer_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            psutil.Process(victim_pid).send_signal(signal.SIGKILL)
            hold.set()
            served: Final = await burst
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_answered(item.outcome)
            received: Final = _received(wire)
            assert len(received) == 20 and all(item.cache_points == 0 for item in received), received
            respawned: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 3,
                seconds=60,
            )[-1]
            hold.clear()
            for _ in range(20):
                for _ in range(held.qsize()):
                    held.get_nowait()
                follow_up_task: Final = asyncio.create_task(_held_burst(url, candidate.key, 6, prompt))
                await asyncio.to_thread(eventually, held.qsize, lambda size: size == 6, 60)
                on_respawn: Final = _open_peer_connections(respawned, wire.url)
                hold.set()
                follow_up: Final = await follow_up_task
                hold.clear()
                assert len(follow_up) == 6, follow_up
                for item in follow_up:
                    _assert_answered(item.outcome)
                later: Final = _received(wire)
                assert len(later) == 6 and all(item.cache_points == 0 for item in later), later
                if on_respawn > 0:
                    break
            else:
                raise AssertionError("The respawned worker never took a held stream")
            _assert_rows_once(served)


@pytest.mark.timeout(900)
def test_c03_a_restart_re_registers_a_stored_deployment_flag_from_the_database(
    gateway: Gateway, tmp_path: Path
) -> None:
    arn: Final = _arn("restart")
    with wire_server(_peer()) as wire, gateway.scenario() as scenario:
        config: Final = _owned_config(wire, tmp_path, names=("claude-control",))
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            name: Final = _deployment(
                owned.gateway,
                scenario,
                wire,
                f"bedrock/{arn}",
                {"supports_prompt_cache_breakpoint": False},
                cleanup=False,
            )
            before, before_received = _observe_at(
                owned.gateway, wire, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER)
            )
            _assert_answered(before)
            assert before_received.cache_points == 0, before_received.body
            _success_row(before.response_id)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as restarted:
            _settled(restarted.gateway, name, wire)
            after, after_received = _observe_at(
                restarted.gateway, wire, "chat", _chat_body(name, _prompt(), _SYSTEM_AND_USER)
            )
            _assert_answered(after)
            assert after_received.model == arn, after_received.model
            assert after_received.cache_points == 0, after_received.body
            _success_row(after.response_id)
            control, control_received = _observe_at(
                restarted.gateway, wire, "chat", _chat_body("claude-control", _prompt(), _SYSTEM_AND_USER)
            )
            _assert_answered(control)
            assert control_received.cache_points == 2, control_received.body
            restarted.gateway.post("/model/delete", {"id": _model_id(name)})


@pytest.mark.timeout(900)
async def test_c04_a_slow_peer_under_ten_concurrent_streams_completes_every_call_once(rig: _Rig) -> None:
    rig.wire.drain()
    calls: Final = tuple(_Call("chat", index, True) for index in range(10))
    started: Final = time.monotonic()
    served: Final = await _burst(
        _proxy_url(rig.gateway), rig.gateway.key, "kimi-us", calls, f"{_prompt()} {_SLOW_MARKER}"
    )
    elapsed: Final = time.monotonic() - started
    assert len(served) == 10
    for item in served:
        _assert_answered(item.outcome)
    received: Final = _received(rig.wire)
    assert len(received) == 10 and all(item.cache_points == 0 for item in received), received
    assert elapsed < 30, elapsed
    _assert_rows_once(served)

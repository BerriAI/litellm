import json
import os
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

import httpx
import openai
import pytest
import yaml
from integration._support import claude_code as cc
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.worker_readiness import wait_for_worker_readiness
from pydantic import JsonValue, TypeAdapter

from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD
from litellm.responses.utils import ResponsesAPIRequestUtils

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 120), pytest.mark.shares_worker_state]

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CLAUDE: Final = cc.OPUS
_GEMINI: Final = "gemini-3.8-flash"
_OPENAI: Final = "gpt-5.6"
_PROXY_WORKERS: Final = max(1, int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1")))
_CACHE_BUST: Final[Mapping[str, JsonValue]] = {"cache": {"no-cache": True}}
_COMPACTING: Final[list[JsonValue]] = [{"type": "compaction", "compact_threshold": 150000}]
_COMPACT_BETA: Final = "compact-2026-01-12"
_CITY: Final = "Tokyo"
_MASKED_CITY: Final = "[CITY]"
_MASK_INPUT: Final = "compaction-mask-input"
_MASK_OUTPUT: Final = "compaction-mask-output"
_BLOCK_OUTPUT: Final = "compaction-block-output"
_REDACT_HEADER: Final = {"x-litellm-enable-message-redaction": "true"}
_UNREDACT_HEADER: Final = {"litellm-disable-message-redaction": "true"}
_SCRIPTED_REJECTION: Final = json.dumps(
    {"type": "error", "error": {"type": "invalid_request_error", "message": "scripted rejection"}}
).encode()
_ASK: Final = re.compile(r"ask-([0-9a-f]{32})")

Route: TypeAlias = Literal["claude-compacting", "claude-plain", "gemini", "openai", "openai-compact", "openai-opaque"]


def _new_marker() -> str:
    return uuid.uuid4().hex


def _ask(marker: str, *, compact: bool = False) -> str:
    return f"ask-{marker} compact-now-{marker}" if compact else f"ask-{marker}"


def _fail(marker: str) -> str:
    return f"fail-now-{marker}"


def _think(marker: str) -> str:
    return f"think-now-{marker}"


def _thinking(marker: str) -> dict[str, JsonValue]:
    return {"type": "thinking", "thinking": f"thought-{marker}", "signature": f"signature-{marker}"}


def _answer(marker: str) -> str:
    return f"answer-{marker}"


def _summary(marker: str) -> str:
    return f"kept-{marker} {'a' * 1500} middle-{marker} {'b' * 1500} the user lives in {_CITY}"


def _block(summary: str) -> dict[str, JsonValue]:
    return {"type": "compaction", "content": summary}


def _compaction_item(summary: str) -> dict[str, JsonValue]:
    return {
        "type": "compaction",
        "id": f"cmp_{uuid.uuid4().hex}",
        "encrypted_content": json.dumps(_block(summary), separators=(",", ":")),
    }


def _newest_marker(raw: bytes) -> str:
    found: Final = _ASK.findall(raw.decode())
    assert found, raw[:400]
    return str(found[-1])


def _anthropic_usage() -> dict[str, JsonValue]:
    return {"input_tokens": 20, "output_tokens": 5}


def _anthropic_block_frames(index: int, block: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
    def start(content_block: Mapping[str, JsonValue]) -> bytes:
        return cc.sse_frame(
            "content_block_start", {"type": "content_block_start", "index": index, "content_block": dict(content_block)}
        )

    def delta(fields: Mapping[str, JsonValue]) -> bytes:
        return cc.sse_frame(
            "content_block_delta", {"type": "content_block_delta", "index": index, "delta": dict(fields)}
        )

    stop: Final = cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": index})
    match block["type"]:
        case "compaction":
            summary: Final = string_value(block["content"])
            half: Final = len(summary) // 2
            return (
                start({"type": "compaction", "content": ""}),
                delta({"type": "compaction_delta", "content": summary[:half]}),
                delta({"type": "compaction_delta", "content": summary[half:]}),
                stop,
            )
        case "thinking":
            return (
                start({"type": "thinking", "thinking": ""}),
                delta({"type": "thinking_delta", "thinking": block["thinking"]}),
                delta({"type": "signature_delta", "signature": block["signature"]}),
                stop,
            )
        case _:
            return (start({"type": "text", "text": ""}), delta({"type": "text_delta", "text": block["text"]}), stop)


def _anthropic_stream(identity: str, content: Sequence[Mapping[str, JsonValue]]) -> tuple[bytes, ...]:
    start: Final = cc.sse_frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": _CLAUDE,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 1},
            },
        },
    )
    blocks: Final = tuple(
        chain.from_iterable(_anthropic_block_frames(index, block) for index, block in enumerate(content))
    )
    end: Final = cc.sse_frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
    )
    return (start, *blocks, end, cc.sse_frame("message_stop", {"type": "message_stop"}))


def _claude_reply(body: Mapping[str, JsonValue], raw: bytes) -> Reply:
    marker: Final = _newest_marker(raw)
    if _fail(marker).encode() in raw:
        return Reply(status=400, body=_SCRIPTED_REJECTION)
    compaction: Final = (_block(_summary(marker)),) if f"compact-now-{marker}".encode() in raw else ()
    thinking: Final = (_thinking(marker),) if _think(marker).encode() in raw else ()
    content: Final = (*compaction, *thinking, {"type": "text", "text": _answer(marker)})
    identity: Final = f"msg_{uuid.uuid4().hex}"
    if body.get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=_anthropic_stream(identity, content))
    return Reply(body=cc.message_reply(identity, _CLAUDE, content, _anthropic_usage()))


def _gemini_reply(raw: bytes) -> Reply:
    frame: Final = {
        "responseId": f"gemini-{uuid.uuid4().hex}",
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": _answer(_newest_marker(raw))}]},
                "index": 0,
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25},
        "modelVersion": _GEMINI,
    }
    return Reply(body=json.dumps(frame).encode())


_OPENAI_USAGE: Final = {
    "input_tokens": 20,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 5,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 25,
}


def _openai_reply(raw: bytes) -> Reply:
    tag: Final = uuid.uuid4().hex
    message: Final = {
        "id": f"msg_{tag}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": _answer(_newest_marker(raw)), "annotations": []}],
    }
    response: Final = {
        "id": f"resp_{tag}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": _OPENAI,
        "output": [message],
        "usage": _OPENAI_USAGE,
    }
    return Reply(body=json.dumps(response).encode())


def _openai_compact_reply() -> Reply:
    compacted: Final = {
        "id": f"resp_compact_{uuid.uuid4().hex}",
        "created_at": 1,
        "object": "response.compaction",
        "output": [],
        "usage": _OPENAI_USAGE,
    }
    return Reply(body=json.dumps(compacted).encode())


def _vendor(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=json.dumps({"object": "list", "data": [{"id": _OPENAI, "object": "model"}]}).encode())
    path: Final = urlsplit(request.target).path
    if path.endswith("/v1/messages/count_tokens"):
        return Reply(body=json.dumps({"input_tokens": _word_count(_JSON_OBJECT.validate_json(request.body))}).encode())
    if path.endswith("/v1/messages"):
        return _claude_reply(_JSON_OBJECT.validate_json(request.body), request.body)
    if path.endswith(":generateContent"):
        return _gemini_reply(request.body)
    if path.endswith("/responses/compact"):
        return _openai_compact_reply()
    assert path.endswith("/responses"), request.target
    return _openai_reply(request.body)


def _provider_calls(wire: Wire, marker: str) -> tuple[Request, ...]:
    needle: Final = f"ask-{marker}".encode()
    return tuple(request for request in wire.drain() if request.method == "POST" and needle in request.body)


def _only_call(wire: Wire, marker: str) -> tuple[Request, dict[str, JsonValue]]:
    calls: Final = _provider_calls(wire, marker)
    assert len(calls) == 1, [call.target for call in calls]
    return calls[0], _JSON_OBJECT.validate_json(calls[0].body)


def _objects(value: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    assert isinstance(value, list), value
    return tuple(object_value(item) for item in value)


def _message_texts(message: Mapping[str, JsonValue], key: str = "content") -> tuple[str, ...]:
    content: Final = message.get(key)
    if isinstance(content, str):
        return (content,)
    if not isinstance(content, list):
        return ()
    return tuple(string_value(part["text"]) for part in _objects(content) if isinstance(part.get("text"), str))


def _anthropic_texts(body: Mapping[str, JsonValue], role: str) -> tuple[str, ...]:
    messages: Final = tuple(message for message in _objects(body["messages"]) if message["role"] == role)
    return tuple(chain.from_iterable(map(_message_texts, messages)))


def _word_count(body: Mapping[str, JsonValue]) -> int:
    texts: Final = (*_anthropic_texts(body, "user"), *_anthropic_texts(body, "assistant"))
    return sum(len(text.split()) for text in texts)


def _anthropic_block_types(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    contents: Final = tuple(message["content"] for message in _objects(body["messages"]))
    blocks: Final = chain.from_iterable(_objects(content) for content in contents if isinstance(content, list))
    return tuple(block.get("type") for block in blocks)


def _gemini_texts(body: Mapping[str, JsonValue], role: str) -> tuple[str, ...]:
    turns: Final = tuple(turn for turn in _objects(body["contents"]) if turn["role"] == role)
    return tuple(chain.from_iterable(_message_texts(turn, "parts") for turn in turns))


def _openai_user_texts(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    items: Final = tuple(item for item in _objects(body["input"]) if item.get("role") == "user")
    return tuple(chain.from_iterable(map(_message_texts, items)))


def _openai_item_types(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    return tuple(item.get("type") for item in _objects(body["input"]))


def _register(scenario: Scenario, wire: Wire, provider: Literal["claude", "gemini", "openai"]) -> str:
    match provider:
        case "claude":
            model = scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        case "gemini":
            model = scenario.model(model=f"gemini/{_GEMINI}", api_base=wire.url, api_key="synthetic-gemini-key")
        case "openai":
            model = scenario.model(model=f"openai/{_OPENAI}", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key")
    wait_for_worker_readiness(scenario.gateway, model, 1, _PROXY_WORKERS)
    return model


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _decoded(encrypted_content: str) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(encrypted_content)


def _responses(
    gateway: Gateway, body: Mapping[str, JsonValue], headers: Mapping[str, str] | None = None
) -> dict[str, JsonValue]:
    response: Final = gateway.request("POST", "/v1/responses", {**body, **_CACHE_BUST}, headers=headers)
    assert response.status_code == 200, response.text
    return _JSON_OBJECT.validate_json(response.content)


def _sse_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    payloads: Final = tuple(
        line.removeprefix("data:").strip() for line in text.splitlines() if line.startswith("data:")
    )
    return tuple(_JSON_OBJECT.validate_json(payload) for payload in payloads if payload != "[DONE]")


def _stream(
    gateway: Gateway, path: str, body: Mapping[str, JsonValue], headers: Mapping[str, str] | None = None
) -> tuple[dict[str, JsonValue], ...]:
    sent_headers: Final = {"Authorization": f"Bearer {gateway.key}", **(headers or {})}
    with gateway.client.stream(
        "POST", path, json={**body, **_CACHE_BUST, "stream": True}, headers=sent_headers
    ) as response:
        text: Final = response.read().decode()
    assert response.status_code == 200, text
    return _sse_events(text)


def _delta_provider_fields(chunk: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        fields
        for choice in _objects(chunk.get("choices", []))
        if isinstance(fields := object_value(choice.get("delta", {})).get("provider_specific_fields"), dict)
    )


def _streamed_compaction_blocks(chunks: Sequence[Mapping[str, JsonValue]]) -> tuple[JsonValue, ...]:
    fields: Final = chain.from_iterable(_delta_provider_fields(chunk) for chunk in chunks)
    return tuple(field["compaction_blocks"] for field in fields if "compaction_blocks" in field)


def _output_items(response: Mapping[str, JsonValue], item_type: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(item for item in _objects(response["output"]) if item["type"] == item_type)


def _listed(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list), value
    return value


def _only(items: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    assert len(items) == 1, items
    return items[0]


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_claude_compaction_block_is_returned_as_a_compaction_output_item(gateway: Gateway, stream: bool) -> None:
    marker: Final = _new_marker()
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire, "claude")
        client: Final = _openai_client(gateway)
        request: Final = {
            "input": _ask(marker, compact=True),
            "extra_body": {"context_management": _COMPACTING, **_CACHE_BUST},
        }
        if not stream:
            reply: Final = client.responses.create(model=model, **request)
            assert [item.type for item in reply.output] == ["compaction", "message"]
            compaction: Final = reply.output[0]
            assert compaction.type == "compaction"
            assert _decoded(compaction.encrypted_content) == _block(_summary(marker))
            assert reply.output_text == _answer(marker)
        else:
            events: Final = tuple(client.responses.create(model=model, stream=True, **request))
            added: Final = [e for e in events if e.type == "response.output_item.added" and e.item.type == "compaction"]
            done: Final = [e for e in events if e.type == "response.output_item.done" and e.item.type == "compaction"]
            completed: Final = [e for e in events if e.type == "response.completed"]
            assert len(added) == 1 and len(done) == 1 and len(completed) == 1, [e.type for e in events]
            assert added[0].item.encrypted_content == ""
            assert done[0].output_index == added[0].output_index
            assert done[0].item.id == added[0].item.id
            assert _decoded(done[0].item.encrypted_content) == _block(_summary(marker))
            finished: Final = [item for item in completed[0].response.output if item.type == "compaction"]
            assert [(item.id, item.encrypted_content) for item in finished] == [
                (done[0].item.id, done[0].item.encrypted_content)
            ]
            message_indexes: Final = {
                e.output_index for e in events if e.type == "response.output_item.added" and e.item.type == "message"
            }
            assert message_indexes and added[0].output_index not in message_indexes
            assert completed[0].response.output_text == _answer(marker)
        call, body = _only_call(wire, marker)
        assert _COMPACT_BETA in call.headers.get("anthropic-beta", ""), dict(call.headers)


def _history(marker: str, compaction: Mapping[str, JsonValue]) -> list[JsonValue]:
    return [
        {"role": "developer", "content": f"rules-{marker}"},
        {"role": "user", "content": f"old-{marker}"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": f"old-answer-{marker}", "annotations": []}],
        },
        dict(compaction),
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": f"recap-{marker}", "annotations": []}],
        },
        {"role": "user", "content": _ask(marker)},
    ]


_ROUTES: Final[tuple[Route, ...]] = (
    "claude-compacting",
    "claude-plain",
    "gemini",
    "openai",
    "openai-compact",
    "openai-opaque",
)


def _route_provider(route: Route) -> Literal["claude", "gemini", "openai"]:
    if route.startswith("claude"):
        return "claude"
    return "gemini" if route == "gemini" else "openai"


def _assert_reached(
    route: Route, marker: str, sent: Mapping[str, JsonValue], call: Request, body: Mapping[str, JsonValue]
) -> None:
    summary: Final = _summary(marker)
    dumped: Final = json.dumps(body)
    if route == "openai-opaque":
        assert urlsplit(call.target).path.endswith("/responses"), call.target
        assert sent in _objects(body["input"]), body["input"]
        assert f"old-{marker}" in dumped
        return
    if route != "claude-compacting":
        assert f"old-{marker}" not in dumped and f"old-answer-{marker}" not in dumped, dumped[:2000]
    assert f"rules-{marker}" in dumped
    assert f"recap-{marker}" in dumped
    match route:
        case "claude-compacting":
            assistant: Final = tuple(
                message for message in _objects(body["messages"]) if message["role"] == "assistant"
            )
            blocks: Final = tuple(
                chain.from_iterable(
                    _objects(message["content"]) for message in assistant if isinstance(message["content"], list)
                )
            )
            assert _block(summary) in blocks, body["messages"]
            assert summary not in _anthropic_texts(body, "user")
            assert f"rules-{marker}" in json.dumps(body["system"])
            assert _COMPACT_BETA in call.headers.get("anthropic-beta", ""), dict(call.headers)
        case "claude-plain":
            assert "compaction" not in _anthropic_block_types(body), body["messages"]
            assert summary in _anthropic_texts(body, "user"), body["messages"]
        case "gemini":
            assert summary in _gemini_texts(body, "user"), body["contents"]
        case "openai" | "openai-compact":
            suffix: Final = "/responses/compact" if route == "openai-compact" else "/responses"
            assert urlsplit(call.target).path.endswith(suffix), call.target
            assert "compaction" not in _openai_item_types(body), body["input"]
            assert summary in _openai_user_texts(body), body["input"]


@pytest.mark.parametrize("route", _ROUTES)
def test_compaction_item_reaches_each_route_in_the_shape_it_reads(gateway: Gateway, route: Route) -> None:
    marker: Final = _new_marker()
    compaction: Final = (
        {"type": "compaction", "id": f"cmp_{marker}", "encrypted_content": f"gAAAAA-opaque-{marker}"}
        if route == "openai-opaque"
        else _compaction_item(_summary(marker))
    )
    compacting: Final[dict[str, JsonValue]] = (
        {"context_management": _COMPACTING} if route == "claude-compacting" else {}
    )
    dropping: Final[dict[str, JsonValue]] = (
        {"context_management": _COMPACTING, "drop_params": True} if route == "gemini" else {}
    )
    path: Final = "/v1/responses/compact" if route == "openai-compact" else "/v1/responses"
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire, _route_provider(route))
        response: Final = gateway.request(
            "POST",
            path,
            {"model": model, "input": _history(marker, compaction), **compacting, **dropping, **_CACHE_BUST},
        )
        assert response.status_code == 200, response.text
        if route != "openai-compact":
            assert _answer(marker) in response.text, response.text
        call, body = _only_call(wire, marker)
        _assert_reached(route, marker, compaction, call, body)


def _input_tokens(gateway: Gateway, model: str, items: list[JsonValue]) -> int:
    counted: Final = gateway.post("/v1/responses/input_tokens", {"model": model, "input": items})
    tokens: Final = counted["input_tokens"]
    assert isinstance(tokens, int), counted
    return tokens


def test_input_tokens_count_from_the_newest_compaction(gateway: Gateway) -> None:
    marker: Final = _new_marker()
    summary: Final = _summary(marker)
    old_history: Final = " ".join(f"history-{index}" for index in range(600))
    compacted: Final[list[JsonValue]] = [
        {"role": "user", "content": old_history},
        _compaction_item(summary),
        {"role": "user", "content": _ask(marker)},
    ]
    equivalent: Final[list[JsonValue]] = [
        {"role": "user", "content": summary},
        {"role": "user", "content": _ask(marker)},
    ]
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire, "claude")
        assert _input_tokens(gateway, model, compacted) == _input_tokens(gateway, model, equivalent)
        counted: Final = tuple(
            request.body for request in wire.drain() if urlsplit(request.target).path.endswith("/count_tokens")
        )
    assert len(counted) == 2, counted
    assert all(b"history-" not in body for body in counted), counted[0][:2000]


def test_streamed_compaction_item_is_done_before_the_reasoning_after_it(gateway: Gateway) -> None:
    marker: Final = _new_marker()
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire, "claude")
        events: Final = _stream(
            gateway,
            "/v1/responses",
            {
                "model": model,
                "input": f"{_ask(marker, compact=True)} {_think(marker)}",
                "context_management": _COMPACTING,
            },
        )
        done: Final = tuple(
            (position, event["output_index"], object_value(event["item"])["type"])
            for position, event in enumerate(events)
            if event["type"] == "response.output_item.done"
        )
        completed: Final = _only([event for event in events if event["type"] == "response.completed"])
        assert [(index, kind) for _, index, kind in done] == [(0, "compaction"), (1, "reasoning"), (2, "message")], [
            event["type"] for event in events
        ]
        assert [item["type"] for item in _objects(object_value(completed["response"])["output"])] == [
            "compaction",
            "reasoning",
            "message",
        ]
        reasoning_done: Final = done[1][0]
        sequence_numbers: Final = [
            number for event in events[: reasoning_done + 1] if isinstance(number := event.get("sequence_number"), int)
        ]
        assert sequence_numbers == sorted(set(sequence_numbers)), sequence_numbers
        _only_call(wire, marker)


def test_chat_stream_reports_the_whole_compaction_block(gateway: Gateway) -> None:
    marker: Final = _new_marker()
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire, "claude")
        chunks: Final = _stream(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": _ask(marker, compact=True)}],
                "context_management": {"edits": [{"type": "compact_20260112"}]},
            },
        )
        reported: Final = _streamed_compaction_blocks(chunks)
        assert reported, chunks
        assert _objects(reported[-1])[-1] == _block(_summary(marker)), reported[-1]
        assert len(_provider_calls(wire, marker)) == 1


def _guardrail_peer(request: Request) -> Reply:
    payload: Final = _JSON_OBJECT.validate_json(request.body)
    texts: Final = payload.get("texts") or []
    assert isinstance(texts, list), payload
    if urlsplit(request.target).path.startswith("/block/"):
        found: Final = any(isinstance(text, str) and _CITY in text for text in texts)
        verdict: Final = {"action": "BLOCKED", "blocked_reason": "city in summary"} if found else {"action": "NONE"}
        return Reply(body=json.dumps(verdict).encode())
    masked: Final = [text.replace(_CITY, _MASKED_CITY) if isinstance(text, str) else text for text in texts]
    return Reply(body=json.dumps({"action": "GUARDRAIL_INTERVENED", "texts": masked}).encode())


def _guardrail(name: str, mode: str, api_base: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": mode,
            "default_on": False,
            "api_base": api_base,
            "api_key": "synthetic-guardrail-key",
            **fields,
        },
    }


def _deployment(name: str, model: str, api_base: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": model, "api_base": api_base, **fields}}


_STORED_CLAUDE: Final = "compaction-stored-claude"
_SMALL_CLAUDE: Final = "compaction-small-claude"
_STORED_GEMINI: Final = "compaction-stored-gemini"
_STORED_OPENAI: Final = "compaction-stored-openai"
_SMALL_CONTEXT: Final = 2000


def _stored_config(directory: Path, provider_url: str, guardrail_url: str, **litellm_settings: JsonValue) -> Path:
    stock: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    claude: Final = {"api_key": cc.ANTHROPIC_API_KEY}
    config: Final[dict[str, JsonValue]] = {
        **stock,
        "model_list": [
            _deployment(_STORED_CLAUDE, f"anthropic/{_CLAUDE}", provider_url, **claude),
            {
                **_deployment(_SMALL_CLAUDE, f"anthropic/{_CLAUDE}", provider_url, **claude),
                "model_info": {"max_input_tokens": _SMALL_CONTEXT},
            },
            _deployment(_STORED_GEMINI, f"gemini/{_GEMINI}", provider_url, api_key="synthetic-gemini-key"),
            _deployment(_STORED_OPENAI, f"openai/{_OPENAI}", f"{provider_url}/v1", api_key="synthetic-openai-key"),
        ],
        "general_settings": {
            **object_value(stock["general_settings"]),
            "store_prompts_in_spend_logs": True,
            "disable_responses_id_security": True,
        },
        "litellm_settings": {**object_value(stock["litellm_settings"]), **litellm_settings},
        "router_settings": {**object_value(stock["router_settings"]), "num_retries": 0, "enable_pre_call_checks": True},
        "guardrails": [
            _guardrail(_MASK_INPUT, "pre_call", f"{guardrail_url}/input"),
            _guardrail(_MASK_OUTPUT, "post_call", f"{guardrail_url}/output"),
            _guardrail(_BLOCK_OUTPUT, "post_call", f"{guardrail_url}/block", streaming_buffer_until_moderated=True),
        ],
    }
    path: Final = directory / "compaction-stored.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class _StoredRig:
    gateway: Gateway
    provider: Wire
    guardrail: Wire


def _stored_rig(directory: Path, **litellm_settings: JsonValue) -> Iterator[_StoredRig]:
    with (
        wire_server(_vendor) as provider,
        wire_server(_guardrail_peer) as guardrail,
        gateway_from_environment() as gateway,
    ):
        config: Final = _stored_config(directory, provider.url, guardrail.url, **litellm_settings)
        with owned_proxy(gateway, directory, {}, config=config, workers=2) as owned:
            yield _StoredRig(owned, provider, guardrail)


@pytest.fixture(scope="module")
def stored(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_StoredRig]:
    yield from _stored_rig(tmp_path_factory.mktemp("compaction-stored"))


@pytest.fixture(scope="module")
def redacting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_StoredRig]:
    yield from _stored_rig(tmp_path_factory.mktemp("compaction-redacting"), turn_off_message_logging=True)


def _logged_row(request_id: str, status: Literal["success", "failure"]) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, proxy_server_request, response, metadata, status FROM "LiteLLM_SpendLogs"'
            " WHERE request_id=%s",
            (request_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert rows[0]["status"] == status, rows[0]
    return rows[0]


def _spend_row(response_id: str) -> dict[str, JsonValue]:
    decoded: Final = ResponsesAPIRequestUtils.decode_responses_api_response_id(response_id).get("response_id")
    return _logged_row(decoded if isinstance(decoded, str) else response_id, "success")


_REPLAY_ROUTES: Final = (
    pytest.param(_STORED_CLAUDE, True, id="claude-compacting"),
    pytest.param(_STORED_CLAUDE, False, id="claude-plain"),
    pytest.param(_STORED_GEMINI, True, id="gemini"),
)


@pytest.mark.parametrize(("follow_up_model", "compacting"), _REPLAY_ROUTES)
def test_previous_response_id_after_a_compaction_turn_replays_the_whole_summary(
    stored: _StoredRig, follow_up_model: str, compacting: bool
) -> None:
    first: Final = _new_marker()
    second: Final = _new_marker()
    summary: Final = _summary(first)
    long_text: Final = f"long-{first} {'z' * 3000} tail-{first}"
    compaction_request: Final[dict[str, JsonValue]] = {"context_management": _COMPACTING}
    opened: Final = _responses(
        stored.gateway,
        {
            "model": _STORED_CLAUDE,
            "input": [{"role": "user", "content": long_text}, {"role": "user", "content": _ask(first, compact=True)}],
            **compaction_request,
        },
    )
    assert _decoded(string_value(_only(_output_items(opened, "compaction"))["encrypted_content"])) == _block(summary)
    row: Final = _spend_row(string_value(opened["id"]))
    assert summary in json.dumps(row["response"]), json.dumps(row["response"])[:600]
    stored_request: Final = json.dumps(row["proxy_server_request"])
    assert LITELLM_TRUNCATED_PAYLOAD_FIELD in stored_request and "z" * 3000 not in stored_request
    _only_call(stored.provider, first)

    follow_up: Final = _responses(
        stored.gateway,
        {
            "model": follow_up_model,
            "input": _ask(second),
            "previous_response_id": opened["id"],
            **(compaction_request if compacting else {}),
            **({"drop_params": True} if follow_up_model == _STORED_GEMINI else {}),
        },
    )
    assert _answer(second) in json.dumps(follow_up)
    call, body = _only_call(stored.provider, second)
    dumped: Final = json.dumps(body)
    assert f"long-{first}" not in dumped and f"ask-{first}" not in dumped, dumped[:2000]
    assert _answer(first) in dumped
    if follow_up_model == _STORED_GEMINI:
        assert summary in _gemini_texts(body, "user"), body["contents"]
    elif compacting:
        replayed: Final = _objects(body["messages"])[0]
        assert replayed["role"] == "assistant", body["messages"]
        assert _objects(replayed["content"])[0] == _block(summary), replayed
        assert _COMPACT_BETA in call.headers.get("anthropic-beta", ""), dict(call.headers)
    else:
        assert "compaction" not in _anthropic_block_types(body), body["messages"]
        assert summary in _anthropic_texts(body, "user"), body["messages"]


def test_router_pre_call_check_counts_from_the_newest_compaction(stored: _StoredRig) -> None:
    compacted_marker: Final = _new_marker()
    control_marker: Final = _new_marker()
    old_history: Final = " ".join(f"history-{index}" for index in range(3 * _SMALL_CONTEXT))

    def send(marker: str, items: list[JsonValue]) -> httpx.Response:
        return stored.gateway.request(
            "POST",
            "/v1/responses",
            {"model": _SMALL_CLAUDE, "input": [*items, {"role": "user", "content": _ask(marker)}], **_CACHE_BUST},
        )

    compacted: Final = send(
        compacted_marker,
        [{"role": "user", "content": old_history}, _compaction_item(f"short summary {compacted_marker}")],
    )
    assert compacted.status_code == 200, compacted.text
    _only_call(stored.provider, compacted_marker)
    control: Final = send(control_marker, [{"role": "user", "content": old_history}])
    assert control.status_code == 400, control.text
    assert _provider_calls(stored.provider, control_marker) == ()


_GUARDED_ROUTES: Final = (
    pytest.param(_STORED_OPENAI, id="openai"),
    pytest.param(_STORED_CLAUDE, id="claude-compacting"),
)


@pytest.mark.parametrize("model", _GUARDED_ROUTES)
def test_pre_call_guardrail_rewrites_the_compaction_summary(stored: _StoredRig, model: str) -> None:
    marker: Final = _new_marker()
    summary: Final = _summary(marker)
    masked: Final = summary.replace(_CITY, _MASKED_CITY)
    compacting: Final[dict[str, JsonValue]] = {"context_management": _COMPACTING} if model == _STORED_CLAUDE else {}
    _responses(
        stored.gateway,
        {
            "model": model,
            "input": [_compaction_item(summary), {"role": "user", "content": _ask(marker)}],
            "guardrails": [_MASK_INPUT],
            **compacting,
        },
    )
    scanned: Final = tuple(
        _JSON_OBJECT.validate_json(request.body)
        for request in stored.guardrail.drain()
        if f"kept-{marker}".encode() in request.body
    )
    assert any(summary in _listed(payload["texts"]) for payload in scanned), scanned
    call, body = _only_call(stored.provider, marker)
    assert _CITY not in json.dumps(body), json.dumps(body)[:2000]
    if model == _STORED_OPENAI:
        assert masked in _openai_user_texts(body), body["input"]
    else:
        assert _objects(_objects(body["messages"])[0]["content"])[0] == _block(masked), body["messages"]


def test_post_call_guardrail_rewrites_the_returned_compaction_summary(stored: _StoredRig) -> None:
    marker: Final = _new_marker()
    returned: Final = _responses(
        stored.gateway,
        {
            "model": _STORED_CLAUDE,
            "input": _ask(marker, compact=True),
            "context_management": _COMPACTING,
            "guardrails": [_MASK_OUTPUT],
        },
    )
    masked: Final = _block(_summary(marker).replace(_CITY, _MASKED_CITY))
    assert _decoded(string_value(_only(_output_items(returned, "compaction"))["encrypted_content"])) == masked
    _only_call(stored.provider, marker)


def test_buffered_stream_guardrail_blocks_on_the_compaction_summary(stored: _StoredRig) -> None:
    marker: Final = _new_marker()
    body: Final[dict[str, JsonValue]] = {
        "model": _STORED_CLAUDE,
        "input": _ask(marker, compact=True),
        "context_management": _COMPACTING,
        "guardrails": [_BLOCK_OUTPUT],
        "stream": True,
        **_CACHE_BUST,
    }
    headers: Final = {"Authorization": f"Bearer {stored.gateway.key}"}
    with stored.gateway.client.stream("POST", "/v1/responses", json=body, headers=headers) as response:
        text: Final = response.read().decode()
    assert response.status_code == 200, text
    peer_calls: Final = tuple(
        (urlsplit(request.target).path, _JSON_OBJECT.validate_json(request.body).get("texts"))
        for request in stored.guardrail.drain()
        if marker.encode() in request.body
    )
    assert any(path.startswith("/block/") and _summary(marker) in _listed(texts) for path, texts in peer_calls), [
        (path, [str(value)[:80] for value in _listed(texts or [])]) for path, texts in peer_calls
    ]
    event_types: Final = tuple(event["type"] for event in _sse_events(text))
    assert "response.failed" in event_types, event_types
    assert "response.completed" not in event_types, event_types
    assert f"kept-{marker}" not in text and _CITY not in text, text[:2000]
    _only_call(stored.provider, marker)


def test_redacted_spend_row_holds_no_compaction_summary(stored: _StoredRig) -> None:
    marker: Final = _new_marker()
    returned: Final = _responses(
        stored.gateway,
        {"model": _STORED_CLAUDE, "input": _ask(marker, compact=True), "context_management": _COMPACTING},
        headers=_REDACT_HEADER,
    )
    assert _decoded(string_value(_only(_output_items(returned, "compaction"))["encrypted_content"])) == _block(
        _summary(marker)
    )
    row: Final = _spend_row(string_value(returned["id"]))
    assert f"kept-{marker}" not in json.dumps(row["response"]), json.dumps(row["response"])[:2000]
    _only_call(stored.provider, marker)


def test_stored_compaction_input_item_keeps_its_whole_summary_and_nothing_else(stored: _StoredRig) -> None:
    first: Final = _new_marker()
    second: Final = _new_marker()
    summary: Final = _summary(first)
    padded_item: Final = {
        **_compaction_item(summary),
        "encrypted_content": json.dumps({**_block(summary), "padding": f"pad-{first} {'p' * 3000}"}),
    }
    opened: Final = _responses(
        stored.gateway,
        {
            "model": _STORED_CLAUDE,
            "input": [padded_item, {"role": "user", "content": _ask(first)}],
            "context_management": _COMPACTING,
        },
    )
    stored_input: Final = _objects(
        object_value(_spend_row(string_value(opened["id"]))["proxy_server_request"])["input"]
    )
    assert _decoded(string_value(stored_input[0]["encrypted_content"])) == _block(summary), stored_input[0]
    _only_call(stored.provider, first)

    _responses(
        stored.gateway,
        {
            "model": _STORED_CLAUDE,
            "input": _ask(second),
            "previous_response_id": opened["id"],
            "context_management": _COMPACTING,
        },
    )
    _, body = _only_call(stored.provider, second)
    assert summary in _anthropic_texts(body, "user"), body["messages"]
    assert f"pad-{first}" not in json.dumps(body), body["messages"]


def test_redacted_chat_stream_spend_row_holds_no_compaction_delta(stored: _StoredRig) -> None:
    marker: Final = _new_marker()
    summary: Final = _summary(marker)
    chunks: Final = _stream(
        stored.gateway,
        "/v1/chat/completions",
        {
            "model": _STORED_CLAUDE,
            "messages": [{"role": "user", "content": _ask(marker, compact=True)}],
            "context_management": {"edits": [{"type": "compact_20260112"}]},
        },
        headers=_REDACT_HEADER,
    )
    streamed_blocks: Final = _streamed_compaction_blocks(chunks)
    assert streamed_blocks and _objects(streamed_blocks[-1])[-1] == _block(summary), chunks
    secrets: Final = (f"kept-{marker}", summary[len(summary) // 2 :])
    logged: Final = {column: json.dumps(value) for column, value in _spend_row(string_value(chunks[0]["id"])).items()}
    leaked: Final = {column: text for column, text in logged.items() if any(secret in text for secret in secrets)}
    assert not leaked, {column: text[max(0, text.find(secrets[0]) - 600) :][:1500] for column, text in leaked.items()}
    _only_call(stored.provider, marker)


def _failing_request(path: str, marker: str) -> dict[str, JsonValue]:
    ask: Final = f"{_ask(marker)} {_fail(marker)}"
    if path == "/v1/responses":
        return {
            "model": _STORED_CLAUDE,
            "input": [_compaction_item(_summary(marker)), {"role": "user", "content": ask}],
        }
    return {"model": _STORED_CLAUDE, "max_tokens": 64, "messages": [{"role": "user", "content": ask}]}


def _failed_call_row(
    rig: _StoredRig,
    path: str,
    marker: str,
    headers: Mapping[str, str],
    key: str | None = None,
    **fields: JsonValue,
) -> dict[str, JsonValue]:
    body: Final = {**_failing_request(path, marker), **fields}
    response: Final = rig.gateway.request("POST", path, body, headers=headers, key=key)
    assert response.status_code == 400, response.text
    _only_call(rig.provider, marker)
    return _logged_row(response.headers["x-litellm-call-id"], "failure")


_LEGACY_REDACT_HEADER: Final = {"litellm-enable-message-redaction": "true"}
_REDACTED_FAILURES: Final = (
    pytest.param("/v1/responses", _REDACT_HEADER, id="responses"),
    pytest.param("/v1/responses", _LEGACY_REDACT_HEADER, id="responses-legacy-header"),
    pytest.param("/v1/messages", _REDACT_HEADER, id="messages"),
    pytest.param("/v1/chat/completions", _REDACT_HEADER, id="chat"),
)


@pytest.mark.parametrize(("path", "headers"), _REDACTED_FAILURES)
def test_failed_call_spend_row_holds_no_request_text_with_the_redaction_header(
    stored: _StoredRig, path: str, headers: Mapping[str, str]
) -> None:
    marker: Final = _new_marker()
    logged: Final = json.dumps(_failed_call_row(stored, path, marker, headers))
    assert marker not in logged, logged[max(0, logged.find(marker) - 600) :][:1500]


def test_failed_responses_call_spend_row_keeps_the_whole_compaction_summary_without_the_header(
    stored: _StoredRig,
) -> None:
    marker: Final = _new_marker()
    row: Final = _failed_call_row(stored, "/v1/responses", marker, {})
    stored_input: Final = _objects(object_value(row["proxy_server_request"])["input"])
    assert _decoded(string_value(stored_input[0]["encrypted_content"])) == _block(_summary(marker)), stored_input[0]
    assert f"{_ask(marker)} {_fail(marker)}" in _message_texts(stored_input[1]), stored_input[1]


def test_failed_responses_call_stores_the_caller_metadata_without_the_request_headers(stored: _StoredRig) -> None:
    marker: Final = _new_marker()
    row: Final = _failed_call_row(stored, "/v1/responses", marker, _REDACT_HEADER, metadata={"caller": marker})
    stored_metadata: Final = object_value(object_value(row["proxy_server_request"])["metadata"])
    assert stored_metadata.get("caller") == marker, stored_metadata
    assert "headers" not in stored_metadata, stored_metadata


def test_failed_responses_call_keeps_the_caller_metadata_headers_over_the_redaction_header(stored: _StoredRig) -> None:
    marker: Final = _new_marker()
    caller_headers: Final = {"caller": marker}
    row: Final = _failed_call_row(stored, "/v1/responses", marker, _REDACT_HEADER, metadata={"headers": caller_headers})
    stored_request: Final = object_value(row["proxy_server_request"])
    assert object_value(stored_request["metadata"])["headers"] == caller_headers, stored_request["metadata"]
    assert _ask(marker) in json.dumps(stored_request), json.dumps(stored_request)[:1500]


_GLOBALLY_REDACTED_FAILURES: Final = (
    pytest.param("/v1/responses", _UNREDACT_HEADER, False, id="responses-disable-header"),
    pytest.param("/v1/chat/completions", _UNREDACT_HEADER, True, id="chat-disable-header"),
    pytest.param("/v1/responses", {}, False, id="responses-no-header"),
)


@pytest.mark.parametrize(("path", "headers", "kept"), _GLOBALLY_REDACTED_FAILURES)
def test_opted_out_key_failed_call_spend_row_takes_the_disable_header_over_global_redaction_only_on_chat(
    redacting: _StoredRig, path: str, headers: Mapping[str, str], kept: bool
) -> None:
    marker: Final = _new_marker()
    with redacting.gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"allow_client_message_redaction_opt_out": True})
        row: Final = _failed_call_row(redacting, path, marker, headers, key)
    stored_request: Final = json.dumps(row["proxy_server_request"])
    assert (_ask(marker) in stored_request) is kept, stored_request[:1500]

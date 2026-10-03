import asyncio
import json
import re
import signal
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import unquote, urlsplit

import httpx
import openai
import psutil
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.anthropic_thinking import (
    BEDROCK_MODEL,
    JSON_LIST,
    JSON_OBJECT,
    MODEL,
    NO_CACHE,
    SIGNATURE,
    THINKING,
    THINKING_PARTS,
    Event,
    accumulate,
    answer,
    chunks_of,
    content_text,
    deltas_of,
    identity,
    marker_of,
    message_body,
    message_events,
    prompt,
    reasoning_text,
    redacted_events,
    signature_only,
    signed_blocks,
    standard_events,
    standard_peer,
    stream_reply,
    streams,
    text_events,
    thinking_block,
    thinking_events,
)
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.chat import ChatCompletionChunk
from pydantic import JsonValue

_SECOND_SIGNATURE: Final = "scripted-signature-" + "t" * 32
_LONG_SIGNATURE: Final = "k" * 5120
_REDACTED: Final = "scripted-redacted-" + "r" * 32
_VERTEX_PROJECT: Final = "scripted-project"
_VERTEX_LOCATION: Final = "us-east5"
_VERTEX_MODEL_PATH: Final = (
    f"/v1/projects/{_VERTEX_PROJECT}/locations/{_VERTEX_LOCATION}/publishers/anthropic/models/{MODEL}"
)
_CONFIG_MODEL: Final = "anthropic-signature-chaos"
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")

Provider = Literal["anthropic", "bedrock_invoke", "claude_platform", "vertex_ai", "snowflake", "azure_ai"]
Endpoint = Literal["chat", "messages", "responses"]

_TARGETS: Final = MappingProxyType(
    {
        "anthropic": "/v1/messages",
        "bedrock_invoke": f"/model/{BEDROCK_MODEL}/invoke-with-response-stream",
        "claude_platform": "/v1/messages",
        "vertex_ai": f"{_VERTEX_MODEL_PATH}:streamRawPredict",
        "snowflake": "/api/v2/cortex/v1/messages",
        "azure_ai": "/anthropic/v1/messages",
    }
)


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
            "project_id": _VERTEX_PROJECT,
            "private_key_id": "scripted",
            "private_key": private_key,
            "client_email": f"scripted@{_VERTEX_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": f"{token_url}/_oauth/token",
        }
    )


def _deployment(scenario: Scenario, provider: Provider, wire_url: str, upstream_url: str) -> str:
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
        case "vertex_ai":
            return scenario.model(
                model=f"vertex_ai/{MODEL}",
                api_base=f"{wire_url}{_VERTEX_MODEL_PATH}",
                api_key=None,
                vertex_project=_VERTEX_PROJECT,
                vertex_location=_VERTEX_LOCATION,
                vertex_credentials=_service_account_json(upstream_url.rstrip("/")),
            )
        case "snowflake":
            return scenario.model(model=f"snowflake/{MODEL}", api_base=wire_url, api_key="scripted-snowflake-key")
        case "azure_ai":
            return scenario.model(model=f"azure_ai/{MODEL}", api_base=wire_url, api_key="scripted-azure-key")


def _chat_body(
    model: str,
    marker: str,
    *,
    cache_control: Mapping[str, JsonValue] = NO_CACHE,
    messages: Sequence[Mapping[str, JsonValue]] | None = None,
) -> dict[str, JsonValue]:
    turn: Final = list(messages) if messages else [{"role": "user", "content": prompt(marker)}]
    return {"model": model, "messages": turn, "stream": True, "max_tokens": 64, **cache_control}


def _stream_chat(gateway: Gateway, body: Mapping[str, JsonValue], *, key: str | None = None) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", body, key=key)


def _sdk_delta(chunk: ChatCompletionChunk) -> Event:
    if not chunk.choices:
        return {}
    return JSON_OBJECT.validate_python(chunk.choices[0].delta.model_dump(exclude_none=True))


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _spend_row(request_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status, model_group FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _assert_signed_once(deltas: Sequence[Event], marker: str, *, signature: JsonValue = SIGNATURE) -> None:
    assert signed_blocks(deltas) == (signature_only(signature),), deltas
    assert accumulate(deltas) == (thinking_block(THINKING, signature),), deltas
    assert reasoning_text(deltas) == THINKING, deltas
    assert content_text(deltas) == answer(marker), deltas


def _replay_messages(marker: str, follow_up: str, deltas: Sequence[Event]) -> tuple[dict[str, JsonValue], ...]:
    assistant: Event = {
        "role": "assistant",
        "content": content_text(deltas),
        "thinking_blocks": list(accumulate(deltas)),
    }
    return ({"role": "user", "content": prompt(marker)}, assistant, {"role": "user", "content": prompt(follow_up)})


def _assistant_turn(request: Request) -> tuple[Event, ...]:
    messages: Final = JSON_LIST.validate_python(JSON_OBJECT.validate_json(request.body)["messages"])
    assistant: Final = JSON_OBJECT.validate_python(messages[1])
    assert assistant["role"] == "assistant", request.body
    return tuple(JSON_OBJECT.validate_python(part) for part in JSON_LIST.validate_python(assistant["content"]))


@pytest.mark.parametrize(
    "provider",
    ["anthropic", "bedrock_invoke", "claude_platform", "vertex_ai", "snowflake", "azure_ai"],
)
def test_signature_chunk_carries_no_thinking_text_on_every_anthropic_wire_provider(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, provider, wire.url, gateway.upstream_url)
        response: Final = _stream_chat(gateway, _chat_body(model, marker))
        assert response.status_code == 200, response.text
        assert response.text.rstrip().endswith("data: [DONE]"), response.text
        chunks: Final = chunks_of(response.text)
        _assert_signed_once(deltas_of(chunks), marker)
        assert [urlsplit(unquote(request.target)).path for request in wire.drain()] == [_TARGETS[provider]], (
            response.text
        )
        row: Final = _spend_row(str(chunks[0]["id"]))
        assert (row["model_group"], row["status"]) == (model, "success"), row


def test_openai_sdk_sync_stream_accumulates_the_thinking_once(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        chunks: Final = tuple(
            _openai_client(gateway).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt(marker)}],
                stream=True,
                max_tokens=64,
                extra_body=NO_CACHE,
            )
        )
        _assert_signed_once(tuple(_sdk_delta(chunk) for chunk in chunks), marker)
        assert len(wire.drain()) == 1
        assert _spend_row(chunks[0].id)["model_group"] == model


async def test_openai_sdk_async_stream_accumulates_the_thinking_once(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        client: Final = openai.AsyncOpenAI(
            base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0
        )
        stream: Final = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt(marker)}],
            stream=True,
            max_tokens=64,
            extra_body=NO_CACHE,
        )
        chunks: Final = tuple([chunk async for chunk in stream])
        _assert_signed_once(tuple(_sdk_delta(chunk) for chunk in chunks), marker)
        assert len(wire.drain()) == 1
        assert (await asyncio.to_thread(_spend_row, chunks[0].id))["model_group"] == model


def test_non_streaming_completion_keeps_the_signed_thinking_block_intact(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        completion: Final = _openai_client(gateway).chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt(marker)}], max_tokens=64, extra_body=NO_CACHE
        )
        message: Final = JSON_OBJECT.validate_python(completion.choices[0].message.model_dump(exclude_none=True))
        assert message["thinking_blocks"] == [thinking_block(THINKING, SIGNATURE)], message
        assert message["reasoning_content"] == THINKING, message
        assert message["content"] == answer(marker), message
        received: Final = wire.drain()
        assert len(received) == 1 and not streams(received[0]), received
        assert _spend_row(completion.id)["model_group"] == model


def _reasoning_item(output: Sequence[Event]) -> Event:
    reasoning: Final = tuple(item for item in output if item["type"] == "reasoning")
    assert len(reasoning) == 1, output
    return reasoning[0]


def _reasoning_text(item: Mapping[str, JsonValue]) -> str:
    parts: Final = tuple(JSON_OBJECT.validate_python(part) for part in JSON_LIST.validate_python(item["content"]))
    return "".join(str(part["text"]) for part in parts)


def test_responses_stream_encrypts_the_thinking_once(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        events: Final = tuple(
            _openai_client(gateway).responses.create(
                model=model,
                input=prompt(marker),
                stream=True,
                include=["reasoning.encrypted_content"],
                max_output_tokens=64,
                extra_body=NO_CACHE,
            )
        )
        completed: Final = tuple(event for event in events if event.type == "response.completed")
        assert len(completed) == 1, [event.type for event in events]
        output: Final = tuple(JSON_OBJECT.validate_python(item.model_dump()) for item in completed[0].response.output)
        item: Final = _reasoning_item(output)
        assert json.loads(str(item["encrypted_content"])) == [thinking_block(THINKING, SIGNATURE)], item
        assert _reasoning_text(item) == THINKING, item
        received: Final = wire.drain()
        assert len(received) == 1 and streams(received[0]), received


def test_responses_non_stream_encrypts_the_signed_block_as_received(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = _openai_client(gateway).responses.create(
            model=model,
            input=prompt(marker),
            include=["reasoning.encrypted_content"],
            max_output_tokens=64,
            extra_body=NO_CACHE,
        )
        output: Final = tuple(JSON_OBJECT.validate_python(item.model_dump()) for item in response.output)
        item: Final = _reasoning_item(output)
        assert json.loads(str(item["encrypted_content"])) == [thinking_block(THINKING, SIGNATURE)], item
        assert _reasoning_text(item) == THINKING, item
        received: Final = wire.drain()
        assert len(received) == 1 and not streams(received[0]), received


def test_cache_hit_replays_the_answer_from_one_upstream_call_and_never_doubles_the_thinking(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        body: Final = _chat_body(model, marker, cache_control={})
        first: Final = _stream_chat(gateway, body)
        assert first.status_code == 200, first.text
        first_chunks: Final = chunks_of(first.text)
        _assert_signed_once(deltas_of(first_chunks), marker)
        assert _spend_row(str(first_chunks[0]["id"]))["model_group"] == model
        second: Final = _stream_chat(gateway, body)
        assert second.status_code == 200, second.text
        second_deltas: Final = deltas_of(chunks_of(second.text))
        assert content_text(second_deltas) == answer(marker), second.text
        assert accumulate(second_deltas) in ((), (thinking_block(THINKING, SIGNATURE),)), second.text
        assert len(wire.drain()) == 1, second.text


def test_replaying_the_accumulated_turn_sends_the_thinking_once_with_its_signature(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    follow_up: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        first: Final = _stream_chat(gateway, _chat_body(model, marker))
        assert first.status_code == 200, first.text
        deltas: Final = deltas_of(chunks_of(first.text))
        second: Final = _stream_chat(
            gateway, _chat_body(model, follow_up, messages=_replay_messages(marker, follow_up, deltas))
        )
        assert second.status_code == 200, second.text
        assert content_text(deltas_of(chunks_of(second.text))) == answer(follow_up), second.text
        received: Final = wire.drain()
        assert len(received) == 2, [request.body for request in received]
        assert _assistant_turn(received[1]) == (
            thinking_block(THINKING, SIGNATURE),
            {"type": "text", "text": answer(marker)},
        ), received[1].body


def test_two_signed_blocks_each_keep_their_own_text_through_a_replay(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    follow_up: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        found: Final = marker_of(request)
        if not streams(request):
            return Reply(body=message_body(found))
        events: Final = message_events(
            found,
            (
                thinking_events(0, ("one ", "two"), (SIGNATURE,)),
                thinking_events(1, ("three ", "four"), (_SECOND_SIGNATURE,)),
                text_events(2, answer(found)),
            ),
        )
        return stream_reply(request, events)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        first: Final = _stream_chat(gateway, _chat_body(model, marker))
        assert first.status_code == 200, first.text
        deltas: Final = deltas_of(chunks_of(first.text))
        assert signed_blocks(deltas) == (signature_only(SIGNATURE), signature_only(_SECOND_SIGNATURE)), deltas
        assert accumulate(deltas) == (
            thinking_block("one two", SIGNATURE),
            thinking_block("three four", _SECOND_SIGNATURE),
        ), deltas
        assert reasoning_text(deltas) == "one twothree four", deltas
        second: Final = _stream_chat(
            gateway, _chat_body(model, follow_up, messages=_replay_messages(marker, follow_up, deltas))
        )
        assert second.status_code == 200, second.text
        received: Final = wire.drain()
        assert len(received) == 2, [request.body for request in received]
        assert _assistant_turn(received[1]) == (
            thinking_block("one two", SIGNATURE),
            thinking_block("three four", _SECOND_SIGNATURE),
            {"type": "text", "text": answer(marker)},
        ), received[1].body


def test_redacted_block_before_a_signed_block_replays_each_once(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    follow_up: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        found: Final = marker_of(request)
        if not streams(request):
            return Reply(body=message_body(found))
        events: Final = message_events(
            found,
            (
                redacted_events(0, _REDACTED),
                thinking_events(1, THINKING_PARTS, (SIGNATURE,)),
                text_events(2, answer(found)),
            ),
        )
        return stream_reply(request, events)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        first: Final = _stream_chat(gateway, _chat_body(model, marker))
        assert first.status_code == 200, first.text
        deltas: Final = deltas_of(chunks_of(first.text))
        assert accumulate(deltas) == (
            {"type": "redacted_thinking", "data": _REDACTED},
            thinking_block(THINKING, SIGNATURE),
        ), deltas
        second: Final = _stream_chat(
            gateway, _chat_body(model, follow_up, messages=_replay_messages(marker, follow_up, deltas))
        )
        assert second.status_code == 200, second.text
        received: Final = wire.drain()
        assert len(received) == 2, [request.body for request in received]
        assert _assistant_turn(received[1]) == (
            {"type": "redacted_thinking", "data": _REDACTED},
            thinking_block(THINKING, SIGNATURE),
            {"type": "text", "text": answer(marker)},
        ), received[1].body


def test_signature_only_block_without_thinking_deltas_is_relayed_as_is(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        return stream_reply(request, standard_events(marker_of(request), parts=()))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = _stream_chat(gateway, _chat_body(model, marker))
        assert response.status_code == 200, response.text
        deltas: Final = deltas_of(chunks_of(response.text))
        assert signed_blocks(deltas) == (signature_only(SIGNATURE),), deltas
        assert accumulate(deltas) == (signature_only(SIGNATURE),), deltas
        assert reasoning_text(deltas) == "", deltas
        assert content_text(deltas) == answer(marker), deltas
        assert len(wire.drain()) == 1


def test_two_identical_requests_with_no_cache_each_land_their_own_spend_row(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        responses: Final = tuple(_stream_chat(gateway, _chat_body(model, marker)) for _ in range(2))
        ids: Final = tuple(str(chunks_of(response.text)[0]["id"]) for response in responses)
        for response in responses:
            assert response.status_code == 200, response.text
            _assert_signed_once(deltas_of(chunks_of(response.text)), marker)
        assert len(set(ids)) == 2, ids
        assert len(wire.drain()) == 2
        for request_id in ids:
            assert _spend_row(request_id)["model_group"] == model


@pytest.mark.parametrize("signature", [123, [], ""], ids=["integer", "list", "empty"])
def test_unusable_signature_values_yield_no_signed_block_and_keep_the_stream_intact(
    gateway: Gateway, signature: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        return stream_reply(request, standard_events(marker_of(request), signatures=(signature,)))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = _stream_chat(gateway, _chat_body(model, marker))
        assert response.status_code == 200, response.text
        assert response.text.rstrip().endswith("data: [DONE]"), response.text
        deltas: Final = deltas_of(chunks_of(response.text))
        assert signed_blocks(deltas) == (), deltas
        assert reasoning_text(deltas) == THINKING, deltas
        assert content_text(deltas) == answer(marker), deltas
        assert len(wire.drain()) == 1
        assert gateway.client.get("/health/liveliness").status_code == 200


def test_five_kilobyte_signature_is_relayed_verbatim_without_thinking_text(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        return stream_reply(request, standard_events(marker_of(request), signatures=(_LONG_SIGNATURE,)))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = _stream_chat(gateway, _chat_body(model, marker))
        assert response.status_code == 200, response.text
        _assert_signed_once(deltas_of(chunks_of(response.text)), marker, signature=_LONG_SIGNATURE)
        assert len(wire.drain()) == 1


def test_duplicate_signature_deltas_never_repeat_the_thinking_text(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        return stream_reply(request, standard_events(marker_of(request), signatures=(SIGNATURE, SIGNATURE)))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = _stream_chat(gateway, _chat_body(model, marker))
        assert response.status_code == 200, response.text
        deltas: Final = deltas_of(chunks_of(response.text))
        assert signed_blocks(deltas) == (signature_only(SIGNATURE), signature_only(SIGNATURE)), deltas
        assert "".join(str(block["thinking"]) for block in accumulate(deltas)) == THINKING, deltas
        assert reasoning_text(deltas) == THINKING, deltas
        assert len(wire.drain()) == 1


def test_non_string_thinking_delta_is_ignored_and_the_signed_block_still_lands_once(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        return stream_reply(request, standard_events(marker_of(request), parts=("alpha ", 7, "beta")))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = _stream_chat(gateway, _chat_body(model, marker))
        assert response.status_code == 200, response.text
        assert response.text.rstrip().endswith("data: [DONE]"), response.text
        _assert_signed_once(deltas_of(chunks_of(response.text)), marker)
        assert len(wire.drain()) == 1


def test_upstream_authentication_error_reaches_the_caller_and_leaves_the_proxy_healthy(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        body: Final = {"type": "error", "error": {"type": "authentication_error", "message": "scripted invalid key"}}
        return Reply(status=401, body=json.dumps(body).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = _stream_chat(gateway, _chat_body(model, marker))
        assert response.status_code == 401, response.text
        assert "scripted invalid key" in response.text, response.text
        assert len(wire.drain()) >= 1
        assert gateway.client.get("/health/liveliness").status_code == 200
        control: Final = uuid.uuid4().hex
        with wire_server(standard_peer) as healthy, gateway.scenario() as again:
            working: Final = _deployment(again, "anthropic", healthy.url, gateway.upstream_url)
            recovered: Final = _stream_chat(gateway, _chat_body(working, control))
            assert recovered.status_code == 200, recovered.text
            _assert_signed_once(deltas_of(chunks_of(recovered.text)), control)


def test_unauthenticated_stream_is_refused_before_the_upstream_is_called(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(standard_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = _stream_chat(gateway, _chat_body(model, marker), key=f"sk-not-a-key-{marker}")
        assert response.status_code == 401, response.text
        assert wire.drain() == ()


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(model: str, call: _Call) -> dict[str, JsonValue]:
    match call.endpoint:
        case "chat":
            return _chat_body(model, call.marker) | {"stream": call.stream}
        case "messages":
            return {
                "model": model,
                "max_tokens": 64,
                "stream": call.stream,
                "messages": [{"role": "user", "content": prompt(call.marker)}],
            }
        case "responses":
            return {
                "model": model,
                "input": prompt(call.marker),
                "stream": call.stream,
                "max_output_tokens": 64,
                **NO_CACHE,
            }


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    try:
        async with client.stream(
            "POST", _path(call.endpoint), json=_body(model, call), headers={"Authorization": f"Bearer {key}"}
        ) as response:
            raw: Final = await response.aread()
            return _Served(call=call, status=response.status_code, text=raw.decode())
    except httpx.TransportError as error:
        return _Served(call=call, status=0, text=repr(error))


async def _burst(base_url: str, key: str, model: str, calls: Sequence[_Call]) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, key, model, call) for call in calls)))


def _calls(count: int, endpoints: Sequence[Endpoint]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=endpoints[index % len(endpoints)], stream=index % 2 == 0, marker=uuid.uuid4().hex)
        for index in range(count)
    )


def _completed_id(item: _Served) -> str | None:
    match item.call.endpoint:
        case "chat":
            first: Final = chunks_of(item.text)[0] if item.call.stream else JSON_OBJECT.validate_json(item.text)
            return str(first["id"])
        case "messages":
            return identity(item.call.marker)
        case "responses":
            return None


def _success_rows(model: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE model_group=%s AND status=%s', (model, "success")
    )


async def test_mid_thinking_upstream_aborts_in_a_mixed_burst_leave_every_completed_call_logged_once(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(24, ("chat", "messages", "responses"))
    aborted: Final = frozenset(call.marker for index, call in enumerate(calls) if index % 4 == 0)

    def respond(request: Request) -> Reply:
        marker: Final = marker_of(request)
        if not streams(request):
            return Reply(body=message_body(marker))
        return stream_reply(request, standard_events(marker), abort_after=3 if marker in aborted else None)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert gateway.client.get("/health/liveliness").status_code == 200
        completed: Final = tuple(item for item in served if item.call.marker not in aborted)
        for item in served:
            if item.call.marker in aborted:
                assert answer(item.call.marker) not in item.text, item.text
            else:
                assert item.status == 200, item.text
                assert answer(item.call.marker) in item.text, item.text
        assert len(completed) == 18, [item.call for item in completed]
        for item in completed:
            if item.call.endpoint == "chat" and item.call.stream:
                _assert_signed_once(deltas_of(chunks_of(item.text)), item.call.marker)
        assert len(wire.drain()) == 24
        rows: Final = await asyncio.to_thread(
            eventually, lambda: _success_rows(model), lambda found: len(found) == len(completed), 70
        )
        logged: Final = tuple(str(row["request_id"]) for row in rows)
        for item in completed:
            request_id: Final = _completed_id(item)
            assert request_id is None or logged.count(request_id) == 1, (request_id, logged)


def _chaos_config(wire: Wire, directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {
                "model": f"anthropic/{MODEL}",
                "api_base": wire.url,
                "api_key": "scripted-anthropic-key",
            },
        }
    ]
    path: Final = directory / "anthropic-signature-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(180)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_streaming_signed_thinking_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, ("chat",))
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        held_markers.put(marker_of(request))
        assert release.wait(timeout=60), "The burst was never released"
        return standard_peer(request)

    with wire_server(held) as wire:
        path: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, calls)
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            completed: Final = tuple(item for item in served if item.status == 200)
            assert len(completed) == held_by[survivor_pid], (held_by, [item.status for item in served])
            for item in completed:
                if item.call.stream:
                    _assert_signed_once(deltas_of(chunks_of(item.text)), item.call.marker)
                else:
                    assert answer(item.call.marker) in item.text, item.text
            follow_up: Final = _Call(endpoint="chat", stream=True, marker=uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, (follow_up,))
            assert answered.status == 200, answered.text
            _assert_signed_once(deltas_of(chunks_of(answered.text)), follow_up.marker)
            assert len(wire.drain()) == 21

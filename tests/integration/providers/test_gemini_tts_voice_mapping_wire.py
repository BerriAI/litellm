"""Gemini and Vertex TTS deployments map OpenAI voice names onto Gemini prebuilt voices.

The scripted peer on 127.0.0.1 answers an audio part when the ``voiceName`` it receives is one of
Gemini's prebuilt voices and the vendor's ``No matching speaker voice found`` 400 otherwise, so a row
is green only when the voice the peer received is the mapped one. Every row runs through the rig proxy.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._gemini_tts_voice_support import (
    AUDIO_TOKENS,
    BACKEND,
    GEMINI_API_KEY,
    NO_VOICE,
    PCM_B64,
    REJECTED_PREFIX,
    VERTEX_MODEL_PATH,
    anthropic_client,
    async_openai_client,
    audio_data,
    audio_param,
    chat_body,
    chunked_peer,
    error_message,
    frame_types,
    gemini_deployment,
    gemini_peer,
    live_deployment,
    live_scenario,
    live_setups,
    live_turn,
    marker,
    openai_client,
    received_text,
    received_voice,
    spend_row_by_call,
    text_in,
    vertex_deployment,
    voice_in,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(180)

BUCKET: Final = "integration-tts-bucket"
FIVE_KB: Final = "v" * 5120
BURST: Final = 24
SLOW_BURST: Final = 12
OUTAGE_AFTER: Final = 6
SLOW_SECONDS: Final = 1.0


@dataclass(frozen=True, slots=True)
class Answer:
    status: int
    text: str
    response_id: str
    audio: str | None
    content: str
    call_id: str


ChatClient = Callable[..., Answer]


def _failed(status: int, text: str) -> Answer:
    return Answer(status, text, "", None, "", "")


def _delta_text(chunk: Mapping[str, JsonValue]) -> str:
    choices: Final = chunk["choices"]
    if not isinstance(choices, list) or not choices:
        return ""
    content: Final = object_value(object_value(choices[0])["delta"]).get("content")
    return content if isinstance(content, str) else ""


def _message_text(body: Mapping[str, JsonValue]) -> str:
    choices: Final = body["choices"]
    assert isinstance(choices, list) and len(choices) == 1, body
    content: Final = object_value(object_value(choices[0])["message"]).get("content")
    return content if isinstance(content, str) else ""


def _sse_chunks(text: str) -> tuple[dict[str, JsonValue], ...]:
    lines: Final = tuple(line[6:] for line in text.splitlines() if line.startswith("data: "))
    return tuple(JSON_OBJECT.validate_json(line) for line in lines if line != "[DONE]")


def _error_text(response: httpx.Response) -> str:
    response.read()
    return response.text


async def _async_error_text(response: httpx.Response) -> str:
    await response.aread()
    return response.text


def _httpx_chat(gateway: Gateway, model: str, voice: JsonValue, text: str, *, stream: bool) -> Answer:
    response: Final = gateway.request("POST", "/v1/chat/completions", chat_body(model, voice, text, stream=stream))
    call_id: Final = response.headers.get("x-litellm-call-id", "")
    if response.status_code != 200:
        return _failed(response.status_code, response.text)
    if stream:
        chunks: Final = _sse_chunks(response.text)
        content: Final = "".join(_delta_text(chunk) for chunk in chunks)
        return Answer(200, response.text, "", None, content, call_id)
    body: Final = JSON_OBJECT.validate_json(response.content)
    return Answer(200, response.text, string_value(body["id"]), audio_data(body), _message_text(body), call_id)


def _sdk_chat(gateway: Gateway, model: str, voice: str, text: str, *, stream: bool) -> Answer:
    client: Final = openai_client(gateway)
    messages: Final = [{"role": "user", "content": text}]
    try:
        if stream:
            raw: Final = client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                modalities=["audio"],
                audio={"voice": voice, "format": "pcm16"},
                stream=True,
            )
            chunks: Final = list(raw.parse())
            content: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
            return Answer(200, "", "", None, content, raw.headers.get("x-litellm-call-id", ""))
        completed: Final = client.chat.completions.with_raw_response.create(
            model=model, messages=messages, modalities=["audio"], audio={"voice": voice, "format": "pcm16"}
        )
        completion: Final = completed.parse()
        message: Final = completion.choices[0].message
        audio: Final = message.audio.data if message.audio is not None else None
        return Answer(
            200,
            completion.model_dump_json(),
            completion.id,
            audio,
            message.content or "",
            completed.headers.get("x-litellm-call-id", ""),
        )
    except openai.APIStatusError as error:
        return _failed(error.status_code, _error_text(error.response))


async def _async_sdk_chat(gateway: Gateway, model: str, voice: str, text: str, *, stream: bool) -> Answer:
    client: Final = async_openai_client(gateway)
    messages: Final = [{"role": "user", "content": text}]
    try:
        if stream:
            raw: Final = await client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                modalities=["audio"],
                audio={"voice": voice, "format": "pcm16"},
                stream=True,
            )
            chunks: Final = [chunk async for chunk in raw.parse()]
            content: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
            return Answer(200, "", "", None, content, raw.headers.get("x-litellm-call-id", ""))
        completed: Final = await client.chat.completions.with_raw_response.create(
            model=model, messages=messages, modalities=["audio"], audio={"voice": voice, "format": "pcm16"}
        )
        completion: Final = completed.parse()
        message: Final = completion.choices[0].message
        audio: Final = message.audio.data if message.audio is not None else None
        return Answer(
            200,
            completion.model_dump_json(),
            completion.id,
            audio,
            message.content or "",
            completed.headers.get("x-litellm-call-id", ""),
        )
    except openai.APIStatusError as error:
        return _failed(error.status_code, await _async_error_text(error.response))
    finally:
        await client.close()


def _sdk_async_chat(gateway: Gateway, model: str, voice: str, text: str, *, stream: bool) -> Answer:
    return asyncio.run(_async_sdk_chat(gateway, model, voice, text, stream=stream))


CLIENTS: Final[Mapping[str, ChatClient]] = {"httpx": _httpx_chat, "sdk": _sdk_chat, "sdk-async": _sdk_async_chat}


def _deployment(scenario: Scenario, provider: str, wire_url: str, upstream_url: str, **extra: JsonValue) -> str:
    if provider == "gemini":
        return gemini_deployment(scenario, wire_url, **extra)
    return vertex_deployment(scenario, wire_url, upstream_url, **extra)


def _received(wire: Wire, text: str) -> Request:
    matching: Final = tuple(request for request in wire.drain() if received_text(request) == text)
    assert len(matching) == 1, [request.target for request in matching]
    return matching[0]


def _assert_target(request: Request, provider: str, *, stream: bool) -> None:
    suffix: Final = ":streamGenerateContent?alt=sse" if stream else ":generateContent"
    if provider == "gemini":
        assert request.target == f"/models/{BACKEND}{suffix}", request.target
        assert request.headers["x-goog-api-key"] == GEMINI_API_KEY, request.headers
        return
    assert request.target == f"{VERTEX_MODEL_PATH}{suffix}", request.target
    assert request.headers["authorization"] == "Bearer scripted-token", request.headers


def _assert_logged(answer: Answer) -> None:
    row: Final = spend_row_by_call(answer.call_id)
    assert (row["call_type"], row["completion_tokens"]) == ("acompletion", AUDIO_TOKENS), row


MAPPED: Final = (
    pytest.param("gemini", "httpx", False, "alloy", "Kore", id="gemini-httpx-alloy"),
    pytest.param("vertex", "httpx", False, "alloy", "Kore", id="vertex-httpx-alloy"),
    pytest.param("gemini", "sdk", False, "ALLOY", "Kore", id="gemini-sdk-ALLOY"),
    pytest.param("gemini", "sdk-async", False, "fable", "Umbriel", id="gemini-sdk-async-fable"),
    pytest.param("vertex", "sdk", False, "onyx", "Orus", id="vertex-sdk-onyx"),
    pytest.param("gemini", "httpx", True, "echo", "Charon", id="gemini-httpx-stream-echo"),
    pytest.param("vertex", "sdk", True, "onyx", "Orus", id="vertex-sdk-stream-onyx"),
    pytest.param("vertex", "sdk-async", True, "alloy", "Kore", id="vertex-sdk-async-stream-alloy"),
)


@pytest.mark.parametrize(("provider", "client", "stream", "voice", "mapped"), MAPPED)
def test_openai_voice_reaches_the_peer_as_the_gemini_voice(
    gateway: Gateway, provider: str, client: str, stream: bool, voice: str, mapped: str
) -> None:
    with wire_server(gemini_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, provider, wire.url, gateway.upstream_url)
        text: Final = marker()
        answer: Final = CLIENTS[client](gateway, model, voice, text, stream=stream)
        assert answer.status == 200, answer.text
        request: Final = _received(wire, text)
        _assert_target(request, provider, stream=stream)
        assert received_voice(request) == mapped, request.body
        assert answer.content == text, answer.text
        assert answer.audio == (None if stream else PCM_B64), answer.text
        _assert_logged(answer)


@pytest.mark.parametrize(("provider", "voice"), [("gemini", "Kore"), ("vertex", "nova")])
def test_gemini_voice_names_pass_through_verbatim(gateway: Gateway, provider: str, voice: str) -> None:
    with wire_server(gemini_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, provider, wire.url, gateway.upstream_url)
        text: Final = marker()
        answer: Final = _httpx_chat(gateway, model, voice, text, stream=False)
        assert answer.status == 200, answer.text
        assert received_voice(_received(wire, text)) == voice
        assert answer.audio == PCM_B64, answer.text
        _assert_logged(answer)


def _raw_chat(gateway: Gateway, body: str) -> httpx.Response:
    headers: Final = {"Authorization": f"Bearer {gateway.key}", "content-type": "application/json"}
    return gateway.client.post("/v1/chat/completions", content=body.encode(), headers=headers)


def _raw_body(model: str, text: str, audio_literal: str) -> str:
    return (
        f'{{"model": {json.dumps(model)}, "messages": [{{"role": "user", "content": {json.dumps(text)}}}], '
        f'"modalities": ["audio"], "audio": {audio_literal}}}'
    )


def test_vendor_rejection_of_an_unknown_voice_reaches_the_caller_and_the_proxy_keeps_serving(gateway: Gateway) -> None:
    with wire_server(gemini_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_deployment(scenario, wire.url)
        text: Final = marker()
        refused: Final = gateway.request("POST", "/v1/chat/completions", chat_body(model, "Custom-Voice", text))
        assert refused.status_code == 400, refused.text
        assert f"{REJECTED_PREFIX}Custom-Voice" in error_message(refused), refused.text
        assert received_voice(_received(wire, text)) == "Custom-Voice"
        follow_up: Final = marker()
        served: Final = _httpx_chat(gateway, model, "Kore", follow_up, stream=False)
        assert served.status == 200, served.text
        assert received_voice(_received(wire, follow_up)) == "Kore"


MALFORMED: Final = (
    pytest.param('{"voice": 7, "format": "pcm16"}', "7", id="int"),
    pytest.param('{"voice": null, "format": "pcm16"}', "None", id="null"),
    pytest.param('{"voice": "", "format": "pcm16"}', "", id="empty"),
    pytest.param(f'{{"voice": "{FIVE_KB}", "format": "pcm16"}}', FIVE_KB, id="5kb"),
    pytest.param('{"voice": ["alloy"], "format": "pcm16"}', "['alloy']", id="list"),
)


@pytest.mark.parametrize(("audio_literal", "rejected_as"), MALFORMED)
def test_malformed_voice_values_are_refused_without_taking_the_proxy_down(
    gateway: Gateway, audio_literal: str, rejected_as: str
) -> None:
    with wire_server(gemini_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_deployment(scenario, wire.url)
        text: Final = marker()
        refused: Final = _raw_chat(gateway, _raw_body(model, text, audio_literal))
        assert refused.status_code == 400, refused.text
        assert f"{REJECTED_PREFIX}{rejected_as} and language: " in error_message(refused), refused.text
        received: Final = _received(wire, text)
        assert received_voice(received) == json.loads(audio_literal)["voice"], received.body
        follow_up: Final = marker()
        served: Final = _httpx_chat(gateway, model, "alloy", follow_up, stream=False)
        assert served.status == 200, served.text
        assert received_voice(_received(wire, follow_up)) == "Kore"


def test_duplicated_voice_key_keeps_the_last_value(gateway: Gateway) -> None:
    with wire_server(gemini_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_deployment(scenario, wire.url)
        text: Final = marker()
        response: Final = _raw_chat(
            gateway, _raw_body(model, text, '{"voice": "alloy", "voice": "fable", "format": "pcm16"}')
        )
        assert response.status_code == 200, response.text
        assert received_voice(_received(wire, text)) == "Umbriel"
        assert audio_data(JSON_OBJECT.validate_json(response.content)) == PCM_B64


def test_audio_without_a_voice_sends_no_voice_config(gateway: Gateway) -> None:
    with wire_server(gemini_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_deployment(scenario, wire.url)
        text: Final = marker()
        body: Final = {**chat_body(model, "alloy", text), "audio": {"format": "pcm16"}}
        response: Final = gateway.request("POST", "/v1/chat/completions", body)
        assert response.status_code == 200, response.text
        assert received_voice(_received(wire, text)) == NO_VOICE
        assert audio_data(JSON_OBJECT.validate_json(response.content)) == PCM_B64


def _messages_bridge(gateway: Gateway, model: str, text: str, *, stream: bool) -> Answer:
    client: Final = anthropic_client(gateway)
    try:
        if stream:
            raw: Final = client.messages.with_raw_response.create(
                model=model, max_tokens=64, messages=[{"role": "user", "content": text}], stream=True
            )
            collected: Final = "".join(
                event.delta.text
                for event in raw.parse()
                if event.type == "content_block_delta" and event.delta.type == "text_delta"
            )
            return Answer(200, "", "", None, collected, raw.headers.get("x-litellm-call-id", ""))
        completed: Final = client.messages.with_raw_response.create(
            model=model, max_tokens=64, messages=[{"role": "user", "content": text}]
        )
        message: Final = completed.parse()
        content: Final = "".join(block.text for block in message.content if block.type == "text")
        return Answer(
            200, message.model_dump_json(), message.id, None, content, completed.headers.get("x-litellm-call-id", "")
        )
    except anthropic.APIStatusError as error:
        return _failed(error.status_code, _error_text(error.response))


def _responses_bridge(gateway: Gateway, model: str, text: str, *, stream: bool) -> Answer:
    client: Final = openai_client(gateway)
    try:
        if stream:
            raw: Final = client.responses.with_raw_response.create(model=model, input=text, stream=True)
            events: Final = list(raw.parse())
            collected: Final = "".join(event.delta for event in events if event.type == "response.output_text.delta")
            assert events[-1].type == "response.completed", [event.type for event in events]
            return Answer(200, "", "", None, collected, raw.headers.get("x-litellm-call-id", ""))
        completed: Final = client.responses.with_raw_response.create(model=model, input=text)
        response: Final = completed.parse()
        return Answer(
            200,
            response.model_dump_json(),
            response.id,
            None,
            response.output_text,
            completed.headers.get("x-litellm-call-id", ""),
        )
    except openai.APIStatusError as error:
        return _failed(error.status_code, _error_text(error.response))


def _completions_bridge(gateway: Gateway, model: str, text: str, *, stream: bool) -> Answer:
    response: Final = gateway.request("POST", "/v1/completions", {"model": model, "prompt": text, "max_tokens": 64})
    if response.status_code != 200:
        return _failed(response.status_code, response.text)
    body: Final = JSON_OBJECT.validate_json(response.content)
    choices: Final = body["choices"]
    assert isinstance(choices, list) and len(choices) == 1, response.text
    return Answer(
        200,
        response.text,
        string_value(body["id"]),
        None,
        string_value(object_value(choices[0])["text"]),
        response.headers.get("x-litellm-call-id", ""),
    )


BRIDGES: Final[Mapping[str, Callable[..., Answer]]] = {
    "messages": _messages_bridge,
    "responses": _responses_bridge,
    "completions": _completions_bridge,
}


@pytest.mark.parametrize(
    ("bridge", "stream"),
    [("messages", False), ("messages", True), ("responses", False), ("responses", True), ("completions", False)],
    ids=["messages", "messages-stream", "responses", "responses-stream", "completions"],
)
def test_bridged_endpoints_carry_the_deployment_voice(gateway: Gateway, bridge: str, stream: bool) -> None:
    with wire_server(gemini_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_deployment(scenario, wire.url, audio=audio_param("alloy"), modalities=["audio"])
        text: Final = marker()
        answer: Final = BRIDGES[bridge](gateway, model, text, stream=stream)
        assert answer.status == 200, answer.text
        request: Final = _received(wire, text)
        _assert_target(request, "gemini", stream=stream)
        assert received_voice(request) == "Kore", request.body
        assert answer.content == text, answer.text
        row: Final = spend_row_by_call(answer.call_id)
        assert row["completion_tokens"] == AUDIO_TOKENS, row


def _gcs_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target.startswith(f"/upload/storage/v1/b/{BUCKET}/o?uploadType=media&name="), request.target
    assert request.headers["authorization"] == "Bearer scripted-token", request.headers
    name: Final = request.target.rsplit("name=", 1)[1]
    body: Final = {
        "kind": "storage#object",
        "id": f"{BUCKET}/{name}/1",
        "name": name,
        "bucket": BUCKET,
        "size": str(len(request.body)),
        "timeCreated": "2026-10-08T00:00:00.000Z",
        "contentType": "application/json",
    }
    return Reply(body=json.dumps(body).encode())


def _batch_line(custom_id: str, text: str, voice: str) -> str:
    body: Final = {
        "model": BACKEND,
        "messages": [{"role": "user", "content": text}],
        "modalities": ["audio"],
        "audio": audio_param(voice),
    }
    return json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body})


def _uploaded_rows(request: Request) -> tuple[dict[str, JsonValue], ...]:
    lines: Final = tuple(line for line in request.body.decode().splitlines() if line.strip())
    return tuple(object_value(JSON_OBJECT.validate_json(line)["request"]) for line in lines)


def _decoded_file_id(file_id: str) -> tuple[str, str]:
    assert file_id.startswith("file-"), file_id
    encoded: Final = file_id[len("file-") :]
    decoded: Final = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    assert decoded.startswith("litellm:"), decoded
    raw, model = decoded[len("litellm:") :].rsplit(";model,", 1)
    return raw, model


def test_vertex_batch_file_rows_carry_the_mapped_voice(gateway: Gateway) -> None:
    with chunked_peer(_gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_deployment(
            scenario, wire.url, gateway.upstream_url, api_base=wire.url, gcs_bucket_name=BUCKET
        )
        first: Final = marker()
        second: Final = marker()
        content: Final = f"{_batch_line('row-1', first, 'alloy')}\n{_batch_line('row-2', second, 'Kore')}\n".encode()
        response: Final = gateway.request_multipart(
            "/v1/files", {"purpose": "batch", "model": model}, {"file": ("rows.jsonl", content, "application/jsonl")}
        )
        assert response.status_code == 200, response.text
        uploaded: Final = JSON_OBJECT.validate_json(response.content)
        requests: Final = wire.drain()
        assert len(requests) == 1, [request.target for request in requests]
        assert (uploaded["object"], uploaded["purpose"], uploaded["bytes"]) == ("file", "batch", len(requests[0].body))
        raw, file_model = _decoded_file_id(string_value(uploaded["id"]))
        assert raw.startswith(f"gs://{BUCKET}/") and file_model == model, response.text
        rows: Final = _uploaded_rows(requests[0])
        assert tuple(voice_in(row) for row in rows) == ("Kore", "Kore"), requests[0].body
        assert tuple(text_in(row) for row in rows) == (first, second)


def test_vertex_live_default_mode_sends_one_setup_without_the_client_voice(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = live_scenario(scenario)
        key: Final = scenario.key()
        model: Final = live_deployment(scenario, handle.scenario_id, gateway.upstream_url)
        frames: Final = live_turn(gateway, model, key, "fable")
        types: Final = frame_types(frames)
        assert types[0] == "session.created" and types[-1] == "response.done", frames
        setups: Final = live_setups(gateway, handle.scenario_id)
        assert len(setups) == 1, setups
        assert voice_in(setups[0]) == NO_VOICE, setups[0]


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return int(reserve.getsockname()[1])


@dataclass(frozen=True, slots=True)
class Fired:
    provider: str
    stream: bool
    voice: str
    text: str
    status: int
    text_body: str
    call_id: str


VOICES: Final = ("alloy", "fable", "Kore")


async def _fire(
    gateway: Gateway, models: Mapping[str, str], count: int, *, tolerate_transport_errors: bool = False
) -> tuple[Fired, ...]:
    async def one(client: httpx.AsyncClient, index: int) -> Fired:
        provider: Final = "gemini" if index % 2 == 0 else "vertex"
        stream: Final = index % 4 >= 2
        voice: Final = VOICES[index % len(VOICES)]
        text: Final = marker()
        body: Final = chat_body(models[provider], voice, text, stream=stream)
        try:
            response: Final = await client.post(
                "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {gateway.key}"}
            )
        except httpx.TransportError as error:
            if not tolerate_transport_errors:
                raise
            return Fired(provider, stream, voice, text, 0, repr(error), "")
        call_id: Final = response.headers.get("x-litellm-call-id", "")
        if response.status_code != 200:
            return Fired(provider, stream, voice, text, response.status_code, response.text, call_id)
        content: Final = (
            "".join(_delta_text(chunk) for chunk in _sse_chunks(response.text))
            if stream
            else _message_text(JSON_OBJECT.validate_json(response.content))
        )
        assert content == text, response.text
        return Fired(provider, stream, voice, text, 200, response.text, call_id)

    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
        return tuple(await asyncio.gather(*(one(client, index) for index in range(count))))


def _by_text(received: tuple[Request, ...]) -> dict[str, tuple[Request, ...]]:
    texts: Final = {received_text(request) for request in received}
    return {text: tuple(request for request in received if received_text(request) == text) for text in texts}


def _expected_voice(voice: str) -> str:
    return {"alloy": "Kore", "fable": "Umbriel"}.get(voice, voice)


def _assert_served(items: tuple[Fired, ...], received: tuple[Request, ...]) -> None:
    by_text: Final = _by_text(received)
    for item in items:
        assert item.status == 200, (item.provider, item.voice, item.text_body)
        assert len(by_text.get(item.text, ())) == 1, item.text
        assert received_voice(by_text[item.text][0]) == _expected_voice(item.voice), item.text
        row: Final = spend_row_by_call(item.call_id)
        assert row["call_type"] == "acompletion", row


def test_peer_outage_mid_burst_ends_each_request_once_and_recovers_on_the_same_port(gateway: Gateway) -> None:
    port: Final = _free_port()
    with gateway.scenario() as scenario:
        url: Final = f"http://127.0.0.1:{port}"
        models: Final = {
            "gemini": gemini_deployment(scenario, url),
            "vertex": vertex_deployment(scenario, url, gateway.upstream_url),
        }
        with wire_server(gemini_peer, port=port) as wire:
            healthy: Final = asyncio.run(_fire(gateway, models, BURST))
            _assert_served(healthy, wire.drain())
            racing: Final = _start_in_thread(lambda: asyncio.run(_fire(gateway, models, BURST)))
            eventually(lambda: wire.received.qsize(), lambda size: size >= OUTAGE_AFTER, 30)
        down: Final = asyncio.run(_fire(gateway, models, BURST))
        raced: Final = racing()
        raced_received: Final = wire.drain()
        with wire_server(gemini_peer, port=port) as restarted:
            recovered: Final = asyncio.run(_fire(gateway, models, BURST))
            recovered_received: Final = restarted.drain()
    assert all(len(requests) == 1 for requests in _by_text(raced_received).values())
    _assert_served(tuple(item for item in raced if item.status == 200), raced_received)
    lost: Final = tuple(item for item in raced if item.status != 200)
    for item in (*lost, *down):
        assert item.status >= 500, (item.status, item.text_body)
        assert '"error"' in item.text_body, item.text_body
    assert all(item.status != 200 for item in down), [item.status for item in down]
    _assert_served(recovered, recovered_received)
    assert {received_text(request) for request in recovered_received} == {item.text for item in recovered}


def _start_in_thread(work: Callable[[], tuple[Fired, ...]]) -> Callable[[], tuple[Fired, ...]]:
    results: Final[list[tuple[Fired, ...]]] = []
    thread: Final = threading.Thread(target=lambda: results.append(work()))
    thread.start()

    def finish() -> tuple[Fired, ...]:
        thread.join(timeout=120)
        assert not thread.is_alive(), "the racing burst never finished"
        return results[0]

    return finish


def _slow_peer(request: Request) -> Reply:
    reply: Final = gemini_peer(request)
    if reply.chunks is not None:
        return dataclasses.replace(reply, pause_between_chunks=SLOW_SECONDS)
    time.sleep(SLOW_SECONDS)
    return reply


def test_slow_peer_under_a_burst_answers_every_request_once_while_liveliness_stays_up(gateway: Gateway) -> None:
    with wire_server(_slow_peer) as wire, gateway.scenario() as scenario:
        models: Final = {
            "gemini": gemini_deployment(scenario, wire.url),
            "vertex": vertex_deployment(scenario, wire.url, gateway.upstream_url),
        }
        burst: Final = _start_in_thread(lambda: asyncio.run(_fire(gateway, models, SLOW_BURST)))
        eventually(lambda: wire.received.qsize(), lambda size: size >= SLOW_BURST, 30)
        liveliness: Final = gateway.request("GET", "/health/liveliness")
        assert liveliness.status_code == 200, liveliness.text
        served: Final = burst()
        received: Final = wire.drain()
    assert len(received) == SLOW_BURST, [received_text(request) for request in received]
    _assert_served(served, received)

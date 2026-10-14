import asyncio
import base64
import json
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import accumulate
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import openai
import psutil
import pytest
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_PROVIDER_KEY: Final = "synthetic-tts-key"
_FIRST: Final = b"audio-frame-0:" + bytes(range(64))
_REST: Final = (
    b"audio-frame-1:" + bytes(range(64, 192)),
    b"audio-frame-2:" + bytes(range(192, 256)),
)
_CLIP: Final = _FIRST + b"".join(_REST)
_FIRST_BYTE_SECONDS: Final = 20
_SPEECH: Final = "/v1/audio/speech"
_SPEND_SQL: Final = (
    'SELECT status, call_type, model_group, spend, metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s'
)
_ELEVENLABS_VOICE: Final = "21m00Tcm4TlvDq8ikWAM"
_AZURE_DEPLOYMENT: Final = "gpt-4o-mini-tts"


@dataclass(frozen=True, slots=True)
class Provider:
    model: str
    parameters: Mapping[str, JsonValue]
    streamed_path: str
    buffered_path: str


PROVIDERS: Final = MappingProxyType(
    {
        "openai": Provider(
            "openai/gpt-4o-mini-tts",
            {"api_key": _PROVIDER_KEY},
            "/audio/speech",
            "/audio/speech",
        ),
        "azure": Provider(
            f"azure/{_AZURE_DEPLOYMENT}",
            {"api_key": _PROVIDER_KEY, "api_version": "2025-04-01-preview"},
            f"/openai/deployments/{_AZURE_DEPLOYMENT}/audio/speech",
            f"/openai/deployments/{_AZURE_DEPLOYMENT}/audio/speech",
        ),
        "elevenlabs": Provider(
            "elevenlabs/eleven_v3",
            {"api_key": _PROVIDER_KEY},
            f"/v1/text-to-speech/{_ELEVENLABS_VOICE}/stream",
            f"/v1/text-to-speech/{_ELEVENLABS_VOICE}",
        ),
        "aws_polly": Provider(
            "aws_polly/neural",
            {
                "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                "aws_secret_access_key": "scripted-secret",
                "aws_region_name": "us-east-1",
            },
            "/v1/speech",
            "/v1/speech",
        ),
        "azure_speech": Provider(
            "azure/speech/azure-tts",
            {"api_key": _PROVIDER_KEY},
            "/cognitiveservices/v1",
            "/cognitiveservices/v1",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Heard:
    status: int
    content_type: str
    body: bytes


def _text() -> str:
    return f"streamed speech {uuid.uuid4().hex}"


def _speech(model: str, text: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "input": text, "voice": "alloy", **extra}


def _audio_peer(release: threading.Event | None = None, abort_after: int | None = None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        return Reply(
            content_type="audio/mpeg",
            chunks=(_FIRST, *_REST),
            gate_after_first=release,
            abort_after=abort_after,
        )

    return respond


def _deployment(scenario: Scenario, wire: Wire, provider: Provider, **extra: JsonValue) -> str:
    return scenario.model(
        model=provider.model,
        api_base=wire.url,
        model_info={"mode": "audio_speech"},
        **provider.parameters,
        **extra,
    )


def _headers(gateway: Gateway, call_id: str, key: str | None = None) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {gateway.key if key is None else key}",
        "x-litellm-call-id": call_id,
    }


def _first_bytes(chunks: Iterator[bytes], size: int) -> bytes:
    return next(heard for heard in accumulate(chunks) if len(heard) >= size)


async def _afirst_bytes(chunks: AsyncIterator[bytes], size: int, heard: bytes = b"") -> bytes:
    if len(heard) >= size:
        return heard
    return await _afirst_bytes(chunks, size, heard + await anext(chunks))


def _listen(
    gateway: Gateway,
    body: Mapping[str, JsonValue],
    call_id: str,
    release: threading.Event,
) -> Heard:
    with gateway.client.stream(
        "POST",
        _SPEECH,
        json=dict(body),
        headers=_headers(gateway, call_id),
        timeout=httpx.Timeout(60, read=_FIRST_BYTE_SECONDS),
    ) as response:
        chunks: Final = response.iter_raw()
        first: Final = _first_bytes(chunks, len(_FIRST))
        release.set()
        rest: Final = b"".join(chunks)
        return Heard(response.status_code, response.headers.get("content-type", ""), first + rest)


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(_SPEND_SQL, (call_id,)),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    assert len(rows) == 1, rows
    return rows[0]


def _buffered_spend(gateway: Gateway, model: str, text: str) -> float:
    call_id: Final = str(uuid.uuid4())
    response: Final = gateway.request("POST", _SPEECH, _speech(model, text), headers={"x-litellm-call-id": call_id})
    assert response.status_code == 200, response.text[:400]
    spend: Final = _spend_row(call_id)["spend"]
    assert isinstance(spend, float) and spend > 0, spend
    return spend


def _assert_billed(gateway: Gateway, model: str, call_id: str, text: str) -> dict[str, JsonValue]:
    row: Final = _spend_row(call_id)
    assert (row["status"], row["call_type"], row["model_group"]) == ("success", "aspeech", model), row
    assert row["spend"] == pytest.approx(_buffered_spend(gateway, model, text)), row
    return object_value(row["metadata"])


def _failure_spend(model: str, call_id: str) -> JsonValue:
    row: Final = _spend_row(call_id)
    assert (row["status"], row["model_group"]) == ("failure", model), row
    return row["spend"]


def _only_request(wire: Wire) -> Request:
    requests: Final = wire.drain()
    assert len(requests) == 1, [request.target for request in requests]
    return requests[0]


def _assert_reached(wire: Wire, path: str, text: str) -> Request:
    request: Final = _only_request(wire)
    assert (request.method, urlsplit(request.target).path) == ("POST", path), request.target
    assert text.encode() in request.body, request.body[:400]
    return request


def _held_upstream_connections(wire: Wire) -> int:
    port: Final = urlsplit(wire.url).port
    return sum(
        1
        for connection in psutil.Process().net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.laddr and connection.laddr.port == port
    )


@pytest.mark.parametrize("provider", tuple(PROVIDERS))
def test_streamed_speech_reaches_the_caller_before_the_clip_ends_and_bills_it_once(
    gateway: Gateway, provider: str
) -> None:
    release: Final = threading.Event()
    with wire_server(_audio_peer(release)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS[provider])
        text: Final = _text()
        call_id: Final = str(uuid.uuid4())
        heard: Final = _listen(gateway, _speech(model, text, stream=True), call_id, release)
        assert (heard.status, heard.body) == (200, _CLIP), heard
        assert heard.content_type.startswith("audio/"), heard.content_type
        _assert_reached(wire, PROVIDERS[provider].streamed_path, text)
        _assert_billed(gateway, model, call_id, text)


def test_openai_sdk_sync_client_hears_streamed_speech_before_the_clip_ends(
    gateway: Gateway,
) -> None:
    release: Final = threading.Event()
    with wire_server(_audio_peer(release)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["openai"])
        text: Final = _text()
        call_id: Final = str(uuid.uuid4())
        client: Final = openai.OpenAI(
            base_url=f"{gateway.client.base_url}/v1",
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=httpx.Timeout(60, read=_FIRST_BYTE_SECONDS)),
        )
        with client.audio.speech.with_streaming_response.create(
            model=model,
            input=text,
            voice="alloy",
            extra_body={"stream": True},
            extra_headers={"x-litellm-call-id": call_id},
        ) as response:
            chunks: Final = response.iter_bytes()
            first: Final = _first_bytes(chunks, len(_FIRST))
            release.set()
            assert first + b"".join(chunks) == _CLIP
        client.close()
        _assert_reached(wire, PROVIDERS["openai"].streamed_path, text)
        _assert_billed(gateway, model, call_id, text)


async def _async_sdk_clip(gateway: Gateway, model: str, text: str, call_id: str, release: threading.Event) -> bytes:
    client: Final = openai.AsyncOpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(60, read=_FIRST_BYTE_SECONDS)),
    )
    try:
        async with client.audio.speech.with_streaming_response.create(
            model=model,
            input=text,
            voice="alloy",
            extra_body={"stream": True},
            extra_headers={"x-litellm-call-id": call_id},
        ) as response:
            chunks: Final = response.iter_bytes()
            first: Final = await _afirst_bytes(chunks, len(_FIRST))
            release.set()
            return first + b"".join([chunk async for chunk in chunks])
    finally:
        await client.close()


def test_openai_sdk_async_client_hears_streamed_speech_before_the_clip_ends(
    gateway: Gateway,
) -> None:
    release: Final = threading.Event()
    with wire_server(_audio_peer(release)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["openai"])
        text: Final = _text()
        call_id: Final = str(uuid.uuid4())
        assert asyncio.run(_async_sdk_clip(gateway, model, text, call_id, release)) == _CLIP
        _assert_reached(wire, PROVIDERS["openai"].streamed_path, text)
        _assert_billed(gateway, model, call_id, text)


@pytest.mark.parametrize("stream", [None, False], ids=["omitted", "false"])
@pytest.mark.parametrize("provider", ["openai", "elevenlabs"])
def test_buffered_speech_is_unchanged(gateway: Gateway, provider: str, stream: bool | None) -> None:
    with wire_server(_audio_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS[provider])
        text: Final = _text()
        call_id: Final = str(uuid.uuid4())
        extra: Final[dict[str, JsonValue]] = {} if stream is None else {"stream": stream}
        response: Final = gateway.request(
            "POST",
            _SPEECH,
            _speech(model, text, **extra),
            headers={"x-litellm-call-id": call_id},
        )
        assert (response.status_code, response.content) == (200, _CLIP), response.text[:400]
        _assert_reached(wire, PROVIDERS[provider].buffered_path, text)
        _assert_billed(gateway, model, call_id, text)


def _mistral_peer(request: Request) -> Reply:
    return Reply(body=json.dumps({"audio_data": base64.b64encode(_CLIP).decode()}).encode())


def test_stream_true_on_a_provider_without_audio_streaming_answers_buffered_and_bills_it(
    gateway: Gateway,
) -> None:
    with wire_server(_mistral_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="mistral/voxtral-mini-tts-latest",
            api_base=wire.url,
            api_key=_PROVIDER_KEY,
            model_info={"mode": "audio_speech"},
        )
        text: Final = _text()
        call_id: Final = str(uuid.uuid4())
        response: Final = gateway.request(
            "POST",
            _SPEECH,
            _speech(model, text, stream=True),
            headers={"x-litellm-call-id": call_id},
        )
        assert (response.status_code, response.content) == (200, _CLIP), response.text[:400]
        _assert_reached(wire, "/v1/audio/speech", text)
        _assert_billed(gateway, model, call_id, text)


@pytest.mark.parametrize(
    "stream",
    ["true", 1, None, [], {}, "s" * 5120],
    ids=["string-true", "int-1", "null", "empty-list", "empty-object", "5kb-string"],
)
def test_a_stream_value_other_than_literal_true_answers_buffered(gateway: Gateway, stream: JsonValue) -> None:
    with wire_server(_audio_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["elevenlabs"])
        text: Final = _text()
        call_id: Final = str(uuid.uuid4())
        response: Final = gateway.request(
            "POST",
            _SPEECH,
            _speech(model, text, stream=stream),
            headers={"x-litellm-call-id": call_id},
        )
        assert (response.status_code, response.content) == (200, _CLIP), response.text[:400]
        request: Final = _assert_reached(wire, PROVIDERS["elevenlabs"].buffered_path, text)
        assert "stream" not in object_value(json.loads(request.body)), request.body[:400]
        _assert_billed(gateway, model, call_id, text)


def test_a_duplicated_stream_true_key_streams(gateway: Gateway) -> None:
    with wire_server(_audio_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["elevenlabs"])
        text: Final = _text()
        call_id: Final = str(uuid.uuid4())
        body: Final = json.dumps(_speech(model, text, stream=True))[:-1] + ', "stream": true}'
        response: Final = gateway.client.post(
            _SPEECH,
            content=body.encode(),
            headers={**_headers(gateway, call_id), "content-type": "application/json"},
        )
        assert (response.status_code, response.content) == (200, _CLIP), response.text[:400]
        _assert_reached(wire, PROVIDERS["elevenlabs"].streamed_path, text)
        _assert_billed(gateway, model, call_id, text)


def test_a_hang_up_mid_clip_closes_the_provider_stream_and_bills_the_clip_with_a_499_marker(
    gateway: Gateway,
) -> None:
    release: Final = threading.Event()
    with wire_server(_audio_peer(release)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["openai"])
        text: Final = _text()
        call_id: Final = str(uuid.uuid4())
        with gateway.client.stream(
            "POST",
            _SPEECH,
            json=_speech(model, text, stream=True),
            headers=_headers(gateway, call_id),
            timeout=httpx.Timeout(60, read=_FIRST_BYTE_SECONDS),
        ) as response:
            assert response.status_code == 200
            assert _first_bytes(response.iter_raw(), len(_FIRST)) == _FIRST
        eventually(lambda: _held_upstream_connections(wire), lambda held: held == 0)
        release.set()
        _assert_reached(wire, PROVIDERS["openai"].streamed_path, text)
        metadata: Final = _assert_billed(gateway, model, call_id, text)
        assert object_value(metadata["error_information"])["error_code"] == "499", metadata


def test_a_provider_cut_off_mid_clip_cuts_the_caller_off_and_bills_a_failure_at_full_cost(
    gateway: Gateway,
) -> None:
    with (
        wire_server(_audio_peer(abort_after=1)) as wire,
        wire_server(_audio_peer()) as healthy,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, wire, PROVIDERS["openai"])
        twin: Final = _deployment(scenario, healthy, PROVIDERS["openai"])
        text: Final = _text()
        call_id: Final = str(uuid.uuid4())
        with gateway.client.stream(
            "POST",
            _SPEECH,
            json=_speech(model, text, stream=True),
            headers=_headers(gateway, call_id),
        ) as response:
            assert response.status_code == 200
            chunks: Final = response.iter_raw()
            assert _first_bytes(chunks, len(_FIRST)) == _FIRST
            with pytest.raises(httpx.RemoteProtocolError):
                b"".join(chunks)
        _assert_reached(wire, PROVIDERS["openai"].streamed_path, text)
        assert _failure_spend(model, call_id) == pytest.approx(_buffered_spend(gateway, twin, text))


_REJECTIONS: Final = MappingProxyType(
    {
        "openai": json.dumps(
            {
                "error": {
                    "message": "Incorrect API key provided: synthetic",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            }
        ),
        "elevenlabs": json.dumps(
            {
                "detail": {
                    "status": "invalid_api_key",
                    "message": "Invalid API key: synthetic",
                }
            }
        ),
    }
)


@pytest.mark.parametrize("provider", tuple(_REJECTIONS))
def test_a_provider_rejection_before_any_audio_reaches_the_caller_with_its_status_and_message(
    gateway: Gateway, provider: str
) -> None:
    def respond(request: Request) -> Reply:
        return Reply(status=401, body=_REJECTIONS[provider].encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS[provider])
        text: Final = _text()
        call_id: Final = str(uuid.uuid4())
        response: Final = gateway.request(
            "POST",
            _SPEECH,
            _speech(model, text, stream=True),
            headers={"x-litellm-call-id": call_id},
        )
        assert response.status_code == 401, response.text[:400]
        assert "API key" in string_value(object_value(JSON_OBJECT.validate_json(response.content)["error"])["message"])
        _assert_reached(wire, PROVIDERS[provider].streamed_path, text)
        assert _failure_spend(model, call_id) == 0.0


def test_a_repeated_identical_streamed_request_reaches_the_provider_and_bills_each_call_once(
    gateway: Gateway,
) -> None:
    with wire_server(_audio_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["openai"])
        text: Final = _text()
        call_ids: Final = (str(uuid.uuid4()), str(uuid.uuid4()))
        responses: Final = tuple(
            gateway.request(
                "POST",
                _SPEECH,
                _speech(model, text, stream=True),
                headers={"x-litellm-call-id": call_id},
            )
            for call_id in call_ids
        )
        assert [(response.status_code, response.content) for response in responses] == [(200, _CLIP)] * 2
        received: Final = wire.drain()
        assert [urlsplit(request.target).path for request in received] == ["/audio/speech"] * 2
        for call_id in call_ids:
            _assert_billed(gateway, model, call_id, text)


def _admitted(gateway: Gateway, model: str, key: str) -> int:
    return gateway.request("POST", _SPEECH, _speech(model, _text(), stream=True), key=key).status_code


def test_a_key_max_parallel_requests_slot_is_held_until_the_streamed_clip_ends(
    gateway: Gateway,
) -> None:
    release: Final = threading.Event()
    with wire_server(_audio_peer(release)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["openai"])
        key: Final = scenario.key(max_parallel_requests=1)
        with gateway.client.stream(
            "POST",
            _SPEECH,
            json=_speech(model, _text(), stream=True),
            headers=_headers(gateway, str(uuid.uuid4()), key),
            timeout=httpx.Timeout(60, read=_FIRST_BYTE_SECONDS),
        ) as held:
            chunks: Final = held.iter_raw()
            first: Final = _first_bytes(chunks, len(_FIRST))
            blocked: Final = gateway.request("POST", _SPEECH, _speech(model, _text(), stream=True), key=key)
            release.set()
            assert first + b"".join(chunks) == _CLIP
        assert blocked.status_code == 429, blocked.text[:400]
        eventually(lambda: _admitted(gateway, model, key), lambda status: status == 200)
        assert len(wire.drain()) == 2


def test_a_key_max_parallel_requests_slot_is_freed_after_each_completed_clip(
    gateway: Gateway,
) -> None:
    with wire_server(_audio_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["openai"])
        key: Final = scenario.key(max_parallel_requests=1)
        for _ in range(3):
            eventually(lambda: _admitted(gateway, model, key), lambda status: status == 200)
        assert len(wire.drain()) == 3


def test_a_key_max_parallel_requests_slot_is_freed_by_a_hang_up(
    gateway: Gateway,
) -> None:
    release: Final = threading.Event()
    with wire_server(_audio_peer(release)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["openai"])
        key: Final = scenario.key(max_parallel_requests=1)
        with gateway.client.stream(
            "POST",
            _SPEECH,
            json=_speech(model, _text(), stream=True),
            headers=_headers(gateway, str(uuid.uuid4()), key),
            timeout=httpx.Timeout(60, read=_FIRST_BYTE_SECONDS),
        ) as held:
            assert _first_bytes(held.iter_raw(), len(_FIRST)) == _FIRST
        eventually(lambda: _held_upstream_connections(wire), lambda count: count == 0)
        release.set()
        eventually(lambda: _admitted(gateway, model, key), lambda status: status == 200)
        assert len(wire.drain()) == 2


def test_a_deployment_max_parallel_requests_slot_is_held_until_the_streamed_clip_ends(
    gateway: Gateway,
) -> None:
    release: Final = threading.Event()
    with wire_server(_audio_peer(release)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["openai"], max_parallel_requests=1)
        with gateway.client.stream(
            "POST",
            _SPEECH,
            json=_speech(model, _text(), stream=True),
            headers=_headers(gateway, str(uuid.uuid4())),
            timeout=httpx.Timeout(60, read=_FIRST_BYTE_SECONDS),
        ) as held:
            chunks: Final = held.iter_raw()
            first: Final = _first_bytes(chunks, len(_FIRST))
            blocked: Final = gateway.request("POST", _SPEECH, _speech(model, _text(), stream=True))
            release.set()
            assert first + b"".join(chunks) == _CLIP
        assert blocked.status_code == 429, blocked.text[:400]
        eventually(lambda: _admitted(gateway, model, gateway.key), lambda status: status == 200)
        assert len(wire.drain()) == 2


def test_a_deployment_max_parallel_requests_slot_is_freed_by_a_hang_up(
    gateway: Gateway,
) -> None:
    release: Final = threading.Event()
    with wire_server(_audio_peer(release)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, PROVIDERS["openai"], max_parallel_requests=1)
        with gateway.client.stream(
            "POST",
            _SPEECH,
            json=_speech(model, _text(), stream=True),
            headers=_headers(gateway, str(uuid.uuid4())),
            timeout=httpx.Timeout(60, read=_FIRST_BYTE_SECONDS),
        ) as held:
            assert _first_bytes(held.iter_raw(), len(_FIRST)) == _FIRST
        eventually(lambda: _held_upstream_connections(wire), lambda count: count == 0)
        release.set()
        eventually(lambda: _admitted(gateway, model, gateway.key), lambda status: status == 200)
        assert len(wire.drain()) == 2


_CHAT_ID: Final = "chatcmpl-nonboolstream"
_CHAT_USAGE: Final = {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}


def _chat_chunk(delta: Mapping[str, JsonValue], finish_reason: str | None, **extra: JsonValue) -> bytes:
    chunk: Final = {
        "id": _CHAT_ID,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
        **extra,
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _chat_peer(request: Request) -> Reply:
    if object_value(JSON_OBJECT.validate_json(request.body)).get("stream"):
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _chat_chunk({"role": "assistant", "content": "hello"}, None),
                _chat_chunk({}, "stop", usage=_CHAT_USAGE),
                b"data: [DONE]\n\n",
            ),
        )
    completion: Final = {
        "id": _CHAT_ID,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": _CHAT_USAGE,
    }
    return Reply(body=json.dumps(completion).encode())


@pytest.mark.parametrize("stream", ["true", 1], ids=["string-true", "int-1"])
def test_a_chat_completion_with_a_non_boolean_stream_value_answers_and_lands_one_success_row(
    gateway: Gateway, stream: JsonValue
) -> None:
    with wire_server(_chat_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        call_id: Final = str(uuid.uuid4())
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"say hello {call_id}"}],
                "stream": stream,
            },
            headers={"x-litellm-call-id": call_id},
        )
        assert response.status_code == 200, response.text[:400]
        assert "hello" in response.text, response.text[:400]
        row: Final = _spend_row(call_id)
        assert (row["status"], row["call_type"], row["model_group"]) == (
            "success",
            "acompletion",
            model,
        ), row
        assert isinstance(row["spend"], float) and row["spend"] > 0, row

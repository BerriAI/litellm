import io
import json
import math
import mimetypes
import uuid
import wave
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from hashlib import sha256
from typing import Final, cast

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

_WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)
_WAV_CONTENT_TYPE: Final = mimetypes.guess_type("a.wav")[0] or "application/octet-stream"
_VERBOSE_TEXT_PARTS: Final = (
    ("include[]", "logprobs"),
    ("language", "en"),
    ("model", "whisper-1"),
    ("prompt", "hi"),
    ("response_format", "verbose_json"),
    ("temperature", "0.2"),
    ("timestamp_granularities[]", "word"),
    ("timestamp_granularities[]", "segment"),
)
_VERBOSE_TRANSCRIPT: Final = {
    "task": "transcribe",
    "language": "english",
    "duration": 1.5,
    "text": "hello world",
    "words": [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
    ],
    "segments": [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 1.0,
            "text": "hello world",
            "tokens": [1, 2],
            "temperature": 0.2,
            "avg_logprob": -0.1,
            "compression_ratio": 1.0,
            "no_speech_prob": 0.01,
        }
    ],
    "usage": {
        "type": "tokens",
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
        "input_token_details": {"text_tokens": 2, "audio_tokens": 10},
    },
}
_JSON_TRANSCRIPT: Final = {"text": "hello world", "usage": _VERBOSE_TRANSCRIPT["usage"]}
_PLAIN_TRANSCRIPTS: Final = {
    "text": "hello world\n",
    "srt": "1\n00:00:00,000 --> 00:00:01,000\nhello world\n",
    "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello world\n",
}


class _Word(BaseModel):
    model_config = ConfigDict(extra="forbid")
    word: str
    start: float
    end: float


class _Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: tuple[int, ...]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class _InputTokenDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_tokens: int
    audio_tokens: int


class _Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_token_details: _InputTokenDetails


class _VerboseTranscript(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str
    language: str
    duration: float
    text: str
    words: tuple[_Word, ...]
    segments: tuple[_Segment, ...]
    usage: _Usage


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def _text_parts(parts: tuple[Message, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                (part.get_param("name", header="content-disposition"), part.get_payload(decode=True).decode())
                for part in parts
                if part.get_filename() is None
            ),
            key=lambda field: field[0],
        )
    )


def _file_parts(parts: tuple[Message, ...]) -> tuple[tuple[str, str, str, bytes], ...]:
    return tuple(
        sorted(
            (
                (
                    part.get_param("name", header="content-disposition"),
                    part.get_filename(),
                    part.get_content_type(),
                    part.get_payload(decode=True),
                )
                for part in parts
                if part.get_filename() is not None
            ),
            key=lambda field: field[0],
        )
    )


def _sdk(gateway: Gateway, path: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=str(gateway.client.base_url).rstrip("/") + path,
        api_key=api_key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


def test_audio_transcription_verbose_json_forwards_every_form_field_and_returns_the_transcript(
    gateway: Gateway,
) -> None:
    call_id: Final = f"audio-transcription-{uuid.uuid4().hex}"
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == _VERBOSE_TEXT_PARTS
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, wav),)
        return Reply(body=json.dumps(_VERBOSE_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-openai-key",
            input_cost_per_token=0.000001,
            input_cost_per_audio_token=0.000003,
            output_cost_per_token=0.000002,
        )
        with _sdk(gateway, "/v1", gateway.key) as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", wav, _WAV_CONTENT_TYPE),
                temperature=0.2,
                language="en",
                prompt="hi",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                include=["logprobs"],
                extra_headers={"x-litellm-call-id": call_id},
            )
        assert response.status_code == 200, response.text
        assert json.loads(response.content) == _VERBOSE_TRANSCRIPT, response.text
        _VerboseTranscript.model_validate_json(response.content)
        assert response.headers["x-litellm-call-id"] == call_id, response.text
        usage: Final = cast("dict[str, object]", _VERBOSE_TRANSCRIPT["usage"])
        token_details: Final = cast("dict[str, int]", usage["input_token_details"])
        assert math.isclose(
            float(response.headers["x-litellm-response-cost"]),
            token_details["audio_tokens"] * 0.000003
            + token_details["text_tokens"] * 0.000001
            + cast(int, usage["output_tokens"]) * 0.000002,
            rel_tol=1e-9,
        ), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/transcriptions")]


def test_audio_transcription_reaches_upstream_identically_on_prefixed_and_unprefixed_routes(
    gateway: Gateway,
) -> None:
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == _VERBOSE_TEXT_PARTS
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, wav),)
        return Reply(body=json.dumps(_VERBOSE_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        prefixed: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        unprefixed: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        key: Final = scenario.key(models=[prefixed, unprefixed])

        def transcribe(base_path: str, model: str) -> dict[str, object]:
            with _sdk(gateway, base_path, key) as sdk:
                return json.loads(
                    sdk.audio.transcriptions.with_raw_response.create(
                        model=model,
                        file=("a.wav", wav, _WAV_CONTENT_TYPE),
                        temperature=0.2,
                        language="en",
                        prompt="hi",
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"],
                        include=["logprobs"],
                    ).content
                )

        bodies: Final = tuple(
            transcribe(base_path, model) for base_path, model in (("/v1", prefixed), ("", unprefixed))
        )
        assert bodies[0] == bodies[1] == _VERBOSE_TRANSCRIPT
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/v1/audio/transcriptions"),
            ("POST", "/v1/audio/transcriptions"),
        ]


def test_audio_transcription_json_format_returns_text_and_usage(gateway: Gateway) -> None:
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "whisper-1"), ("response_format", "json"))
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, wav),)
        return Reply(body=json.dumps(_JSON_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        with _sdk(gateway, "/v1", gateway.key) as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", wav, _WAV_CONTENT_TYPE),
                response_format="json",
            )
        assert response.status_code == 200, response.text
        assert json.loads(response.content) == _JSON_TRANSCRIPT, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/transcriptions")]


def test_audio_transcription_forwards_nested_and_repeated_bracket_fields_verbatim(gateway: Gateway) -> None:
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (
            ("chunking_strategy[prefix_padding_ms]", "300"),
            ("chunking_strategy[silence_duration_ms]", "500"),
            ("chunking_strategy[threshold]", "0.5"),
            ("chunking_strategy[type]", "server_vad"),
            ("known_speaker_names[]", "alice"),
            ("known_speaker_names[]", "bob"),
            ("known_speaker_references[]", "data:audio/wav;base64,AAAA"),
            ("known_speaker_references[]", "data:audio/wav;base64,BBBB"),
            ("model", "gpt-4o-transcribe"),
            ("response_format", "json"),
        )
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, wav),)
        return Reply(body=json.dumps(_JSON_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-transcribe", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        known_speaker_names: Final = ("alice", "bob")
        known_speaker_references: Final = (
            "data:audio/wav;base64,AAAA",
            "data:audio/wav;base64,BBBB",
        )
        with _sdk(gateway, "/v1", gateway.key) as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", wav, _WAV_CONTENT_TYPE),
                response_format="json",
                chunking_strategy={
                    "type": "server_vad",
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "threshold": 0.5,
                },
                known_speaker_names=known_speaker_names,
                known_speaker_references=known_speaker_references,
            )
        assert response.status_code == 200, response.text
        assert json.loads(response.content) == _JSON_TRANSCRIPT, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/transcriptions")]


def test_audio_transcription_stream_returns_server_sent_events(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: stream=true transcriptions reach the upstream but its SSE body comes back as application/json "
        '{"text": "data: ..."}, so OpenAI SDK stream=True callers get no transcript events'
    )
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "gpt-4o-transcribe"), ("stream", "true"))
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, wav),)
        return Reply(
            chunks=(
                b'data: {"type":"transcript.text.delta","delta":"hello"}\n\n',
                b'data: {"type":"transcript.text.delta","delta":" world"}\n\n',
                b'data: {"type":"transcript.text.done","text":"hello world",'
                b'"usage":{"type":"tokens","input_tokens":12,"output_tokens":4,"total_tokens":16}}\n\n',
            ),
            content_type="text/event-stream",
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-transcribe", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        with _sdk(gateway, "/v1", gateway.key) as sdk:
            with sdk.audio.transcriptions.with_streaming_response.create(
                model=model,
                file=("a.wav", wav, _WAV_CONTENT_TYPE),
                stream=True,
            ) as raw:
                assert raw.headers["content-type"].startswith("text/event-stream")
            events: Final = tuple(
                event.to_dict()
                for event in sdk.audio.transcriptions.create(
                    model=model,
                    file=("a.wav", wav, _WAV_CONTENT_TYPE),
                    stream=True,
                )
            )
        assert events == (
            {"type": "transcript.text.delta", "delta": "hello"},
            {"type": "transcript.text.delta", "delta": " world"},
            {
                "type": "transcript.text.done",
                "text": "hello world",
                "usage": {
                    "type": "tokens",
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "total_tokens": 16,
                },
            },
        )
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/v1/audio/transcriptions"),
            ("POST", "/v1/audio/transcriptions"),
        ]


def test_audio_transcription_applies_json_string_metadata_without_forwarding_it(gateway: Gateway) -> None:
    tag: Final = f"audio-metadata-{uuid.uuid4().hex}"
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "whisper-1"), ("response_format", "json"))
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, wav),)
        return Reply(body=json.dumps(_JSON_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = gateway.request_multipart(
            "/v1/audio/transcriptions",
            {
                "model": model,
                "response_format": "json",
                "metadata": json.dumps({"tags": [tag]}),
            },
            {"file": ("a.wav", wav, _WAV_CONTENT_TYPE)},
        )
        assert response.status_code == 200, response.text
        assert response.json() == _JSON_TRANSCRIPT, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/transcriptions")]
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
                (response.headers["x-litellm-call-id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        request_tags: Final = rows[0]["request_tags"]
        assert isinstance(request_tags, list), rows
        assert tuple(
            value for value in request_tags if isinstance(value, str) and not value.startswith("User-Agent: ")
        ) == (tag,), rows


@pytest.mark.parametrize("response_format", ("text", "srt", "vtt"))
def test_audio_transcription_forwards_plain_response_format_to_upstream(gateway: Gateway, response_format: str) -> None:
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "whisper-1"), ("response_format", response_format))
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, wav),)
        return Reply(body=_PLAIN_TRANSCRIPTS[response_format].encode(), content_type="text/plain")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        with _sdk(gateway, "/v1", gateway.key) as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", wav, _WAV_CONTENT_TYPE),
                response_format=response_format,
            )
        assert response.status_code == 200, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/transcriptions")]


@pytest.mark.parametrize("response_format", ("text", "srt", "vtt"))
def test_audio_transcription_plain_response_format_returns_the_raw_transcript(
    gateway: Gateway, response_format: str
) -> None:
    pytest.skip(
        "BUG: response_format=text/srt/vtt transcriptions return application/json "
        '{"text": ...} instead of the raw transcript, so the OpenAI SDK returns a JSON string as the transcript'
    )
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        return Reply(body=_PLAIN_TRANSCRIPTS[response_format].encode(), content_type="text/plain")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        with _sdk(gateway, "/v1", gateway.key) as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", wav, _WAV_CONTENT_TYPE),
                response_format=response_format,
            )
            assert response.headers["content-type"].startswith("text/plain"), response.text
            transcript: Final = sdk.audio.transcriptions.create(
                model=model,
                file=("a.wav", wav, _WAV_CONTENT_TYPE),
                response_format=response_format,
            )
        assert transcript == _PLAIN_TRANSCRIPTS[response_format]


def test_audio_transcription_applies_json_string_litellm_metadata_without_forwarding_it(
    gateway: Gateway,
) -> None:
    tag: Final = f"audio-metadata-{uuid.uuid4().hex}"
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "whisper-1"), ("response_format", "json"))
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, wav),)
        return Reply(body=json.dumps(_JSON_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = gateway.request_multipart(
            "/v1/audio/transcriptions",
            {
                "model": model,
                "response_format": "json",
                "litellm_metadata": json.dumps({"tags": [tag]}),
            },
            {"file": ("a.wav", wav, _WAV_CONTENT_TYPE)},
        )
        assert response.status_code == 200, response.text
        assert response.json() == _JSON_TRANSCRIPT, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/transcriptions")]
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
                (response.headers["x-litellm-call-id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        request_tags: Final = rows[0]["request_tags"]
        assert isinstance(request_tags, list), rows
        assert tuple(
            value for value in request_tags if isinstance(value, str) and not value.startswith("User-Agent: ")
        ) == (tag,), rows


@pytest.mark.parametrize("metadata_field", ("metadata", "litellm_metadata"))
def test_audio_transcription_consumes_bracketed_form_metadata_tags(gateway: Gateway, metadata_field: str) -> None:
    pytest.skip(
        "BUG: multipart metadata[tags]/litellm_metadata[tags] on /v1/audio/transcriptions reaches the upstream "
        "as an extra text part and the tag never lands in LiteLLM_SpendLogs.request_tags"
    )
    tag: Final = f"audio-metadata-{uuid.uuid4().hex}"
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "whisper-1"), ("response_format", "json"))
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, wav),)
        return Reply(body=json.dumps(_JSON_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = gateway.request_multipart(
            "/v1/audio/transcriptions",
            {
                "model": model,
                "response_format": "json",
                f"{metadata_field}[tags]": tag,
            },
            {"file": ("a.wav", wav, _WAV_CONTENT_TYPE)},
        )
        assert response.status_code == 200, response.text
        assert response.json() == _JSON_TRANSCRIPT, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/transcriptions")]
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
                (response.headers["x-litellm-call-id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        request_tags: Final = rows[0]["request_tags"]
        assert isinstance(request_tags, list), rows
        assert tuple(
            value for value in request_tags if isinstance(value, str) and not value.startswith("User-Agent: ")
        ) == (tag,), rows


def test_audio_transcription_preserves_the_client_file_content_type(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: the proxy discards the client's file part content type and re-derives it from the filename, "
        "so file=('recording', ..., 'audio/wav') reaches the upstream as application/octet-stream"
    )
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        parts: Final = _multipart_parts(request)
        assert _file_parts(parts) == (("file", "recording", "audio/wav", wav),)
        return Reply(body=json.dumps(_JSON_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = gateway.request_multipart(
            "/v1/audio/transcriptions",
            {"model": model, "response_format": "json"},
            {"file": ("recording", wav, "audio/wav")},
        )
        assert response.status_code == 200, response.text
        assert response.json() == _JSON_TRANSCRIPT, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/transcriptions")]


def test_audio_transcription_bills_audio_tokens_at_the_input_rate_when_no_audio_rate_is_set(
    gateway: Gateway,
) -> None:
    pytest.skip(
        "BUG: with only input_cost_per_token set, transcription audio tokens are billed at 0, so "
        "x-litellm-response-cost is 1e-05 instead of 2e-05 for 10 audio, 2 text and 4 output tokens"
    )
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "whisper-1"), ("response_format", "json"))
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, wav),)
        return Reply(body=json.dumps(_JSON_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-openai-key",
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
        )
        response: Final = gateway.request_multipart(
            "/v1/audio/transcriptions",
            {"model": model, "response_format": "json"},
            {"file": ("a.wav", wav, _WAV_CONTENT_TYPE)},
        )
        assert response.status_code == 200, response.text
        assert response.json() == _JSON_TRANSCRIPT, response.text
        usage: Final = cast("dict[str, object]", _JSON_TRANSCRIPT["usage"])
        token_details: Final = cast("dict[str, int]", usage["input_token_details"])
        assert math.isclose(
            float(response.headers["x-litellm-response-cost"]),
            token_details["audio_tokens"] * 0.000001
            + token_details["text_tokens"] * 0.000001
            + cast(int, usage["output_tokens"]) * 0.000002,
            rel_tol=1e-9,
        ), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/transcriptions")]


_GPT_TRANSCRIBE_VERBOSE_JSON_REFUSAL: Final = {
    "error": {
        "message": "response_format 'verbose_json' is not compatible with model 'gpt-transcribe'",
        "type": "invalid_request_error",
        "param": "response_format",
        "code": "unsupported_value",
    }
}
_UNDECODABLE_M4A_BYTES: Final = b"\x00\x00\x00\x20ftypM4A \x00\x00\x02\x00M4A isommp42\x00\x00\x00\x08free"
_DEPLOYMENT_COST_PER_SECOND: Final = 0.0003


def _openai_gpt_transcribe_reply(request: Request, reported_seconds: float) -> Reply:
    fields: Final = dict(_text_parts(_multipart_parts(request)))
    if fields.get("response_format") == "verbose_json":
        return Reply(status=400, body=json.dumps(_GPT_TRANSCRIBE_VERBOSE_JSON_REFUSAL).encode())
    return Reply(
        body=json.dumps({"text": "hello world", "usage": {"type": "duration", "seconds": reported_seconds}}).encode()
    )


def _uploaded_files(request: Request) -> tuple[tuple[str, str, bytes], ...]:
    return tuple((name, filename, payload) for name, filename, _, payload in _file_parts(_multipart_parts(request)))


def _silent_wav(seconds: float) -> bytes:
    output: Final = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * round(16000 * seconds))
    return output.getvalue()


def _advertised_cost_per_second(gateway: Gateway, model: str) -> float:
    entries: Final = gateway.get("/v2/model/info", {"model": model})["data"]
    assert isinstance(entries, list) and len(entries) == 1, entries
    price: Final = object_value(object_value(entries[0])["model_info"])["input_cost_per_second"]
    assert isinstance(price, float) and price > 0, entries
    return price


def _response_cost(response: httpx.Response) -> float:
    return float(response.headers["x-litellm-response-cost"]) if "x-litellm-response-cost" in response.headers else 0.0


def _logged_spend(call_id: str) -> float:
    rows: Final = eventually(
        lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return float(str(rows[0]["spend"]))


def test_gpt_transcribe_without_response_format_reaches_openai_without_verbose_json(gateway: Gateway) -> None:
    upload: Final = _UNDECODABLE_M4A_BYTES + uuid.uuid4().bytes

    with (
        wire_server(lambda request: _openai_gpt_transcribe_reply(request, 2)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/gpt-transcribe", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        with _sdk(gateway, "/v1", gateway.key) as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model, file=("a.m4a", upload, "audio/mp4")
            )
        requests: Final = wire.drain()

    assert response.status_code == 200, response.text
    assert JSON_OBJECT.validate_json(response.content)["text"] == "hello world", response.text
    assert [(request.method, request.target) for request in requests] == [("POST", "/v1/audio/transcriptions")]
    assert _text_parts(_multipart_parts(requests[0])) == (("model", "gpt-transcribe"),)
    assert _uploaded_files(requests[0]) == (("file", "a.m4a", upload),)


@pytest.mark.parametrize(
    "configured_cost_per_second",
    (None, _DEPLOYMENT_COST_PER_SECOND),
    ids=("cost_map_price", "deployment_price"),
)
def test_gpt_transcribe_unmeasurable_upload_bills_the_seconds_openai_reports(
    gateway: Gateway, configured_cost_per_second: float | None
) -> None:
    reported_seconds: Final = 2.0
    upload: Final = _UNDECODABLE_M4A_BYTES + uuid.uuid4().bytes

    with (
        wire_server(lambda request: _openai_gpt_transcribe_reply(request, reported_seconds)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = (
            scenario.model(
                model="openai/gpt-transcribe",
                api_base=f"{wire.url}/v1",
                api_key="synthetic-openai-key",
                input_cost_per_second=configured_cost_per_second,
            )
            if configured_cost_per_second is not None
            else scenario.model(
                model="openai/gpt-transcribe", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
            )
        )
        key: Final = scenario.key(models=[model])
        cost_per_second: Final = (
            configured_cost_per_second
            if configured_cost_per_second is not None
            else _advertised_cost_per_second(gateway, model)
        )
        response: Final = gateway.request_multipart(
            "/v1/audio/transcriptions",
            {"model": model, "response_format": "json"},
            {"file": ("a.m4a", upload, "audio/mp4")},
            key=key,
        )
        expected_cost: Final = reported_seconds * cost_per_second

        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["text"] == "hello world", response.text
        assert math.isclose(_response_cost(response), expected_cost), (_response_cost(response), expected_cost)
        assert math.isclose(_logged_spend(response.headers["x-litellm-call-id"]), expected_cost)
        key_rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s',
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) > 0,
            seconds=70,
        )
        assert math.isclose(float(str(key_rows[0]["spend"])), expected_cost), key_rows
        requests: Final = wire.drain()

    assert [(request.method, request.target) for request in requests] == [("POST", "/v1/audio/transcriptions")]
    assert _uploaded_files(requests[0]) == (("file", "a.m4a", upload),)


def test_gpt_transcribe_measured_upload_bills_the_measured_length_over_reported_seconds(gateway: Gateway) -> None:
    measured_seconds: Final = 1.0
    upload: Final = _silent_wav(measured_seconds)

    with (
        wire_server(lambda request: _openai_gpt_transcribe_reply(request, 7)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/gpt-transcribe",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-openai-key",
            input_cost_per_second=_DEPLOYMENT_COST_PER_SECOND,
        )
        response: Final = gateway.request_multipart(
            "/v1/audio/transcriptions",
            {"model": model, "response_format": "json"},
            {"file": ("a.wav", upload, _WAV_CONTENT_TYPE)},
        )
        expected_cost: Final = measured_seconds * _DEPLOYMENT_COST_PER_SECOND

        assert response.status_code == 200, response.text
        assert math.isclose(_response_cost(response), expected_cost), (_response_cost(response), expected_cost)
        assert math.isclose(_logged_spend(response.headers["x-litellm-call-id"]), expected_cost)
        requests: Final = wire.drain()

    assert [(request.method, request.target) for request in requests] == [("POST", "/v1/audio/transcriptions")]
    assert _uploaded_files(requests[0]) == (("file", "a.wav", upload),)

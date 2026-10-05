import json
import uuid
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

_WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)
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


def _sdk(gateway: Gateway, path: str) -> OpenAI:
    return OpenAI(
        base_url=str(gateway.client.base_url).rstrip("/") + path,
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


def test_audio_transcription_verbose_json_forwards_every_form_field_and_returns_the_transcript(
    gateway: Gateway,
) -> None:
    call_id: Final = f"audio-transcription-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == _VERBOSE_TEXT_PARTS
        assert _file_parts(parts) == (("file", "a.wav", "audio/x-wav", _WAV_BYTES),)
        return Reply(body=json.dumps(_VERBOSE_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-openai-key",
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
        )
        with _sdk(gateway, "/v1") as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", _WAV_BYTES, "audio/wav"),
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
        assert float(response.headers["x-litellm-response-cost"]) >= 0, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/v1/audio/transcriptions")
        ]


def test_audio_transcription_reaches_upstream_identically_on_prefixed_and_unprefixed_routes(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == _VERBOSE_TEXT_PARTS
        assert _file_parts(parts) == (("file", "a.wav", "audio/x-wav", _WAV_BYTES),)
        return Reply(body=json.dumps(_VERBOSE_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        prefixed: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        unprefixed: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        bodies: Final[list[dict]] = []
        for base_path, model in (("/v1", prefixed), ("", unprefixed)):
            with _sdk(gateway, base_path) as sdk:
                bodies.append(
                    json.loads(
                        sdk.audio.transcriptions.with_raw_response.create(
                            model=model,
                            file=("a.wav", _WAV_BYTES, "audio/wav"),
                            temperature=0.2,
                            language="en",
                            prompt="hi",
                            response_format="verbose_json",
                            timestamp_granularities=["word", "segment"],
                            include=["logprobs"],
                        ).content
                    )
                )
        assert bodies[0] == bodies[1] == _VERBOSE_TRANSCRIPT
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/v1/audio/transcriptions"),
            ("POST", "/v1/audio/transcriptions"),
        ]


def test_audio_transcription_json_format_returns_text_and_usage(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "whisper-1"), ("response_format", "json"))
        assert _file_parts(parts) == (("file", "a.wav", "audio/x-wav", _WAV_BYTES),)
        return Reply(body=json.dumps(_JSON_TRANSCRIPT).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        with _sdk(gateway, "/v1") as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", _WAV_BYTES, "audio/wav"),
                response_format="json",
            )
        assert response.status_code == 200, response.text
        assert json.loads(response.content) == _JSON_TRANSCRIPT, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/v1/audio/transcriptions")
        ]


@pytest.mark.parametrize("response_format", ("text", "srt", "vtt"))
def test_audio_transcription_forwards_plain_response_format_to_upstream(
    gateway: Gateway, response_format: str
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "whisper-1"), ("response_format", response_format))
        assert _file_parts(parts) == (("file", "a.wav", "audio/x-wav", _WAV_BYTES),)
        return Reply(body=_PLAIN_TRANSCRIPTS[response_format].encode(), content_type="text/plain")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        with _sdk(gateway, "/v1") as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", _WAV_BYTES, "audio/wav"),
                response_format=response_format,
            )
        assert response.status_code == 200, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/v1/audio/transcriptions")
        ]


@pytest.mark.parametrize("response_format", ("text", "srt", "vtt"))
def test_audio_transcription_plain_response_format_returns_the_raw_transcript(
    gateway: Gateway, response_format: str
) -> None:
    pytest.skip(
        "BUG: response_format=text/srt/vtt transcriptions return application/json "
        '{"text": ...} instead of the raw transcript, so the OpenAI SDK returns a JSON string as the transcript'
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        return Reply(body=_PLAIN_TRANSCRIPTS[response_format].encode(), content_type="text/plain")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        with _sdk(gateway, "/v1") as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", _WAV_BYTES, "audio/wav"),
                response_format=response_format,
            )
            assert response.headers["content-type"].startswith("text/plain"), response.text
            transcript: Final = sdk.audio.transcriptions.create(
                model=model,
                file=("a.wav", _WAV_BYTES, "audio/wav"),
                response_format=response_format,
            )
        assert transcript == _PLAIN_TRANSCRIPTS[response_format]

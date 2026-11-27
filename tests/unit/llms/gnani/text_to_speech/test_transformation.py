from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

_MODEL: Final = "gnani/timbre-v3.0"
_TEXT: Final = "नमस्ते <pauses> आज हम गणित सीखेंगे।"
_BODY: Final = TypeAdapter(Mapping[str, object])


def test_speech_sends_native_fields_and_returns_provider_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNANI_API_KEY", "env-key")
    monkeypatch.setenv("GNANI_API_BASE", "https://example.test")
    audio: Final = b"OggS\x00\xff"

    def synthesize(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.test/api/v1/tts/inference"
        assert request.headers["X-API-Key-ID"] == "env-key"
        assert _BODY.validate_json(request.content) == {
            "model": "timbre-v3.0",
            "text": _TEXT,
            "voice": "Shivani",
            "speed": 1.2,
            "language": "hi-IN",
            "audio_config": {"container": "ogg", "sample_rate": 24000},
        }
        return httpx.Response(200, headers={"Content-Type": "audio/ogg"}, content=audio)

    with httpx.Client(transport=httpx.MockTransport(synthesize)) as client:
        response: Final = litellm.speech(
            model=_MODEL,
            input=_TEXT,
            voice="Shivani",
            response_format="opus",
            speed=1.2,
            extra_body={"language": "hi-IN", "audio_config": {"sample_rate": 24000}, "model": "other"},
            client=HTTPHandler(client=client),
        )
    assert response.content == audio
    assert response.content_type == "audio/ogg"


@pytest.mark.asyncio
@pytest.mark.parametrize(("speed", "expected"), (("slow", 0.85), ("medium", 1.0), ("fast", 1.15), (0.92, 0.92)))
async def test_aspeech_preserves_native_pcm_and_maps_speed_shortcuts(speed: str | float, expected: float) -> None:
    config: Final = {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}

    def synthesize(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key-ID"] == "deployment-key"
        body: Final = _BODY.validate_json(request.content)
        assert body["speed"] == expected
        assert body["language"] == "auto"
        assert body["audio_config"] == config
        return httpx.Response(200, headers={"Content-Type": "application/octet-stream"}, content=b"\xfe\xff")

    async with httpx.AsyncClient(transport=httpx.MockTransport(synthesize)) as client:
        response: Final = await litellm.aspeech(
            model="gnani/timbre-v2.5",
            input="Hello",
            voice="Nalini",
            speed=speed,
            api_key="deployment-key",
            audio_config=config,
            client=AsyncHTTPHandler(client=client),
        )
    assert response.content == b"\xfe\xff"
    assert response.content_type == "audio/pcm"


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    (
        (400, "invalid speed", litellm.BadRequestError),
        (401, "invalid key", litellm.AuthenticationError),
        (429, "one active request", litellm.RateLimitError),
    ),
)
def test_provider_errors_keep_the_status_and_message(status: int, message: str, expected: type[Exception]) -> None:
    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"message": message} if status == 400 else {"detail": {"message": message}})

    with httpx.Client(transport=httpx.MockTransport(reject)) as client, pytest.raises(expected) as caught:
        litellm.speech(
            model=_MODEL, input="Hello", voice="Shivani", api_key="test-key", client=HTTPHandler(client=client)
        )
    assert message in str(caught.value)


@pytest.mark.parametrize(
    ("fields", "expected"),
    (
        ({"stream": True}, "stream"),
        ({"stream_format": "sse"}, "complete audio"),
        ({"response_format": "flac"}, "format"),
        ({"response_format": "wav", "audio_config": {"container": "mp3"}}, "conflicts"),
        ({"extra_headers": {"x-api-key-id": "other"}}, "api_key"),
    ),
)
def test_rejects_unfulfillable_speech_requests(fields: Mapping[str, object], expected: str) -> None:
    with pytest.raises((litellm.UnsupportedParamsError, litellm.BadRequestError), match=expected):
        litellm.speech(model=_MODEL, input="Hello", voice="Shivani", api_key="test-key", **fields)


@pytest.mark.parametrize(("mime", "content"), (("audio/wav", b""), ("application/json", b'{"success":false}')))
def test_invalid_success_payload_is_not_returned_as_audio(mime: str, content: bytes) -> None:
    def synthesize(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": mime}, content=content)

    with (
        httpx.Client(transport=httpx.MockTransport(synthesize)) as client,
        pytest.raises(litellm.BadGatewayError) as error,
    ):
        litellm.speech(
            model=_MODEL, input="Hello", voice="Shivani", api_key="test-key", client=HTTPHandler(client=client)
        )
    assert error.value.status_code == 502

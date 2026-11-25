import asyncio
import json
from email import policy
from email.parser import BytesParser
from typing import Final

import httpx
import pytest
from respx import MockRouter

import litellm
from litellm.cost_calculator import completion_cost
from litellm.types.utils import TranscriptionResponse

_MODEL: Final = "oruk/oruk-fourier-2b"
_BASE: Final = "https://speech-api.example.test/v1"
_AUDIO: Final = ("sample.wav", b"RIFF-sample-audio", "audio/wav")


def _form(request: httpx.Request) -> dict[str, bytes]:
    envelope: Final = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    message: Final = BytesParser(policy=policy.default).parsebytes(envelope)
    return {
        str(part.get_param("name", header="content-disposition")): part.get_payload(decode=True)
        for part in message.iter_parts()
    }


def test_sdk_sends_oruk_multipart_and_parses_json(respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORUK_API_BASE", _BASE)
    monkeypatch.setenv("ORUK_API_KEY", "env-key")
    route: Final = respx_mock.post(f"{_BASE}/audio/transcriptions").respond(
        json={"text": "Hello from Oruk Fourier 2B."}
    )
    result: Final = litellm.transcription(
        model=_MODEL,
        file=_AUDIO,
        api_key="request-key",
        language="en-us",
        response_format="json",
    )
    request: Final = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer request-key"
    assert b'filename="sample.wav"' in request.content
    assert _form(request) == {
        "file": _AUDIO[1],
        "model": b"oruk-fourier-2b",
        "language": b"en-US",
        "response_format": b"json",
    }
    assert isinstance(result, TranscriptionResponse)
    assert result.text == "Hello from Oruk Fourier 2B."


@pytest.mark.parametrize(
    ("raw_language", "expected"),
    [
        ("en", b"en"),
        ("en-us", b"en-US"),
        ("EN-GB", b"en-GB"),
        ("English", b"english"),
        ("AUTO", b"auto"),
    ],
)
def test_supported_language_hints_normalize(
    raw_language: str, expected: bytes, respx_mock: MockRouter
) -> None:
    route: Final = respx_mock.post(f"{_BASE}/audio/transcriptions").respond(json={"text": "ok"})
    litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE, language=raw_language)
    assert _form(route.calls.last.request)["language"] == expected


def test_verbose_json_preserves_duration_language_and_calculates_cost(respx_mock: MockRouter) -> None:
    respx_mock.post(f"{_BASE}/audio/transcriptions").respond(
        json={
            "id": "tr_01",
            "object": "audio.transcription",
            "model": "oruk-fourier-2b",
            "task": "transcription",
            "text": "Sixty seconds of English speech.",
            "language": "en",
            "duration": 60.0,
            "audio_seconds": 60.0,
            "usage": {
                "audio_seconds": 60.0,
                "billable_seconds": 60.0,
                "billing_unit": "audio_second",
                "rate_per_minute_usd": "0.0080",
                "estimated_cost_usd": "0.0080",
                "pricing_version": "2026-11",
            },
            "model_revision": "8c2a8871f1304aa2f634da4fb95e24438c4945b9",
        }
    )
    result: Final = litellm.transcription(
        model=_MODEL,
        file=_AUDIO,
        api_key="test",
        api_base=_BASE,
        response_format="verbose_json",
    )
    assert isinstance(result, TranscriptionResponse)
    assert result.text == "Sixty seconds of English speech."
    assert getattr(result, "language", None) == "en"
    assert getattr(result, "duration", None) == 60.0
    cost: Final = completion_cost(completion_response=result, model=_MODEL)
    assert cost == pytest.approx(0.008, rel=1e-5)


def test_plain_text_response_format(respx_mock: MockRouter) -> None:
    respx_mock.post(f"{_BASE}/audio/transcriptions").respond(
        content=b"Plain text transcript",
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )
    result: Final = litellm.transcription(
        model=_MODEL,
        file=_AUDIO,
        api_key="test",
        api_base=_BASE,
        response_format="text",
    )
    assert isinstance(result, TranscriptionResponse)
    assert result.text == "Plain text transcript"


@pytest.mark.asyncio
async def test_concurrent_async_calls_keep_payloads_separate(
    respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("ORUK_API_KEY", "env-key")

    def transcript(request: httpx.Request) -> httpx.Response:
        fields: Final = _form(request)
        assert request.headers["Authorization"] == "Bearer env-key"
        return httpx.Response(200, json={"text": (fields["language"] + b":" + fields["file"]).decode()})

    respx_mock.post(f"{_BASE}/audio/transcriptions").mock(side_effect=transcript)
    first, second = await asyncio.gather(
        litellm.atranscription(model=_MODEL, file=("a.wav", b"first"), api_base=_BASE, language="en"),
        litellm.atranscription(model=_MODEL, file=("b.wav", b"second"), api_base=_BASE, language="en-GB"),
    )
    assert isinstance(first, TranscriptionResponse) and first.text == "en:first"
    assert isinstance(second, TranscriptionResponse) and second.text == "en-GB:second"


def test_missing_key_fails_before_upload(respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORUK_API_KEY", raising=False)
    with pytest.raises(litellm.AuthenticationError, match="ORUK_API_KEY"):
        litellm.transcription(model=_MODEL, file=_AUDIO, api_base=_BASE)
    assert not respx_mock.calls


def test_empty_audio_fails_before_upload(respx_mock: MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="non-empty audio"):
        litellm.transcription(model=_MODEL, file=("empty.wav", b""), api_key="test", api_base=_BASE)
    assert not respx_mock.calls


@pytest.mark.parametrize(
    "options",
    [{"language": "fr"}, {"response_format": "srt"}, {"prompt": "Oruk"}, {"stream": True}],
)
def test_unsupported_options_rejected(options: dict[str, object], respx_mock: MockRouter) -> None:
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE, **options)
    assert not respx_mock.calls


def test_drop_params_drops_unsupported_options(respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(f"{_BASE}/audio/transcriptions").respond(json={"text": "ok"})
    litellm.transcription(
        model=_MODEL,
        file=_AUDIO,
        api_key="test",
        api_base=_BASE,
        language="fr",
        response_format="srt",
        prompt="Oruk",
        temperature=0.0,
        stream=True,
        drop_params=True,
    )
    assert _form(route.calls.last.request) == {
        "model": b"oruk-fourier-2b",
        "file": _AUDIO[1],
    }


@pytest.mark.parametrize("stream", [None, "", False, "false"])
def test_unset_or_false_stream_is_ignored(stream: str | bool | None, respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(f"{_BASE}/audio/transcriptions").respond(json={"text": "ok"})
    litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE, stream=stream)
    assert "stream" not in _form(route.calls.last.request)


@pytest.mark.parametrize("body", [b"not JSON", b"{}", b'{"text": 123}'])
def test_malformed_json_response_is_bad_gateway(respx_mock: MockRouter, body: bytes) -> None:
    respx_mock.post(f"{_BASE}/audio/transcriptions").respond(
        content=body,
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(litellm.BadGatewayError, match="Invalid Oruk transcription response"):
        litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE)


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, litellm.BadRequestError),
        (401, litellm.AuthenticationError),
        (403, litellm.PermissionDeniedError),
        (422, litellm.BadRequestError),
        (429, litellm.RateLimitError),
        (500, litellm.InternalServerError),
        (504, litellm.Timeout),
    ],
)
def test_http_errors_map_to_litellm_exceptions(
    respx_mock: MockRouter, status: int, error: type[Exception]
) -> None:
    respx_mock.post(f"{_BASE}/audio/transcriptions").respond(status, json={"error": {"message": "provider reason"}})
    with pytest.raises(error, match="provider reason") as failure:
        litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE)
    assert getattr(failure.value, "status_code", None) == status


def test_extra_headers_forwarded_without_overwriting_authorization(respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(f"{_BASE}/audio/transcriptions").respond(json={"text": "ok"})
    litellm.transcription(
        model=_MODEL,
        file=_AUDIO,
        api_key="real-key",
        api_base=_BASE,
        extra_headers={"X-Request-ID": "req-123", "Authorization": "Bearer spoofed"},
    )
    request: Final = route.calls.last.request
    assert request.headers["X-Request-ID"] == "req-123"
    assert request.headers.get_list("Authorization") == ["Bearer real-key"]


def test_discovery_and_environment_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORUK_API_KEY", raising=False)
    assert litellm.validate_environment(model=_MODEL)["missing_keys"] == ["ORUK_API_KEY"]
    monkeypatch.setenv("ORUK_API_KEY", "env-key")
    assert litellm.validate_environment(model=_MODEL) == {"keys_in_environment": True, "missing_keys": []}
    assert _MODEL in litellm.get_valid_models(custom_llm_provider="oruk", check_provider_endpoint=False)
    assert litellm.get_supported_openai_params(model=_MODEL, request_type="transcription") == [
        "language",
        "response_format",
    ]
    assert json.loads(json.dumps(litellm.model_cost[_MODEL]))["input_cost_per_second"] == pytest.approx(0.0001333333)

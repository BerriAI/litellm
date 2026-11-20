import asyncio
import json
from email import policy
from email.parser import BytesParser
from typing import Final

import httpx
import pytest
from respx import MockRouter

import litellm
from litellm.types.utils import TranscriptionResponse

_MODEL: Final = "gnani/prisma-v3.0"
_BASE: Final = "https://asr.example.test"
_AUDIO: Final = ("lecture.ogg", b"OggS-sample", "audio/ogg")


def _form(request: httpx.Request) -> dict[str, bytes]:
    envelope: Final = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    message: Final = BytesParser(policy=policy.default).parsebytes(envelope)
    return {
        str(part.get_param("name", header="content-disposition")): part.get_payload(decode=True)
        for part in message.iter_parts()
    }


def test_sdk_sends_gnani_multipart_and_normalizes_text(respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNANI_API_BASE", _BASE)
    monkeypatch.setenv("GNANI_API_KEY", "env-key")
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(
        json={"success": True, "transcript": "नमस्ते", "request_id": "req"}
    )
    result: Final = litellm.transcription(
        model=_MODEL,
        file=_AUDIO,
        api_key="request-key",
        language="hi,en-IN",
        response_format="json",
        format="codemix",
        keyterms=["एक", "one"],
        topic="math",
        previous_text="नमस्ते",
        speaker_names=["Ada"],
        enable_substitution=True,
        substitution_map=[{"patterns": ["one"], "replacement": "1"}],
    )
    request: Final = route.calls.last.request
    assert request.headers["X-API-Key-ID"] == "request-key"
    assert b'filename="lecture.ogg"' in request.content
    assert _form(request) == {
        "audio_file": _AUDIO[1],
        "model": _MODEL.split("/")[1].encode(),
        "language_code": b"hi-IN,en-IN",
        "format": b"codemix",
        "keyterms": json.dumps(["एक", "one"], ensure_ascii=False).encode(),
        "topic": b"math",
        "previous_text": "नमस्ते".encode(),
        "speaker_names": b'["Ada"]',
        "enable_substitution": b"true",
        "substitution_map": b'[{"patterns": ["one"], "replacement": "1"}]',
    }
    assert isinstance(result, TranscriptionResponse)
    assert result.model_dump(exclude_none=True) == {"text": "नमस्ते", "request_id": "req"}


def test_response_returns_gnani_language_request_id_and_truncation(respx_mock: MockRouter) -> None:
    respx_mock.post(f"{_BASE}/stt/v3").respond(
        json={
            "success": True,
            "transcript": "नमस्ते",
            "request_id": "req-1",
            "detected_language_code": "hi-IN",
            "truncated": True,
        }
    )
    result: Final = litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE)
    assert result.model_dump(exclude_none=True) == {
        "text": "नमस्ते",
        "language": "hi-IN",
        "request_id": "req-1",
        "truncated": True,
    }


def test_prisma_v25_returns_the_formatted_transcript(respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(
        json={"success": True, "transcript": "₹500", "output": {"literal": "five hundred rupees"}}
    )
    result: Final = litellm.transcription(
        model="gnani/prisma-v2.5",
        file=_AUDIO,
        api_base=_BASE,
        api_key="test",
        language="en",
        format="transcribe",
        bias_list=["rupees"],
        bias_score=1.5,
        itn_native_numerals=False,
    )
    assert _form(route.calls.last.request) == {
        "model": b"prisma-v2.5",
        "language_code": b"en-IN",
        "audio_file": _AUDIO[1],
        "format": b"transcribe",
        "bias_list": b'["rupees"]',
        "bias_score": b"1.5",
        "itn_native_numerals": b"false",
    }
    assert result.text == "₹500"


@pytest.mark.parametrize("language", [None, "indic-auto", "INDIC-AUTO", "en,hi", " "])
def test_prisma_v25_requires_one_explicit_language(language: str | None, respx_mock: MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="one explicit language"):
        litellm.transcription(model="gnani/prisma-v2.5", file=_AUDIO, api_base=_BASE, api_key="test", language=language)
    assert not respx_mock.calls


def test_prisma_v25_reports_unsupported_context(respx_mock: MockRouter) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match="topic") as failure:
        litellm.transcription(
            model="gnani/prisma-v2.5",
            file=_AUDIO,
            api_base=_BASE,
            api_key="test",
            language="hi",
            extra_body={"topic": "math"},
        )
    assert "GnaniException" not in str(failure.value)
    assert not respx_mock.calls


@pytest.mark.parametrize(
    ("model", "options", "kept"),
    [
        ("gnani/prisma-v2.5", {"language": "hi", "topic": "math", "bias_list": ["x"]}, {"bias_list": b'["x"]'}),
        ("gnani/prisma-v3.0", {"bias_score": 1.5, "keyterms": ["x"]}, {"keyterms": b'["x"]'}),
    ],
)
def test_drop_params_drops_fields_the_model_does_not_support(
    model: str, options: dict[str, object], kept: dict[str, bytes], respx_mock: MockRouter
) -> None:
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(json={"success": True, "transcript": "ok"})
    litellm.transcription(model=model, file=_AUDIO, api_key="test", api_base=_BASE, drop_params=True, **options)
    sent: Final = _form(route.calls.last.request)
    assert {key: sent[key] for key in kept} == kept
    assert not {"topic", "bias_score"} & sent.keys()


@pytest.mark.asyncio
async def test_concurrent_calls_keep_language_and_audio_separate(
    respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("GNANI_API_KEY", "env-key")

    def transcript(request: httpx.Request) -> httpx.Response:
        fields: Final = _form(request)
        assert request.headers["X-API-Key-ID"] == "env-key"
        return httpx.Response(
            200, json={"success": True, "transcript": (fields["language_code"] + fields["audio_file"]).decode()}
        )

    respx_mock.post(f"{_BASE}/stt/v3").mock(side_effect=transcript)
    first, second = await asyncio.gather(
        litellm.atranscription(model=_MODEL, file=("a.wav", b"-first"), api_base=_BASE),
        litellm.atranscription(model=_MODEL, file=("b.wav", b"-second"), api_base=_BASE, language="en"),
    )
    assert isinstance(first, TranscriptionResponse) and first.text == "indic-auto-first"
    assert isinstance(second, TranscriptionResponse) and second.text == "en-IN-second"


def test_silence_is_a_successful_empty_transcription(respx_mock: MockRouter) -> None:
    respx_mock.post(f"{_BASE}/stt/v3").respond(json={"success": True, "transcript": ""})
    result: Final = litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE)
    assert isinstance(result, TranscriptionResponse) and result.text == ""


@pytest.mark.parametrize(
    "body",
    [b"not JSON", b'{"success":false,"transcript":""}', b'{"success":true}', b'{"success":true,"transcript":42}'],
)
def test_malformed_success_is_a_gateway_error(respx_mock: MockRouter, body: bytes) -> None:
    respx_mock.post(f"{_BASE}/stt/v3").respond(content=body)
    with pytest.raises(litellm.BadGatewayError, match="Invalid Gnani transcription response"):
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
def test_provider_failure_keeps_its_reason(respx_mock: MockRouter, status: int, error: type[Exception]) -> None:
    respx_mock.post(f"{_BASE}/stt/v3").respond(status, json={"error": {"message": "provider reason"}})
    with pytest.raises(error, match="provider reason") as failure:
        litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE)
    assert getattr(failure.value, "status_code", None) == status


@pytest.mark.parametrize(
    "options",
    [
        {"stream": True},
        {"response_format": "verbose_json"},
        {"temperature": 0},
        {"chunking_strategy": "auto"},
        {"include": ["logprobs"]},
        {"timestamp_granularities": ["word"]},
        {"known_speaker_names": ["Ada"]},
        {"known_speaker_references": ["data:audio/wav;base64,AA=="]},
    ],
)
def test_unsupported_openai_options_are_rejected(options: dict[str, object], respx_mock: MockRouter) -> None:
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE, **options)
    assert not respx_mock.calls


def test_prisma_v25_rejects_prompt(respx_mock: MockRouter) -> None:
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.transcription(
            model="gnani/prisma-v2.5", file=_AUDIO, api_key="test", api_base=_BASE, language="hi", prompt="context"
        )
    assert not respx_mock.calls


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"prompt": "earlier clip"}, b"earlier clip"),
        ({"prompt": "earlier clip", "previous_text": "explicit"}, b"explicit"),
        ({"prompt": "earlier clip", "extra_body": {"previous_text": "explicit"}}, b"explicit"),
    ],
)
def test_prompt_becomes_previous_text_unless_set(
    options: dict[str, object], expected: bytes, respx_mock: MockRouter
) -> None:
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(json={"success": True, "transcript": "ok"})
    litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE, **options)
    sent: Final = _form(route.calls.last.request)
    assert sent["previous_text"] == expected
    assert "prompt" not in sent


@pytest.mark.parametrize("stream", [None, "", False, "false"])
def test_unset_or_false_stream_is_not_rejected(stream: str | bool | None, respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(json={"success": True, "transcript": "ok"})
    litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE, stream=stream)
    assert "stream" not in _form(route.calls.last.request)


@pytest.mark.parametrize(("model", "language"), [("gnani/prisma-v3.0", None), ("gnani/prisma-v2.5", "hi")])
def test_fields_litellm_does_not_know_reach_gnani(model: str, language: str | None, respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(json={"success": True, "transcript": "ok"})
    litellm.transcription(
        model=model,
        file=_AUDIO,
        api_key="test",
        api_base=_BASE,
        language=language,
        diarize=True,
        extra_body={"keyterm": "Lincoln"},
    )
    sent: Final = _form(route.calls.last.request)
    assert {key: sent.get(key) for key in ("diarize", "keyterm")} == {"diarize": b"true", "keyterm": b"Lincoln"}


@pytest.mark.parametrize(
    ("model", "options", "expected"),
    [
        ("gnani/prisma-v2.5", {"language_code": "en-IN"}, b"en-IN"),
        ("gnani/prisma-v3.0", {"language_code": "hi-in"}, b"hi-IN"),
        ("gnani/prisma-v3.0", {"language": "en", "language_code": "hi-IN"}, b"hi-IN"),
        ("gnani/prisma-v3.0", {"language": "en", "language_code": ""}, b"en-IN"),
    ],
)
def test_gnani_language_code_is_normalized_and_wins_over_language(
    model: str, options: dict[str, object], expected: bytes, respx_mock: MockRouter
) -> None:
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(json={"success": True, "transcript": "ok"})
    litellm.transcription(model=model, file=_AUDIO, api_key="test", api_base=_BASE, **options)
    assert _form(route.calls.last.request)["language_code"] == expected


def test_global_drop_params_drops_unsupported_options(respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", True)
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(json={"success": True, "transcript": "ok"})
    litellm.transcription(
        model=_MODEL,
        file=_AUDIO,
        api_key="test",
        api_base=_BASE,
        response_format="srt",
        temperature=0,
        stream=True,
        languages=["en"],
        keywords=["x"],
        timestamp_granularities=["word"],
        chunking_strategy="auto",
        include=["logprobs"],
        known_speaker_names=["Ada"],
        known_speaker_references=["data:audio/wav;base64,AA=="],
    )
    assert _form(route.calls.last.request) == {
        "language_code": b"indic-auto",
        "model": b"prisma-v3.0",
        "audio_file": _AUDIO[1],
    }


def test_missing_key_fails_before_upload(respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    with pytest.raises(litellm.AuthenticationError, match="GNANI_API_KEY") as failure:
        litellm.transcription(model=_MODEL, file=_AUDIO, api_base=_BASE)
    assert "GnaniException" not in str(failure.value)
    assert not respx_mock.calls


@pytest.mark.parametrize(
    ("model", "params"),
    [(_MODEL, ["language", "response_format", "prompt"]), ("gnani/prisma-v2.5", ["language", "response_format"])],
)
def test_discovery_lists_transcription_for_gnani(
    model: str, params: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    assert litellm.validate_environment(model=model)["missing_keys"] == ["GNANI_API_KEY"]
    monkeypatch.setenv("GNANI_API_KEY", "env-key")
    assert litellm.validate_environment(model=model) == {"keys_in_environment": True, "missing_keys": []}
    assert model in litellm.get_valid_models(custom_llm_provider="gnani", check_provider_endpoint=False)
    assert litellm.get_supported_openai_params(model=model, request_type="transcription") == params


def test_dropping_unsupported_options_preserves_native_hints(respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(json={"success": True, "transcript": "ok"})
    litellm.transcription(
        model=_MODEL,
        file=_AUDIO,
        api_key="test",
        api_base=_BASE,
        response_format="srt",
        drop_params=True,
        extra_body={"keyterms": ["one"], "model": "unexpected"},
    )
    assert _form(route.calls.last.request) == {
        "keyterms": b'["one"]',
        "language_code": b"indic-auto",
        "model": b"prisma-v3.0",
        "audio_file": _AUDIO[1],
    }


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("", b"indic-auto"),
        (" , ", b"indic-auto"),
        ("hi,", b"hi-IN"),
        ("INDIC-AUTO", b"indic-auto"),
        ("Indic-Auto", b"indic-auto"),
        ("hi-in", b"hi-IN"),
        ("EN-in,ta", b"en-IN,ta-IN"),
    ],
)
def test_prisma_v3_normalizes_language_hints(language: str, expected: bytes, respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(json={"success": True, "transcript": "ok"})
    litellm.transcription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE, language=language)
    assert _form(route.calls.last.request)["language_code"] == expected


def test_empty_audio_stops_before_upload(respx_mock: MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError) as failure:
        litellm.transcription(model=_MODEL, file=("clip.wav", b""), api_key="test", api_base=_BASE)
    assert "GnaniException" not in str(failure.value)
    assert not respx_mock.calls


def test_unlisted_prisma_model_gets_current_model_behavior(respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(f"{_BASE}/stt/v3").respond(json={"success": True, "transcript": "ok"})
    litellm.transcription(
        model="gnani/prisma-v3.5", file=_AUDIO, api_key="test", api_base=_BASE, prompt="earlier", keyterms=["x"]
    )
    assert _form(route.calls.last.request) == {
        "model": b"prisma-v3.5",
        "language_code": b"indic-auto",
        "previous_text": b"earlier",
        "keyterms": b'["x"]',
        "audio_file": _AUDIO[1],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"), [(httpx.ConnectError, litellm.APIConnectionError), (httpx.ReadTimeout, litellm.Timeout)]
)
async def test_transport_failure_reaches_sdk(
    respx_mock: MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[httpx.TransportError],
    expected: type[Exception],
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    respx_mock.post(f"{_BASE}/stt/v3").mock(side_effect=failure("network unavailable"))
    with pytest.raises(expected):
        await litellm.atranscription(model=_MODEL, file=_AUDIO, api_key="test", api_base=_BASE, timeout=3)

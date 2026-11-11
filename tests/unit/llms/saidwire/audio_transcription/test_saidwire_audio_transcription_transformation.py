import io
import wave
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException

TRANSCRIPTIONS_URL: Final = "https://api.saidwire.com/v1/audio/transcriptions"
JSON_BODY: Final = {"text": "The quick brown fox.", "usage": {"type": "duration", "seconds": 12.0}}


def _wav() -> tuple[str, bytes]:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return ("clip.wav", buf.getvalue())


def _form(request: httpx.Request) -> bytes:
    return request.read()


@pytest.fixture(autouse=True)
def _no_saidwire_env(monkeypatch):
    monkeypatch.delenv("SAIDWIRE_API_KEY", raising=False)
    monkeypatch.delenv("SAIDWIRE_API_BASE", raising=False)


def test_transcription_posts_to_saidwire_with_bearer_key(respx_mock: respx.MockRouter):
    route = respx_mock.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=JSON_BODY))

    response = litellm.transcription(model="saidwire/saidwire-en", file=_wav(), api_key="sk_test_abc")

    assert response.text == JSON_BODY["text"]
    assert route.call_count == 1
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk_test_abc"
    body = _form(request)
    assert b'name="model"\r\n\r\nsaidwire-en' in body
    assert b'filename="clip.wav"' in body


def test_transcription_reads_key_and_base_from_env(respx_mock: respx.MockRouter, monkeypatch):
    monkeypatch.setenv("SAIDWIRE_API_KEY", "sk_env_key")
    monkeypatch.setenv("SAIDWIRE_API_BASE", "https://us-east.example.test/v1/")
    route = respx_mock.post("https://us-east.example.test/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=JSON_BODY)
    )

    litellm.transcription(model="saidwire/saidwire-en", file=_wav())

    assert route.calls.last.request.headers["Authorization"] == "Bearer sk_env_key"


def test_transcription_without_a_key_names_the_env_var(respx_mock: respx.MockRouter):
    route = respx_mock.post(TRANSCRIPTIONS_URL)

    with pytest.raises(BaseLLMException, match="SAIDWIRE_API_KEY") as exc_info:
        litellm.transcription(model="saidwire/saidwire-en", file=_wav())

    assert exc_info.value.status_code == 401
    assert route.call_count == 0


def test_caller_headers_cannot_replace_the_provider_key(respx_mock: respx.MockRouter):
    route = respx_mock.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=JSON_BODY))

    litellm.transcription(
        model="saidwire/saidwire-en",
        file=_wav(),
        api_key="sk_real",
        extra_headers={"Authorization": "Bearer someone-else"},
    )

    assert route.calls.last.request.headers["Authorization"] == "Bearer sk_real"


def test_openai_params_reach_the_form_and_granularities_use_array_syntax(respx_mock: respx.MockRouter):
    route = respx_mock.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=JSON_BODY))

    litellm.transcription(
        model="saidwire/saidwire-en",
        file=_wav(),
        api_key="k",
        language="en",
        prompt="Saidwire",
        temperature=0,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
    )

    body = _form(route.calls.last.request)
    assert b'name="language"\r\n\r\nen' in body
    assert b'name="prompt"\r\n\r\nSaidwire' in body
    assert b'name="response_format"\r\n\r\nverbose_json' in body
    assert body.count(b'name="timestamp_granularities[]"') == 2


def test_verbose_json_keeps_segments_words_and_duration(respx_mock: respx.MockRouter):
    verbose: Final = {
        "task": "transcribe",
        "language": "english",
        "duration": 2.0,
        "text": "hello world",
        "segments": [{"id": 0, "start": 0.1, "end": 1.9, "text": " hello world"}],
        "words": [{"word": "hello", "start": 0.1, "end": 0.6}, {"word": "world", "start": 0.7, "end": 1.9}],
        "usage": {"type": "duration", "seconds": 2.0},
    }
    respx_mock.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=verbose))

    response = litellm.transcription(
        model="saidwire/saidwire-en", file=_wav(), api_key="k", response_format="verbose_json"
    )

    assert response["segments"] == verbose["segments"]
    assert response["words"] == verbose["words"]
    assert response["duration"] == verbose["duration"]
    assert response["language"] == verbose["language"]


@pytest.mark.parametrize("fmt", ["text", "srt", "vtt"])
def test_plain_text_formats_come_back_as_text(respx_mock: respx.MockRouter, fmt: str):
    body: Final = "1\n00:00:00,100 --> 00:00:01,900\nhello world\n"
    respx_mock.post(TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/plain; charset=utf-8"})
    )

    response = litellm.transcription(model="saidwire/saidwire-en", file=_wav(), api_key="k", response_format=fmt)

    assert response.text == body


def test_cost_uses_billed_seconds_and_the_cost_map_rate(respx_mock: respx.MockRouter):
    respx_mock.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=JSON_BODY))

    response = litellm.transcription(model="saidwire/saidwire-en", file=_wav(), api_key="k")

    rate: Final = litellm.model_cost["saidwire/saidwire-en"]["input_cost_per_second"]
    assert response._hidden_params["response_cost"] == pytest.approx(JSON_BODY["usage"]["seconds"] * rate)


@pytest.mark.parametrize("status", [400, 401, 429, 503])
def test_upstream_errors_keep_status_and_message(respx_mock: respx.MockRouter, status: int):
    respx_mock.post(TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(
            status, json={"error": {"message": "upstream said no", "type": "x", "param": None, "code": "y"}}
        )
    )

    with pytest.raises(BaseLLMException, match="upstream said no") as exc_info:
        litellm.transcription(model="saidwire/saidwire-en", file=_wav(), api_key="k", max_retries=0)

    assert exc_info.value.status_code == status


@pytest.mark.asyncio
async def test_atranscription_posts_to_saidwire(respx_mock: respx.MockRouter, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=JSON_BODY))

    response = await litellm.atranscription(model="saidwire/saidwire-en", file=_wav(), api_key="sk_async")

    assert response.text == JSON_BODY["text"]
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk_async"


def test_saidwire_is_listed_for_its_models_and_transcription_params():
    assert "saidwire/saidwire-en" in litellm.models_by_provider["saidwire"]
    assert "timestamp_granularities" in litellm.get_supported_openai_params(
        model="saidwire-en", custom_llm_provider="saidwire", request_type="transcription"
    )

"""Live e2e: POST /v1/audio/speech returns audio, non-streamed and streamed.

The non-streamed call asserts an audio (not JSON) body. The streamed call consumes
the response the way a player would and asserts customer-observable streaming:
chunked transfer encoding (a buffered body would carry a content-length) with
non-zero audio bytes.
"""

from __future__ import annotations

import pytest
from e2e_config import unique_marker
from e2e_http import assert_client_error, require_successful_call
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from pydantic import BaseModel

pytestmark = pytest.mark.e2e


class _OptionalSpeechBody(BaseModel):
    model: str | None = None
    input: str | None = None
    voice: str | None = None


def _register_tts(
    endpoints_client: EndpointsClient, resources: ResourceManager
) -> tuple[str, str]:
    model = f"e2e-speech-{unique_marker()}"
    model_id = endpoints_client.create_model(
        model,
        LiteLLMParamsBody(model="openai/gpt-4o-mini-tts", api_key="os.environ/OPENAI_API_KEY"),
    )
    resources.defer(lambda: endpoints_client.delete_model(model_id))
    return model, resources.key()


class TestAudioSpeech:
    @pytest.mark.covers("llm.audio_speech.openai.basic.nonstream.works")
    def test_audio_speech_returns_audio(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register_tts(endpoints_client, resources)
        result = endpoints_client.audio_speech(key, model, "Hello!")
        require_successful_call(result)
        assert "audio" in (result.content_type or ""), (
            f"/audio/speech content-type is not audio: {result.content_type!r}"
        )
        assert result.body, "/audio/speech returned an empty body"

    @pytest.mark.covers("llm.audio_speech.openai.basic.stream.works")
    def test_audio_speech_streams_audio_chunks(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register_tts(endpoints_client, resources)
        result = endpoints_client.audio_speech_stream(
            key,
            model,
            "Streaming speech should arrive in several audio chunks so a client can "
            "begin playback well before the whole clip has finished generating.",
        )
        assert result.ok, (
            f"/audio/speech stream failed (status {result.status_code}); body={result.error_body}"
        )
        assert "audio" in (result.content_type or ""), (
            f"/audio/speech content-type is not audio: {result.content_type!r}"
        )
        assert result.chunked, (
            f"/audio/speech did not stream: transfer-encoding={result.transfer_encoding!r}, "
            f"content-length={result.content_length!r} (a buffered body is not a stream)"
        )
        assert result.content_length is None, (
            f"/audio/speech advertised content-length={result.content_length!r} on a "
            f"streamed response (a buffered body is not a stream)"
        )
        assert result.total_bytes > 0, "/audio/speech stream returned no audio bytes"

    @pytest.mark.skip(reason="stage red: product gap, /v1/audio/speech 500s on missing input instead of 400")
    @pytest.mark.covers("llm.audio_speech.openai.input_validation.nonstream.works")
    def test_missing_input_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register_tts(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/audio/speech",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalSpeechBody(model=model, voice="alloy"),
        )
        assert_client_error(result, "speech missing input")

    @pytest.mark.skip(reason="stage red: product gap, /v1/audio/speech 500s on missing model instead of 400")
    @pytest.mark.covers("llm.audio_speech.openai.input_validation.nonstream.works")
    def test_missing_model_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        _, key = _register_tts(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/audio/speech",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalSpeechBody(input="hello", voice="alloy"),
        )
        assert_client_error(result, "speech missing model")

    @pytest.mark.skip(reason="stage red: product gap, /v1/audio/speech 500s on invalid voice instead of surfacing the provider 4xx")
    @pytest.mark.covers("llm.audio_speech.openai.input_validation.nonstream.works")
    def test_invalid_voice_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register_tts(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/audio/speech",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalSpeechBody(model=model, input="hello", voice="invalid_voice_xyz"),
        )
        assert_client_error(result, "speech invalid voice")

    @pytest.mark.skip(reason="stage red: product gap, /v1/audio/speech 500s on empty input instead of surfacing the provider 4xx")
    @pytest.mark.covers("llm.audio_speech.openai.input_validation.nonstream.works")
    def test_empty_input_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register_tts(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/audio/speech",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalSpeechBody(model=model, input="", voice="alloy"),
        )
        assert_client_error(result, "speech empty input")

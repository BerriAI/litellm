"""Live e2e: POST /v1/audio/speech returns audio, non-streamed and streamed.

The non-streamed call asserts an audio (not JSON) body. The streamed call consumes
the response the way a player would and asserts customer-observable streaming:
chunked transfer encoding (a buffered body would carry a content-length) with
non-zero audio bytes.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


class TestAudioSpeech:
    @pytest.mark.covers("llm.audio_speech.openai.basic.nonstream.works")
    def test_audio_speech_returns_audio(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-speech-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/gpt-4o-mini-tts", api_key="os.environ/OPENAI_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

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
        model = f"e2e-speech-stream-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/gpt-4o-mini-tts", api_key="os.environ/OPENAI_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.audio_speech_stream(
            key,
            model,
            "Streaming speech should arrive in several audio chunks so a client can "
            "begin playback well before the whole clip has finished generating.",
        )
        assert result.ok, (
            f"/audio/speech stream failed (status {result.status_code})"
        )
        assert "audio" in (result.content_type or ""), (
            f"/audio/speech content-type is not audio: {result.content_type!r}"
        )
        assert result.chunked, (
            f"/audio/speech did not stream: transfer-encoding={result.transfer_encoding!r}, "
            f"content-length={result.content_length!r} (a buffered body is not a stream)"
        )
        assert result.total_bytes > 0, "/audio/speech stream returned no audio bytes"

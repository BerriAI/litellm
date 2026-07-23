"""Live e2e: POST /v1/audio/speech returns audio, non-streamed and streamed.

Both calls go through the real OpenAI SDK (LIT-4577). The non-streamed call
asserts an audio (not JSON) body. The streamed call consumes the response the
way a player would and asserts customer-observable streaming: chunked transfer
encoding (a buffered body would carry a content-length) with non-zero audio
bytes.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients, response_header

pytestmark = pytest.mark.e2e


def _register(proxy: ProxyClient, resources: ResourceManager, prefix: str) -> str:
    model = f"{prefix}-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(model="openai/gpt-4o-mini-tts", api_key="os.environ/OPENAI_API_KEY"),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


class TestAudioSpeech:
    @pytest.mark.covers("llm.audio_speech.openai.basic.nonstream.works")
    def test_audio_speech_returns_audio(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, "e2e-speech")
        client = sdk.openai(resources.key())

        response = client.audio.speech.with_raw_response.create(
            model=model, voice="alloy", input="Hello!"
        )
        content_type = response_header(response.headers, "content-type")
        assert "audio" in (content_type or ""), (
            f"/audio/speech content-type is not audio: {content_type!r}"
        )
        assert response.content, "/audio/speech returned an empty body"

    @pytest.mark.covers("llm.audio_speech.openai.basic.stream.works")
    def test_audio_speech_streams_audio_chunks(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, "e2e-speech-stream")
        client = sdk.openai(resources.key())

        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice="alloy",
            input=(
                "Streaming speech should arrive in several audio chunks so a client can "
                "begin playback well before the whole clip has finished generating."
            ),
        ) as response:
            content_type = response_header(response.headers, "content-type")
            transfer_encoding = response_header(response.headers, "transfer-encoding")
            content_length = response_header(response.headers, "content-length")
            total_bytes = sum(len(chunk) for chunk in response.iter_bytes(chunk_size=8192))

        assert "audio" in (content_type or ""), (
            f"/audio/speech content-type is not audio: {content_type!r}"
        )
        assert "chunked" in (transfer_encoding or ""), (
            f"/audio/speech did not stream: transfer-encoding={transfer_encoding!r}, "
            f"content-length={content_length!r} (a buffered body is not a stream)"
        )
        assert content_length is None, (
            f"/audio/speech advertised content-length={content_length!r} on a "
            f"streamed response (a buffered body is not a stream)"
        )
        assert total_bytes > 0, "/audio/speech stream returned no audio bytes"

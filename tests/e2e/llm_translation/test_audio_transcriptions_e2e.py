"""Live e2e: POST /v1/audio/transcriptions turns speech into text.

Registers an OpenAI speech-to-text deployment at runtime and uploads a spoken
weather question (the realtime suite's 24kHz WAV fixture) through the real
OpenAI SDK (LIT-4577), asserting the returned transcript is non-empty and
mentions the word it was asked about.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

WEATHER_WAV = (
    Path(__file__).resolve().parent / "realtime" / "fixtures" / "weather_question_24k.wav"
)


class TestAudioTranscriptions:
    @pytest.mark.covers("llm.audio_transcriptions.openai.basic.nonstream.works")
    def test_audio_transcriptions_returns_text(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = f"e2e-transcribe-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/gpt-4o-mini-transcribe", api_key="os.environ/OPENAI_API_KEY"
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        client = sdk.openai(resources.key())

        transcription = client.audio.transcriptions.create(
            model=model, file=(WEATHER_WAV.name, WEATHER_WAV.read_bytes(), "audio/wav")
        )
        text = transcription.text.strip()
        assert text, "/audio/transcriptions returned an empty transcript"
        assert "weather" in text.lower(), (
            f"transcript of a spoken weather question does not mention weather: {text!r}"
        )

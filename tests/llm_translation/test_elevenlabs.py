import os

import pytest

import litellm
from base_audio_transcription_unit_tests import BaseLLMAudioTranscriptionTest

os.environ.setdefault("ELEVENLABS_API_KEY", "test-elevenlabs-key")


class TestElevenLabsAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> dict:
        return {
            "model": "elevenlabs/scribe_v1",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.ELEVENLABS

class TestElevenLabsTextToSpeechTransformation:
    @pytest.fixture(scope="class")
    def config(self):
        from litellm.llms.elevenlabs.text_to_speech.transformation import (
            ElevenLabsTextToSpeechConfig,
        )

        return ElevenLabsTextToSpeechConfig()

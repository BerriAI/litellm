import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from base_audio_transcription_unit_tests import BaseLLMAudioTranscriptionTest


class TestDeepgramAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> dict:
        return {
            "model": "deepgram/nova-2",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.DEEPGRAM

import os
from typing import Dict

import litellm
import pytest

from tests.llm_translation.base_audio_transcription_unit_tests import (
    BaseLLMAudioTranscriptionTest,
)

@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set, skipping Mistral audio transcription tests",
)
class TestMistralAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> Dict:
        return {
            "model": "mistral/voxtral-mini-latest",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.MISTRAL


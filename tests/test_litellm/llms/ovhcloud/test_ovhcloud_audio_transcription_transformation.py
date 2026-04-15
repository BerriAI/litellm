import os
from typing import Dict

import litellm
import pytest

from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.ovhcloud.audio_transcription.transformation import OVHCloudAudioTranscriptionConfig
from litellm.types.utils import TranscriptionResponse
from litellm.utils import ProviderConfigManager
from tests.llm_translation.base_audio_transcription_unit_tests import (
    BaseLLMAudioTranscriptionTest,
)
from unittest.mock import MagicMock
import httpx


@pytest.mark.skipif(
    not os.getenv("OVHCLOUD_API_KEY"),
    reason="OVHCLOUD_API_KEY not set, skipping OVHCloud audio transcription tests",
)
class TestOVHCloudAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> Dict:
        return {
            "model": "ovhcloud/whisper-large-v3-turbo",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.OVHCLOUD

    # Override the async base test with a sync no-op to avoid
    # 'async def functions are not natively supported' failures when
    # running this file in isolation without pytest-asyncio.
    def test_audio_transcription_async(self):  # type: ignore[override]
        pytest.skip(
            "Async audio transcription test for OVHCloud is skipped in this suite; "
            "async test plugins (e.g. pytest-asyncio/anyio) are not configured here."
        )


@pytest.mark.skipif(
    not os.getenv("OVHCLOUD_API_KEY"),
    reason="OVHCLOUD_API_KEY not set, skipping OVHCloud audio transcription config test",
)
def test_ovhcloud_audio_transcription_config_installed():
    """
    Ensure OVHCloud audio transcription config is registered with ProviderConfigManager.
    """
    model = "ovhcloud/whisper-large-v3-turbo"
    provider = litellm.LlmProviders.OVHCLOUD

    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model=model,
        provider=provider,
    )

    assert config is not None
    assert isinstance(config, BaseAudioTranscriptionConfig)

def test_ovhcloud_audio_transcription_response_transform_diarized():
    """Test that diarized responses preserve segments and language."""
    config = OVHCloudAudioTranscriptionConfig()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {
        "text": "Hello, how are you? I am fine.",
        "language": "en",
        "segments": [
            {
                "text": "Hello, how are you?",
                "start": 0.3,
                "end": 2.1
            },
            {
                "text": "I am fine.",
                "start": 2.5,
                "end": 3.8
            },
        ],
        "usage": {
            "type": "duration",
            "duration": 5,
            "seconds": 5,
        },
    }

    response = config.transform_audio_transcription_response(mock_response)

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "Hello, how are you? I am fine."
    assert response["segments"] is not None
    assert len(response["segments"]) == 2
    assert response["segments"][0]["text"] == "Hello, how are you?"
    assert response["segments"][1]["text"] == "I am fine."
    assert response["language"] == "en"
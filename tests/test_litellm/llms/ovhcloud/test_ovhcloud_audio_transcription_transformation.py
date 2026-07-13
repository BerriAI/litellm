import os
from typing import Dict

import litellm
import pytest

from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.utils import ProviderConfigManager
from tests.llm_translation.base_audio_transcription_unit_tests import (
    BaseLLMAudioTranscriptionTest,
)


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



class TestOVHCloudDurationFieldMigration:
    """Tests for OVHCloud duration -> seconds field migration."""

    def test_seconds_field_mapped_to_duration(self):
        """New `seconds` field should be normalized to `duration`."""
        from litellm.llms.ovhcloud.audio_transcription.transformation import (
            OVHCloudAudioTranscriptionConfig,
        )
        from unittest.mock import MagicMock

        config = OVHCloudAudioTranscriptionConfig()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "Hello world",
            "seconds": 3.14,
        }

        result = config.transform_audio_transcription_response(mock_response)

        assert result.text == "Hello world"
        assert result._hidden_params["duration"] == 3.14

    def test_legacy_duration_field_still_works(self):
        """Legacy `duration` field should still be accepted."""
        from litellm.llms.ovhcloud.audio_transcription.transformation import (
            OVHCloudAudioTranscriptionConfig,
        )
        from unittest.mock import MagicMock

        config = OVHCloudAudioTranscriptionConfig()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "Hello world",
            "duration": 2.71,
        }

        result = config.transform_audio_transcription_response(mock_response)

        assert result.text == "Hello world"
        assert result._hidden_params["duration"] == 2.71



    def test_seconds_zero_mapped_to_duration(self):
        """seconds=0.0 must not be treated as falsy and lost."""
        from litellm.llms.ovhcloud.audio_transcription.transformation import (
            OVHCloudAudioTranscriptionConfig,
        )
        from unittest.mock import MagicMock

        config = OVHCloudAudioTranscriptionConfig()
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "silence", "seconds": 0.0}
        result = config.transform_audio_transcription_response(mock_response)
        assert result._hidden_params["duration"] == 0.0        
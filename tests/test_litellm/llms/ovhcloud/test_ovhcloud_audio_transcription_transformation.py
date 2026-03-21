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




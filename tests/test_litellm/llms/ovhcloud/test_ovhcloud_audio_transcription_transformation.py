import os
from typing import Dict
from unittest.mock import MagicMock

import litellm
import pytest

from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.ovhcloud.audio_transcription.transformation import (
    OVHCloudAudioTranscriptionConfig,
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


class TestOVHCloudVerboseJsonResponse:
    """Unit tests for verbose_json response handling (no API key required)."""

    def _make_mock_response(self, body: dict) -> MagicMock:
        mock = MagicMock()
        mock.json.return_value = body
        mock.text = str(body)
        mock.status_code = 200
        mock.headers = {}
        return mock

    def test_plain_json_response_returns_text(self):
        """Basic response_format=json only returns text."""
        config = OVHCloudAudioTranscriptionConfig()
        mock_response = self._make_mock_response({"text": "Hello world"})
        result = config.transform_audio_transcription_response(mock_response)
        assert result.text == "Hello world"

    def test_verbose_json_response_preserves_segments(self):
        """response_format=verbose_json segments must be present in the response."""
        config = OVHCloudAudioTranscriptionConfig()
        segments = [
            {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello"},
            {"id": 1, "start": 1.5, "end": 3.0, "text": "world"},
        ]
        mock_response = self._make_mock_response(
            {
                "text": "Hello world",
                "task": "transcribe",
                "language": "english",
                "duration": 3.0,
                "segments": segments,
            }
        )
        result = config.transform_audio_transcription_response(mock_response)
        assert result.text == "Hello world"
        assert result["segments"] == segments
        assert result["language"] == "english"
        assert result["duration"] == 3.0
        assert result["task"] == "transcribe"

    def test_verbose_json_response_preserves_words(self):
        """timestamp_granularities=['word'] words must be present in the response."""
        config = OVHCloudAudioTranscriptionConfig()
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        mock_response = self._make_mock_response(
            {
                "text": "Hello world",
                "language": "english",
                "duration": 1.0,
                "words": words,
                "segments": [],
            }
        )
        result = config.transform_audio_transcription_response(mock_response)
        assert result["words"] == words

    def test_missing_verbose_json_fields_are_not_set(self):
        """Fields absent from the response must not be added to the result."""
        config = OVHCloudAudioTranscriptionConfig()
        mock_response = self._make_mock_response({"text": "Hello"})
        result = config.transform_audio_transcription_response(mock_response)
        assert result.text == "Hello"
        assert result.get("segments") is None
        assert result.get("words") is None
        assert result.get("language") is None

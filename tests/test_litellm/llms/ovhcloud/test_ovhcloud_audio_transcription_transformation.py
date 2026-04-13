import json
import os
from typing import Dict
from unittest.mock import MagicMock

import httpx
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


def _make_mock_response(body: dict) -> httpx.Response:
    """Helper to build a mock httpx.Response with a JSON body."""
    mock = MagicMock(spec=httpx.Response)
    mock.json.return_value = body
    mock.text = json.dumps(body)
    mock.status_code = 200
    mock.headers = {}
    return mock


class TestOVHCloudTranscriptionResponseTransformation:
    """Unit tests for OVHCloudAudioTranscriptionConfig.transform_audio_transcription_response."""

    def setup_method(self):
        self.config = OVHCloudAudioTranscriptionConfig()

    def test_verbose_json_segments_preserved(self):
        """
        Regression test for https://github.com/BerriAI/litellm/issues/25633.

        When response_format=verbose_json, the OVHCloud Whisper endpoint returns
        `segments` (and optionally `words`, `language`, `duration`).  The
        transformer must pass all fields through to TranscriptionResponse.
        """
        verbose_response = {
            "task": "transcribe",
            "language": "english",
            "duration": 3.14,
            "text": "Hello world.",
            "segments": [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Hello world.",
                    "tokens": [50364, 2425, 1002, 13],
                    "temperature": 0.0,
                    "avg_logprob": -0.3,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.01,
                }
            ],
        }

        raw_response = _make_mock_response(verbose_response)
        result = self.config.transform_audio_transcription_response(raw_response)

        assert result.text == "Hello world."
        # segments must be preserved
        assert hasattr(result, "segments") or "segments" in result._hidden_params
        segments = getattr(result, "segments", None) or result._hidden_params.get(
            "segments"
        )
        assert segments is not None
        assert len(segments) == 1

    def test_plain_text_response(self):
        """A minimal JSON response with only `text` should still work."""
        raw_response = _make_mock_response({"text": "Simple transcript."})
        result = self.config.transform_audio_transcription_response(raw_response)
        assert result.text == "Simple transcript."

    def test_transcript_field_normalised_to_text(self):
        """If OVHCloud returns `transcript` instead of `text`, it should be normalised."""
        raw_response = _make_mock_response({"transcript": "Normalised text."})
        result = self.config.transform_audio_transcription_response(raw_response)
        assert result.text == "Normalised text."

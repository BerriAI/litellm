import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.ovhcloud.audio_transcription.transformation import OVHCloudAudioTranscriptionConfig
from litellm.types.utils import TranscriptionResponse



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


def _transform(payload: object) -> TranscriptionResponse:
    return OVHCloudAudioTranscriptionConfig().transform_audio_transcription_response(httpx.Response(200, json=payload))


@pytest.mark.parametrize(
    ("payload", "expected_text", "expected_hidden_params"),
    [
        (
            {"text": "hello", "seconds": 3.5, "duration": 9},
            "hello",
            {"text": "hello", "seconds": 3.5, "duration": 3.5},
        ),
        (
            {"transcript": "from transcript", "seconds": None, "duration": 4},
            "from transcript",
            {"transcript": "from transcript", "seconds": None, "duration": 4},
        ),
        ({"language": "en"}, "", {"language": "en"}),
    ],
)
def test_transform_audio_transcription_response_normalizes_text_and_duration(
    payload: dict[str, object], expected_text: str, expected_hidden_params: dict[str, object]
):
    response = _transform(payload)

    assert response.text == expected_text
    assert response._hidden_params == expected_hidden_params


@pytest.mark.parametrize("payload", [7, "spoken secret", [{"text": "spoken secret"}]])
def test_transform_audio_transcription_response_rejects_non_object_bodies(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert "spoken secret" not in str(exc_info.value)

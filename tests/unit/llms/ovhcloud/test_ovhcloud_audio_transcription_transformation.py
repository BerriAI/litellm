



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

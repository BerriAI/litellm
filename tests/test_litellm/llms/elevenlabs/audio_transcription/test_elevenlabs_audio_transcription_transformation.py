import os
import sys
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.elevenlabs.audio_transcription.transformation import (
    ElevenLabsAudioTranscriptionConfig,
)


def _mock_raw_response(response_json: dict) -> MagicMock:
    raw = MagicMock()
    raw.json.return_value = response_json
    raw.text = str(response_json)
    return raw


class TestElevenLabsTransformResponse:
    def test_sets_duration_from_last_timestamp(self):
        """Duration must be derived from word timestamps so per-second cost
        tracking (input_cost_per_second x duration) bills real audio length —
        without it every ElevenLabs transcription logs $0 spend."""
        config = ElevenLabsAudioTranscriptionConfig()
        response = config.transform_audio_transcription_response(
            _mock_raw_response(
                {
                    "text": "hello world",
                    "language_code": "en",
                    "words": [
                        {"type": "word", "text": "hello", "start": 0.1, "end": 0.5},
                        {"type": "spacing", "text": " ", "start": 0.5, "end": 0.6},
                        {"type": "word", "text": "world", "start": 0.6, "end": 1.9},
                        {"type": "audio_event", "text": "(door)", "start": 2.0, "end": 3.4},
                    ],
                }
            )
        )
        assert response.text == "hello world"
        assert response.duration == 3.4

    def test_no_words_leaves_duration_unset(self):
        config = ElevenLabsAudioTranscriptionConfig()
        response = config.transform_audio_transcription_response(
            _mock_raw_response({"text": "hello", "language_code": "en"})
        )
        assert getattr(response, "duration", None) is None

    def test_malformed_word_timestamps_do_not_raise(self):
        config = ElevenLabsAudioTranscriptionConfig()
        response = config.transform_audio_transcription_response(
            _mock_raw_response(
                {
                    "text": "hello",
                    "language_code": "en",
                    "words": [
                        {"type": "word", "text": "hello", "start": 0.0, "end": None},
                        {"type": "word", "text": "x", "start": 0.0, "end": "bad"},
                        {"type": "word", "text": "y", "start": 0.0, "end": 2.5},
                    ],
                }
            )
        )
        assert response.duration == 2.5

import io

import httpx
import numpy as np
import pytest
import soundfile

from litellm.llms.elevenlabs.audio_transcription.transformation import (
    ElevenLabsAudioTranscriptionConfig,
)
from litellm.types.utils import TranscriptionUsageDurationObject


def _response(payload: dict) -> httpx.Response:
    return httpx.Response(status_code=200, json=payload)


def _wav_bytes(channels: int) -> bytes:
    buf = io.BytesIO()
    samples = np.zeros((1600, channels) if channels > 1 else 1600, dtype="float32")
    soundfile.write(buf, samples, 16000, format="WAV")
    return buf.getvalue()


class TestElevenLabsRequestParams:
    def setup_method(self):
        self.config = ElevenLabsAudioTranscriptionConfig()

    def test_response_format_is_supported(self):
        assert "response_format" in self.config.get_supported_openai_params("scribe_v2")

    def test_diarized_json_maps_to_diarize(self):
        out = self.config.map_openai_params(
            non_default_params={"response_format": "diarized_json"},
            optional_params={},
            model="scribe_v2",
            drop_params=False,
        )
        assert out["diarize"] is True

    def test_non_diarized_response_format_does_not_enable_diarize(self):
        out = self.config.map_openai_params(
            non_default_params={"response_format": "json"},
            optional_params={},
            model="scribe_v2",
            drop_params=False,
        )
        assert "diarize" not in out

    def test_language_maps_to_language_code(self):
        out = self.config.map_openai_params(
            non_default_params={"language": "pt"},
            optional_params={},
            model="scribe_v2",
            drop_params=False,
        )
        assert out["language_code"] == "pt"


class TestDiarizationMechanismSelection:
    """diarized_json picks acoustic diarize (mono) or channel split (stereo)."""

    def setup_method(self):
        self.config = ElevenLabsAudioTranscriptionConfig()

    def _form_data(self, audio: bytes, optional_params: dict) -> dict:
        return self.config.transform_audio_transcription_request(
            model="scribe_v2",
            audio_file=audio,
            optional_params=optional_params,
            litellm_params={},
        ).data

    def test_mono_uses_acoustic_diarize(self):
        data = self._form_data(_wav_bytes(1), {"diarize": True})
        assert data["diarize"] == "True"
        assert "use_multi_channel" not in data

    def test_stereo_switches_to_multichannel_and_drops_diarize(self):
        data = self._form_data(_wav_bytes(2), {"diarize": True})
        assert data["use_multi_channel"] == "True"
        assert data["multichannel_output_style"] == "combined"
        assert "diarize" not in data

    def test_explicit_multichannel_wins_over_detection_on_mono(self):
        data = self._form_data(
            _wav_bytes(1), {"diarize": True, "use_multi_channel": True}
        )
        assert data["use_multi_channel"] == "True"
        assert "diarize" not in data

    def test_explicit_multichannel_false_keeps_diarize_on_stereo(self):
        data = self._form_data(
            _wav_bytes(2), {"diarize": True, "use_multi_channel": False}
        )
        assert data["diarize"] == "True"
        assert "use_multi_channel" not in data

    def test_unreadable_audio_falls_back_to_diarize(self):
        data = self._form_data(b"not audio at all", {"diarize": True})
        assert data["diarize"] == "True"
        assert "use_multi_channel" not in data

    def test_no_diarization_requested_sends_neither(self):
        data = self._form_data(_wav_bytes(2), {})
        assert "diarize" not in data
        assert "use_multi_channel" not in data


class TestElevenLabsDiarizedResponse:
    def setup_method(self):
        self.config = ElevenLabsAudioTranscriptionConfig()

    def test_diarize_single_channel_builds_speaker_segments(self):
        payload = {
            "language_code": "por",
            "text": "Bom dia. Oi?",
            "audio_duration_secs": 70.5,
            "words": [
                {
                    "text": "ev",
                    "start": 1.0,
                    "end": 2.0,
                    "type": "audio_event",
                    "speaker_id": "speaker_0",
                },
                {
                    "text": "Bom",
                    "start": 3.0,
                    "end": 3.4,
                    "type": "word",
                    "speaker_id": "speaker_1",
                },
                {
                    "text": "dia",
                    "start": 3.4,
                    "end": 3.8,
                    "type": "word",
                    "speaker_id": "speaker_1",
                },
                {
                    "text": "Oi",
                    "start": 5.0,
                    "end": 5.3,
                    "type": "word",
                    "speaker_id": "speaker_0",
                },
            ],
        }

        result = self.config.transform_audio_transcription_response(_response(payload))

        segments = result["segments"]
        assert [s["speaker"] for s in segments] == ["speaker_1", "speaker_0"]
        assert segments[0]["text"] == "Bom dia"
        assert segments[0]["start"] == 3.0 and segments[0]["end"] == 3.8
        assert segments[0]["type"] == "transcript.text.segment"
        # audio_event is excluded from segments
        assert all("ev" not in s["text"] for s in segments)
        # diarized responses carry usage/duration, not the flat word list
        assert not hasattr(result, "words") or result.get("words") is None

    def test_fractional_duration_serializes_as_typed_usage(self):
        payload = {
            "language_code": "por",
            "text": "oi",
            "audio_duration_secs": 70.5,
            "words": [
                {
                    "text": "oi",
                    "start": 1.0,
                    "end": 1.5,
                    "type": "word",
                    "speaker_id": "speaker_0",
                }
            ],
        }

        result = self.config.transform_audio_transcription_response(_response(payload))

        assert isinstance(result["usage"], TranscriptionUsageDurationObject)
        assert result["usage"].seconds == 70.5
        assert result["duration"] == 70.5
        # must not emit pydantic serialization warnings for the usage union
        result.model_dump()

    def test_multichannel_uses_channel_as_speaker(self):
        payload = {
            "audio_duration_secs": 40.0,
            "transcripts": [
                {
                    "channel_index": 0,
                    "language_code": "por",
                    "text": "lado A",
                    "words": [
                        {
                            "text": "lado",
                            "start": 1.0,
                            "end": 1.5,
                            "type": "word",
                            "speaker_id": "speaker_0",
                            "channel_index": 0,
                        },
                        {
                            "text": "A",
                            "start": 1.5,
                            "end": 1.8,
                            "type": "word",
                            "speaker_id": "speaker_0",
                            "channel_index": 0,
                        },
                    ],
                },
                {
                    "channel_index": 1,
                    "language_code": "por",
                    "text": "lado B",
                    "words": [
                        {
                            "text": "lado",
                            "start": 10.0,
                            "end": 10.5,
                            "type": "word",
                            "speaker_id": "speaker_0",
                            "channel_index": 1,
                        },
                        {
                            "text": "B",
                            "start": 10.5,
                            "end": 10.8,
                            "type": "word",
                            "speaker_id": "speaker_0",
                            "channel_index": 1,
                        },
                    ],
                },
            ],
        }

        result = self.config.transform_audio_transcription_response(_response(payload))

        segments = result["segments"]
        # each channel becomes a distinct speaker even though ElevenLabs labels
        # both channels' speaker_id "speaker_0" independently
        assert {s["speaker"] for s in segments} == {"speaker_0", "speaker_1"}
        # time-ordered across channels
        assert [s["start"] for s in segments] == sorted(s["start"] for s in segments)
        assert result["duration"] == 40.0

    def test_multichannel_without_channel_index_keeps_speakers_distinct(self):
        payload = {
            "audio_duration_secs": 20.0,
            "transcripts": [
                {
                    "text": "a",
                    "words": [{"text": "a", "start": 1.0, "end": 1.2, "type": "word"}],
                },
                {
                    "text": "b",
                    "words": [{"text": "b", "start": 2.0, "end": 2.2, "type": "word"}],
                },
            ],
        }

        result = self.config.transform_audio_transcription_response(_response(payload))

        speakers = {s["speaker"] for s in result["segments"]}
        assert speakers == {"speaker_0", "speaker_1"}
        assert "speaker_None" not in speakers

    def test_malformed_response_raises_value_error(self):
        bad = httpx.Response(status_code=200, content=b"not json")

        with pytest.raises(ValueError):
            self.config.transform_audio_transcription_response(bad)

    def test_plain_response_stays_flat(self):
        payload = {
            "language_code": "en",
            "text": "Four score",
            "words": [
                {"text": "Four", "start": 0.0, "end": 0.5, "type": "word"},
                {"text": " ", "start": 0.5, "end": 0.5, "type": "spacing"},
                {"text": "score", "start": 0.5, "end": 1.0, "type": "word"},
            ],
        }

        result = self.config.transform_audio_transcription_response(_response(payload))

        assert result["words"] == [
            {"word": "Four", "start": 0.0, "end": 0.5},
            {"word": "score", "start": 0.5, "end": 1.0},
        ]
        assert not hasattr(result, "segments") or result.get("segments") is None
        assert result["language"] == "en"

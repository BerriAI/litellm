import json

import pytest

from litellm.llms.gemini.realtime.transformation import GeminiRealtimeConfig


@pytest.mark.parametrize(
    ("transcription", "expected"),
    [
        ("invalid", {}),
        ({}, {}),
        ({"language": "", "keywords": ["", 42]}, {}),
        (
            {"language": "pl", "keywords": ["", 42, "term"]},
            {"languageCodes": ["pl-PL"], "customVocabulary": ["term"]},
        ),
    ],
)
def test_session_update_maps_transcription_values(transcription, expected):
    messages = GeminiRealtimeConfig().transform_realtime_request(
        json.dumps(
            {
                "type": "session.update",
                "session": {"input_audio_transcription": transcription},
            }
        ),
        "gemini-3.5-transcribe-live",
        session_configuration_request=None,
    )

    setup = json.loads(messages[0])
    assert setup["setup"]["inputAudioTranscription"] == expected


def test_session_update_maps_language_and_keywords():
    messages = GeminiRealtimeConfig().transform_realtime_request(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "input_audio_transcription": {
                        "language": "pl",
                        "keywords": ["alpha", "beta"],
                    }
                },
            }
        ),
        "gemini-3.5-transcribe-live",
        session_configuration_request=None,
    )

    setup = json.loads(messages[0])
    assert setup["setup"]["inputAudioTranscription"] == {
        "languageCodes": ["pl-PL"],
        "customVocabulary": ["alpha", "beta"],
    }

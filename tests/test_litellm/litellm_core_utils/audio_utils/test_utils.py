import pytest

from litellm.litellm_core_utils.audio_utils.utils import resolve_speech_media_type


@pytest.mark.parametrize(
    ("upstream_content_type", "response_format", "expected"),
    [
        ("audio/wav", None, "audio/wav"),
        ("AUDIO/WAV", None, "audio/wav"),
        ("audio/flac; charset=binary", "mp3", "audio/flac"),
        ("application/json", "flac", "audio/flac"),
        ("application/octet-stream", "pcm", "audio/pcm"),
        (None, "wav", "audio/wav"),
        (None, "WAV", "audio/wav"),
        (None, "opus", "audio/opus"),
        (None, "aac", "audio/aac"),
        (None, "mp3", "audio/mpeg"),
        (None, "mp4", "audio/mpeg"),
        (None, "bogus", "audio/mpeg"),
        (None, None, "audio/mpeg"),
        ("", None, "audio/mpeg"),
    ],
)
def test_resolve_speech_media_type(upstream_content_type, response_format, expected):
    resolved = resolve_speech_media_type(
        upstream_content_type=upstream_content_type,
        response_format=response_format,
    )
    assert resolved == expected

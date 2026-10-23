from typing import Final

import pytest
from pydantic import JsonValue

from litellm.llms.base_llm.realtime.beta_compat import (
    beta_transcription_session,
    ga_transcription_client_secret_request,
)


@pytest.mark.parametrize(
    ("beta_format", "ga_format"),
    [
        ("pcm16", {"type": "audio/pcm", "rate": 24000}),
        ("g711_ulaw", {"type": "audio/pcmu"}),
        ("g711_alaw", {"type": "audio/pcma"}),
        ("unknown", "unknown"),
        ({"type": "audio/unknown", "rate": 48000}, {"type": "audio/unknown", "rate": 48000}),
    ],
)
@pytest.mark.parametrize("vad", [None, {"type": "server_vad", "threshold": 0.5}])
def test_transcription_settings_survive_conversion_to_ga_and_back(
    beta_format: JsonValue, ga_format: JsonValue, vad: JsonValue
) -> None:
    transcription: Final[JsonValue] = {"model": "deployment", "language": "en", "prompt": ""}
    noise: Final[JsonValue] = {"type": "near_field"} if vad is not None else None
    beta: Final[dict[str, JsonValue]] = {
        "input_audio_format": beta_format,
        "input_audio_transcription": transcription,
        "turn_detection": vad,
        "input_audio_noise_reduction": noise,
        "include": ["item.input_audio_transcription.logprobs"],
        "client_secret": {"expires_at": {"anchor": "created_at", "seconds": 600}},
    }
    ga_session: Final[JsonValue] = {
        "type": "transcription",
        "include": beta["include"],
        "audio": {
            "input": {
                "format": ga_format,
                "transcription": transcription,
                "turn_detection": vad,
                "noise_reduction": noise,
            }
        },
    }
    assert ga_transcription_client_secret_request(beta) == {
        "session": ga_session,
        "expires_after": {"anchor": "created_at", "seconds": 600},
    }
    key: Final[JsonValue] = {"value": "ek_upstream", "expires_at": 1234567890}
    assert beta_transcription_session({"session": ga_session, "value": "ek_upstream", "expires_at": 1234567890}) == {
        **beta,
        "client_secret": key,
    }


def test_absent_optional_settings_remain_absent() -> None:
    assert ga_transcription_client_secret_request({}) == {"session": {"type": "transcription"}}
    assert beta_transcription_session({"value": "ek_upstream"}) == {
        "client_secret": {"value": "ek_upstream", "expires_at": None}
    }

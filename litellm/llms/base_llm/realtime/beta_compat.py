from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter

from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.llms.base_llm.realtime.transcription_protocol import json_mapping

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_BETA_FORMATS: Final = MappingProxyType({ga["type"]: beta for beta, ga in RealTimeStreaming.audio_formats.items()})
_BETA_FIELDS: Final = MappingProxyType(
    {
        "transcription": "input_audio_transcription",
        "turn_detection": "turn_detection",
        "noise_reduction": "input_audio_noise_reduction",
        "format": "input_audio_format",
    }
)


def ga_transcription_client_secret_request(beta_session: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    session: Final = {key: value for key, value in beta_session.items() if key not in ("client_secret", "model")}
    ga_session: Final = _JSON.validate_python(
        RealTimeStreaming.remap_beta_session_to_ga({**session, "type": "transcription"})
    )
    expires_after: Final = json_mapping(beta_session.get("client_secret"), "client_secret").get("expires_at")
    return {"session": ga_session, **({"expires_after": expires_after} if expires_after is not None else {})}


def beta_transcription_session(ga_client_secret: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    session: Final = json_mapping(ga_client_secret.get("session"), "session")
    audio: Final = json_mapping(session.get("audio"), "session.audio")
    audio_input: Final = json_mapping(audio.get("input"), "session.audio.input")
    return {
        **{key: value for key, value in session.items() if key not in ("type", "audio")},
        **{
            beta: _beta_audio_format(audio_input[ga]) if ga == "format" else audio_input[ga]
            for ga, beta in _BETA_FIELDS.items()
            if ga in audio_input
        },
        "client_secret": {"value": ga_client_secret.get("value"), "expires_at": ga_client_secret.get("expires_at")},
    }


def _beta_audio_format(ga_format: JsonValue) -> JsonValue:
    format_type: Final = ga_format.get("type") if isinstance(ga_format, dict) else None
    return _BETA_FORMATS.get(format_type, ga_format) if isinstance(format_type, str) else ga_format

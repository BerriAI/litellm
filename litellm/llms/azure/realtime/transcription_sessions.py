from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import JsonValue, TypeAdapter

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_AUDIO_FORMATS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "pcm16": {"type": "audio/pcm", "rate": 24000},
        "g711_ulaw": {"type": "audio/pcmu"},
        "g711_alaw": {"type": "audio/pcma"},
    }
)
_LEGACY_AUDIO_FORMATS: Final[Mapping[str, str]] = MappingProxyType(
    {"audio/pcm": "pcm16", "audio/pcmu": "g711_ulaw", "audio/pcma": "g711_alaw"}
)


def azure_transcription_client_secret_request(
    transcription_session: Mapping[str, JsonValue], model: str
) -> Mapping[str, JsonValue]:
    raw_secret: Final = transcription_session.get("client_secret")
    expires_after: Final = transcription_session.get(
        "expires_after", raw_secret.get("expires_at") if isinstance(raw_secret, dict) else raw_secret
    )
    raw_audio: Final = transcription_session.get("audio")
    audio: Final = raw_audio if isinstance(raw_audio, dict) else {}
    raw_audio_input: Final = audio.get("input")
    existing_audio_input: Final = raw_audio_input if isinstance(raw_audio_input, dict) else {}
    raw_audio_transcription: Final = existing_audio_input.get("transcription")
    audio_transcription: Final = raw_audio_transcription if isinstance(raw_audio_transcription, dict) else {}
    raw_transcription: Final = transcription_session.get("input_audio_transcription")
    transcription: Final = raw_transcription if isinstance(raw_transcription, dict) else {}
    raw_format: Final = transcription_session.get("input_audio_format")
    audio_input: Final[JsonValue] = {
        **existing_audio_input,
        **(
            {
                "format": _AUDIO_FORMATS[raw_format]
                if isinstance(raw_format, str) and raw_format in _AUDIO_FORMATS
                else raw_format
            }
            if "input_audio_format" in transcription_session
            else {}
        ),
        "transcription": {**audio_transcription, **transcription, "model": model},
        **(
            {"noise_reduction": transcription_session["input_audio_noise_reduction"]}
            if "input_audio_noise_reduction" in transcription_session
            else {}
        ),
        **(
            {"turn_detection": transcription_session["turn_detection"]}
            if "turn_detection" in transcription_session
            else {}
        ),
    }
    session: Final[JsonValue] = {
        **{
            key: value
            for key, value in transcription_session.items()
            if key
            not in (
                "model",
                "audio",
                "client_secret",
                "expires_after",
                "modalities",
                "input_audio_format",
                "input_audio_transcription",
                "input_audio_noise_reduction",
                "turn_detection",
            )
        },
        "type": "transcription",
        "audio": {
            **{key: value for key, value in audio.items() if key != "input"},
            "input": audio_input,
        },
    }
    return MappingProxyType(
        {"session": session, **({"expires_after": expires_after} if expires_after is not None else {})}
    )


def azure_transcription_client_secret_response(response: httpx.Response) -> httpx.Response:
    if response.is_error:
        return response
    payload: Final = _JSON.validate_python(response.json())
    if not isinstance(payload, dict):
        return response
    raw_session: Final = payload.get("session")
    session: Final = raw_session if isinstance(raw_session, dict) else {}
    raw_audio: Final = session.get("audio")
    audio: Final = raw_audio if isinstance(raw_audio, dict) else {}
    raw_input: Final = audio.get("input")
    audio_input: Final = raw_input if isinstance(raw_input, dict) else {}
    raw_format: Final = audio_input.get("format")
    format_type: Final = raw_format.get("type") if isinstance(raw_format, dict) else None
    input_audio_format: Final = (
        _LEGACY_AUDIO_FORMATS.get(format_type, format_type) if isinstance(format_type, str) else raw_format
    )
    client_secret: Final = {
        "value": payload.get("value"),
        **({"expires_at": payload["expires_at"]} if "expires_at" in payload else {}),
    }
    response_body: Final = {
        **{key: value for key, value in session.items() if key not in ("type", "audio", "expires_at")},
        **({"input_audio_format": input_audio_format} if input_audio_format is not None else {}),
        **({"input_audio_transcription": audio_input["transcription"]} if "transcription" in audio_input else {}),
        **({"input_audio_noise_reduction": audio_input["noise_reduction"]} if "noise_reduction" in audio_input else {}),
        **({"turn_detection": audio_input["turn_detection"]} if "turn_detection" in audio_input else {}),
        "client_secret": client_secret,
    }
    return httpx.Response(
        response.status_code,
        headers={
            key: value
            for key, value in response.headers.items()
            if key.lower() not in ("content-length", "content-encoding", "transfer-encoding")
        },
        json=response_body,
        request=response.request,
    )

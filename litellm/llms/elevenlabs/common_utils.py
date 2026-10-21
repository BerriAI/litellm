import json
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.constants import ELEVENLABS_DEFAULT_API_BASE, ELEVENLABS_TTS_WEBSOCKET_DEFAULT_MODEL
from litellm.exceptions import ModelNotMappedError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    ClientTextDecision,
    ForwardedClientText,
    RefusedClientText,
)
from litellm.types.utils import LlmProviders

ElevenLabsTextToSpeechStream: TypeAlias = Literal["stream-input", "multi-stream-input"]

ELEVENLABS_API_KEY_HEADER: Final = "xi-api-key"
ELEVENLABS_TTS_WEBSOCKET_ROUTE_SUFFIXES: Final = ("/stream-input", "/multi-stream-input")
ELEVENLABS_TTS_CALLER_CREDENTIAL_PARAMS: Final = frozenset({"authorization", "single_use_token"})
ELEVENLABS_TTS_CALLER_CREDENTIAL_FIELDS: Final = frozenset({"xi_api_key", "xi-api-key", "authorization"})
ELEVENLABS_TTS_AUDIO_FIELDS: Final = frozenset({"audio", "alignment", "normalizedAlignment"})
ELEVENLABS_TTS_ALIGNED_CHARACTERS_KEY: Final = "aligned_characters"
_MODEL_ID_PARAM: Final = "model_id"
_VOICE_ID: Final = re.compile(r"[A-Za-z0-9_-]+")
_WEBSOCKET_SCHEMES: Final = MappingProxyType({"https": "wss", "http": "ws", "wss": "wss", "ws": "ws"})
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_UNREADABLE_CLIENT_TEXT: Final = RefusedClientText(
    code=1007, reason="Each text frame must be a JSON object whose text field, when present, is a string"
)


class ElevenLabsException(BaseLLMException):
    pass


def elevenlabs_tts_requested_model(query_string: str) -> str:
    return httpx.QueryParams(query_string).get(_MODEL_ID_PARAM) or ELEVENLABS_TTS_WEBSOCKET_DEFAULT_MODEL


def is_elevenlabs_voice_id(voice_id: str) -> bool:
    return _VOICE_ID.fullmatch(voice_id) is not None


def _upstream_params(query_string: str) -> httpx.QueryParams:
    """Auth and pricing read the first ``model_id``, so ElevenLabs must not see a second one, and the proxy's own
    key is the only upstream credential, so a caller's ``authorization`` or ``single_use_token`` never leaves."""
    items: Final = httpx.QueryParams(query_string).multi_items()
    first_model_index: Final = next((index for index, (key, _) in enumerate(items) if key == _MODEL_ID_PARAM), None)
    return httpx.QueryParams(
        tuple(
            (key, value)
            for index, (key, value) in enumerate(items)
            if key not in ELEVENLABS_TTS_CALLER_CREDENTIAL_PARAMS
            and (key != _MODEL_ID_PARAM or index == first_model_index)
        )
    )


def elevenlabs_tts_websocket_target(
    api_base: str | None, voice_id: str, stream: ElevenLabsTextToSpeechStream, query_string: str
) -> str:
    base_url: Final = (api_base or ELEVENLABS_DEFAULT_API_BASE).rstrip("/")
    stream_url: Final = httpx.URL(f"{base_url}/v1/text-to-speech/{voice_id}/{stream}")
    websocket_url: Final = stream_url.copy_with(scheme=_WEBSOCKET_SCHEMES.get(stream_url.scheme, stream_url.scheme))
    params: Final = _upstream_params(query_string)
    query: Final = (
        params
        if params.get(_MODEL_ID_PARAM)
        else params.remove(_MODEL_ID_PARAM).add(_MODEL_ID_PARAM, ELEVENLABS_TTS_WEBSOCKET_DEFAULT_MODEL)
    )
    return f"{websocket_url}?{query}"


def elevenlabs_tts_model(upstream_url: str) -> str:
    models: Final = parse_qs(urlparse(upstream_url).query).get(_MODEL_ID_PARAM)
    return models[0] if models else ELEVENLABS_TTS_WEBSOCKET_DEFAULT_MODEL


def elevenlabs_tts_registry_key(upstream_url: str) -> str:
    return f"{LlmProviders.ELEVENLABS.value}/{elevenlabs_tts_model(upstream_url)}"


def elevenlabs_tts_is_priced(upstream_url: str) -> bool:
    """Only an exact registry hit with a per-character rate counts: the speech cost path bills a session by
    ``input_cost_per_character`` and treats a missing or zero rate as unpriced."""
    registry_key: Final = elevenlabs_tts_registry_key(upstream_url)
    try:
        model_info: Final = litellm.get_model_info(
            model=registry_key, custom_llm_provider=LlmProviders.ELEVENLABS.value
        )
    except ModelNotMappedError:
        return False
    return model_info["key"] == registry_key and bool(model_info.get("input_cost_per_character"))


def _json_object(text: str) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_json(text)
    except ValidationError:
        return None


def elevenlabs_tts_client_text(text: str) -> ClientTextDecision:
    """The upstream gets the parsed frame re-serialized, never the raw text, so ElevenLabs reads exactly the
    ``text`` that was counted and no caller credential field survives an escape or a parser disagreement."""
    frame: Final = _json_object(text)
    if frame is None:
        return _UNREADABLE_CLIENT_TEXT
    frame_text: Final = frame.get("text")
    if frame_text is not None and not isinstance(frame_text, str):
        return _UNREADABLE_CLIENT_TEXT
    forwarded: Final = json.dumps(
        {key: value for key, value in frame.items() if key not in ELEVENLABS_TTS_CALLER_CREDENTIAL_FIELDS},
        ensure_ascii=False,
    )
    return ForwardedClientText(text=forwarded, recorded={"text": frame_text} if frame_text is not None else None)


class _Alignment(BaseModel):
    model_config = ConfigDict(frozen=True)
    chars: tuple[str, ...]


_ALIGNMENT: Final = TypeAdapter(_Alignment)


def _aligned_characters(alignment: object) -> int:
    try:
        return len(_ALIGNMENT.validate_python(alignment).chars)
    except ValidationError:
        return 0


def elevenlabs_tts_recorded_upstream_frame(frame: Mapping[str, object]) -> Mapping[str, object]:
    without_audio: Final = {key: value for key, value in frame.items() if key not in ELEVENLABS_TTS_AUDIO_FIELDS}
    if not frame.get("audio"):
        return without_audio
    return {**without_audio, ELEVENLABS_TTS_ALIGNED_CHARACTERS_KEY: _aligned_characters(frame.get("alignment"))}


def elevenlabs_tts_characters(
    client_frames: Sequence[Mapping[str, object]], upstream_frames: Sequence[Mapping[str, object]]
) -> int:
    """ElevenLabs bills the characters it synthesized, which its alignment lists, so text a session sent but never
    heard (interrupted, unflushed, timed out) is free; audio that came back with no alignment at all bills the text
    the client sent instead."""
    audio_frames: Final = tuple(frame for frame in upstream_frames if ELEVENLABS_TTS_ALIGNED_CHARACTERS_KEY in frame)
    aligned: Final = sum(
        count for frame in audio_frames if isinstance(count := frame[ELEVENLABS_TTS_ALIGNED_CHARACTERS_KEY], int)
    )
    if aligned or not audio_frames:
        return aligned
    return sum(len(text) for frame in client_frames if isinstance(text := frame.get("text"), str))

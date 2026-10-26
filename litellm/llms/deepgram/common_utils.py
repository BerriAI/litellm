import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm.constants import DEEPGRAM_DEFAULT_API_BASE, DEEPGRAM_LISTEN_DEFAULT_MODEL, DEEPGRAM_SPEAK_DEFAULT_MODEL
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import LlmProviders, ModelInfo

_WEBSOCKET_SCHEMES: Final = MappingProxyType({"https": "wss", "http": "ws", "wss": "wss", "ws": "ws"})
DEEPGRAM_LISTEN_CALLBACK_PARAMS: Final = frozenset({"callback", "callback_method"})
DEEPGRAM_LISTEN_STREAMING_PRICING_PREFIX: Final = "streaming/"
DEEPGRAM_LISTEN_MULTILINGUAL_LANGUAGE: Final = "multi"
DEEPGRAM_LISTEN_MULTILINGUAL_PRICING_SUFFIX: Final = "-multilingual"
DEEPGRAM_LISTEN_ADDON_PRICING_PARAMS: Final = MappingProxyType(
    {
        "redact": "redact",
        "keyterm": "keyterm",
        "detect_entities": "detect_entities",
        "diarize": "diarize",
        "diarize_model": "diarize",
    }
)
# Deepgram bills every voice of an Aura generation at one rate, so each generation has one cost-map row
DEEPGRAM_SPEAK_PRICING_FAMILIES: Final = (
    (re.compile(r"aura-2-[a-z]+-[a-z]{2}"), "aura-2"),
    (re.compile(r"aura-[a-z]+-[a-z]{2}"), "aura"),
)
_DISABLED_PARAM_VALUES: Final = frozenset({"", "false"})
_SINGLE_VALUED_PARAMS: Final = frozenset({"model", "language"})


class DeepgramException(BaseLLMException):
    pass


def _requested_model(query_string: str, default_model: str) -> str:
    return httpx.QueryParams(query_string).get("model") or default_model


def deepgram_listen_requested_model(query_string: str) -> str:
    return _requested_model(query_string, DEEPGRAM_LISTEN_DEFAULT_MODEL)


def _first_occurrences(query_string: str) -> httpx.QueryParams:
    """Authorization and pricing read the first ``model`` and ``language`` value; Deepgram must not see a second one."""
    items: Final = httpx.QueryParams(query_string).multi_items()
    return httpx.QueryParams(
        tuple(
            (key, value)
            for index, (key, value) in enumerate(items)
            if key not in _SINGLE_VALUED_PARAMS or all(earlier != key for earlier, _ in items[:index])
        )
    )


def _deepgram_websocket_target(api_base: str | None, path: str, query_string: str, default_model: str) -> str:
    http_url: Final = httpx.URL(join_api_base(api_base or DEEPGRAM_DEFAULT_API_BASE, path))
    websocket_url: Final = http_url.copy_with(scheme=_WEBSOCKET_SCHEMES.get(http_url.scheme, http_url.scheme))
    params: Final = _first_occurrences(query_string)
    query: Final = params if params.get("model") else params.remove("model").add("model", default_model)
    return f"{websocket_url}?{query}"


def deepgram_listen_websocket_target(api_base: str | None, query_string: str) -> str:
    return _deepgram_websocket_target(api_base, "listen", query_string, DEEPGRAM_LISTEN_DEFAULT_MODEL)


def deepgram_listen_callback_params(query_string: str) -> tuple[str, ...]:
    return tuple(sorted(DEEPGRAM_LISTEN_CALLBACK_PARAMS.intersection(httpx.QueryParams(query_string).keys())))


def _upstream_model(upstream_url: str, default_model: str) -> str:
    models: Final = parse_qs(urlparse(upstream_url).query).get("model")
    return models[0] if models else default_model


def deepgram_listen_model(upstream_url: str) -> str:
    return _upstream_model(upstream_url, DEEPGRAM_LISTEN_DEFAULT_MODEL)


def _param_enabled(values: Sequence[str]) -> bool:
    return any(value.strip().lower() not in _DISABLED_PARAM_VALUES for value in values)


def deepgram_listen_pricing_model(upstream_url: str) -> str:
    """Registry key, without the provider prefix, for the per-second base rate Deepgram bills a streaming session at:
    the multilingual streaming entry when ``language=multi``, otherwise the model's own streaming entry. Pre-recorded
    entries are never a substitute: Deepgram prices the two products differently."""
    streaming: Final = f"{DEEPGRAM_LISTEN_STREAMING_PRICING_PREFIX}{deepgram_listen_model(upstream_url)}"
    language: Final = parse_qs(urlparse(upstream_url).query).get("language", ("",))[0]
    if language.strip().lower() == DEEPGRAM_LISTEN_MULTILINGUAL_LANGUAGE:
        return f"{streaming}{DEEPGRAM_LISTEN_MULTILINGUAL_PRICING_SUFFIX}"
    return streaming


def deepgram_listen_registry_key(upstream_url: str) -> str:
    return f"{LlmProviders.DEEPGRAM.value}/{deepgram_listen_pricing_model(upstream_url)}"


def deepgram_listen_is_priced(upstream_url: str) -> bool:
    """Only an exact registry hit counts: the cost calculator resolves a missing ``streaming/<model>`` row to the
    pre-recorded ``<model>`` row, which is not the rate Deepgram bills a WebSocket session at."""
    return _exact_registry_entry(deepgram_listen_registry_key(upstream_url)) is not None


def _exact_registry_entry(registry_key: str) -> ModelInfo | None:
    try:
        model_info: Final = litellm.get_model_info(model=registry_key, custom_llm_provider=LlmProviders.DEEPGRAM.value)
    except Exception:
        return None
    return model_info if model_info["key"] == registry_key else None


def deepgram_listen_addon_pricing_models(upstream_url: str) -> tuple[str, ...]:
    params: Final = parse_qs(urlparse(upstream_url).query)
    return tuple(
        sorted(
            frozenset(
                f"{DEEPGRAM_LISTEN_STREAMING_PRICING_PREFIX}{addon}"
                for param, addon in DEEPGRAM_LISTEN_ADDON_PRICING_PARAMS.items()
                if _param_enabled(params.get(param, ()))
            )
        )
    )


def _channel_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 1 else None


def _results_channel_count(frame: Mapping[str, object]) -> int | None:
    match frame.get("channel_index"):
        case [object(), int() as channels]:
            return _channel_count(channels)
        case _:
            return None


def _declared_channel_count(upstream_url: str) -> int | None:
    declared: Final = parse_qs(urlparse(upstream_url).query).get("channels")
    if not declared or not declared[0].isdigit():
        return None
    return _channel_count(int(declared[0]))


def deepgram_listen_channel_count(websocket_messages: Sequence[Mapping[str, object]], upstream_url: str) -> int:
    metadata_channels: Final = tuple(
        channels
        for frame in websocket_messages
        if frame.get("type") == "Metadata"
        if (channels := _channel_count(frame.get("channels"))) is not None
    )
    if metadata_channels:
        return metadata_channels[-1]
    results_channels: Final = tuple(
        channels
        for frame in websocket_messages
        if frame.get("type") == "Results"
        if (channels := _results_channel_count(frame)) is not None
    )
    if results_channels:
        return max(results_channels)
    return _declared_channel_count(upstream_url) or 1


def _seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _results_frame_end(frame: Mapping[str, object]) -> float | None:
    start: Final = _seconds(frame.get("start"))
    duration: Final = _seconds(frame.get("duration"))
    return None if start is None or duration is None else start + duration


def _final_transcript(frame: Mapping[str, object]) -> str | None:
    if frame.get("is_final") is not True:
        return None
    channel: Final = frame.get("channel")
    alternatives: Final = channel.get("alternatives") if isinstance(channel, Mapping) else None
    first: Final = alternatives[0] if isinstance(alternatives, list) and alternatives else None
    transcript: Final = first.get("transcript") if isinstance(first, Mapping) else None
    return transcript if isinstance(transcript, str) and transcript else None


def deepgram_listen_audio_seconds(websocket_messages: Sequence[Mapping[str, object]]) -> float:
    metadata_durations: Final = tuple(
        duration
        for frame in websocket_messages
        if frame.get("type") == "Metadata"
        if (duration := _seconds(frame.get("duration"))) is not None and duration > 0
    )
    if metadata_durations:
        return metadata_durations[-1]
    return max(
        (
            end
            for frame in websocket_messages
            if frame.get("type") == "Results"
            if (end := _results_frame_end(frame)) is not None
        ),
        default=0.0,
    )


def deepgram_listen_transcript(websocket_messages: Sequence[Mapping[str, object]]) -> str:
    return " ".join(
        transcript
        for frame in websocket_messages
        if frame.get("type") == "Results"
        if (transcript := _final_transcript(frame)) is not None
    )


def deepgram_speak_requested_model(query_string: str) -> str:
    return _requested_model(query_string, DEEPGRAM_SPEAK_DEFAULT_MODEL)


def deepgram_speak_websocket_target(api_base: str | None, query_string: str) -> str:
    return _deepgram_websocket_target(api_base, "speak", query_string, DEEPGRAM_SPEAK_DEFAULT_MODEL)


def deepgram_speak_model(upstream_url: str) -> str:
    return _upstream_model(upstream_url, DEEPGRAM_SPEAK_DEFAULT_MODEL)


def deepgram_speak_registry_key(upstream_url: str) -> str:
    """The row a speak session is billed at: its Aura generation's row, else a row named after the voice itself"""
    model: Final = deepgram_speak_model(upstream_url)
    pricing_model: Final = next(
        (family for pattern, family in DEEPGRAM_SPEAK_PRICING_FAMILIES if pattern.fullmatch(model)), model
    )
    return f"{LlmProviders.DEEPGRAM.value}/{pricing_model}"


def deepgram_speak_character_rate(upstream_url: str) -> float | None:
    model_info: Final = _exact_registry_entry(deepgram_speak_registry_key(upstream_url))
    return None if model_info is None else model_info.get("input_cost_per_character")


def deepgram_speak_is_priced(upstream_url: str) -> bool:
    return deepgram_speak_character_rate(upstream_url) is not None


class _DeepgramSpeakMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["Speak"]
    text: str


def deepgram_speak_characters(frame: str | bytes) -> int:
    """Characters Deepgram bills for one client frame: the length of a Speak message's text, else 0. It never raises,
    because the relay runs it on every frame and a traceback from a malformed frame would carry text that must not be
    logged"""
    try:
        return len(_DeepgramSpeakMessage.model_validate_json(frame).text)
    except ValidationError:
        return 0

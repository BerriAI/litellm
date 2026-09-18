import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlparse

import httpx

from litellm.constants import DEEPGRAM_DEFAULT_API_BASE, DEEPGRAM_LISTEN_DEFAULT_MODEL
from litellm.llms.base_llm.chat.transformation import BaseLLMException

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
_DISABLED_PARAM_VALUES: Final = frozenset({"", "false"})


class DeepgramException(BaseLLMException):
    pass


def deepgram_listen_requested_model(query_string: str) -> str:
    return httpx.QueryParams(query_string).get("model") or DEEPGRAM_LISTEN_DEFAULT_MODEL


def deepgram_listen_websocket_target(api_base: str | None, query_string: str) -> str:
    listen_url: Final = httpx.URL(f"{(api_base or DEEPGRAM_DEFAULT_API_BASE).rstrip('/')}/listen")
    websocket_url: Final = listen_url.copy_with(scheme=_WEBSOCKET_SCHEMES.get(listen_url.scheme, listen_url.scheme))
    params: Final = httpx.QueryParams(query_string)
    query: Final = (
        query_string if params.get("model") else str(params.remove("model").add("model", DEEPGRAM_LISTEN_DEFAULT_MODEL))
    )
    return f"{websocket_url}?{query}"


def deepgram_listen_callback_params(query_string: str) -> tuple[str, ...]:
    return tuple(sorted(DEEPGRAM_LISTEN_CALLBACK_PARAMS.intersection(httpx.QueryParams(query_string).keys())))


def deepgram_listen_model(upstream_url: str) -> str:
    models: Final = parse_qs(urlparse(upstream_url).query).get("model")
    return models[0] if models else DEEPGRAM_LISTEN_DEFAULT_MODEL


def _param_enabled(values: Sequence[str]) -> bool:
    return any(value.strip().lower() not in _DISABLED_PARAM_VALUES for value in values)


def deepgram_listen_base_pricing_models(upstream_url: str) -> tuple[str, ...]:
    """Registry keys to try, in order, for the per-second base rate of a streaming session: the streaming entry for
    the language mode Deepgram bills (multilingual when ``language=multi``), then the plain streaming entry, then
    the pre-recorded entry for models that have no streaming price of their own."""
    model: Final = deepgram_listen_model(upstream_url)
    params: Final = parse_qs(urlparse(upstream_url).query)
    streaming: Final = f"{DEEPGRAM_LISTEN_STREAMING_PRICING_PREFIX}{model}"
    multilingual: Final = params.get("language", ("",))[-1].strip().lower() == DEEPGRAM_LISTEN_MULTILINGUAL_LANGUAGE
    return (
        (f"{streaming}{DEEPGRAM_LISTEN_MULTILINGUAL_PRICING_SUFFIX}", streaming, model)
        if multilingual
        else (streaming, model)
    )


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
    channel_index: Final = frame.get("channel_index")
    if not isinstance(channel_index, list) or len(channel_index) != 2:
        return None
    return _channel_count(channel_index[1])


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

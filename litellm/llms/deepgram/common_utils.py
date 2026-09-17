import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlparse

import httpx

from litellm.constants import DEEPGRAM_DEFAULT_API_BASE, DEEPGRAM_LISTEN_DEFAULT_MODEL
from litellm.llms.base_llm.chat.transformation import BaseLLMException

_WEBSOCKET_SCHEMES: Final = MappingProxyType({"https": "wss", "http": "ws", "wss": "wss", "ws": "ws"})


class DeepgramException(BaseLLMException):
    pass


def deepgram_listen_websocket_target(api_base: str | None, query_string: str) -> str:
    listen_url: Final = httpx.URL(f"{(api_base or DEEPGRAM_DEFAULT_API_BASE).rstrip('/')}/listen")
    websocket_url: Final = listen_url.copy_with(scheme=_WEBSOCKET_SCHEMES.get(listen_url.scheme, listen_url.scheme))
    params: Final = httpx.QueryParams(query_string)
    query: Final = (
        query_string if params.get("model") else str(params.remove("model").add("model", DEEPGRAM_LISTEN_DEFAULT_MODEL))
    )
    return f"{websocket_url}?{query}"


def deepgram_listen_model(upstream_url: str) -> str:
    models: Final = parse_qs(urlparse(upstream_url).query).get("model")
    return models[0] if models else DEEPGRAM_LISTEN_DEFAULT_MODEL


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
        if (duration := _seconds(frame.get("duration"))) is not None
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

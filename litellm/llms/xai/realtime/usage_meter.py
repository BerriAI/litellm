"""Counts billable xAI Grok Voice realtime usage from raw wire frames.

Audio format reference: https://docs.x.ai/voice-realtime.ws.json
"""

from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.llms.xai import RealtimeMeteredUsage

_PCM_DEFAULT_RATE: Final = 24000
_G711_BYTES_PER_SECOND: Final = 8000
_AUDIO_CONTENT_PART_TYPES: Final = frozenset({"input_audio", "audio"})
_CONTENT_PART_LIST: Final = TypeAdapter(list[object])


def _decoded_base64_length(value: object) -> int:
    """Decoded byte length of a base64 string without decoding it."""
    if not isinstance(value, str):
        return 0
    padding: Final = min(len(value) - len(value.rstrip("=")), 2)
    return len(value) * 3 // 4 - padding


def _audio_format_bytes_per_second(fmt: Mapping[str, object]) -> float:
    fmt_type: Final = fmt.get("type")
    if fmt_type is None or fmt_type == "audio/pcm":
        rate: Final = fmt.get("rate")
        rate_hz: Final = (
            int(rate)
            if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0
            else _PCM_DEFAULT_RATE
        )
        return float(2 * rate_hz)
    if fmt_type in ("audio/pcmu", "audio/pcma"):
        return float(_G711_BYTES_PER_SECOND)
    return 0.0


def _beta_audio_format_bytes_per_second(name: object) -> float | None:
    if name == "pcm16":
        return float(2 * _PCM_DEFAULT_RATE)
    if name in ("g711_ulaw", "g711_alaw"):
        return float(_G711_BYTES_PER_SECOND)
    return None


def _content_part_mappings(item: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    content: Final = item.get("content")
    raw_parts: Final = _CONTENT_PART_LIST.validate_python(content) if isinstance(content, list) else ()
    return tuple(part_map for part in raw_parts if (part_map := as_str_mapping(part)) is not None)


def _is_billable_text_item(item: Mapping[str, object], parts: tuple[Mapping[str, object], ...]) -> bool:
    if item.get("type") == "function_call_output":
        return False
    return not any(part.get("type") in _AUDIO_CONTENT_PART_TYPES for part in parts)


class XAIRealtimeUsageMeter:
    """Counts billable xAI realtime usage from raw client and backend frames."""

    def __init__(self) -> None:
        self._push_to_talk: bool = False
        self._audio_seconds_sent: float = 0.0
        self._audio_seconds_received: float = 0.0
        self._text_messages: int = 0
        self._input_bytes_per_second: float = float(2 * _PCM_DEFAULT_RATE)
        self._output_bytes_per_second: float = float(2 * _PCM_DEFAULT_RATE)

    def observe_client_event(self, event: Mapping[str, object]) -> None:
        event_type: Final = event.get("type")
        if event_type == "session.update":
            session: Final = as_str_mapping(event.get("session"))
            if session is not None:
                self._apply_client_turn_detection(session)
                self._apply_session_config(session)
        elif event_type == "input_audio_buffer.append":
            if self._input_bytes_per_second > 0:
                self._audio_seconds_sent += _decoded_base64_length(event.get("audio")) / self._input_bytes_per_second
        elif event_type == "conversation.item.create":
            item: Final = as_str_mapping(event.get("item"))
            if item is not None:
                parts: Final = _content_part_mappings(item)
                if _is_billable_text_item(item, parts):
                    self._text_messages += 1
                if self._input_bytes_per_second > 0:
                    self._audio_seconds_sent += (
                        sum(
                            _decoded_base64_length(part.get("audio"))
                            for part in parts
                            if part.get("type") in _AUDIO_CONTENT_PART_TYPES
                        )
                        / self._input_bytes_per_second
                    )

    def observe_backend_event(self, event: Mapping[str, object]) -> None:
        event_type: Final = event.get("type")
        if event_type in ("session.created", "session.updated"):
            session: Final = as_str_mapping(event.get("session"))
            if session is not None:
                self._apply_session_config(session)
        elif event_type in ("response.output_audio.delta", "response.audio.delta"):
            if self._output_bytes_per_second > 0:
                self._audio_seconds_received += (
                    _decoded_base64_length(event.get("delta")) / self._output_bytes_per_second
                )

    def snapshot(self) -> RealtimeMeteredUsage:
        return RealtimeMeteredUsage(
            push_to_talk=self._push_to_talk,
            audio_seconds_sent=self._audio_seconds_sent,
            audio_seconds_received=self._audio_seconds_received,
            text_messages=self._text_messages,
        )

    def _apply_session_config(self, session: Mapping[str, object]) -> None:
        audio: Final = as_str_mapping(session.get("audio"))
        audio_input: Final = as_str_mapping(audio.get("input")) if audio is not None else None

        input_bps: Final = _beta_audio_format_bytes_per_second(session.get("input_audio_format"))
        if input_bps is not None:
            self._input_bytes_per_second = input_bps
        output_bps: Final = _beta_audio_format_bytes_per_second(session.get("output_audio_format"))
        if output_bps is not None:
            self._output_bytes_per_second = output_bps

        if audio is None:
            return
        if audio_input is not None:
            input_format: Final = as_str_mapping(audio_input.get("format"))
            if input_format is not None:
                self._input_bytes_per_second = _audio_format_bytes_per_second(input_format)
        output_config: Final = as_str_mapping(audio.get("output"))
        if output_config is not None:
            output_format: Final = as_str_mapping(output_config.get("format"))
            if output_format is not None:
                self._output_bytes_per_second = _audio_format_bytes_per_second(output_format)

    def _apply_client_turn_detection(self, session: Mapping[str, object]) -> None:
        audio: Final = as_str_mapping(session.get("audio"))
        audio_input: Final = as_str_mapping(audio.get("input")) if audio is not None else None
        if "turn_detection" in session:
            self._apply_turn_detection(session["turn_detection"])
        elif audio_input is not None and "turn_detection" in audio_input:
            self._apply_turn_detection(audio_input["turn_detection"])

    def _apply_turn_detection(self, value: object) -> None:
        if value is None:
            self._push_to_talk = True
            return
        config: Final = as_str_mapping(value)
        if config is None or "type" not in config:
            return
        self._push_to_talk = config["type"] is None

"""Per-session xAI Grok Voice realtime usage meter.

xAI realtime ``response.done`` events carry ``usage: {}``, so token billing is
impossible and duration/text billing must be counted from the raw wire frames:
client ``input_audio_buffer.append`` and billable ``conversation.item.create``
frames, and backend ``response.output_audio.delta`` frames. Session config
(client ``session.update``, backend ``session.created`` / ``session.updated``)
decides the audio format used to convert base64 payload bytes into seconds and
whether the session is push-to-talk (``turn_detection: null``), which changes
how the session's audio is billed downstream in the cost calculator.

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
    """Bytes per second for a GA ``{type, rate}`` audio format object."""
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
    """Bytes per second for a flat beta ``input_audio_format`` / ``output_audio_format`` string."""
    if name == "pcm16":
        return float(2 * _PCM_DEFAULT_RATE)
    if name in ("g711_ulaw", "g711_alaw"):
        return float(_G711_BYTES_PER_SECOND)
    return None


def _is_billable_text_item(item: Mapping[str, object]) -> bool:
    if item.get("type") == "function_call_output":
        return False
    content: Final = item.get("content")
    parts: Final = _CONTENT_PART_LIST.validate_python(content) if isinstance(content, list) else ()
    return not any(
        (part_map := as_str_mapping(part)) is not None and part_map.get("type") in _AUDIO_CONTENT_PART_TYPES
        for part in parts
    )


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
                self._apply_session_config(session)
        elif event_type == "input_audio_buffer.append":
            if self._input_bytes_per_second > 0:
                self._audio_seconds_sent += _decoded_base64_length(event.get("audio")) / self._input_bytes_per_second
        elif event_type == "conversation.item.create":
            item: Final = as_str_mapping(event.get("item"))
            if item is not None and _is_billable_text_item(item):
                self._text_messages += 1

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
        if "turn_detection" in session:
            self._apply_turn_detection(session["turn_detection"])
        else:
            audio: Final = as_str_mapping(session.get("audio"))
            audio_input: Final = as_str_mapping(audio.get("input")) if audio is not None else None
            if audio_input is not None and "turn_detection" in audio_input:
                self._apply_turn_detection(audio_input["turn_detection"])

        input_bps: Final = _beta_audio_format_bytes_per_second(session.get("input_audio_format"))
        if input_bps is not None:
            self._input_bytes_per_second = input_bps
        output_bps: Final = _beta_audio_format_bytes_per_second(session.get("output_audio_format"))
        if output_bps is not None:
            self._output_bytes_per_second = output_bps

        audio_config: Final = as_str_mapping(session.get("audio"))
        if audio_config is None:
            return
        input_config: Final = as_str_mapping(audio_config.get("input"))
        if input_config is not None:
            input_format: Final = as_str_mapping(input_config.get("format"))
            if input_format is not None:
                self._input_bytes_per_second = _audio_format_bytes_per_second(input_format)
        output_config: Final = as_str_mapping(audio_config.get("output"))
        if output_config is not None:
            output_format: Final = as_str_mapping(output_config.get("format"))
            if output_format is not None:
                self._output_bytes_per_second = _audio_format_bytes_per_second(output_format)

    def _apply_turn_detection(self, value: object) -> None:
        if value is None:
            self._push_to_talk = True
        elif isinstance(value, Mapping):
            self._push_to_talk = False

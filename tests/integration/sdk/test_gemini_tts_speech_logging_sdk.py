from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.wire import wire_server
from integration.providers._gemini_tts_voice_support import (
    AUDIO_TOKENS,
    BACKEND,
    GEMINI_API_KEY,
    PCM,
    PROMPT_TOKENS,
    gemini_peer,
    marker,
    received_text,
    wav_payload,
)

import litellm
from litellm.integrations.custom_logger import CustomLogger

BINARY_RESPONSE: Final = "HttpxBinaryResponseContent"


@dataclass(frozen=True, slots=True)
class _Logged:
    call_type: object
    response_type: str
    tokens: tuple[object, object]


class _Recorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._lock: Final = threading.Lock()
        self.events: tuple[_Logged, ...] = ()

    def log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        payload: Final = kwargs.get("standard_logging_object")
        logged: Final = payload if isinstance(payload, dict) else {}
        event: Final = _Logged(
            call_type=kwargs.get("call_type"),
            response_type=type(response_obj).__name__,
            tokens=(logged.get("prompt_tokens"), logged.get("completion_tokens")),
        )
        with self._lock:
            self.events = (*self.events, event)


def test_sync_speech_logs_one_success_event_carrying_the_reply_token_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: Final = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    with wire_server(gemini_peer) as wire:
        monkeypatch.setenv("GEMINI_API_BASE", wire.url)
        text: Final = marker()
        audio: Final = litellm.speech(model=f"gemini/{BACKEND}", input=text, voice="Kore", api_key=GEMINI_API_KEY)
        assert wav_payload(audio.content) == PCM
        assert sum(1 for request in wire.drain() if received_text(request) == text) == 1
    events: Final = eventually(
        lambda: recorder.events,
        lambda found: any(event.response_type == BINARY_RESPONSE for event in found),
        seconds=30,
    )
    assert events == (_Logged("speech", BINARY_RESPONSE, (PROMPT_TOKENS, AUDIO_TOKENS)),)

"""Shared helpers for the second round of Google AI Studio contracts (`test_dec1_gemini2_*`).

Round one (`dec1_gemini_support.py`) owns the rig: one `gemini/<model>` deployment whose `api_base` is an
owned loopback peer, the synthetic `x-goog-api-key`, and the Gemini reply builders. This module only adds
what the round two files need on top of it: arbitrary SSE frame sequences (tool calls, safety stops,
mid-stream error frames), `functionCall` parts with thought signatures, readers for `generationConfig`
and streamed chunks, a responder that fails a set number of times before answering, and SpendLogs
helpers. Nothing here builds a URL on a Google host.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import httpx
from integration._support.client import Gateway, eventually, list_value, object_value
from integration._support.wire import Reply, Request
from integration.providers.dec1_gemini_support import FLASH, gemini_echo, path_of, spend_rows, usage_metadata
from integration.providers.dec1_vertex_support import at, json_of, sse_payloads
from pydantic import JsonValue

WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
TIME_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Look up the local time for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
SAMPLING_OFF_MODEL: Final = "gemini-3.6-flash"
"""Flagged `supports_sampling_params: false` in `model_prices_and_context_window.json` (`gemini/gemini-3.6-flash`)."""
TTS_MODEL: Final = "gemini-2.5-flash-preview-tts"
"""A model whose name contains `tts`, so `audio` is a supported param (`GoogleAIStudioGeminiConfig.is_model_gemini_audio_model`)."""
SIGNATURE: Final = "c2lnLWRlYzEtZ2VtaW5pMg=="
"""A synthetic base64 thought signature."""


def function_call(
    name: str, args: Mapping[str, JsonValue], *, signature: str | None = None, call_id: str | None = None
) -> dict[str, JsonValue]:
    """A Gemini `functionCall` part, optionally carrying `thoughtSignature` and a native call `id`."""
    call: Final[dict[str, JsonValue]] = {"name": name, "args": dict(args)}
    if call_id is not None:
        call["id"] = call_id
    part: Final[dict[str, JsonValue]] = {"functionCall": call}
    if signature is not None:
        part["thoughtSignature"] = signature
    return part


def frame(
    parts: Sequence[JsonValue] = (),
    *,
    finish: str | None = None,
    usage: Mapping[str, JsonValue] | None = None,
    **extra: JsonValue,
) -> dict[str, JsonValue]:
    """One `streamGenerateContent` frame with one candidate."""
    candidate: Final[dict[str, JsonValue]] = {"content": {"role": "model", "parts": list(parts)}, "index": 0}
    if finish is not None:
        candidate["finishReason"] = finish
    built: Final[dict[str, JsonValue]] = {"candidates": [candidate], "modelVersion": FLASH, **extra}
    if usage is not None:
        built["usageMetadata"] = dict(usage)
    return built


def sse_reply(frames: Sequence[Mapping[str, JsonValue]], headers: Mapping[str, str] | None = None) -> Reply:
    """An SSE answer made of the given frames, each as one `data:` event."""
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(dict(item))}\r\n\r\n".encode() for item in frames),
        headers=dict(headers or {}),
    )


def stream_or_json(stream: Reply, unary: Reply) -> Callable[[Request], Reply]:
    """Answer `:streamGenerateContent` with `stream` and every other model call with `unary`."""

    def respond(request: Request) -> Reply:
        return stream if ":streamGenerateContent" in path_of(request) else unary

    return respond


def generation_config(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(body.get("generationConfig") or {})


def chunks(text: str) -> tuple[dict[str, JsonValue], ...]:
    """The chat chunks of a streamed answer that carry a `choices` list."""
    return tuple(item for item in sse_payloads(text) if item.get("choices"))


def deltas(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(object_value(object_value(list_value(item["choices"])[0]).get("delta") or {}) for item in chunks(text))


def finish_reasons(text: str) -> tuple[str, ...]:
    return tuple(
        str(object_value(list_value(item["choices"])[0])["finish_reason"])
        for item in chunks(text)
        if object_value(list_value(item["choices"])[0]).get("finish_reason")
    )


def streamed_tool_calls(text: str) -> list[dict[str, JsonValue]]:
    return [object_value(call) for delta in deltas(text) for call in list_value(delta.get("tool_calls") or [])]


def error_message(response: httpx.Response) -> str:
    body: Final = json_of(response)
    return str(at(body, "error", "message"))


def fails_then_answers(
    failure: Reply, failures: int = 1, then: Callable[[Request], Reply] = gemini_echo
) -> Callable[[Request], Reply]:
    """Answer the first `failures` model calls with `failure`, then with `then`."""
    lock: Final = threading.Lock()
    seen: Final[list[int]] = []

    def respond(request: Request) -> Reply:
        with lock:
            seen.append(1)
            count: Final = len(seen)
        return failure if count <= failures else then(request)

    return respond


def one_row(gateway: Gateway, model_group: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: spend_rows(gateway, model_group), lambda found: len(found) >= 1, seconds=90)
    assert len(rows) == 1, rows
    return rows[0]


def spend_of(row: Mapping[str, JsonValue]) -> float:
    return float(str(row["spend"]))


def usage_with(prompt: int, candidates: int, **extra: JsonValue) -> dict[str, JsonValue]:
    return usage_metadata(prompt, candidates, **extra)

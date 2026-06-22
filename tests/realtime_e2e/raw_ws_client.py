"""Minimal raw-WebSocket client for the litellm proxy realtime endpoint.

This is the Layer 1 assertion backbone: it speaks the OpenAI GA realtime event
schema directly so the same test body validates every provider through the
proxy's normalization layer. No audio, no VAD, no third-party client -> the
assertions stay deterministic.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable
from urllib.parse import urlencode

import websockets


@dataclass(frozen=True, slots=True)
class RealtimeEvent:
    type: str
    raw: dict


def _decode(message: str | bytes) -> dict:
    text = message.decode("utf-8") if isinstance(message, bytes) else message
    return json.loads(text)


class RealtimeConnection:
    def __init__(self, ws: websockets.ClientConnection) -> None:
        self._ws = ws

    async def send_event(self, event: dict) -> None:
        await self._ws.send(json.dumps(event))

    async def collect_until(
        self,
        stop: Callable[[RealtimeEvent], bool],
        timeout: float,
    ) -> tuple[RealtimeEvent, ...]:
        """Read events until ``stop`` returns True. Raises TimeoutError if the
        stop event never arrives within ``timeout`` seconds -> a real failure."""

        async def _collect() -> list[RealtimeEvent]:
            collected: list[RealtimeEvent] = []
            async for message in self._ws:
                data = _decode(message)
                event = RealtimeEvent(type=str(data.get("type", "")), raw=data)
                collected.append(event)
                if stop(event):
                    break
            return collected

        return tuple(await asyncio.wait_for(_collect(), timeout=timeout))


@asynccontextmanager
async def connect_realtime(
    *,
    proxy_ws_url: str,
    api_key: str,
    model: str,
    intent: str | None = None,
    guardrails: str | None = None,
    open_timeout: float = 15.0,
) -> AsyncIterator[RealtimeConnection]:
    params = {"model": model}
    if intent is not None:
        params["intent"] = intent
    if guardrails is not None:
        params["guardrails"] = guardrails
    url = f"{proxy_ws_url.rstrip('/')}/v1/realtime?{urlencode(params)}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with websockets.connect(
        url, additional_headers=headers, open_timeout=open_timeout
    ) as ws:
        yield RealtimeConnection(ws)


def is_type(event_type: str) -> Callable[[RealtimeEvent], bool]:
    return lambda event: event.type == event_type


def of_type(
    events: tuple[RealtimeEvent, ...], event_type: str
) -> tuple[RealtimeEvent, ...]:
    return tuple(e for e in events if e.type == event_type)


def last_of_type(
    events: tuple[RealtimeEvent, ...], event_type: str
) -> RealtimeEvent | None:
    matches = of_type(events, event_type)
    return matches[-1] if matches else None


def concat_deltas(events: tuple[RealtimeEvent, ...], delta_type: str) -> str:
    return "".join(e.raw.get("delta", "") for e in of_type(events, delta_type))


def response_transcript(events: tuple[RealtimeEvent, ...]) -> str:
    """Concatenated assistant output across whichever delta channel the provider
    used (text modality vs audio-transcript)."""
    return concat_deltas(events, "response.text.delta") or concat_deltas(
        events, "response.audio_transcript.delta"
    )


def function_call_item(events: tuple[RealtimeEvent, ...]) -> dict | None:
    """The first completed function_call output item, if the model called a tool."""
    for event in of_type(events, "response.output_item.done"):
        item = event.raw.get("item", {})
        if item.get("type") == "function_call":
            return item
    return None


def user_text_event(text: str) -> dict:
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def function_call_output_event(call_id: str, output: dict) -> dict:
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(output),
        },
    }

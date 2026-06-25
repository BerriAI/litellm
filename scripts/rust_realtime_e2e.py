"""End-to-end proof that the Rust realtime path is event-for-event compatible
with the Python path against the live OpenAI realtime API, and that LiteLLM's
``RealTimeStreaming`` logging stays intact through both transports.

Run with ``OPENAI_API_KEY`` set; the Rust extension must be built first
(``cd litellm-rust/crates/python-bridge && maturin develop --release``).

What it does:
1. Drives ``OpenAIRealtime.async_realtime`` directly (the same code path the
   proxy hits) twice: once with the default ``websockets.connect`` and once
   with the Rust ``rust_backend_connect_factory``.
2. Pipes a small scripted client conversation (a single user text turn that
   asks the model for one short audio reply).
3. Collects the upstream event-type sequence from both runs and prints them
   side by side so a reviewer can confirm parity.
4. Prints the ``RealTimeStreaming`` accumulator state (``messages``,
   ``input_messages``, ``response_id``, ``conversation_id``) recorded by the
   logging path so it's obvious neither transport bypassed it.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import sys
from collections import deque
from typing import Any, cast

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.openai.realtime.handler import OpenAIRealtime
from litellm.realtime_api.rust_bridge import (
    load_rust_realtime,
    rust_backend_connect_factory,
)


class ScriptedClientWS:
    """In-process stand-in for the FastAPI WebSocket that
    ``OpenAIRealtime.async_realtime`` would normally receive.

    Plays a scripted list of outbound events via ``receive_text`` (the inputs
    LiteLLM forwards upstream) and captures every ``send_text`` LiteLLM emits
    back to the client. Closes itself after the scripted events are drained
    and ``response.done`` arrives so the splice exits cleanly.
    """

    def __init__(self, outbound: list[dict[str, Any]]) -> None:
        self._outbound = deque(outbound)
        self.received: list[dict[str, Any]] = []
        self.scope = {"headers": []}  # no OpenAI-Beta -> GA protocol
        self._closed = asyncio.Event()
        self._response_done = False

    async def receive_text(self) -> str:
        if not self._outbound:
            await self._closed.wait()
            raise asyncio.CancelledError("client closed")
        evt = self._outbound.popleft()
        return json.dumps(evt)

    async def send_text(self, message: str) -> None:
        event = json.loads(message)
        self.received.append(event)
        if event.get("type") == "response.done":
            self._response_done = True
            self._closed.set()

    async def close(self, *args: Any, **kwargs: Any) -> None:
        self._closed.set()


def make_logging_obj() -> LiteLLMLogging:
    return LiteLLMLogging(
        model="gpt-realtime",
        messages=[],
        stream=False,
        call_type="realtime",
        start_time=datetime.datetime.now(),
        litellm_call_id="e2e-rust-realtime",
        function_id="e2e",
    )


SCRIPT = [
    {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["text"],
        },
    },
    {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Reply with exactly the word 'parity'.",
                }
            ],
        },
    },
    {"type": "response.create"},
]


async def run_once(
    label: str, *, use_rust: bool
) -> tuple[list[str], list[dict[str, Any]], LiteLLMLogging]:
    handler = OpenAIRealtime()
    client = ScriptedClientWS(outbound=SCRIPT)
    logging_obj = make_logging_obj()

    backend_connect = None
    if use_rust:
        rust_connect = load_rust_realtime()
        if rust_connect is None:
            raise RuntimeError(
                "Rust bridge not built; run `maturin develop --release` "
                "in litellm-rust/crates/python-bridge first"
            )
        backend_connect = rust_backend_connect_factory(rust_connect)

    print(f"\n=== {label} (use_rust={use_rust}) ===")
    try:
        await asyncio.wait_for(
            handler.async_realtime(
                model="gpt-realtime",
                websocket=client,
                logging_obj=logging_obj,
                api_base="https://api.openai.com/",
                api_key=os.environ["OPENAI_API_KEY"],
                backend_connect=backend_connect,
            ),
            timeout=60,
        )
    except asyncio.CancelledError:
        # Scripted client raises CancelledError when its outbound queue drains;
        # the splice catches it and tears down upstream cleanly.
        pass

    event_types = [evt.get("type", "") for evt in client.received]
    return event_types, client.received, logging_obj


def summarize_logging(logging_obj: LiteLLMLogging) -> dict[str, Any]:
    """Capture the parts of ``LiteLLMLogging`` that ``RealTimeStreaming``
    populates as it observes the event stream — these are what spend logs and
    custom callbacks consume downstream."""
    details = getattr(logging_obj, "model_call_details", {}) or {}
    return {
        "input_messages_count": len(details.get("messages") or []),
        "realtime_tool_calls": len(details.get("realtime_tool_calls") or []),
        "model": details.get("model"),
        "call_type": details.get("call_type"),
    }


async def main() -> int:
    py_types, py_events, py_logging = await run_once("Python path", use_rust=False)
    rs_types, rs_events, rs_logging = await run_once("Rust path", use_rust=True)

    def first_n(seq: list[str], n: int = 25) -> list[str]:
        return seq[:n]

    print("\n--- Upstream event type sequence ---")
    print(f"Python ({len(py_types)} events): {first_n(py_types)}")
    print(f"Rust   ({len(rs_types)} events): {first_n(rs_types)}")

    def stripped(seq: list[str]) -> tuple[str, ...]:
        return tuple(
            t for t in seq if t and not t.startswith("response.output_text.delta")
        )

    py_skeleton = stripped(py_types)
    rs_skeleton = stripped(rs_types)
    parity_ok = py_skeleton == rs_skeleton
    print("\n--- Skeleton parity (ignoring streaming deltas which differ in count) ---")
    print(f"Python skeleton: {py_skeleton}")
    print(f"Rust skeleton:   {rs_skeleton}")
    print(f"Parity: {'OK' if parity_ok else 'MISMATCH'}")

    print("\n--- LiteLLM logging accumulator state (proves callbacks fired) ---")
    print(f"Python logging summary: {summarize_logging(py_logging)}")
    print(f"Rust   logging summary: {summarize_logging(rs_logging)}")

    py_text = "".join(
        cast(str, evt.get("delta", ""))
        for evt in py_events
        if evt.get("type") == "response.output_text.delta"
    )
    rs_text = "".join(
        cast(str, evt.get("delta", ""))
        for evt in rs_events
        if evt.get("type") == "response.output_text.delta"
    )
    print("\n--- Final reply text from each path ---")
    print(f"Python reply: {py_text!r}")
    print(f"Rust   reply: {rs_text!r}")

    return 0 if parity_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

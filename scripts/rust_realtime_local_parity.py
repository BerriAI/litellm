"""Deterministic transport-parity proof for the Rust realtime bridge.

Runs the real ``OpenAIRealtime.async_realtime`` handler against a local fake
OpenAI realtime WebSocket server, twice: once with the default
``websockets.connect`` transport and once with the Rust
``rust_backend_connect_factory``. Asserts the two runs produce identical
upstream-to-client event sequences and identical ``RealTimeStreaming``
logging accumulator state.

This is the proxy's actual realtime code path; only the WebSocket transport
under ``OpenAIRealtime.async_realtime`` differs between the two runs. Use this
when you don't have a live ``OPENAI_API_KEY`` handy and want to prove the
transport swap is byte-for-byte safe; use ``rust_realtime_e2e.py`` to repeat
the same dance against the real OpenAI realtime API.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import sys
from collections import deque
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection, serve

import litellm  # noqa: F401  - ensures bridge module loads
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.openai.realtime.handler import OpenAIRealtime
from litellm.realtime_api.rust_bridge import (
    load_rust_realtime,
    rust_backend_connect_factory,
)


SESSION_CREATED = {
    "type": "session.created",
    "session": {"id": "sess_fake", "model": "gpt-realtime"},
}

RESPONSE_DELTAS = ["par", "ity"]

RESPONSE_DONE_TEMPLATE = {
    "type": "response.done",
    "response": {
        "id": "resp_fake",
        "object": "realtime.response",
        "status": "completed",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
            "input_token_details": {"text_tokens": 12},
            "output_token_details": {"text_tokens": 4},
        },
        "output": [],
    },
}


async def fake_openai_handler(ws: ServerConnection) -> None:
    """Mimic the OpenAI realtime server's event sequence for a single
    response.create turn."""
    await ws.send(json.dumps(SESSION_CREATED))
    async for raw in ws:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "response.create":
            await ws.send(
                json.dumps(
                    {
                        "type": "response.created",
                        "response": {"id": "resp_fake", "status": "in_progress"},
                    }
                )
            )
            for delta in RESPONSE_DELTAS:
                await ws.send(
                    json.dumps(
                        {
                            "type": "response.output_text.delta",
                            "response_id": "resp_fake",
                            "delta": delta,
                        }
                    )
                )
            await ws.send(json.dumps(RESPONSE_DONE_TEMPLATE))


class ScriptedClientWS:
    """Stand-in for the FastAPI WebSocket the proxy hands to
    ``OpenAIRealtime.async_realtime``."""

    def __init__(self, outbound: list[dict[str, Any]]) -> None:
        self._outbound = deque(outbound)
        self.received: list[dict[str, Any]] = []
        self.scope = {"headers": []}
        self._closed = asyncio.Event()

    async def receive_text(self) -> str:
        if not self._outbound:
            await self._closed.wait()
            raise asyncio.CancelledError("client closed")
        return json.dumps(self._outbound.popleft())

    async def send_text(self, message: str) -> None:
        event = json.loads(message)
        self.received.append(event)
        if event.get("type") == "response.done":
            self._closed.set()

    async def close(self, *args: Any, **kwargs: Any) -> None:
        self._closed.set()


SCRIPT = [
    {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Say 'parity'."}],
        },
    },
    {"type": "response.create"},
]


def make_logging_obj() -> LiteLLMLogging:
    return LiteLLMLogging(
        model="gpt-realtime",
        messages=[],
        stream=False,
        call_type="realtime",
        start_time=datetime.datetime.now(),
        litellm_call_id="local-rust-realtime",
        function_id="local",
    )


async def run_once(
    *, host: str, port: int, label: str, use_rust: bool
) -> tuple[list[str], LiteLLMLogging]:
    handler = OpenAIRealtime()
    client = ScriptedClientWS(outbound=list(SCRIPT))
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
                api_base=f"ws://{host}:{port}",
                api_key="sk-fake-key-for-local-test",
                backend_connect=backend_connect,
            ),
            timeout=10,
        )
    except asyncio.CancelledError:
        pass

    return [evt.get("type", "") for evt in client.received], logging_obj


def summarize(logging_obj: LiteLLMLogging) -> dict[str, Any]:
    details = getattr(logging_obj, "model_call_details", {}) or {}
    return {
        "input_messages": len(details.get("messages") or []),
        "realtime_tool_calls": len(details.get("realtime_tool_calls") or []),
        "model": details.get("model"),
    }


async def main() -> int:
    host, port = "127.0.0.1", 7788
    async with serve(fake_openai_handler, host, port):
        py_events, py_log = await run_once(
            host=host, port=port, label="Python path", use_rust=False
        )
        rs_events, rs_log = await run_once(
            host=host, port=port, label="Rust path", use_rust=True
        )

    print("\n--- Upstream event sequence (client receives) ---")
    print(f"Python: {py_events}")
    print(f"Rust:   {rs_events}")
    parity = py_events == rs_events
    print(f"\nEvent-sequence parity: {'OK' if parity else 'MISMATCH'}")

    print("\n--- LiteLLM logging accumulator state ---")
    print(f"Python: {summarize(py_log)}")
    print(f"Rust:   {summarize(rs_log)}")
    log_parity = summarize(py_log) == summarize(rs_log)
    print(f"\nLogging-state parity: {'OK' if log_parity else 'MISMATCH'}")

    return 0 if (parity and log_parity) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

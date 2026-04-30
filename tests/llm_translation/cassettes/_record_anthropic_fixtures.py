"""Helper script that records Anthropic-shaped cassettes against a local mock.

This is a *one-shot* utility, not a test. It exists so we can deterministically
regenerate the canned Anthropic cassettes shipped under
``tests/llm_translation/cassettes/`` without spending real provider credits and
without needing an ``ANTHROPIC_API_KEY``.

Run it with::

    uv run python tests/llm_translation/cassettes/_record_anthropic_fixtures.py

The script:

1. Spins up a tiny in-process HTTP server that returns canned Anthropic
   ``/v1/messages`` payloads (one non-streaming, one SSE streaming).
2. Records LiteLLM's real outbound HTTP through vcrpy.
3. Rewrites the cassette URL/Host so replay matches genuine
   ``https://api.anthropic.com/v1/messages`` traffic.

If you want to refresh against the *real* Anthropic API instead, use the
``LITELLM_VCR_RECORD_MODE=once`` workflow described in
``tests/llm_translation/vcr_config.py`` — that path needs a real API key.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

import vcr  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import litellm  # noqa: E402

CASSETTE_DIR = Path(__file__).parent
MOCK_HOST = "127.0.0.1"
NON_STREAM_PORT = 18765
STREAM_PORT = 18766
REAL_ANTHROPIC_HOST = "api.anthropic.com"

NON_STREAM_RESPONSE: dict[str, Any] = {
    "id": "msg_01ABCDEFGHIJKLMNOPQRSTUV",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 11,
    },
}

STREAM_EVENTS: list[tuple[str, dict[str, Any]]] = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_01STREAMABCDEFGH",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 14, "output_tokens": 1},
            },
        },
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " from"},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " LiteLLM!"},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]


def _make_handler(mode: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any, **kwargs: Any) -> None:  # silence
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            if mode == "json":
                body = json.dumps(NON_STREAM_RESPONSE).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("anthropic-ratelimit-requests-limit", "4000")
                self.send_header("anthropic-ratelimit-requests-remaining", "3999")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for event_name, data in STREAM_EVENTS:
                    chunk = (
                        f"event: {event_name}\n" f"data: {json.dumps(data)}\n\n"
                    ).encode("utf-8")
                    self.wfile.write(chunk)
                    self.wfile.flush()

    return Handler


def _serve(port: int, mode: str) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((MOCK_HOST, port), _make_handler(mode))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _rewrite_cassette_to_real_host(path: Path, mock_host_port: str) -> None:
    """Replace mock host/port in the cassette with the real Anthropic host."""
    text = path.read_text()
    text = text.replace(f"http://{mock_host_port}", f"https://{REAL_ANTHROPIC_HOST}")
    text = text.replace(mock_host_port, REAL_ANTHROPIC_HOST)
    path.write_text(text)


def _consume(iterable: Iterable[Any]) -> None:
    for _ in iterable:
        pass


def record_non_streaming() -> None:
    cassette = CASSETTE_DIR / "anthropic_basic_completion.yaml"
    server = _serve(NON_STREAM_PORT, "json")
    try:
        my_vcr = vcr.VCR(
            record_mode="all",
            filter_headers=["authorization", "x-api-key", "anthropic-version"],
        )
        with my_vcr.use_cassette(str(cassette)):
            response = litellm.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello!"}],
                api_base=f"http://{MOCK_HOST}:{NON_STREAM_PORT}",
                api_key="sk-ant-recording",
            )
            assert response.choices[0].message.content
    finally:
        server.shutdown()
    _rewrite_cassette_to_real_host(cassette, f"{MOCK_HOST}:{NON_STREAM_PORT}")


def record_streaming() -> None:
    cassette = CASSETTE_DIR / "anthropic_streaming_completion.yaml"
    server = _serve(STREAM_PORT, "stream")
    try:
        my_vcr = vcr.VCR(
            record_mode="all",
            filter_headers=["authorization", "x-api-key", "anthropic-version"],
        )
        with my_vcr.use_cassette(str(cassette)):
            stream = litellm.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello!"}],
                api_base=f"http://{MOCK_HOST}:{STREAM_PORT}",
                api_key="sk-ant-recording",
                stream=True,
            )
            _consume(stream)
    finally:
        server.shutdown()
    _rewrite_cassette_to_real_host(cassette, f"{MOCK_HOST}:{STREAM_PORT}")


def main() -> None:
    os.environ.setdefault("LITELLM_LOG", "WARNING")
    record_non_streaming()
    record_streaming()
    print(f"Wrote cassettes to {CASSETTE_DIR}")


if __name__ == "__main__":
    main()

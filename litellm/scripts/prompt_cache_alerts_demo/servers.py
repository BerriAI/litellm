"""Local stand-ins for the demo, so it needs no provider keys and makes no model calls.

`anthropic` answers POST /v1/messages like the Anthropic Messages API. The cache counters it reports come from a
`[demo:write]`, `[demo:read]` or `[demo:miss]` tag in the last user message, which sits after the cached prefix.
`slack` accepts webhook posts and lists them at GET /received.
"""

import json
import sys
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

PORT: Final = 8080
PREFIX_TOKENS: Final = 2048
COUNTERS: Final[Mapping[str, tuple[int, int]]] = {
    "write": (0, PREFIX_TOKENS),
    "read": (PREFIX_TOKENS, 0),
    "miss": (0, 0),
}


def scenario(body: Mapping[str, object]) -> str:
    messages: Final = body.get("messages")
    last: Final = messages[-1] if isinstance(messages, list) and messages else {}
    content: Final = last.get("content") if isinstance(last, dict) else ""
    text: Final = json.dumps(content)
    return next((name for name in COUNTERS if f"[demo:{name}]" in text), "miss")


def message(body: Mapping[str, object]) -> bytes:
    read, write = COUNTERS[scenario(body)]
    return json.dumps(
        {
            "id": "msg_demo",
            "type": "message",
            "role": "assistant",
            "model": body.get("model"),
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 1,
                "cache_read_input_tokens": read,
                "cache_creation_input_tokens": write,
            },
        }
    ).encode()


class Received:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self._posts: tuple[object, ...] = ()

    def add(self, post: object) -> None:
        with self._lock:
            self._posts = (*self._posts, post)

    def snapshot(self) -> tuple[object, ...]:
        with self._lock:
            return self._posts


RECEIVED: Final = Received()


class Handler(BaseHTTPRequestHandler):
    mode: str = "anthropic"

    def _reply(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> object:
        length: Final = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, b"{}")
        elif self.mode == "slack" and self.path == "/received":
            posts: Final = RECEIVED.snapshot()
            self._reply(200, json.dumps({"count": len(posts), "posts": posts}).encode())
        else:
            self._reply(404, b"{}")

    def do_POST(self) -> None:
        body: Final = self._body()
        if self.mode == "anthropic" and self.path.startswith("/v1/messages") and isinstance(body, dict):
            self._reply(200, message(body))
        elif self.mode == "slack" and self.path == "/webhook":
            RECEIVED.add(body)
            self._reply(200, b"ok")
        else:
            self._reply(404, b"{}")


if __name__ == "__main__":
    Handler.mode = sys.argv[1]
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

"""
Minimal fake Anthropic-compatible backend for integration tests.

Echoes received request bodies so tests can assert what LiteLLM actually
sent to the upstream (thinking blocks, output_config, etc.).
"""

import socket
import threading
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request


class FakeAnthropicBackend:
    """Container for received requests + FastAPI app."""

    def __init__(self) -> None:
        self.received_requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.post("/v1/messages")
        async def anthropic_messages(request: Request) -> Dict[str, Any]:
            body = await request.json()
            with self._lock:
                self.received_requests.append({"path": "/v1/messages", "body": body})
            return {
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": body.get("model", "fake"),
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        @self.app.post("/v1/chat/completions")
        async def openai_chat(request: Request) -> Dict[str, Any]:
            body = await request.json()
            with self._lock:
                self.received_requests.append(
                    {"path": "/v1/chat/completions", "body": body}
                )
            return {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

    def pop_requests(self) -> List[Dict[str, Any]]:
        """Return and clear all received requests."""
        with self._lock:
            reqs = list(self.received_requests)
            self.received_requests.clear()
            return reqs


def start_fake_backend() -> tuple:
    """Start the fake backend on an ephemeral port.

    Returns:
        (base_url, backend, server, thread, sock)
    """
    import asyncio
    import socket
    import time

    backend = FakeAnthropicBackend()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()

    config = uvicorn.Config(backend.app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve(sockets=[sock]))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    start_time = time.time()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("Fake backend failed to start")
        if time.time() - start_time > 15.0:
            raise TimeoutError("Fake backend did not start in time")
        time.sleep(0.05)

    return f"http://{host}:{port}", backend, server, thread, sock


def stop_fake_backend(
    server: uvicorn.Server, thread: threading.Thread, sock: socket.socket
) -> None:
    server.should_exit = True
    thread.join(timeout=10)
    sock.close()

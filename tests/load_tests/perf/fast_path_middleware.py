"""ASGI middleware that hands eligible POST /v1/chat/completions requests to
a Rust extension and returns the response directly, bypassing FastAPI +
starlette + pydantic + the litellm router chain.

Ineligible requests (different path, different method, model not in the
mock-map) are passed through to the inner ASGI app via a receive-wrapper
that replays the body we already read.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Dict

import litellm_fast_path  # Rust extension (built via maturin)

ASGIApp = Callable[[dict, Callable, Callable], Awaitable[None]]


class FastPathMiddleware:
    """Wraps an inner ASGI app. Short-circuits configured mock requests.

    mock_map: dict of {model_name: mock_response_content}. Requests whose
    `model` field is a key in this map get a Rust-built response; others
    fall through.
    """

    def __init__(self, app: ASGIApp, mock_map: Dict[str, str]) -> None:
        self.app = app
        self.mock_map = mock_map

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if not (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/v1/chat/completions"
        ):
            await self.app(scope, receive, send)
            return

        # Accumulate the request body (needed to peek at `model`).
        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                # Lifespan / disconnect / unexpected — delegate to inner app.
                await self.app(scope, _replay_receive(message, receive), send)
                return
            body.extend(message.get("body", b""))
            more_body = message.get("more_body", False)

        response_bytes = litellm_fast_path.try_build_mock_response(
            bytes(body), self.mock_map
        )

        if response_bytes is not None:
            # Fast path hit — send the Rust-built response directly.
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(response_bytes)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": response_bytes})
            return

        # Fast path miss — replay the body to the inner app.
        await self.app(scope, _build_replay_receive(bytes(body)), send)


def _build_replay_receive(body: bytes) -> Callable:
    """Return a receive() callable that yields the pre-read body once, then
    blocks forever (matching ASGI contract: no more body messages after
    more_body=False)."""
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        # After the body, ASGI spec says receive should suspend (disconnect
        # messages may come later). We return a disconnect so the inner app
        # cleans up promptly if it keeps reading.
        return {"type": "http.disconnect"}

    return receive


def _replay_receive(first_message: dict, inner_receive: Callable) -> Callable:
    """Receive wrapper for the odd case where the first message was not
    http.request — hand it back, then delegate to the original receive."""
    delivered = False

    async def receive() -> dict:
        nonlocal delivered
        if not delivered:
            delivered = True
            return first_message
        return await inner_receive()

    return receive

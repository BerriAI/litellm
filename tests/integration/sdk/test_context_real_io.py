import asyncio
import base64
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from typing import Final, Literal

import pytest

import litellm
from litellm.llms.anthropic.messages.context_management.inference_view import CountContextEdits
from litellm.llms.vertex_ai.count_tokens.context import VertexGeminiCountContextError, gemini_count_contents
_NO_EDITS: Final = CountContextEdits(spec=None, model="gemini/gemini-2.5-flash", tools=None)


_PNG_BYTES: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_LOOP_PROOF_TIMEOUT_SECONDS: Final = 5


@dataclass(frozen=True, slots=True)
class _LoopProbe:
    request_arrived: Event = field(default_factory=Event)
    loop_ran: Event = field(default_factory=Event)
    loop_ran_while_download_pending: Event = field(default_factory=Event)


def _png_handler_waiting_for_loop(probe: _LoopProbe) -> type[BaseHTTPRequestHandler]:
    class _PngHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            probe.request_arrived.set()
            if probe.loop_ran.wait(_LOOP_PROOF_TIMEOUT_SECONDS):
                probe.loop_ran_while_download_pending.set()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_PNG_BYTES)))
            self.end_headers()
            self.wfile.write(_PNG_BYTES)

        def log_message(self, format: str, *args: object) -> None:
            return None

    return _PngHandler


@contextmanager
def _png_server_waiting_for_loop(probe: _LoopProbe) -> Generator[str, None, None]:
    with ThreadingHTTPServer(("127.0.0.1", 0), _png_handler_waiting_for_loop(probe)) as server:
        thread: Final = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()


async def _run_loop_once_download_starts(probe: _LoopProbe) -> None:
    assert await asyncio.to_thread(probe.request_arrived.wait, _LOOP_PROOF_TIMEOUT_SECONDS)
    probe.loop_ran.set()


@pytest.mark.parametrize("custom_llm_provider", ("vertex_ai", "gemini"))
async def test_remote_image_downloads_without_blocking_the_event_loop(
    custom_llm_provider: Literal["vertex_ai", "gemini"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    probe: Final = _LoopProbe()
    with _png_server_waiting_for_loop(probe) as base_url:
        image_url: Final = f"{base_url}/{uuid.uuid4().hex}.png"
        _, contents = await asyncio.gather(
            _run_loop_once_download_starts(probe),
            gemini_count_contents(
                (
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe it"},
                            {"type": "image", "source": {"type": "url", "url": image_url}},
                        ],
                    },
                ),
                "anthropic",
                "gemini-2.5-flash",
                custom_llm_provider,
                {},
            ),
        )

    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": "Describe it"},
                {"inline_data": {"data": base64.b64encode(_PNG_BYTES).decode(), "mime_type": "image/png"}},
            ],
        }
    ]
    assert probe.loop_ran_while_download_pending.is_set()


class _MissingImageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return None


@pytest.mark.parametrize("custom_llm_provider", ("vertex_ai", "gemini"))
async def test_image_that_cannot_be_fetched_returns_an_error_value(
    custom_llm_provider: Literal["vertex_ai", "gemini"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    with ThreadingHTTPServer(("127.0.0.1", 0), _MissingImageHandler) as server:
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            contents = await gemini_count_contents(
                (
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": f"http://127.0.0.1:{server.server_address[1]}/{uuid.uuid4().hex}.png",
                                },
                            }
                        ],
                    },
                ),
                "anthropic",
                "gemini-2.5-flash",
                custom_llm_provider,
                {},
            )
        finally:
            server.shutdown()

    assert isinstance(contents, VertexGeminiCountContextError)

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest

from litellm import Router
from tests.test_litellm_rust.messages.data import MESSAGES_EVENTS, MESSAGES_MODEL
from tests.test_litellm_rust.support.recording_server import RecordedRequest, RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark: Final = pytest.mark.requires_rust_extension


@pytest.fixture
def unfinished_messages_peer() -> Iterator[tuple[RecordingServer, threading.Event]]:
    requests: Final[list[RecordedRequest]] = []  # mutable-ok: HTTP handler records observed requests
    disconnected: Final = threading.Event()
    prefix: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS[:1]).payloads()[0]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body: Final = self.rfile.read(int(self.headers.get("content-length", "0")))
            requests.append(
                RecordedRequest(
                    method="POST",
                    path=self.path,
                    headers={name.lower(): value for name, value in self.headers.items()},
                    raw_body=body,
                    body=json.loads(body),
                )
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(prefix) + 1024))
            self.end_headers()
            self.wfile.write(prefix)
            self.wfile.flush()
            self.connection.settimeout(5)
            try:
                if self.rfile.read(1) == b"":
                    disconnected.set()
            except ConnectionResetError:
                disconnected.set()
            except TimeoutError:
                pass

        def log_message(self, format: str, *args: object) -> None:
            pass

    http: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http.daemon_threads = True
    thread: Final = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield RecordingServer(http, requests, [], ResponseSpec(body={})), disconnected
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("read_first", [False, True], ids=["before-first-read", "after-first-read"])
async def test_router_closing_a_messages_stream_releases_the_unfinished_provider_connection(
    unfinished_messages_peer: tuple[RecordingServer, threading.Event], read_first: bool
) -> None:
    server, disconnected = unfinished_messages_peer
    router: Final = Router(
        model_list=[
            {
                "model_name": "messages",
                "litellm_params": {"model": MESSAGES_MODEL, "api_base": server.base_url, "api_key": "fake-key"},
            }
        ],
        num_retries=0,
        disable_cooldowns=True,
    )
    stream: Final = await router.aanthropic_messages(
        model="messages", messages=[{"role": "user", "content": "hello"}], max_tokens=64, stream=True
    )
    if read_first:
        assert await anext(stream) == ResponseSpec(body=None, events=MESSAGES_EVENTS[:1]).payloads()[0]
    await stream.aclose()
    await stream.aclose()

    assert await asyncio.to_thread(disconnected.wait, 2), "closing the Router stream must close upstream"
    assert_served_natively(server, stream)
    with pytest.raises(StopAsyncIteration):
        await anext(stream)

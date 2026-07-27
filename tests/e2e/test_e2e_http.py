"""Harness coverage for the transport's transient-failure retry (no proxy needed).

A pod scale-down makes the load balancer answer 502 for a second or two, which used
to kill whatever test was mid-call. These pin which failures are retried and, just
as importantly, which are not: a 429 or a 500 is the app's own answer that
rate-limit and error-mapping tests assert on, so it must arrive unretried.

The scripted server serves real HTTP on an ephemeral port and records every hit, so
"did it retry" is a request count rather than a timing guess.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic import BaseModel

from e2e_http import URL, NoBody, Result, post, probe, send

DROP_CONNECTION = 0
SLOW_RESPONSE_SECONDS = 2.0
RETRY_BUDGET_CEILING_SECONDS = 40.0


class OkBody(BaseModel):
    ok: bool


class QuietServer(ThreadingHTTPServer):
    """A client that times out or walks away mid-response is the point of some of
    these tests, so the resulting broken pipe is expected, not a server crash."""

    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        return


@dataclass(frozen=True, slots=True)
class ScriptedServer:
    url: URL
    hits: list[int] = field(default_factory=list)


@contextmanager
def scripted_server(statuses: Sequence[int], *, slow: bool = False) -> Generator[ScriptedServer]:
    """Answer with `statuses` in order, repeating the last one forever. A status of
    DROP_CONNECTION closes the socket without answering (what a load balancer does
    when it drops a backend mid-request); `slow` stalls every response so the client
    can hit its own read timeout."""
    scripted = iter(statuses)
    hits: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _answer(self) -> None:
            status = next(scripted, statuses[-1])
            hits.append(status)
            if slow:
                time.sleep(SLOW_RESPONSE_SECONDS)
            if status == DROP_CONNECTION:
                self.close_connection = True
                return
            body = b'{"ok": true}' if status == 200 else b'{"error": "transient"}'
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            _ = self.wfile.write(body)

        do_GET = _answer
        do_POST = _answer

        def log_message(self, format: str, *args: object) -> None:
            return

    server = QuietServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ScriptedServer(url=URL(f"http://127.0.0.1:{server.server_port}/anything"), hits=hits)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post(url: URL, timeout: float = 30.0) -> Result[OkBody]:
    return post(url, headers=NoBody(), json=NoBody(), response_type=OkBody, timeout=timeout)


class TestTransientRetry:
    def test_post_retries_a_bad_gateway_until_it_succeeds(self) -> None:
        with scripted_server((502, 503, 200)) as server:
            result = _post(server.url)
        assert result.kind == "success", result
        assert server.hits == [502, 503, 200]

    def test_post_retries_a_dropped_connection_until_it_succeeds(self) -> None:
        with scripted_server((DROP_CONNECTION, 200)) as server:
            result = _post(server.url)
        assert result.kind == "success", result
        assert server.hits == [DROP_CONNECTION, 200]

    def test_post_gives_up_on_a_permanent_bad_gateway_with_the_same_error_as_before(self) -> None:
        started = time.monotonic()
        with scripted_server((502,)) as server:
            result = _post(server.url)
        elapsed = time.monotonic() - started
        assert result.kind == "unknown", result
        assert result.status_code == 502
        assert "transient" in result.body
        assert len(server.hits) >= 4, server.hits
        assert elapsed < RETRY_BUDGET_CEILING_SECONDS, elapsed

    def test_post_does_not_retry_a_rate_limit(self) -> None:
        with scripted_server((429, 200)) as server:
            result = _post(server.url)
        assert result.kind == "rate_limited", result
        assert server.hits == [429]

    def test_post_does_not_retry_a_server_error(self) -> None:
        with scripted_server((500, 200)) as server:
            result = _post(server.url)
        assert result.kind == "unknown", result
        assert result.status_code == 500
        assert server.hits == [500]

    def test_post_does_not_retry_a_client_error(self) -> None:
        with scripted_server((400, 200)) as server:
            result = _post(server.url)
        assert result.kind == "unknown", result
        assert result.status_code == 400
        assert server.hits == [400]

    def test_post_does_not_retry_a_read_timeout(self) -> None:
        with scripted_server((200,), slow=True) as server:
            result = _post(server.url, timeout=0.25)
        assert result.kind == "network", result
        assert server.hits == [200]

    def test_probe_retries_a_bad_gateway_until_it_succeeds(self) -> None:
        with scripted_server((503, 200)) as server:
            result = probe(server.url, headers=NoBody(), params=NoBody())
        assert result.status_code == 200, result
        assert result.healthy
        assert server.hits == [503, 200]

    def test_send_retries_a_bad_gateway_until_it_succeeds(self) -> None:
        with scripted_server((504, 200)) as server:
            result = send(server.url, headers=NoBody(), json=NoBody())
        assert result.ok, result
        assert server.hits == [504, 200]

    def test_send_leaves_a_streaming_call_alone(self) -> None:
        with scripted_server((502, 200)) as server:
            result = send(server.url, headers=NoBody(), json=NoBody(), stream=True)
        assert result.status_code == 502, result
        assert server.hits == [502]

from __future__ import annotations

import threading
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final, TypeVar


class LocalHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class LocalHttpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def write_chunk(self, chunk: bytes) -> None:
        self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
        self.wfile.write(chunk)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def finish_chunked(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def write_chunked(self, chunks: Iterable[bytes]) -> None:
        for chunk in chunks:
            self.write_chunk(chunk)
        self.finish_chunked()

    def log_message(self, format: str, *args: object) -> None:
        return


ServerT = TypeVar("ServerT", bound=LocalHttpServer)


@contextmanager
def serve_in_thread(server: ServerT, poll_interval: float = 0.5) -> Generator[ServerT]:
    thread: Final = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": poll_interval},
        daemon=True,
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

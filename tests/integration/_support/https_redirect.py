from __future__ import annotations

import socket
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final


def _pipe(source: socket.socket, destination: socket.socket) -> None:
    try:
        for payload in iter(lambda: source.recv(65536), b""):
            destination.sendall(payload)
    except OSError:
        return
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


@contextmanager
def redirect_https_host(host: str, port: int, *, to_port: int) -> Generator[str, None, None]:
    """Yields an HTTPS_PROXY url that sends host:port to 127.0.0.1:to_port and refuses anything else."""
    authority: Final = f"{host}:{port}"

    class _RedirectHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30

        def do_CONNECT(self) -> None:
            if self.path != authority:
                self.send_error(403)
                return

            self.send_response(200, "Connection Established")
            self.end_headers()
            try:
                with socket.create_connection(("127.0.0.1", to_port), timeout=10) as upstream:
                    client_to_upstream: Final = threading.Thread(
                        target=_pipe, args=(self.connection, upstream), daemon=True
                    )
                    upstream_to_client: Final = threading.Thread(
                        target=_pipe, args=(upstream, self.connection), daemon=True
                    )
                    client_to_upstream.start()
                    upstream_to_client.start()
                    client_to_upstream.join(timeout=30)
                    upstream_to_client.join(timeout=30)
            except OSError:
                return

        def log_message(self, format: str, *args: object) -> None:
            pass

    class _RedirectProxy(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 16

    with _RedirectProxy(("127.0.0.1", 0), _RedirectHandler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=6)
            assert not thread.is_alive(), "CONNECT tunnel server survived cleanup"
            server.server_close()

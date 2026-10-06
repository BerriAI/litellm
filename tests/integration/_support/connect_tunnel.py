from __future__ import annotations

import socket
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final


def _relay(source: socket.socket, destination: socket.socket) -> None:
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
def connect_tunnel(
    connect_host: str,
    connect_port: int,
    forward_host: str,
    forward_port: int,
) -> Generator[str, None, None]:
    authority: Final = f"{connect_host}:{connect_port}"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30

        def do_CONNECT(self) -> None:
            if self.path != authority:
                self.send_error(403)
                return

            self.send_response(200, "Connection Established")
            self.end_headers()
            try:
                with socket.create_connection((forward_host, forward_port), timeout=10) as upstream:
                    client_to_upstream: Final = threading.Thread(
                        target=_relay, args=(self.connection, upstream), daemon=True
                    )
                    upstream_to_client: Final = threading.Thread(
                        target=_relay, args=(upstream, self.connection), daemon=True
                    )
                    client_to_upstream.start()
                    upstream_to_client.start()
                    client_to_upstream.join(timeout=30)
                    upstream_to_client.join(timeout=30)
            except OSError:
                return

        def log_message(self, format: str, *args: object) -> None:
            pass

    class OwnedProxy(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 16

    with OwnedProxy(("127.0.0.1", 0), Handler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=6)
            assert not thread.is_alive(), "CONNECT tunnel server survived cleanup"
            server.server_close()

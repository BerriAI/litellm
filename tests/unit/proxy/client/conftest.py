import threading

import pytest


@pytest.fixture
def hanging_server():
    """A server that accepts the connection and never answers, so only a timeout ends the call."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn

    stop: threading.Event = threading.Event()

    class SilentRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _hang(self):
            stop.wait(timeout=30)

        do_GET = _hang
        do_POST = _hang

        def log_message(self, format, *args):
            pass

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedServer(("127.0.0.1", 0), SilentRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

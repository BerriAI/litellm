"""
QW-4: Standalone TCP health check server.

Runs in a background thread on a separate port, completely independent of the
main FastAPI ASGI stack. If the event loop is blocked, middleware panics, or
a guardrail hook crashes, this endpoint still responds — preventing Kubernetes
from killing pods that can still serve inference traffic.

Configuration:
    LITELLM_STANDALONE_HEALTH_PORT: Port for the standalone health server
        (default: 8001, set to 0 to disable)

Usage in Kubernetes:
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8001
      initialDelaySeconds: 5
      periodSeconds: 10
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from litellm._logging import verbose_proxy_logger

_server: Optional[HTTPServer] = None
_thread: Optional[threading.Thread] = None

STANDALONE_HEALTH_PORT = int(os.getenv("LITELLM_STANDALONE_HEALTH_PORT", 8001))


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that responds to GET /health/live."""

    def do_GET(self):  # noqa: N802
        if self.path in ("/health/live", "/health/live/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default stderr logging."""
        pass


def start_standalone_health_server() -> bool:
    """
    Start the standalone health check server in a daemon thread.

    Returns True if the server was started, False if disabled or already running.
    """
    global _server, _thread

    if STANDALONE_HEALTH_PORT == 0:
        verbose_proxy_logger.info(
            "Standalone health server disabled (LITELLM_STANDALONE_HEALTH_PORT=0)"
        )
        return False

    if _thread is not None and _thread.is_alive():
        verbose_proxy_logger.debug("Standalone health server already running")
        return False

    try:
        _server = HTTPServer(("0.0.0.0", STANDALONE_HEALTH_PORT), _HealthHandler)
        _thread = threading.Thread(
            target=_server.serve_forever,
            name="standalone-health-server",
            daemon=True,
        )
        _thread.start()
        verbose_proxy_logger.info(
            "Standalone health server started on port %d", STANDALONE_HEALTH_PORT
        )
        return True
    except OSError as e:
        verbose_proxy_logger.warning(
            "Failed to start standalone health server on port %d: %s",
            STANDALONE_HEALTH_PORT,
            e,
        )
        return False


def stop_standalone_health_server():
    """Stop the standalone health check server."""
    global _server, _thread

    if _server is not None:
        _server.shutdown()
        _server = None
    if _thread is not None:
        _thread.join(timeout=5)
        _thread = None
        verbose_proxy_logger.info("Standalone health server stopped")

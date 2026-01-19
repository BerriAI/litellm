"""
Minimal mock GraySwan monitor server that intentionally responds slowly.

Usage:
    python scripts/mock_grayswan_timeout_server.py --port 8787 --delay 35

Point GRAYSWAN_API_BASE at http://127.0.0.1:8787 so the guardrail hits this
endpoint and times out (the guardrail client has a 30s timeout).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

LOG = logging.getLogger("mock_grayswan")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class SlowHandler(BaseHTTPRequestHandler):
    delay_seconds: float = 35.0

    def log_message(self, fmt: str, *args) -> None:  # noqa: D401
        """Route handler logs through the logging module."""
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _read_body(self) -> Optional[bytes]:
        content_length = self.headers.get("content-length")
        if content_length is None:
            return None
        try:
            length = int(content_length)
        except ValueError:
            return None
        return self.rfile.read(length)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/cygnal/monitor":
            self.send_error(404, "Not Found")
            return

        body = self._read_body()
        LOG.info("Received POST %s body=%s", self.path, body)

        LOG.info("Sleeping for %.1fs to trigger client timeout", self.delay_seconds)
        time.sleep(self.delay_seconds)

        response = {"status": "ok", "delayed": self.delay_seconds}
        response_bytes = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock GraySwan monitor server")
    parser.add_argument("--port", type=int, default=8787, help="Port to listen on")
    parser.add_argument(
        "--delay",
        type=float,
        default=35.0,
        help="Seconds to delay before responding (must exceed guardrail timeout)",
    )
    args = parser.parse_args()

    SlowHandler.delay_seconds = args.delay
    server = HTTPServer(("0.0.0.0", args.port), SlowHandler)
    LOG.info("Starting mock server on port %d with delay %.1fs", args.port, args.delay)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down mock server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

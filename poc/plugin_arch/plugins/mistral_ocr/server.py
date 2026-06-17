"""HTTP wrapper exposing an `LLMPlugin` as the wire interface.

The wire interface is intentionally tiny so the core stays format-agnostic:

  POST /handle         body: raw request bytes from the core
                       returns: PluginResponse bytes (on success) or
                                PluginError JSON envelope (on failure)
  GET  /capabilities   returns: {"models": [...], "endpoints": [...]}
  GET  /healthz        returns: "ok"

This wrapper does not know what kind of plugin it is hosting. Any class that
satisfies `LLMPlugin` can be served by it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

from .base import Capabilities, LLMPlugin, PluginError, PluginRequest, PluginResponse
from .plugin import MistralOCRPlugin

_LOG = logging.getLogger("plugin_server")

ERROR_CONTENT_TYPE: Final[str] = "application/json"


def _error_envelope(err: PluginError) -> bytes:
    return json.dumps(
        {"error": {"code": err.code, "message": err.message, "type": err.type}}
    ).encode("utf-8")


def _capabilities_envelope(c: Capabilities) -> bytes:
    return json.dumps({"models": list(c.models), "endpoints": list(c.endpoints)}).encode(
        "utf-8"
    )


def make_handler(plugin: LLMPlugin) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            _LOG.info("%s - %s", self.address_string(), fmt % args)

        def _write(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/handle":
                err = PluginError(
                    code="not_found",
                    message=f"no such route: {self.path}",
                    type="invalid_request",
                    http_status=404,
                )
                self._write(404, _error_envelope(err), ERROR_CONTENT_TYPE)
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else b""
            ct = self.headers.get("Content-Type", "application/json")
            req = PluginRequest(body=body, content_type=ct)
            result = plugin.handle(req)
            match result:
                case PluginResponse(body=resp_body, content_type=resp_ct, status_code=sc):
                    self._write(sc, resp_body, resp_ct)
                case PluginError() as err:
                    self._write(err.http_status, _error_envelope(err), ERROR_CONTENT_TYPE)

        def do_GET(self) -> None:
            if self.path == "/capabilities":
                self._write(200, _capabilities_envelope(plugin.capabilities()), "application/json")
                return
            if self.path == "/healthz":
                self._write(200, b"ok", "text/plain")
                return
            err = PluginError(
                code="not_found",
                message=f"no such route: {self.path}",
                type="invalid_request",
                http_status=404,
            )
            self._write(404, _error_envelope(err), ERROR_CONTENT_TYPE)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Mistral OCR plugin server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--upstream-url",
        default=None,
        help="override the upstream URL (defaults to Mistral OCR API)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        kwargs = {} if args.upstream_url is None else {"upstream_url": args.upstream_url}
        plugin: LLMPlugin = MistralOCRPlugin(**kwargs)
    except RuntimeError as e:
        print(f"startup error: {e}", file=sys.stderr)
        return 2

    server = ThreadingHTTPServer((args.host, args.port), make_handler(plugin))
    _LOG.info("plugin server listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _LOG.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Forwarding recorder: logs every request as one JSON line, then forwards it to the real destination.

usage: python recorder.py <listen_port> <upstream_base_url> <log_path>

Credential headers are logged by name with the value replaced by `<redacted len=N>`; the destination
still receives them verbatim. Upstream 3xx answers are relayed to the caller instead of being followed,
so the forwarded credentials never leave the configured upstream origin. The log is truncated at start
(one recorder process is one run) and created 0600.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN_PORT, UPSTREAM, LOG_PATH = int(sys.argv[1]), sys.argv[2].rstrip("/"), sys.argv[3]
HOP_HEADERS = frozenset({"host", "content-length", "transfer-encoding", "connection", "accept-encoding"})
SECRET_MARKERS = ("authorization", "api-key", "api_key", "token", "secret", "cookie", "credential", "password")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_: object) -> None:
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def redact(name: str, value: str) -> str:
    return f"<redacted len={len(value)}>" if any(m in name.lower() for m in SECRET_MARKERS) else value


class Recorder(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _handle(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        with open(LOG_PATH, "a", encoding="utf-8") as log:
            log.write(json.dumps({"ts": time.time(), "method": self.command, "path": self.path,
                                  "headers": {k: redact(k, v) for k, v in headers.items()},
                                  "body": body.decode("utf-8", "replace")}) + "\n")
        request = urllib.request.Request(UPSTREAM + self.path, data=body or None, headers=headers,
                                         method=self.command)
        try:
            with OPENER.open(request, timeout=600) as upstream:
                status, payload, upstream_headers = upstream.status, upstream.read(), upstream.headers
        except urllib.error.HTTPError as error:
            status, payload, upstream_headers = error.code, error.read(), error.headers
        self.send_response(status)
        for key, value in upstream_headers.items():
            if key.lower() not in HOP_HEADERS:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

    def log_message(self, *_: object) -> None:
        return


os.close(os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600))
server = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Recorder)
print(f"recorder pid={os.getpid()} listening on 127.0.0.1:{LISTEN_PORT} -> {UPSTREAM} log={LOG_PATH}", flush=True)
server.serve_forever()

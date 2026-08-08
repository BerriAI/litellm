"""Stand-in cost center validation service for the team metadata e2e tests.

Accepts POST /validate with {"operation": ..., "metadata": {...}} and answers
{"ok": true} or {"ok": false, "reason": ...} based on a static allowlist.
GET /health answers 200 for the CI wait loop.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED_COST_CENTERS = {"CC-1001", "CC-1002"}


class CostCenterHandler(BaseHTTPRequestHandler):
    def _respond(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "healthy"})
            return
        self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/validate":
            self._respond(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        cost_center = (body.get("metadata") or {}).get("cost_center")
        if cost_center is None:
            self._respond(200, {"ok": False, "reason": "cost_center missing per cost center service"})
        elif cost_center not in ALLOWED_COST_CENTERS:
            self._respond(200, {"ok": False, "reason": f"cost center {cost_center} rejected by cost center service"})
        else:
            self._respond(200, {"ok": True})

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9414)
    args = parser.parse_args()
    print(f"cost center service listening on {args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), CostCenterHandler).serve_forever()

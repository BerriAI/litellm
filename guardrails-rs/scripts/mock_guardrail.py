#!/usr/bin/env python3
"""
Mock guardrail API for testing the Rust sidecar.

Blocks any request containing "kill" or "harm" in the texts.
Masks emails by replacing them with [EMAIL_REDACTED].
Otherwise passes through.

Usage: python3 scripts/mock_guardrail.py
Runs on http://127.0.0.1:8888
"""

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

BLOCK_PATTERNS = ["kill", "harm", "attack", "destroy"]
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        texts = body.get("texts", [])

        for text in texts:
            if any(p in text.lower() for p in BLOCK_PATTERNS):
                self._respond({"action": "BLOCKED", "blocked_reason": "violent content detected"})
                return

        masked_texts = [EMAIL_RE.sub("[EMAIL_REDACTED]", t) for t in texts]
        if masked_texts != texts:
            self._respond({"action": "GUARDRAIL_INTERVENED", "texts": masked_texts})
            return

        self._respond({"action": "NONE"})

    def _respond(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        print(f"[mock] {args[0]}")


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8888), Handler)
    print("mock guardrail API listening on http://127.0.0.1:8888")
    server.serve_forever()

"""Mock Mistral OCR upstream used for end-to-end seam tests.

It mimics the shape of Mistral's OCR API just enough to prove the core +
plugin wiring works without spending real money. Real Mistral runs are done
by setting MISTRAL_API_KEY and omitting `--upstream-url` on the plugin.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(body.decode("utf-8"))
        except ValueError:
            req = {}
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            self._json(401, {"message": "missing bearer token"})
            return
        resp = {
            "model": req.get("model", "unknown"),
            "pages": [
                {
                    "index": 0,
                    "markdown": "# Hello from mock Mistral\n\nThis is a fake OCR page.",
                    "images": [],
                    "dimensions": {"dpi": 200, "height": 1100, "width": 850},
                }
            ],
            "usage_info": {"pages_processed": 1, "doc_size_bytes": 12345},
        }
        self._json(200, resp)

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    args = parser.parse_args()
    s = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock mistral listening on http://{args.host}:{args.port}")
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        s.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

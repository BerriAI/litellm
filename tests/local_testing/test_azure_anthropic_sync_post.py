"""
``_get_httpx_client`` + ``HTTPHandler.post`` (same pattern as Azure Anthropic sync path:
``_get_httpx_client(params={"timeout": ...})`` then ``post(..., timeout=...)``).

A local server stalls longer than the per-request ``timeout`` but well under the client
default, so the handler must raise :class:`~litellm.exceptions.Timeout` from the per-request
override rather than completing under the (much larger) client default.

Lives under ``local_testing`` (not ``make test-unit``).
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from litellm.exceptions import Timeout as LitellmTimeout
from litellm.llms.custom_httpx.http_handler import (
    MaskedHTTPStatusError,
    _get_httpx_client,
)

_SERVER_DELAY_S = 5
_PER_REQUEST_TIMEOUT_S = 1.0
_CLIENT_DEFAULT_TIMEOUT_S = 60.0


class _SlowHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(_SERVER_DELAY_S)
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
        except OSError:
            pass

    def log_message(self, *args):
        pass


def test_post_delay_exceeds_per_request_timeout_raises():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address

    handler = _get_httpx_client(params={"timeout": _CLIENT_DEFAULT_TIMEOUT_S})
    try:
        with pytest.raises(LitellmTimeout):
            handler.post(
                f"http://{host}:{port}/delay",
                headers={"content-type": "application/json"},
                data=json.dumps({"model": "claude", "messages": []}),
                timeout=_PER_REQUEST_TIMEOUT_S,
            )
    except MaskedHTTPStatusError as e:
        pytest.skip(f"httpbin.org unavailable: {e}")
    finally:
        handler.close()
        server.shutdown()
        server.server_close()

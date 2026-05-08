"""
``_get_httpx_client`` + ``HTTPHandler.post`` (same pattern as Azure Anthropic sync path:
``_get_httpx_client(params={"timeout": ...})`` then ``post(..., timeout=...)``).

Uses https://httpbin.org/delay/10 with ``timeout=5`` — the handler must raise :class:`~litellm.exceptions.Timeout`
before the 10s delay completes. Skips if httpbin is unreachable.

Lives under ``local_testing`` (not ``make test-unit``).
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from litellm.exceptions import Timeout as LitellmTimeout
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

_HTTPBIN_DELAY_S = 10
_PER_REQUEST_TIMEOUT_S = 5.0
_CLIENT_DEFAULT_TIMEOUT_S = 60.0


def test_post_delay_exceeds_per_request_timeout_raises():
    try:
        httpx.get("https://httpbin.org/get", timeout=5.0)
    except Exception as e:
        pytest.skip(f"httpbin.org unreachable: {e}")

    handler = _get_httpx_client(params={"timeout": _CLIENT_DEFAULT_TIMEOUT_S})
    try:
        with pytest.raises(LitellmTimeout):
            handler.post(
                f"https://httpbin.org/delay/{_HTTPBIN_DELAY_S}",
                headers={"content-type": "application/json"},
                data=json.dumps({"model": "claude", "messages": []}),
                timeout=_PER_REQUEST_TIMEOUT_S,
            )
    finally:
        handler.close()

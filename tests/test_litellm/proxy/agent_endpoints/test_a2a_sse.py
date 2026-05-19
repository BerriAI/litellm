"""Tests for A2A streaming SSE serialization (S3-06).

Covers the SSE helpers + the Accept-header detection. End-to-end streaming
goes through the real a2a SDK, so we keep the surface here unit-level —
the integration path is exercised by existing test_a2a_endpoints.py.
"""

import json

from litellm.proxy.agent_endpoints.a2a_endpoints import (
    _serialize_sse_chunk,
    _serialize_sse_error,
    _sse_headers,
    _wants_sse,
)


def test_wants_sse_recognizes_explicit_accept():
    assert _wants_sse("text/event-stream") is True
    # Browsers send a comma-separated list with quality factors.
    assert _wants_sse("text/event-stream, */*;q=0.8") is True
    assert _wants_sse("text/event-stream;charset=utf-8") is True


def test_wants_sse_rejects_other_types():
    assert _wants_sse(None) is False
    assert _wants_sse("") is False
    assert _wants_sse("application/json") is False
    assert _wants_sse("application/x-ndjson") is False
    assert _wants_sse("text/html, application/json") is False


def test_serialize_sse_chunk_dict_input():
    out = _serialize_sse_chunk("req-1", {"role": "assistant", "content": "hi"})
    assert out.startswith("event: a2a.message\n")
    assert out.endswith("\n\n")
    # Single data line, payload is valid JSON.
    data_line = next(
        ln for ln in out.split("\n") if ln.startswith("data: ")
    )
    payload = json.loads(data_line[len("data: ") :])
    assert payload == {"role": "assistant", "content": "hi"}


def test_serialize_sse_chunk_pydantic_input():
    class _Fake:
        def model_dump(self, mode="json", exclude_none=False):
            return {"text": "ok"}

    out = _serialize_sse_chunk("req-1", _Fake())
    assert "event: a2a.message" in out
    assert "\"text\": \"ok\"" in out


def test_serialize_sse_error_carries_jsonrpc_envelope():
    out = _serialize_sse_error("req-1", RuntimeError("boom"))
    assert out.startswith("event: a2a.error\n")
    payload_line = next(ln for ln in out.split("\n") if ln.startswith("data: "))
    payload = json.loads(payload_line[len("data: ") :])
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == "req-1"
    assert payload["error"]["code"] == -32603
    assert "boom" in payload["error"]["message"]


def test_sse_headers_disable_buffering():
    hdrs = _sse_headers()
    assert hdrs["Cache-Control"] == "no-cache"
    assert hdrs["Connection"] == "keep-alive"
    # nginx / Cloudflare buffering off — otherwise events buffer for seconds.
    assert hdrs["X-Accel-Buffering"] == "no"

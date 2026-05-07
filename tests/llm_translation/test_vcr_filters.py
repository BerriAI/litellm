"""Unit tests for the VCR record-time filters that keep cassettes small.

Covers:
- ``_strip_image_b64_payloads`` — replaces base64 image bodies in
  image-gen responses so cassettes don't carry MB-class PNG payloads.
- ``_normalize_multipart_boundary`` — rewrites random multipart
  boundaries to a fixed string so audio-transcription request bodies
  match across record and replay.
"""

from __future__ import annotations

import json
import os
import sys

from vcr.request import Request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_conftest_common import (  # noqa: E402
    VCR_FIXED_MULTIPART_BOUNDARY,
    VCR_IMAGE_B64_PLACEHOLDER,
    _normalize_multipart_boundary,
    _strip_image_b64_payloads,
)


# ---------------------------------------------------------------------------
# Image b64 stripper
# ---------------------------------------------------------------------------


def _image_response(b64_payload: str, body_type: str = "bytes") -> dict:
    body_text = json.dumps({"data": [{"b64_json": b64_payload}]})
    body_string = body_text.encode("utf-8") if body_type == "bytes" else body_text
    return {
        "status": {"code": 200, "message": "OK"},
        "headers": {
            "content-type": ["application/json"],
            "content-length": [str(len(body_text.encode("utf-8")))],
        },
        "body": {"string": body_string},
    }


def test_strip_image_b64_replaces_payload_when_body_is_bytes():
    response = _image_response("A" * 5000, body_type="bytes")
    out = _strip_image_b64_payloads(response)
    payload = json.loads(out["body"]["string"].decode("utf-8"))
    assert payload["data"][0]["b64_json"] == VCR_IMAGE_B64_PLACEHOLDER


def test_strip_image_b64_replaces_payload_when_body_is_str():
    response = _image_response("A" * 5000, body_type="str")
    out = _strip_image_b64_payloads(response)
    payload = json.loads(out["body"]["string"])
    assert payload["data"][0]["b64_json"] == VCR_IMAGE_B64_PLACEHOLDER


def test_strip_image_b64_updates_content_length():
    response = _image_response("A" * 5000)
    out = _strip_image_b64_payloads(response)
    expected_len = len(out["body"]["string"])
    assert out["headers"]["content-length"] == [str(expected_len)]


def test_strip_image_b64_is_idempotent():
    response = _image_response("A" * 5000)
    once = _strip_image_b64_payloads(response)
    twice = _strip_image_b64_payloads(once)
    assert once["body"]["string"] == twice["body"]["string"]


def test_strip_image_b64_handles_nested_data():
    body_text = json.dumps(
        {
            "outer": {
                "data": [
                    {"b64_json": "X" * 4000, "label": "first"},
                    {"b64_json": "Y" * 4000, "label": "second"},
                ]
            }
        }
    )
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"content-type": ["application/json"]},
        "body": {"string": body_text.encode("utf-8")},
    }
    out = _strip_image_b64_payloads(response)
    payload = json.loads(out["body"]["string"].decode("utf-8"))
    assert payload["outer"]["data"][0]["b64_json"] == VCR_IMAGE_B64_PLACEHOLDER
    assert payload["outer"]["data"][1]["b64_json"] == VCR_IMAGE_B64_PLACEHOLDER
    assert payload["outer"]["data"][0]["label"] == "first"


def test_strip_image_b64_leaves_non_image_response_unchanged():
    body_text = json.dumps({"choices": [{"message": {"content": "hello"}}]})
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"content-type": ["application/json"]},
        "body": {"string": body_text.encode("utf-8")},
    }
    out = _strip_image_b64_payloads(response)
    assert json.loads(out["body"]["string"].decode("utf-8")) == json.loads(body_text)


def test_strip_image_b64_leaves_invalid_json_unchanged():
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"content-type": ["application/octet-stream"]},
        "body": {"string": b"\x89PNG\r\n\x1a\n binary stuff not json"},
    }
    out = _strip_image_b64_payloads(response)
    assert out["body"]["string"] == b"\x89PNG\r\n\x1a\n binary stuff not json"


def test_strip_image_b64_skips_short_values():
    """Already-placeholder values aren't re-replaced (idempotency guard)."""
    body_text = json.dumps({"data": [{"b64_json": VCR_IMAGE_B64_PLACEHOLDER}]})
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"content-type": ["application/json"]},
        "body": {"string": body_text.encode("utf-8")},
    }
    out = _strip_image_b64_payloads(response)
    payload = json.loads(out["body"]["string"].decode("utf-8"))
    assert payload["data"][0]["b64_json"] == VCR_IMAGE_B64_PLACEHOLDER


# ---------------------------------------------------------------------------
# Multipart boundary normalizer
# ---------------------------------------------------------------------------


def _multipart_request(boundary: str):
    body_text = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        "Content-Type: audio/wav\r\n"
        "\r\n"
        "fake-audio-bytes\r\n"
        f"--{boundary}--\r\n"
    )
    return Request(
        method="POST",
        uri="https://api.openai.com/v1/audio/transcriptions",
        body=body_text.encode("utf-8"),
        headers={
            "content-type": f"multipart/form-data; boundary={boundary}",
        },
    )


def test_normalize_multipart_rewrites_header_and_body():
    req = _multipart_request("abc123random")
    _normalize_multipart_boundary(req)
    assert (
        req.headers["content-type"]
        == f"multipart/form-data; boundary={VCR_FIXED_MULTIPART_BOUNDARY}"
    )
    assert b"abc123random" not in req.body
    assert VCR_FIXED_MULTIPART_BOUNDARY.encode("utf-8") in req.body


def test_normalize_multipart_is_idempotent():
    req = _multipart_request("abc123random")
    _normalize_multipart_boundary(req)
    body_first = req.body
    header_first = req.headers["content-type"]
    _normalize_multipart_boundary(req)
    assert req.body == body_first
    assert req.headers["content-type"] == header_first


def test_normalize_multipart_two_distinct_boundaries_match_after_normalize():
    """Whisper-style: two requests with different random boundaries should
    end up with byte-identical bodies after normalization."""
    req1 = _multipart_request("boundaryAAA")
    req2 = _multipart_request("boundaryBBB")
    _normalize_multipart_boundary(req1)
    _normalize_multipart_boundary(req2)
    assert req1.body == req2.body
    assert req1.headers["content-type"] == req2.headers["content-type"]


def test_normalize_multipart_skips_non_multipart_requests():
    req = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body=b'{"model":"gpt-4o"}',
        headers={"content-type": "application/json"},
    )
    _normalize_multipart_boundary(req)
    assert req.headers["content-type"] == "application/json"
    assert req.body == b'{"model":"gpt-4o"}'


def test_normalize_multipart_skips_request_without_content_type():
    req = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body=b"unknown body",
        headers={},
    )
    _normalize_multipart_boundary(req)
    assert req.body == b"unknown body"


def test_normalize_multipart_handles_quoted_boundary():
    req = Request(
        method="POST",
        uri="https://api.openai.com/v1/audio/transcriptions",
        body=b"--quoted-boundary--body content--quoted-boundary--",
        headers={"content-type": 'multipart/form-data; boundary="quoted-boundary"'},
    )
    _normalize_multipart_boundary(req)
    assert b"quoted-boundary" not in req.body
    assert VCR_FIXED_MULTIPART_BOUNDARY.encode("utf-8") in req.body

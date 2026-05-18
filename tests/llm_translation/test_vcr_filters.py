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
    FILTERED_REQUEST_HEADERS,
    FILTERED_RESPONSE_HEADER_PREFIXES,
    FILTERED_RESPONSE_HEADERS,
    VCR_FIXED_MULTIPART_BOUNDARY,
    VCR_IMAGE_B64_PLACEHOLDER,
    _normalize_multipart_boundary,
    _scrub_response,
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


# ---------------------------------------------------------------------------
# Response header scrubbing (explicit list + prefix-based blocklist)
# ---------------------------------------------------------------------------


def _response_with_headers(headers: dict) -> dict:
    return {
        "status": {"code": 200, "message": "OK"},
        "headers": headers,
        "body": {"string": b"{}"},
    }


def test_scrub_response_strips_explicit_filtered_headers():
    response = _response_with_headers(
        {
            "Content-Type": ["application/json"],
            "X-Request-Id": ["req-abc"],
            "Set-Cookie": ["session=xyz"],
            "Date": ["Mon, 18 May 2026 18:00:00 GMT"],
        }
    )
    out = _scrub_response(response)
    headers = out["headers"]
    assert "X-Request-Id" not in headers
    assert "Set-Cookie" not in headers
    assert "Date" not in headers
    assert headers["Content-Type"] == ["application/json"]


def test_scrub_response_strips_x_amz_metadata_via_prefix():
    """AWS Bedrock responses come with 10+ ``x-amz-*`` headers per
    request — none of them are asserted on by tests, and none of them
    are tiny. Trim them via the prefix blocklist."""
    response = _response_with_headers(
        {
            "Content-Type": ["application/json"],
            "x-amz-id-2": ["very-long-aws-trace-token=="],
            "x-amz-server-side-encryption": ["AES256"],
            "x-amzn-RequestId": ["req-123"],
            "x-amzn-Trace-Id": ["root=1-abc"],
        }
    )
    out = _scrub_response(response)
    for header in (
        "x-amz-id-2",
        "x-amz-server-side-encryption",
        "x-amzn-RequestId",
        "x-amzn-Trace-Id",
    ):
        assert header not in out["headers"]
    assert out["headers"]["Content-Type"] == ["application/json"]


def test_scrub_response_strips_anthropic_ratelimit_family():
    """Anthropic responses ship 7+ verbose ``anthropic-ratelimit-*``
    headers. They're tiny individually but compound across cassettes.
    """
    response = _response_with_headers(
        {
            "Content-Type": ["application/json"],
            "anthropic-ratelimit-requests-limit": ["50"],
            "anthropic-ratelimit-requests-remaining": ["49"],
            "anthropic-ratelimit-requests-reset": ["2026-05-18T18:00:00Z"],
            "anthropic-ratelimit-tokens-limit": ["100000"],
            "anthropic-ratelimit-tokens-remaining": ["95000"],
        }
    )
    out = _scrub_response(response)
    assert all(not h.lower().startswith("anthropic-ratelimit-") for h in out["headers"])


def test_scrub_response_keeps_content_headers_untouched():
    """Content-Type / Content-Length / Content-Encoding are required
    for vcrpy to replay bodies correctly. Make sure no overzealous
    prefix sweeps them out."""
    response = _response_with_headers(
        {
            "content-type": ["application/json; charset=utf-8"],
            "content-length": ["42"],
            "content-encoding": ["gzip"],
            "transfer-encoding": ["chunked"],
        }
    )
    out = _scrub_response(response)
    for keep in (
        "content-type",
        "content-length",
        "content-encoding",
        "transfer-encoding",
    ):
        assert keep in out["headers"]


def test_anthropic_version_is_no_longer_filtered_from_request_headers():
    """Documented behavior change. ``anthropic-version`` is not a
    secret, is tiny, and parametrized tests rely on it for matching.
    Filtering it caused two parametrizations to share one cassette
    episode and produce false replay hits."""
    assert "anthropic-version" not in (h.lower() for h in FILTERED_REQUEST_HEADERS)


def test_explicit_list_and_prefix_list_are_disjoint_with_keep_set():
    """No header in the ``content-*`` family should be on either list."""
    must_keep = {
        "content-type",
        "content-length",
        "content-encoding",
        "transfer-encoding",
    }
    for name in FILTERED_RESPONSE_HEADERS:
        assert name.lower() not in must_keep
    for prefix in FILTERED_RESPONSE_HEADER_PREFIXES:
        for keep in must_keep:
            assert not keep.startswith(prefix.lower())

"""Unit tests for the VCR record-time filters that keep cassettes small.

Covers:
- ``_strip_image_b64_payloads`` — replaces base64 image bodies in
  image-gen responses so cassettes don't carry MB-class PNG payloads.
- ``_normalize_multipart_boundary`` — rewrites random multipart
  boundaries to a fixed string so audio-transcription request bodies
  match across record and replay.
- ``_normalize_google_oauth_jwt`` — drops volatile JWT claims and the
  signature from Google service-account token requests so the
  ``safe_body`` matcher stays stable across runs.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.parse

from vcr.request import Request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_conftest_common import (  # noqa: E402
    GOOGLE_OAUTH_JWT_ASSERTION_FIELD,
    JWT_NORMALIZED_SIGNATURE,
    JWT_VOLATILE_CLAIMS,
    VCR_FIXED_MULTIPART_BOUNDARY,
    VCR_IMAGE_B64_PLACEHOLDER,
    _normalize_google_oauth_jwt,
    _normalize_jwt,
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


# ---------------------------------------------------------------------------
# Google OAuth JWT normalizer
# ---------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _make_jwt(
    header: dict, payload: dict, signature: str = "raw-signature-bytes"
) -> str:
    header_seg = _b64url_encode(json.dumps(header).encode("utf-8"))
    payload_seg = _b64url_encode(json.dumps(payload).encode("utf-8"))
    sig_seg = _b64url_encode(signature.encode("utf-8"))
    return f"{header_seg}.{payload_seg}.{sig_seg}"


def _service_account_jwt(iat: int, exp: int, *, jti: str = "tok-1") -> str:
    return _make_jwt(
        header={"typ": "JWT", "alg": "RS256", "kid": "abc123"},
        payload={
            "iss": "test-sa@litellm-ci.iam.gserviceaccount.com",
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": "https://oauth2.googleapis.com/token",
            "iat": iat,
            "exp": exp,
            "jti": jti,
        },
        signature=f"sig-bytes-for-iat-{iat}",
    )


def _token_request(jwt_assertion: str) -> Request:
    body_text = urllib.parse.urlencode(
        [
            (GOOGLE_OAUTH_JWT_ASSERTION_FIELD, jwt_assertion),
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
        ]
    )
    return Request(
        method="POST",
        uri="https://oauth2.googleapis.com/token",
        body=body_text.encode("utf-8"),
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "content-length": str(len(body_text.encode("utf-8"))),
        },
    )


def test_normalize_jwt_strips_volatile_claims_and_signature():
    token = _service_account_jwt(iat=1779121027, exp=1779124627, jti="abc")
    normalized = _normalize_jwt(token)
    assert normalized is not None
    header_seg, payload_seg, sig_seg = normalized.split(".")
    padding = "=" * (-len(payload_seg) % 4)
    payload = json.loads(base64.urlsafe_b64decode(payload_seg + padding))
    for claim in JWT_VOLATILE_CLAIMS:
        assert claim not in payload
    assert payload["iss"] == "test-sa@litellm-ci.iam.gserviceaccount.com"
    assert payload["scope"] == "https://www.googleapis.com/auth/cloud-platform"
    assert payload["aud"] == "https://oauth2.googleapis.com/token"
    assert sig_seg == JWT_NORMALIZED_SIGNATURE


def test_normalize_jwt_returns_none_for_malformed_token():
    assert _normalize_jwt("not-a-jwt") is None
    assert _normalize_jwt("only.two") is None
    assert _normalize_jwt("aaa.bbb.ccc") is None


def test_normalize_google_oauth_jwt_makes_two_requests_match():
    """Same logical auth call with different timestamps must produce
    byte-identical bodies after normalization."""
    req_record = _token_request(_service_account_jwt(iat=1779055107, exp=1779058707))
    req_replay = _token_request(_service_account_jwt(iat=1779121027, exp=1779124627))
    assert req_record.body != req_replay.body
    _normalize_google_oauth_jwt(req_record)
    _normalize_google_oauth_jwt(req_replay)
    assert req_record.body == req_replay.body


def test_normalize_google_oauth_jwt_updates_content_length():
    req = _token_request(_service_account_jwt(iat=1, exp=2))
    _normalize_google_oauth_jwt(req)
    expected_len = str(len(req.body))
    assert req.headers["content-length"] == expected_len


def test_normalize_google_oauth_jwt_is_idempotent():
    req = _token_request(_service_account_jwt(iat=1, exp=2))
    _normalize_google_oauth_jwt(req)
    body_first = req.body
    headers_first = dict(req.headers)
    _normalize_google_oauth_jwt(req)
    assert req.body == body_first
    assert dict(req.headers) == headers_first


def test_normalize_google_oauth_jwt_preserves_other_form_fields():
    req = _token_request(_service_account_jwt(iat=1, exp=2))
    _normalize_google_oauth_jwt(req)
    fields = dict(urllib.parse.parse_qsl(req.body.decode("utf-8")))
    assert fields["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
    assert fields[GOOGLE_OAUTH_JWT_ASSERTION_FIELD].endswith(
        f".{JWT_NORMALIZED_SIGNATURE}"
    )


def test_normalize_google_oauth_jwt_handles_str_body():
    from types import SimpleNamespace

    body_text = urllib.parse.urlencode(
        [(GOOGLE_OAUTH_JWT_ASSERTION_FIELD, _service_account_jwt(iat=1, exp=2))]
    )
    req = SimpleNamespace(
        method="POST",
        uri="https://oauth2.googleapis.com/token",
        body=body_text,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    _normalize_google_oauth_jwt(req)
    assert isinstance(req.body, str)
    assert f".{JWT_NORMALIZED_SIGNATURE}" in req.body


def test_normalize_google_oauth_jwt_distinct_scopes_remain_distinct():
    """Different scopes are meaningful auth differences — they must NOT
    collapse after normalization."""
    jwt_a = _make_jwt(
        header={"typ": "JWT", "alg": "RS256"},
        payload={"iss": "a@x.iam", "scope": "scope-a", "iat": 1, "exp": 2},
    )
    jwt_b = _make_jwt(
        header={"typ": "JWT", "alg": "RS256"},
        payload={"iss": "a@x.iam", "scope": "scope-b", "iat": 1, "exp": 2},
    )
    req_a = _token_request(jwt_a)
    req_b = _token_request(jwt_b)
    _normalize_google_oauth_jwt(req_a)
    _normalize_google_oauth_jwt(req_b)
    assert req_a.body != req_b.body


def test_normalize_google_oauth_jwt_skips_non_oauth_request():
    body = b'{"prompt":"hi"}'
    req = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body=body,
        headers={"content-type": "application/json"},
    )
    _normalize_google_oauth_jwt(req)
    assert req.body == body


def test_normalize_google_oauth_jwt_skips_get_request():
    req = Request(
        method="GET",
        uri="https://oauth2.googleapis.com/token",
        body=None,
        headers={},
    )
    _normalize_google_oauth_jwt(req)
    assert req.body is None


def test_normalize_google_oauth_jwt_skips_when_assertion_missing():
    body_text = urllib.parse.urlencode(
        [
            ("grant_type", "refresh_token"),
            ("refresh_token", "rt-abc"),
            ("client_id", "cid"),
        ]
    )
    req = Request(
        method="POST",
        uri="https://oauth2.googleapis.com/token",
        body=body_text.encode("utf-8"),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    original_body = req.body
    _normalize_google_oauth_jwt(req)
    assert req.body == original_body


def test_normalize_google_oauth_jwt_skips_malformed_assertion():
    body_text = urllib.parse.urlencode(
        [
            (GOOGLE_OAUTH_JWT_ASSERTION_FIELD, "not-a-jwt"),
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
        ]
    )
    req = Request(
        method="POST",
        uri="https://oauth2.googleapis.com/token",
        body=body_text.encode("utf-8"),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    original_body = req.body
    _normalize_google_oauth_jwt(req)
    assert req.body == original_body


def test_normalize_google_oauth_jwt_normalizes_accounts_google_host():
    body_text = urllib.parse.urlencode(
        [
            (GOOGLE_OAUTH_JWT_ASSERTION_FIELD, _service_account_jwt(iat=1, exp=2)),
        ]
    )
    req = Request(
        method="POST",
        uri="https://accounts.google.com/o/oauth2/token",
        body=body_text.encode("utf-8"),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    _normalize_google_oauth_jwt(req)
    assert f".{JWT_NORMALIZED_SIGNATURE}".encode("utf-8") in req.body

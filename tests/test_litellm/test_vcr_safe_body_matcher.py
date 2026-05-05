"""Unit tests for the shared VCR helpers in ``tests/_vcr_conftest_common``.

The most important guarantee here is that the custom ``safe_body`` matcher
gracefully handles JSON Lines (and other non-strict-JSON) request bodies
without raising ``json.JSONDecodeError`` — vcrpy's default ``body`` matcher
crashes on those because it unconditionally runs ``json.loads`` for any
``application/json`` request body.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# Tests live in ``tests/test_litellm/`` but ``_vcr_conftest_common`` lives in
# the parent ``tests/`` package. Make sure both are importable regardless of
# how pytest is invoked.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._vcr_conftest_common import (  # noqa: E402
    KEY_FINGERPRINT_HEADER,
    KEY_FINGERPRINT_MATCHER_NAME,
    SAFE_BODY_MATCHER_NAME,
    _before_record_request,
    _key_fingerprint_matcher,
    _safe_body_matcher,
    vcr_config_dict,
)


def _req(body):
    return SimpleNamespace(body=body, headers={"Content-Type": "application/json"})


def test_safe_body_matcher_is_in_match_on():
    cfg = vcr_config_dict()
    assert SAFE_BODY_MATCHER_NAME in cfg["match_on"]
    assert "body" not in cfg["match_on"]


def test_safe_body_matcher_accepts_identical_bytes():
    _safe_body_matcher(_req(b"hello"), _req(b"hello"))


def test_safe_body_matcher_accepts_str_bytes_equivalent():
    _safe_body_matcher(_req("hello"), _req(b"hello"))


def test_safe_body_matcher_handles_jsonl_without_crashing():
    """vcrpy's default ``body`` matcher raises ``JSONDecodeError`` on JSONL.

    The Bedrock batch S3 PUT sends a JSON Lines body under
    ``Content-Type: application/json``. The safe matcher must compare such
    bodies as bytes and never invoke ``json.loads``.
    """
    jsonl = (
        b'{"recordId": "request-1", "modelInput": {}}\n'
        b'{"recordId": "request-2", "modelInput": {}}\n'
    )
    _safe_body_matcher(_req(jsonl), _req(jsonl))


def test_safe_body_matcher_rejects_different_jsonl_bodies():
    a = b'{"recordId": "request-1"}\n{"recordId": "request-2"}\n'
    b = b'{"recordId": "request-1"}\n{"recordId": "request-3"}\n'
    with pytest.raises(AssertionError):
        _safe_body_matcher(_req(a), _req(b))


def test_safe_body_matcher_rejects_different_bytes():
    with pytest.raises(AssertionError):
        _safe_body_matcher(_req(b"a"), _req(b"b"))


def test_safe_body_matcher_treats_none_bodies_as_equal():
    _safe_body_matcher(_req(None), _req(None))


def test_safe_body_matcher_does_not_normalize_json_key_order():
    """The safe matcher is strictly more conservative than vcrpy's default.

    Two semantically-equal JSON bodies with different key order are
    treated as *different* requests (cache miss, never a false hit).
    """
    with pytest.raises(AssertionError):
        _safe_body_matcher(_req(b'{"a":1,"b":2}'), _req(b'{"b":2,"a":1}'))


def test_default_vcrpy_body_matcher_crashes_on_jsonl_for_documentation():
    """Document the behavior we are working around.

    vcrpy's stock body matcher raises ``json.JSONDecodeError`` (not even
    a clean ``AssertionError``) when given a JSONL payload typed as
    ``application/json``. This is precisely the crash that broke
    ``tests/batches_tests/test_bedrock_files_and_batches.py::test_async_create_file``
    and is the reason ``safe_body`` exists.
    """
    import json

    from vcr.matchers import body as vcrpy_body  # type: ignore

    jsonl = b'{"recordId": "request-1"}\n{"recordId": "request-2"}\n'
    with pytest.raises(json.JSONDecodeError):
        vcrpy_body(_req(jsonl), _req(jsonl))


# ---------------------------------------------------------------------------
# Key-fingerprint matcher
# ---------------------------------------------------------------------------


def _req_with_headers(headers, body=b""):
    return SimpleNamespace(headers=dict(headers), body=body)


def test_key_fingerprint_matcher_is_in_match_on():
    cfg = vcr_config_dict()
    assert KEY_FINGERPRINT_MATCHER_NAME in cfg["match_on"]


def test_before_record_request_strips_auth_and_adds_fingerprint():
    """The hook must scrub the secret AND stamp a fingerprint."""
    req = _req_with_headers(
        {
            "Authorization": "Bearer sk-real-key-1234567890",
            "Content-Type": "application/json",
            "x-amz-date": "20240115T120000Z",
        }
    )
    out = _before_record_request(req)
    assert (
        "Authorization" not in out.headers
    ), "Authorization must be removed before the cassette is recorded"
    assert "x-amz-date" not in out.headers, (
        "AWS SigV4 timestamp must be scrubbed (it changes every call and "
        "would defeat caching)"
    )
    fp = out.headers.get(KEY_FINGERPRINT_HEADER)
    assert fp and isinstance(fp, str)
    assert len(fp) >= 8
    assert "sk-real" not in fp, "fingerprint must not leak the secret"


def test_before_record_request_no_auth_yields_stable_no_key_bucket():
    a = _before_record_request(_req_with_headers({"Content-Type": "application/json"}))
    b = _before_record_request(_req_with_headers({}))
    assert a.headers[KEY_FINGERPRINT_HEADER] == b.headers[KEY_FINGERPRINT_HEADER]
    # Two no-auth requests must match so we don't defeat caching for
    # SigV4-style requests where auth lives in headers we've stripped.
    _key_fingerprint_matcher(a, b)


def test_key_fingerprint_matcher_distinguishes_good_and_bad_keys():
    """The whole point: bad-key calls must not replay good-key cassettes."""
    good = _before_record_request(
        _req_with_headers({"Authorization": "Bearer sk-real-good-key"})
    )
    bad = _before_record_request(
        _req_with_headers({"Authorization": "Bearer my-bad-key"})
    )
    assert good.headers[KEY_FINGERPRINT_HEADER] != bad.headers[KEY_FINGERPRINT_HEADER]
    with pytest.raises(AssertionError):
        _key_fingerprint_matcher(good, bad)


def test_key_fingerprint_matcher_matches_repeated_good_key_calls():
    a = _before_record_request(
        _req_with_headers({"Authorization": "Bearer sk-same-key"})
    )
    b = _before_record_request(
        _req_with_headers({"Authorization": "Bearer sk-same-key"})
    )
    _key_fingerprint_matcher(a, b)


def test_key_fingerprint_matcher_distinguishes_x_api_key_callers():
    """Anthropic / Azure use ``x-api-key`` (or ``api-key``) instead of Authorization."""
    a = _before_record_request(_req_with_headers({"x-api-key": "anthropic-real"}))
    b = _before_record_request(_req_with_headers({"x-api-key": "anthropic-bad"}))
    with pytest.raises(AssertionError):
        _key_fingerprint_matcher(a, b)


def test_before_record_request_is_idempotent_under_replay():
    """vcrpy runs ``before_record_request`` on both record and replay paths.

    The fingerprint must be deterministic so a request made today matches
    a cassette recorded yesterday from the same key.
    """
    payload = {"Authorization": "Bearer sk-deterministic"}
    first = _before_record_request(_req_with_headers(payload))
    second = _before_record_request(_req_with_headers(payload))
    assert (
        first.headers[KEY_FINGERPRINT_HEADER] == second.headers[KEY_FINGERPRINT_HEADER]
    )

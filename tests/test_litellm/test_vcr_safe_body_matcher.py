from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

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


def _req_with_headers(headers, body=b""):
    return SimpleNamespace(headers=dict(headers), body=body)


def test_safe_body_matcher_is_in_match_on():
    cfg = vcr_config_dict()
    assert SAFE_BODY_MATCHER_NAME in cfg["match_on"]
    assert "body" not in cfg["match_on"]


def test_safe_body_matcher_accepts_identical_bytes():
    _safe_body_matcher(_req(b"hello"), _req(b"hello"))


def test_safe_body_matcher_accepts_str_bytes_equivalent():
    _safe_body_matcher(_req("hello"), _req(b"hello"))


def test_safe_body_matcher_handles_jsonl_without_crashing():
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
    with pytest.raises(AssertionError):
        _safe_body_matcher(_req(b'{"a":1,"b":2}'), _req(b'{"b":2,"a":1}'))


def test_default_vcrpy_body_matcher_crashes_on_jsonl_for_documentation():
    """Pin the upstream behavior our ``safe_body`` matcher exists to work around."""
    import json

    from vcr.matchers import body as vcrpy_body  # type: ignore

    jsonl = b'{"recordId": "request-1"}\n{"recordId": "request-2"}\n'
    with pytest.raises(json.JSONDecodeError):
        vcrpy_body(_req(jsonl), _req(jsonl))


def test_key_fingerprint_matcher_is_in_match_on():
    cfg = vcr_config_dict()
    assert KEY_FINGERPRINT_MATCHER_NAME in cfg["match_on"]


def test_before_record_request_strips_auth_and_adds_fingerprint():
    req = _req_with_headers(
        {
            "Authorization": "Bearer sk-real-key-1234567890",
            "Content-Type": "application/json",
            "x-amz-date": "20240115T120000Z",
        }
    )
    out = _before_record_request(req)
    assert "Authorization" not in out.headers
    assert "x-amz-date" not in out.headers
    fp = out.headers.get(KEY_FINGERPRINT_HEADER)
    assert fp and isinstance(fp, str)
    assert len(fp) >= 8
    assert "sk-real" not in fp


def test_before_record_request_no_auth_yields_stable_no_key_bucket():
    a = _before_record_request(_req_with_headers({"Content-Type": "application/json"}))
    b = _before_record_request(_req_with_headers({}))
    assert a.headers[KEY_FINGERPRINT_HEADER] == b.headers[KEY_FINGERPRINT_HEADER]
    _key_fingerprint_matcher(a, b)


def test_key_fingerprint_matcher_distinguishes_good_and_bad_keys():
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
    a = _before_record_request(_req_with_headers({"x-api-key": "anthropic-real"}))
    b = _before_record_request(_req_with_headers({"x-api-key": "anthropic-bad"}))
    with pytest.raises(AssertionError):
        _key_fingerprint_matcher(a, b)


def test_before_record_request_is_deterministic_across_distinct_requests():
    payload = {"Authorization": "Bearer sk-deterministic"}
    first = _before_record_request(_req_with_headers(payload))
    second = _before_record_request(_req_with_headers(payload))
    assert (
        first.headers[KEY_FINGERPRINT_HEADER] == second.headers[KEY_FINGERPRINT_HEADER]
    )


def test_before_record_request_is_idempotent_on_the_same_request_object():
    """vcrpy invokes ``before_record_request`` more than once per request.

    ``can_play_response_for`` calls it, then ``__contains__`` /
    ``_responses`` call it again on the result. The second call sees a
    request whose auth headers are already gone, so a naive recompute
    would produce ``"no-key"`` and the matcher would consider the
    request distinct from anything it just stored — manifesting in CI as
    ``UnhandledHTTPRequestError`` from ``play_response``.
    """
    req = _req_with_headers({"Authorization": "Bearer sk-someone"})
    _before_record_request(req)
    fp_after_first = req.headers[KEY_FINGERPRINT_HEADER]
    _before_record_request(req)
    assert req.headers[KEY_FINGERPRINT_HEADER] == fp_after_first
    assert fp_after_first != "no-key"

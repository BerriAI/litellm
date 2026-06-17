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
    TOLERANT_PATH_MATCHER_NAME,
    TOLERANT_QUERY_MATCHER_NAME,
    _before_record_request,
    _is_credential_exchange_request,
    _is_telemetry_request,
    _key_fingerprint_matcher,
    _normalize_volatile_tokens,
    _safe_body_matcher,
    _tolerant_path_matcher,
    _tolerant_query_matcher,
    vcr_config_dict,
)


def _req(body, uri="https://api.openai.com/v1/chat/completions"):
    return SimpleNamespace(
        body=body, uri=uri, headers={"Content-Type": "application/json"}
    )


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


def test_google_oauth_bearer_tokens_collapse_to_one_fingerprint():
    """Rotating ``ya29.*`` access tokens must share one fingerprint so
    Vertex/Gemini cassettes match across runs (cf. AWS SigV4 access-key
    stabilization)."""
    run1 = _before_record_request(
        _req_with_headers({"Authorization": "Bearer ya29.FIRST-token-aaaaaaaa"})
    )
    run2 = _before_record_request(
        _req_with_headers({"Authorization": "Bearer ya29.SECOND-token-bbbbbbbb"})
    )
    assert run1.headers[KEY_FINGERPRINT_HEADER] == run2.headers[KEY_FINGERPRINT_HEADER]
    _key_fingerprint_matcher(run1, run2)


def test_non_google_bearer_tokens_still_distinguished():
    """The ya29 collapse must not make every Bearer token identical."""
    google = _before_record_request(
        _req_with_headers({"Authorization": "Bearer ya29.something"})
    )
    real = _before_record_request(
        _req_with_headers({"Authorization": "Bearer sk-real-openai-key"})
    )
    assert (
        google.headers[KEY_FINGERPRINT_HEADER] != real.headers[KEY_FINGERPRINT_HEADER]
    )


def test_normalize_volatile_tokens_collapses_uuid_and_timestamps():
    a = b'{"content": "news today b92ed205-0fa9-4e79-939c-2365023e9cb3"}'
    b = b'{"content": "news today 1a4e1afa-2915-4dcf-b043-33b991cae879"}'
    assert _normalize_volatile_tokens(a) == _normalize_volatile_tokens(b)

    c = b'{"input": "embed data 1779581429.9713597"}'
    d = b'{"input": "embed data 1779583432.6874988"}'
    assert _normalize_volatile_tokens(c) == _normalize_volatile_tokens(d)

    e = b'{"timestamp": "2026-05-25T03:40:37.262045Z"}'
    f = b'{"timestamp": "2026-05-25T06:10:20.830356Z"}'
    assert _normalize_volatile_tokens(e) == _normalize_volatile_tokens(f)


def test_normalize_volatile_tokens_collapses_bedrock_batch_job_names():
    a = (
        b'{"jobName":"litellm-batch-aaaaaaaa",'
        b'"outputDataConfig":{"s3OutputDataConfig":'
        b'{"s3Uri":"s3://bucket/litellm-batch-outputs/litellm-batch-aaaaaaaa/"}}}'
    )
    b = (
        b'{"jobName":"litellm-batch-bbbbbbbb",'
        b'"outputDataConfig":{"s3OutputDataConfig":'
        b'{"s3Uri":"s3://bucket/litellm-batch-outputs/litellm-batch-bbbbbbbb/"}}}'
    )
    assert _normalize_volatile_tokens(a) == _normalize_volatile_tokens(b)


def test_normalize_volatile_tokens_leaves_deterministic_bodies_unchanged():
    body = b'{"model":"claude-haiku-4-5-20251001","temperature":0.0,"n":2}'
    assert _normalize_volatile_tokens(body) == body


def test_safe_body_matcher_matches_bodies_differing_only_by_cachebuster():
    a = _req(b'{"messages":[{"content":"hi 1779579395.5545585"}],"model":"gpt-4.1"}')
    b = _req(b'{"messages":[{"content":"hi 1779579663.595344"}],"model":"gpt-4.1"}')
    _safe_body_matcher(a, b)  # must not raise


def test_safe_body_matcher_still_rejects_genuinely_different_bodies():
    a = _req(b'{"messages":[{"content":"hello"}]}')
    b = _req(b'{"messages":[{"content":"goodbye"}]}')
    with pytest.raises(AssertionError):
        _safe_body_matcher(a, b)


def test_credential_exchange_request_skips_body_comparison():
    assert _is_credential_exchange_request(
        _req(b"assertion=AAA", uri="https://oauth2.googleapis.com/token")
    )
    assert not _is_credential_exchange_request(
        _req(b"x", uri="https://api.openai.com/v1/chat/completions")
    )
    # Freshly-signed JWT assertions differ every run but must still match.
    a = _req(
        b"grant_type=x&assertion=eyJ0AAAA", uri="https://oauth2.googleapis.com/token"
    )
    b = _req(
        b"grant_type=x&assertion=eyJ0BBBB", uri="https://oauth2.googleapis.com/token"
    )
    _safe_body_matcher(a, b)  # must not raise


def test_match_on_uses_tolerant_query_not_builtin():
    cfg = vcr_config_dict()
    assert TOLERANT_QUERY_MATCHER_NAME in cfg["match_on"]
    assert "query" not in cfg["match_on"]


def test_match_on_uses_tolerant_path_not_builtin():
    cfg = vcr_config_dict()
    assert TOLERANT_PATH_MATCHER_NAME in cfg["match_on"]
    assert "path" not in cfg["match_on"]


def test_tolerant_path_normalizes_bedrock_managed_s3_file_uuid():
    from vcr.request import Request

    a = Request(
        method="PUT",
        uri=(
            "https://s3.us-west-2.amazonaws.com/litellm-proxy-test/"
            "litellm-bedrock-files/us.anthropic.claude-haiku-4-5-20251001-v1-0-"
            "123e4567-e89b-12d3-a456-426614174000.jsonl"
        ),
        body=b"",
        headers={},
    )
    b = Request(
        method="PUT",
        uri=(
            "https://s3.us-west-2.amazonaws.com/litellm-proxy-test/"
            "litellm-bedrock-files/us.anthropic.claude-haiku-4-5-20251001-v1-0-"
            "abcdefab-1234-5678-9abc-def012345678.jsonl"
        ),
        body=b"",
        headers={},
    )
    _tolerant_path_matcher(a, b)


def test_tolerant_path_normalizes_bedrock_batch_s3_file_uuid():
    from vcr.request import Request

    a = Request(
        method="PUT",
        uri=(
            "https://s3.us-west-2.amazonaws.com/litellm-proxy-test/"
            "litellm-bedrock-files-us.anthropic.claude-haiku-4-5-20251001-v1-0-"
            "a48e9ec2-5594-45e3-bdbb-44f5d71c06f3.jsonl"
        ),
        body=b"",
        headers={},
    )
    b = Request(
        method="PUT",
        uri=(
            "https://s3.us-west-2.amazonaws.com/litellm-proxy-test/"
            "litellm-bedrock-files-us.anthropic.claude-haiku-4-5-20251001-v1-0-"
            "123e4567-e89b-12d3-a456-426614174000.jsonl"
        ),
        body=b"",
        headers={},
    )
    _tolerant_path_matcher(a, b)


def test_tolerant_path_still_rejects_different_regular_paths():
    from vcr.request import Request

    a = Request(
        method="GET",
        uri="https://api.openai.com/v1/files/file-a/content",
        body=b"",
        headers={},
    )
    b = Request(
        method="GET",
        uri="https://api.openai.com/v1/files/file-b/content",
        body=b"",
        headers={},
    )
    with pytest.raises(AssertionError):
        _tolerant_path_matcher(a, b)


def test_telemetry_request_detection():
    assert _is_telemetry_request(
        _req(b"x", uri="https://us.cloud.langfuse.com/api/public/ingestion")
    )
    assert _is_telemetry_request(_req(b"x", uri="https://otlp.arize.com/v1/traces"))
    assert not _is_telemetry_request(
        _req(b"x", uri="https://api.openai.com/v1/chat/completions")
    )


def test_safe_body_matcher_skips_telemetry_body():
    a = _req(
        b'{"batch":[{"id":"aaa","timestamp":"2026-05-25T03:40:37Z"}]}',
        uri="https://us.cloud.langfuse.com/api/public/ingestion",
    )
    b = _req(
        b'{"batch":[{"id":"zzz","timestamp":"2026-05-25T09:99:99Z","extra":1}]}',
        uri="https://us.cloud.langfuse.com/api/public/ingestion",
    )
    _safe_body_matcher(a, b)  # must not raise despite wholly different bodies


def test_tolerant_query_skips_telemetry_but_enforces_others():
    from vcr.request import Request

    def _greq(uri):
        return Request(method="GET", uri=uri, body=b"", headers={})

    # Telemetry GET with a fresh trace_id in the query must still match.
    a = _greq(
        "https://us.cloud.langfuse.com/api/public/observations?traceId=litellm-test-AAA"
    )
    b = _greq(
        "https://us.cloud.langfuse.com/api/public/observations?traceId=litellm-test-BBB"
    )
    _tolerant_query_matcher(a, b)  # must not raise

    # Non-telemetry hosts keep vcrpy's strict query comparison.
    c = _greq("https://api.openai.com/v1/models?page=1")
    d = _greq("https://api.openai.com/v1/models?page=2")
    with pytest.raises(AssertionError):
        _tolerant_query_matcher(c, d)


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

"""Unit tests for ``tests.e2e_cassette_proxy.cache_key.derive_cache_key``.

The cache key is the only thing standing between "cache hit" and "cache
miss," so its invariants need to be pinned down precisely. Every test
below is a single equivalence claim: "these two requests should hash
to the same key" or "these two should not."
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from tests.e2e_cassette_proxy.cache_key import (  # noqa: E402
    CACHE_KEY_PREFIX,
    derive_cache_key,
)


def _key(
    method="POST",
    url="https://api.openai.com/v1/chat/completions",
    body=b"",
    headers=None,
):
    return derive_cache_key(method, url, body, headers or {})


def test_should_produce_stable_key_with_prefix():
    key = _key()
    assert key.startswith(CACHE_KEY_PREFIX)
    # 64 hex chars after the prefix.
    assert len(key) == len(CACHE_KEY_PREFIX) + 64


def test_should_match_when_inputs_are_byte_for_byte_identical():
    a = _key(body=b"hello", headers={"content-type": "application/json"})
    b = _key(body=b"hello", headers={"content-type": "application/json"})
    assert a == b


def test_should_match_when_only_auth_headers_differ():
    a = _key(
        headers={"authorization": "Bearer one", "content-type": "application/json"}
    )
    b = _key(
        headers={"authorization": "Bearer two", "content-type": "application/json"}
    )
    assert a == b


def test_should_match_when_only_tracing_headers_differ():
    a = _key(headers={"x-request-id": "abc", "content-type": "application/json"})
    b = _key(headers={"x-request-id": "xyz", "content-type": "application/json"})
    assert a == b


def test_should_match_when_user_agent_differs():
    a = _key(
        headers={"user-agent": "openai-python/1.0", "content-type": "application/json"}
    )
    b = _key(headers={"user-agent": "curl/8", "content-type": "application/json"})
    assert a == b


def test_should_match_when_json_body_differs_only_in_key_order():
    a = _key(
        body=b'{"model":"gpt-4o","messages":[]}',
        headers={"content-type": "application/json"},
    )
    b = _key(
        body=b'{"messages":[],"model":"gpt-4o"}',
        headers={"content-type": "application/json"},
    )
    assert a == b


def test_should_match_when_query_params_only_differ_in_order():
    a = _key(url="https://api.openai.com/v1/x?b=2&a=1")
    b = _key(url="https://api.openai.com/v1/x?a=1&b=2")
    assert a == b


def test_should_match_when_host_capitalization_differs():
    a = _key(url="https://api.openai.com/v1/chat/completions")
    b = _key(url="https://API.OPENAI.COM/v1/chat/completions")
    assert a == b


def test_should_differ_when_method_changes():
    assert _key(method="GET") != _key(method="POST")


def test_should_differ_when_path_changes():
    a = _key(url="https://api.openai.com/v1/chat/completions")
    b = _key(url="https://api.openai.com/v1/responses")
    assert a != b


def test_should_differ_when_body_changes_meaningfully():
    a = _key(
        body=b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}',
        headers={"content-type": "application/json"},
    )
    b = _key(
        body=b'{"model":"gpt-4o","messages":[{"role":"user","content":"bye"}]}',
        headers={"content-type": "application/json"},
    )
    assert a != b


def test_should_differ_when_allowlisted_header_changes():
    # ``anthropic-version`` is on the allowlist — changing it must miss.
    a = _key(headers={"anthropic-version": "2023-06-01"})
    b = _key(headers={"anthropic-version": "2024-01-01"})
    assert a != b


def test_should_match_when_blocklisted_header_added_or_removed():
    a = _key(headers={"content-type": "application/json"})
    b = _key(
        headers={"content-type": "application/json", "x-amz-date": "20260501T000000Z"}
    )
    assert a == b


def test_should_handle_non_json_body_passthrough():
    a = _key(
        body=b"\x00\x01\x02binary", headers={"content-type": "application/octet-stream"}
    )
    b = _key(
        body=b"\x00\x01\x02binary", headers={"content-type": "application/octet-stream"}
    )
    c = _key(body=b"different", headers={"content-type": "application/octet-stream"})
    assert a == b
    assert a != c


def test_should_treat_query_string_difference_as_a_miss():
    a = _key(url="https://api.openai.com/v1/files?purpose=batch")
    b = _key(url="https://api.openai.com/v1/files?purpose=fine-tune")
    assert a != b

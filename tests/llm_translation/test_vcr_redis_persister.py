"""Tests for the Redis-backed vcrpy cassette persister.

These cover the three behaviours we actually rely on in CI:

1. ``save_cassette`` followed by ``load_cassette`` returns the same
   request/response pairs (roundtrip via the real vcrpy serializer).
2. Saved keys expire after ~24h so the cache auto-refreshes against live
   providers without manual ``make`` runs.
3. ``load_cassette`` raises ``CassetteNotFoundError`` on a miss, so vcrpy's
   record-mode machinery falls through to a live HTTP call instead of
   silently matching against an empty cassette.

We also pin the 2xx-only filter so a transient 5xx/429 from the provider
can't be baked into the cache for the rest of the TTL window.
"""

from __future__ import annotations

import os
import sys

import fakeredis
import pytest
from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.request import Request
from vcr.serializers import yamlserializer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _vcr_redis_persister import (  # noqa: E402
    CASSETTE_TTL_SECONDS,
    filter_non_2xx_response,
    make_redis_persister,
    redis_key_for,
)


def _sample_cassette_dict():
    """Build a minimal cassette payload that exercises serialize/deserialize."""
    request = Request(
        method="POST",
        uri="https://api.anthropic.com/v1/messages",
        body=b'{"model":"claude","messages":[{"role":"user","content":"hi"}]}',
        headers={"content-type": "application/json"},
    )
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"content-type": ["application/json"]},
        # vcrpy stores response bodies as bytes; mirror that so the
        # roundtrip assertion exercises real-world serialization shapes.
        "body": {"string": b'{"id":"msg_1","type":"message"}'},
    }
    return {"requests": [request], "responses": [response]}


def _persister_with_fake_redis():
    fake = fakeredis.FakeStrictRedis()
    return fake, make_redis_persister(client=fake)


def test_save_then_load_roundtrips_cassette_content():
    """A saved cassette must come back from ``load_cassette`` identical to
    what was put in. If serialize/deserialize ever drift (e.g. encoding bug)
    every replay-mode test in the suite breaks; this catches it cheaply."""
    fake, persister = _persister_with_fake_redis()
    cassette_path = "tests/llm_translation/cassettes/test_x/test_y.yaml"

    persister.save_cassette(cassette_path, _sample_cassette_dict(), yamlserializer)
    requests, responses = persister.load_cassette(cassette_path, yamlserializer)

    assert len(requests) == 1
    assert len(responses) == 1
    assert requests[0].method == "POST"
    assert requests[0].uri == "https://api.anthropic.com/v1/messages"
    assert responses[0]["status"]["code"] == 200
    assert responses[0]["body"]["string"] == b'{"id":"msg_1","type":"message"}'


def test_saved_key_has_24h_ttl():
    """The whole point of the Redis backend is that entries auto-expire after
    24h so each daily CI run re-records against live providers. If the TTL
    isn't being applied, the cache never refreshes and we silently mask
    upstream API drift."""
    fake, persister = _persister_with_fake_redis()
    cassette_path = "tests/llm_translation/cassettes/test_x/test_ttl.yaml"

    persister.save_cassette(cassette_path, _sample_cassette_dict(), yamlserializer)

    ttl = fake.ttl(redis_key_for(cassette_path))
    assert ttl > 0, "key was saved without an expiry — would never refresh"
    assert ttl <= CASSETTE_TTL_SECONDS
    assert ttl >= CASSETTE_TTL_SECONDS - 5  # allow tiny clock slack


def test_load_missing_key_raises_cassette_not_found():
    """Cache miss must surface as ``CassetteNotFoundError``. vcrpy's record
    machinery catches that exception and falls through to the live HTTP
    call; if we returned empty/None instead, vcrpy would treat it as a
    cassette with zero matching requests and the test would fail with a
    confusing ``CannotOverwriteExistingCassetteException``."""
    _, persister = _persister_with_fake_redis()
    with pytest.raises(CassetteNotFoundError):
        persister.load_cassette("never/recorded.yaml", yamlserializer)


@pytest.mark.parametrize(
    ("status_code", "expect_dropped"),
    [
        (200, False),
        (201, False),
        (204, False),
        (299, False),
        (300, True),
        (400, True),
        (401, True),
        (404, True),
        (429, True),  # rate limit — must never be cached
        (500, True),  # transient 5xx — must never be cached
        (502, True),
        (503, True),
    ],
)
def test_only_2xx_responses_are_cached(status_code, expect_dropped):
    """Pin the cache-poisoning protection: a non-2xx must be dropped from
    the cassette (returned as ``None`` from the hook) so a transient 429
    or 503 doesn't get pinned for the rest of the TTL window. 2xx
    responses must pass through untouched."""
    response = {
        "status": {"code": status_code, "message": "X"},
        "headers": {},
        "body": {"string": ""},
    }
    result = filter_non_2xx_response(response)
    if expect_dropped:
        assert result is None
    else:
        assert result is response

from __future__ import annotations

import os
import sys

import fakeredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.request import Request
from vcr.serializers import yamlserializer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_redis_persister import (  # noqa: E402
    CASSETTE_TTL_SECONDS,
    MAX_EPISODES_PER_CASSETTE,
    filter_non_2xx_response,
    make_redis_persister,
    mark_test_outcome_for_cassette,
    redis_key_for,
)


def _sample_cassette_dict():
    request = Request(
        method="POST",
        uri="https://api.anthropic.com/v1/messages",
        body=b'{"model":"claude","messages":[{"role":"user","content":"hi"}]}',
        headers={"content-type": "application/json"},
    )
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"content-type": ["application/json"]},
        "body": {"string": b'{"id":"msg_1","type":"message"}'},
    }
    return {"requests": [request], "responses": [response]}


def _persister_with_fake_redis():
    fake = fakeredis.FakeStrictRedis()
    return fake, make_redis_persister(client=fake)


def test_save_then_load_roundtrips_cassette_content():
    _, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_y"

    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)
    requests, responses = persister.load_cassette(cassette_id, yamlserializer)

    assert len(requests) == 1
    assert len(responses) == 1
    assert requests[0].method == "POST"
    assert requests[0].uri == "https://api.anthropic.com/v1/messages"
    assert responses[0]["status"]["code"] == 200
    assert responses[0]["body"]["string"] == b'{"id":"msg_1","type":"message"}'


def test_saved_key_has_24h_ttl():
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_ttl"

    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)

    ttl = fake.ttl(redis_key_for(cassette_id))
    assert CASSETTE_TTL_SECONDS - 5 <= ttl <= CASSETTE_TTL_SECONDS


def test_load_missing_key_raises_cassette_not_found():
    _, persister = _persister_with_fake_redis()
    with pytest.raises(CassetteNotFoundError):
        persister.load_cassette("never/recorded", yamlserializer)


def test_redis_key_normalizes_path_passed_by_pytest_recording():
    # pytest-recording passes paths shaped like
    # ``<test_dir>/cassettes/<module>/<test>.yaml``. The persister stores them
    # under a clean test-identifier key — no extension, no ``cassettes/``
    # directory segment — so ``redis-cli keys`` reads as test IDs.
    raw = "tests/llm_translation/cassettes/test_anthropic/test_streaming.yaml"
    assert (
        redis_key_for(raw)
        == "litellm:vcr:cassette:tests/llm_translation/test_anthropic/test_streaming"
    )


class _FlakyRedis:
    """Wraps a fake redis but raises ConnectionError on the chosen op."""

    def __init__(self, inner, fail_on: str):
        self._inner = inner
        self._fail_on = fail_on

    def get(self, *args, **kwargs):
        if self._fail_on == "get":
            raise RedisConnectionError("simulated outage")
        return self._inner.get(*args, **kwargs)

    def set(self, *args, **kwargs):
        if self._fail_on == "set":
            raise RedisConnectionError("simulated outage")
        return self._inner.set(*args, **kwargs)


def test_save_swallows_connection_errors_so_teardown_does_not_fail():
    # Persistence is a cache; an outage shouldn't fail an otherwise-passing test.
    flaky = _FlakyRedis(fakeredis.FakeStrictRedis(), fail_on="set")
    persister = make_redis_persister(client=flaky)

    persister.save_cassette(
        "tests/llm_translation/test_x/test_save_outage",
        _sample_cassette_dict(),
        yamlserializer,
    )


def test_save_skipped_when_test_marked_failed_and_prior_cassette_preserved():
    # A flaky test that fails should NOT overwrite a previously-good cassette.
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_flaky"
    key = redis_key_for(cassette_id)

    # Seed a "known-good" recording from a prior successful run.
    good = _sample_cassette_dict()
    persister.save_cassette(cassette_id, good, yamlserializer)
    good_payload = fake.get(key)
    assert good_payload is not None

    # Simulate a failed run: the hook records "did not pass" before save.
    mark_test_outcome_for_cassette(cassette_id, passed=False)
    bad_response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {},
        "body": {"string": b'{"id":"BAD","type":"message"}'},
    }
    bad = {"requests": good["requests"], "responses": [bad_response]}
    persister.save_cassette(cassette_id, bad, yamlserializer)

    # Prior good payload is still there — the bad save was suppressed.
    assert fake.get(key) == good_payload


def test_save_proceeds_when_test_marked_passed():
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_passed"
    key = redis_key_for(cassette_id)

    mark_test_outcome_for_cassette(cassette_id, passed=True)
    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)

    assert fake.get(key) is not None


def test_save_refused_when_cassette_exceeds_max_episodes():
    # Pathological cassettes (non-deterministic body → unbounded episode growth)
    # should be refused. Any prior good payload stays intact.
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_runaway"
    key = redis_key_for(cassette_id)

    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)
    seed_payload = fake.get(key)

    request = Request(
        method="POST",
        uri="https://api.anthropic.com/v1/messages",
        body=b"x",
        headers={"content-type": "application/json"},
    )
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {},
        "body": {"string": b"{}"},
    }
    bloated = {
        "requests": [request] * (MAX_EPISODES_PER_CASSETTE + 1),
        "responses": [response] * (MAX_EPISODES_PER_CASSETTE + 1),
    }
    persister.save_cassette(cassette_id, bloated, yamlserializer)

    # Refused — the seed payload is unchanged.
    assert fake.get(key) == seed_payload


def test_save_proceeds_at_max_episodes_threshold():
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_at_threshold"
    key = redis_key_for(cassette_id)

    request = Request(
        method="POST",
        uri="https://api.anthropic.com/v1/messages",
        body=b"x",
        headers={"content-type": "application/json"},
    )
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {},
        "body": {"string": b"{}"},
    }
    at_threshold = {
        "requests": [request] * MAX_EPISODES_PER_CASSETTE,
        "responses": [response] * MAX_EPISODES_PER_CASSETTE,
    }
    persister.save_cassette(cassette_id, at_threshold, yamlserializer)

    assert fake.get(key) is not None


def test_save_proceeds_when_outcome_unknown():
    # Used outside a pytest run (e.g. ad-hoc scripts), the outcome gate is
    # bypassed so the persister still works.
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_no_marker"
    key = redis_key_for(cassette_id)

    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)

    assert fake.get(key) is not None


def test_load_treats_connection_errors_as_cassette_miss():
    # An outage on read should fall through to a live call (CassetteNotFound),
    # not surface a redis exception in the test setup.
    flaky = _FlakyRedis(fakeredis.FakeStrictRedis(), fail_on="get")
    persister = make_redis_persister(client=flaky)

    with pytest.raises(CassetteNotFoundError):
        persister.load_cassette(
            "tests/llm_translation/test_x/test_load_outage", yamlserializer
        )


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
        (429, True),
        (500, True),
        (502, True),
        (503, True),
    ],
)
def test_only_2xx_responses_are_cached(status_code, expect_dropped):
    response = {
        "status": {"code": status_code, "message": "X"},
        "headers": {},
        "body": {"string": ""},
    }
    result = filter_non_2xx_response(response)
    assert (result is None) == expect_dropped
    if not expect_dropped:
        assert result is response

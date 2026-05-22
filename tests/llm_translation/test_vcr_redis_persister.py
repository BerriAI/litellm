from __future__ import annotations

import os
import sys

import fakeredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import OutOfMemoryError as RedisOutOfMemoryError
from redis.exceptions import TimeoutError as RedisTimeoutError
from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.request import Request
from vcr.serializers import yamlserializer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_redis_persister import (  # noqa: E402
    CASSETTE_TTL_SECONDS,
    MAX_EPISODES_PER_CASSETTE,
    VCRCassetteCacheWarning,
    cassette_cache_capacity_snapshot,
    cassette_cache_health,
    filter_non_2xx_response,
    make_redis_persister,
    mark_test_outcome_for_cassette,
    redis_key_for,
    reset_cassette_cache_health,
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
    raw = "tests/llm_translation/cassettes/test_anthropic/test_streaming.yaml"
    assert (
        redis_key_for(raw)
        == "litellm:vcr:cassette:tests/llm_translation/test_anthropic/test_streaming"
    )


def test_redis_key_is_stable_across_working_directories(tmp_path, monkeypatch):
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    abs_cassette = os.path.join(
        repo_root,
        "tests/llm_translation/cassettes/test_anthropic/test_streaming.yaml",
    )

    monkeypatch.chdir(repo_root)
    key_from_root = redis_key_for(abs_cassette)

    monkeypatch.chdir(os.path.join(repo_root, "tests", "llm_translation"))
    key_from_subdir = redis_key_for(abs_cassette)

    monkeypatch.chdir(tmp_path)
    key_from_tmp = redis_key_for(abs_cassette)

    assert key_from_root == key_from_subdir == key_from_tmp
    assert (
        key_from_root
        == "litellm:vcr:cassette:tests/llm_translation/test_anthropic/test_streaming"
    )


class _FlakyRedis:
    def __init__(self, inner, fail_on: str, exc=None):
        self._inner = inner
        self._fail_on = fail_on
        self._exc = exc if exc is not None else RedisConnectionError("simulated outage")

    def get(self, *args, **kwargs):
        if self._fail_on == "get":
            raise self._exc
        return self._inner.get(*args, **kwargs)

    def set(self, *args, **kwargs):
        if self._fail_on == "set":
            raise self._exc
        return self._inner.set(*args, **kwargs)


@pytest.mark.parametrize(
    "exc",
    [
        RedisConnectionError("simulated outage"),
        RedisTimeoutError("simulated timeout"),
        RedisOutOfMemoryError("command not allowed when used memory > 'maxmemory'."),
    ],
    ids=["connection_error", "timeout", "out_of_memory"],
)
def test_save_swallows_redis_errors_so_teardown_does_not_fail(exc):
    """Redis-side failures during cassette persistence must never fail
    the test on teardown.

    Regression: previously the persister only swallowed
    ConnectionError/TimeoutError, so OutOfMemoryError (raised by Redis
    Cloud when the cassette cache hit its maxmemory cap) propagated out
    of vcrpy's autouse fixture and failed otherwise-passing tests on
    teardown.
    """
    flaky = _FlakyRedis(fakeredis.FakeStrictRedis(), fail_on="set", exc=exc)
    persister = make_redis_persister(client=flaky)

    persister.save_cassette(
        "tests/llm_translation/test_x/test_save_outage",
        _sample_cassette_dict(),
        yamlserializer,
    )


def test_save_skipped_when_test_marked_failed_and_prior_cassette_preserved():
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_flaky"
    key = redis_key_for(cassette_id)

    good = _sample_cassette_dict()
    persister.save_cassette(cassette_id, good, yamlserializer)
    good_payload = fake.get(key)
    assert good_payload is not None

    mark_test_outcome_for_cassette(cassette_id, passed=False)
    bad_response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {},
        "body": {"string": b'{"id":"BAD","type":"message"}'},
    }
    bad = {"requests": good["requests"], "responses": [bad_response]}
    persister.save_cassette(cassette_id, bad, yamlserializer)

    assert fake.get(key) == good_payload


def test_save_proceeds_when_test_marked_passed():
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_passed"
    key = redis_key_for(cassette_id)

    mark_test_outcome_for_cassette(cassette_id, passed=True)
    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)

    assert fake.get(key) is not None


def test_save_refused_when_cassette_exceeds_max_episodes():
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
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_no_marker"
    key = redis_key_for(cassette_id)

    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)

    assert fake.get(key) is not None


@pytest.mark.parametrize(
    "exc",
    [
        RedisConnectionError("simulated outage"),
        RedisTimeoutError("simulated timeout"),
        RedisOutOfMemoryError("command not allowed when used memory > 'maxmemory'."),
    ],
    ids=["connection_error", "timeout", "out_of_memory"],
)
def test_load_treats_redis_errors_as_cassette_miss(exc):
    flaky = _FlakyRedis(fakeredis.FakeStrictRedis(), fail_on="get", exc=exc)
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


@pytest.fixture
def reset_health():
    reset_cassette_cache_health()
    yield
    reset_cassette_cache_health()


def test_save_failure_increments_health_counter_and_emits_warning(reset_health):
    flaky = _FlakyRedis(
        fakeredis.FakeStrictRedis(),
        fail_on="set",
        exc=RedisOutOfMemoryError(
            "command not allowed when used memory > 'maxmemory'."
        ),
    )
    persister = make_redis_persister(client=flaky)

    with pytest.warns(VCRCassetteCacheWarning, match="OutOfMemoryError"):
        persister.save_cassette(
            "tests/llm_translation/test_x/test_save_outage",
            _sample_cassette_dict(),
            yamlserializer,
        )

    health = cassette_cache_health()
    assert health["save_failures"] == 1
    assert "OutOfMemoryError" in health["save_failure_last_error"]
    assert health["load_failures"] == 0


def test_load_failure_increments_health_counter_and_emits_warning(reset_health):
    flaky = _FlakyRedis(
        fakeredis.FakeStrictRedis(),
        fail_on="get",
        exc=RedisConnectionError("simulated outage"),
    )
    persister = make_redis_persister(client=flaky)

    with pytest.warns(VCRCassetteCacheWarning, match="ConnectionError"):
        with pytest.raises(CassetteNotFoundError):
            persister.load_cassette(
                "tests/llm_translation/test_x/test_load_outage", yamlserializer
            )

    health = cassette_cache_health()
    assert health["load_failures"] == 1
    assert "ConnectionError" in health["load_failure_last_error"]
    assert health["save_failures"] == 0


def test_health_counters_accumulate_across_failures(reset_health):
    flaky = _FlakyRedis(
        fakeredis.FakeStrictRedis(),
        fail_on="set",
        exc=RedisConnectionError("simulated outage"),
    )
    persister = make_redis_persister(client=flaky)

    for i in range(3):
        with pytest.warns(VCRCassetteCacheWarning):
            persister.save_cassette(
                f"tests/llm_translation/test_x/test_outage_{i}",
                _sample_cassette_dict(),
                yamlserializer,
            )

    assert cassette_cache_health()["save_failures"] == 3


def test_successful_save_does_not_emit_warning_or_increment_counter(reset_health):
    _, persister = _persister_with_fake_redis()

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", VCRCassetteCacheWarning)
        persister.save_cassette(
            "tests/llm_translation/test_x/test_happy",
            _sample_cassette_dict(),
            yamlserializer,
        )

    assert cassette_cache_health()["save_failures"] == 0


class _FakeRedisWithInfo:
    def __init__(self, used: int, maxmem: int):
        self._used = used
        self._maxmem = maxmem

    def info(self, section=None):
        return {"used_memory": self._used, "maxmemory": self._maxmem}


def test_capacity_snapshot_returns_used_max_and_pct():
    client = _FakeRedisWithInfo(used=900, maxmem=1000)
    snap = cassette_cache_capacity_snapshot(client=client)
    assert snap == {
        "used_memory_bytes": 900,
        "maxmemory_bytes": 1000,
        "used_pct": 90.0,
    }


def test_capacity_snapshot_returns_none_when_uncapped():
    client = _FakeRedisWithInfo(used=900, maxmem=0)
    assert cassette_cache_capacity_snapshot(client=client) is None


def test_capacity_snapshot_returns_none_when_used_unknown():
    client = _FakeRedisWithInfo(used=0, maxmem=1000)
    assert cassette_cache_capacity_snapshot(client=client) is None


def test_capacity_snapshot_swallows_exceptions():
    class _Boom:
        def info(self, section=None):
            raise RuntimeError("redis offline")

    assert cassette_cache_capacity_snapshot(client=_Boom()) is None

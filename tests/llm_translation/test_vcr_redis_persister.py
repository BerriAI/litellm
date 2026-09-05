from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import fakeredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import OutOfMemoryError as RedisOutOfMemoryError
from redis.exceptions import TimeoutError as RedisTimeoutError
from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.request import Request
from vcr.serializers import yamlserializer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_persister import (  # noqa: E402
    FilesystemBackend,
    make_persister,
    set_cassette_ttl_override,
)
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


def test_load_does_not_refresh_ttl_so_cassettes_lapse_after_write():
    """A successful read must not slide the cassette's expiry forward.

    The TTL deliberately counts down from the last *write*: a cassette that
    is only ever replayed must still lapse ``CASSETTE_TTL_SECONDS`` after it
    was recorded, so the next run past that point re-records live and catches
    provider request/response contract drift. Refreshing the TTL on read
    would keep an actively-used cassette alive forever and that drift check
    would never run.
    """
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_ttl_no_refresh"
    key = redis_key_for(cassette_id)

    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)
    # Simulate a cassette written ~most-of-a-day ago: only a little TTL left.
    fake.expire(key, 60)

    persister.load_cassette(cassette_id, yamlserializer)

    assert fake.ttl(key) <= 60


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


def test_filesystem_save_then_load_roundtrips_with_metadata(tmp_path):
    now = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    cassette_path = tmp_path / "nested" / "cassette.yaml"
    persister = FilesystemBackend(now=lambda: now)

    persister.save_cassette(cassette_path, _sample_cassette_dict(), yamlserializer)
    requests, responses = persister.load_cassette(cassette_path, yamlserializer)
    payload = yamlserializer.deserialize(cassette_path.read_text())

    assert payload["recorded_at"] == "2026-09-02T12:00:00+00:00"
    assert payload["ttl_seconds"] == CASSETTE_TTL_SECONDS
    assert requests[0].body.startswith(b'{"model":"claude"')
    assert responses[0]["body"]["string"] == b'{"id":"msg_1","type":"message"}'


def test_filesystem_expired_cassette_is_a_miss(tmp_path):
    recorded_at = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    cassette_path = tmp_path / "expired.yaml"
    writer = FilesystemBackend(ttl_seconds=60, now=lambda: recorded_at)
    reader = FilesystemBackend(now=lambda: recorded_at + timedelta(seconds=61))
    writer.save_cassette(cassette_path, _sample_cassette_dict(), yamlserializer)

    with pytest.raises(CassetteNotFoundError):
        reader.load_cassette(cassette_path, yamlserializer)


def test_filesystem_cassette_without_recorded_at_is_a_miss(tmp_path):
    cassette_path = tmp_path / "legacy.yaml"
    cassette_path.write_text(
        yamlserializer.serialize({"version": 1, "interactions": []})
    )

    with pytest.raises(CassetteNotFoundError):
        FilesystemBackend().load_cassette(cassette_path, yamlserializer)


@pytest.mark.parametrize("ttl", [0, -1, "inf"])
def test_filesystem_non_positive_and_infinite_ttl_never_expire(tmp_path, ttl):
    recorded_at = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    cassette_path = tmp_path / f"immortal-{ttl}.yaml"
    set_cassette_ttl_override(cassette_path, ttl)
    writer = FilesystemBackend(now=lambda: recorded_at)
    reader = FilesystemBackend(now=lambda: recorded_at + timedelta(days=36500))
    writer.save_cassette(cassette_path, _sample_cassette_dict(), yamlserializer)

    requests, _ = reader.load_cassette(cassette_path, yamlserializer)

    assert len(requests) == 1


def test_redis_ttl_override_controls_set_expiry(monkeypatch):
    monkeypatch.setenv("CASSETTE_TTL_SECONDS", "7200")
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_custom_ttl"
    set_cassette_ttl_override(cassette_id, 3600)

    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)

    assert 3595 <= fake.ttl(redis_key_for(cassette_id)) <= 3600


def test_redis_non_positive_ttl_uses_set_without_expiry():
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_immortal"
    set_cassette_ttl_override(cassette_id, 0)

    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)

    assert fake.ttl(redis_key_for(cassette_id)) == -1


def test_ttl_env_is_baked_into_filesystem_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("CASSETTE_TTL_SECONDS", "7200")
    cassette_path = tmp_path / "env-ttl.yaml"

    FilesystemBackend().save_cassette(
        cassette_path, _sample_cassette_dict(), yamlserializer
    )

    payload = yamlserializer.deserialize(cassette_path.read_text())
    assert payload["ttl_seconds"] == 7200


def test_ttl_override_is_baked_into_filesystem_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("CASSETTE_TTL_SECONDS", "7200")
    cassette_path = tmp_path / "marker-ttl.yaml"
    set_cassette_ttl_override(cassette_path, 3600)

    FilesystemBackend().save_cassette(
        cassette_path, _sample_cassette_dict(), yamlserializer
    )

    payload = yamlserializer.deserialize(cassette_path.read_text())
    assert payload["ttl_seconds"] == 3600


def test_filesystem_corrupt_file_is_a_warned_miss(tmp_path, reset_health):
    cassette_path = tmp_path / "corrupt.yaml"
    cassette_path.write_text("not: [valid")

    with pytest.warns(VCRCassetteCacheWarning):
        with pytest.raises(CassetteNotFoundError):
            FilesystemBackend().load_cassette(cassette_path, yamlserializer)

    assert cassette_cache_health()["load_failures"] == 1


def test_filesystem_atomic_write_preserves_existing_file_on_replace_failure(
    tmp_path, reset_health
):
    cassette_path = tmp_path / "atomic.yaml"
    cassette_path.write_text("existing")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    persister = FilesystemBackend(replace=fail_replace)
    with pytest.warns(VCRCassetteCacheWarning, match="simulated replace failure"):
        persister.save_cassette(cassette_path, _sample_cassette_dict(), yamlserializer)

    assert cassette_path.read_text() == "existing"
    assert list(tmp_path.iterdir()) == [cassette_path]


def test_filesystem_failed_test_preserves_prior_cassette(tmp_path):
    cassette_path = tmp_path / "failed.yaml"
    persister = FilesystemBackend()
    persister.save_cassette(
        cassette_path, _sample_cassette_dict(), yamlserializer
    )
    original = cassette_path.read_bytes()
    mark_test_outcome_for_cassette(cassette_path, passed=False)

    persister.save_cassette(
        cassette_path, _sample_cassette_dict(), yamlserializer
    )

    assert cassette_path.read_bytes() == original


def test_filesystem_episode_cap_preserves_prior_cassette(tmp_path):
    cassette_path = tmp_path / "overflow.yaml"
    persister = FilesystemBackend()
    sample = _sample_cassette_dict()
    persister.save_cassette(cassette_path, sample, yamlserializer)
    original = cassette_path.read_bytes()
    overflow = {
        "requests": sample["requests"] * (MAX_EPISODES_PER_CASSETTE + 1),
        "responses": sample["responses"] * (MAX_EPISODES_PER_CASSETTE + 1),
    }

    persister.save_cassette(cassette_path, overflow, yamlserializer)

    assert cassette_path.read_bytes() == original


def test_filesystem_capacity_snapshot_is_none():
    assert FilesystemBackend().capacity_snapshot() is None


def test_factory_selects_filesystem_by_default(monkeypatch):
    monkeypatch.delenv("CASSETTE_BACKEND", raising=False)
    monkeypatch.delenv("CASSETTE_REDIS_URL", raising=False)

    assert isinstance(make_persister(), FilesystemBackend)


def test_factory_explicit_filesystem_wins_over_redis_url(monkeypatch):
    monkeypatch.setenv("CASSETTE_BACKEND", "filesystem")
    monkeypatch.setenv("CASSETTE_REDIS_URL", "redis://unused")

    assert isinstance(make_persister(), FilesystemBackend)

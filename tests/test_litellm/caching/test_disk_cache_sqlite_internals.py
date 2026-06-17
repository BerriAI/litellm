"""Deterministic unit tests for the sqlite3 store backing DiskCache.

These use an injected clock so TTL/expiry boundaries are exact (no sleeps),
pinning the comparison and arithmetic that real-time tests can only probe
loosely. Candidate-only: they reference _SqliteCache directly.
"""

import json

import pytest

from litellm.caching.disk_cache import DiskCache, _SqliteCache, _decode, _encode


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def store(tmp_path):
    return _SqliteCache(str(tmp_path / "s"), time_fn=Clock(1000.0))


def test_no_expire_never_expires(tmp_path):
    clock = Clock(0.0)
    s = _SqliteCache(str(tmp_path / "s"), time_fn=clock)
    s.set("k", "v")
    clock.t = 10**9
    assert s.get("k") == "v"


def test_expire_boundary_is_inclusive(tmp_path):
    clock = Clock(100.0)
    s = _SqliteCache(str(tmp_path / "s"), time_fn=clock)
    s.set("k", "v", expire=10)  # expire_time = 110
    clock.t = 109.999
    assert s.get("k") == "v"  # strictly before deadline -> alive
    clock.t = 110.0
    assert s.get("k") is None  # at deadline -> expired (kills `<` mutant)
    clock.t = 110.001
    assert s.get("k") is None


def test_expire_time_is_set_plus_expire(tmp_path):
    clock = Clock(100.0)
    s = _SqliteCache(str(tmp_path / "s"), time_fn=clock)
    s.set("k", "v", expire=10)
    clock.t = 109.0
    assert s.get("k") == "v"  # alive at 109 -> expire_time must be 110, not 90
    clock.t = 111.0
    assert s.get("k") is None


def test_expire_zero_and_negative_immediately_expired(tmp_path):
    clock = Clock(100.0)
    s = _SqliteCache(str(tmp_path / "s"), time_fn=clock)
    s.set("z", "v", expire=0)
    assert s.get("z") is None
    s.set("n", "v", expire=-5)
    assert s.get("n") is None


@pytest.mark.parametrize(
    "value",
    [
        b"\x00raw",
        "plain",
        {"a": 1, "b": [1, 2, 3]},
        [1, 2, 3],
        0,
        5,
        -3,
        1.5,
        True,
        False,
        None,
        "",
    ],
)
def test_encode_decode_roundtrip(store, value):
    store.set("k", value)
    got = store.get("k")
    assert got == value
    assert type(got) is type(value)


def test_encode_modes():
    assert _encode(b"x") == (b"x", "b")
    assert _encode("x") == ("x", "s")
    assert _encode({"a": 1}) == (json.dumps({"a": 1}), "j")
    assert _encode(5) == ("5", "j")


def test_decode_modes():
    assert _decode("x", "s") == "x"
    assert _decode(b"x", "b") == b"x"
    assert _decode("[1, 2]", "j") == [1, 2]


def test_pop_returns_value_and_deletes(store):
    store.set("k", {"a": 1})
    assert store.pop("k") == {"a": 1}
    assert store.get("k") is None


def test_pop_missing_returns_none(store):
    assert store.pop("absent") is None


def test_pop_expired_returns_none(tmp_path):
    clock = Clock(100.0)
    s = _SqliteCache(str(tmp_path / "s"), time_fn=clock)
    s.set("k", "v", expire=10)
    clock.t = 200.0
    assert s.pop("k") is None


def test_insert_or_replace_overwrites(store):
    store.set("k", {"v": 1})
    store.set("k", {"v": 2})
    assert store.get("k") == {"v": 2}


def test_clear_empties(store):
    store.set("a", 1)
    store.set("b", 2)
    store.clear()
    assert store.get("a") is None
    assert store.get("b") is None


def test_set_overwrite_clears_prior_expiry(tmp_path):
    clock = Clock(100.0)
    s = _SqliteCache(str(tmp_path / "s"), time_fn=clock)
    s.set("k", "v", expire=10)
    s.set("k", "v2")  # no expiry now
    clock.t = 10**6
    assert s.get("k") == "v2"  # prior expiry must not linger


def test_diskcache_injects_clock(tmp_path):
    clock = Clock(0.0)
    dc = DiskCache(disk_cache_dir=str(tmp_path / "d"), time_fn=clock)
    dc.set_cache("k", json.dumps({"x": 1}), ttl=10)
    clock.t = 5
    assert dc.get_cache("k") == {"x": 1}
    clock.t = 20
    assert dc.get_cache("k") is None

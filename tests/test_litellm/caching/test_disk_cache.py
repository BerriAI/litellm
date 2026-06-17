"""Backend-agnostic behavioral battery for litellm.caching.disk_cache.DiskCache.

This same file runs unchanged against the diskcache-backed implementation
(baseline) and the stdlib-sqlite3 implementation (candidate). Every assertion
characterizes behavior empirically pinned from diskcache 5.6.3 (see
_verify_run/variables_reproduction.md). If both runs are green and the
value-fidelity snapshot is byte-identical, the backend swap preserved behavior.

No network, no LLM. Pure interface, property, stateful, sad-path, concurrency.
"""

import concurrent.futures
import json
import os
import shutil
import tempfile
import time

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from litellm.caching.disk_cache import DiskCache


@pytest.fixture()
def cache(tmp_path):
    return DiskCache(disk_cache_dir=str(tmp_path / "dc"))


# --------------------------------------------------------------------------- #
# Interface characterization (matches the probe exactly)
# --------------------------------------------------------------------------- #


def test_set_get_json_dict_string_returns_dict(cache):
    cache.set_cache("a", json.dumps({"x": 1}))
    assert cache.get_cache("a") == {"x": 1}


def test_set_get_plain_string_returns_string(cache):
    cache.set_cache("b", "plain")
    assert cache.get_cache("b") == "plain"


def test_get_missing_returns_none(cache):
    assert cache.get_cache("nope") is None


def test_hot_path_dict_shape_roundtrips(cache):
    """The shape litellm actually stores: {"timestamp": float, "response": json-str}."""
    value = {
        "timestamp": 1718600000.123456,
        "response": json.dumps({"id": "x", "choices": []}),
    }
    cache.set_cache("hot", value)
    assert cache.get_cache("hot") == value


def test_set_with_ttl_present_before_expiry(cache):
    cache.set_cache("c", json.dumps({"y": 2}), ttl=100)
    assert cache.get_cache("c") == {"y": 2}


def test_ttl_zero_is_immediately_expired(cache):
    cache.set_cache("z", json.dumps({"v": 1}), ttl=0)
    assert cache.get_cache("z") is None


def test_negative_ttl_is_immediately_expired(cache):
    cache.set_cache("n", json.dumps({"v": 1}), ttl=-5)
    assert cache.get_cache("n") is None


def test_ttl_expires_after_wall_clock(cache):
    cache.set_cache("t", json.dumps({"v": 1}), ttl=1)
    assert cache.get_cache("t") == {"v": 1}
    time.sleep(1.3)
    assert cache.get_cache("t") is None


def test_increment_from_empty_then_again(cache):
    assert cache.increment_cache("counter", 5) == 5
    assert cache.increment_cache("counter", 3) == 8
    assert cache.get_cache("counter") == 8


def test_batch_get_mixed_presence(cache):
    cache.set_cache("a", json.dumps({"x": 1}))
    cache.set_cache("b", "plain")
    assert cache.batch_get_cache(["a", "missing", "b"]) == [{"x": 1}, None, "plain"]


def test_delete_existing_then_missing_no_raise(cache):
    cache.set_cache("b", "plain")
    cache.delete_cache("b")
    assert cache.get_cache("b") is None
    cache.delete_cache("does-not-exist")  # must not raise


def test_flush_clears_everything(cache):
    cache.set_cache("a", json.dumps({"x": 1}))
    cache.set_cache("b", "plain")
    cache.flush_cache()
    assert cache.get_cache("a") is None
    assert cache.get_cache("b") is None


@pytest.mark.parametrize(
    "stored,expected",
    [
        ("", None),  # falsy raw value -> None (truthiness gate)
        (json.dumps(0), 0),
        (json.dumps(False), False),
        (json.dumps([]), []),
        (json.dumps({"k": "v"}), {"k": "v"}),
    ],
)
def test_truthiness_and_json_decode_edge_cases(cache, stored, expected):
    cache.set_cache("k", stored)
    assert cache.get_cache("k") == expected


def test_persistence_across_new_instance(tmp_path):
    d = str(tmp_path / "persist")
    first = DiskCache(disk_cache_dir=d)
    first.set_cache("p", json.dumps({"kept": True}))
    second = DiskCache(disk_cache_dir=d)
    assert second.get_cache("p") == {"kept": True}


def test_default_dir_branch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = DiskCache()  # exercises the default ".litellm_cache" branch
    c.set_cache("x", json.dumps({"ok": 1}))
    assert c.get_cache("x") == {"ok": 1}
    assert os.path.isdir(tmp_path / ".litellm_cache")


def test_large_value_roundtrips(cache):
    big = {"timestamp": 1.0, "response": "A" * 1_000_000}
    cache.set_cache("big", big)
    assert cache.get_cache("big") == big


def test_unicode_value_roundtrips(cache):
    value = {"timestamp": 1.0, "response": json.dumps({"text": "héllo 🌍 你好"})}
    cache.set_cache("u", value)
    assert cache.get_cache("u") == value


def test_unicode_key_roundtrips(cache):
    cache.set_cache("clé-🔑", json.dumps({"x": 1}))
    assert cache.get_cache("clé-🔑") == {"x": 1}


def test_bytes_value_roundtrips(cache):
    cache.set_cache("raw", b"\x00\x01rawbytes")
    assert cache.get_cache("raw") == b"\x00\x01rawbytes"


# --------------------------------------------------------------------------- #
# Async interface (delegates to sync, but exercise the async entry points)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_async_set_get(cache):
    await cache.async_set_cache("a", json.dumps({"x": 1}))
    assert await cache.async_get_cache("a") == {"x": 1}


@pytest.mark.asyncio
async def test_async_set_cache_pipeline_with_and_without_ttl(cache):
    await cache.async_set_cache_pipeline([("k1", "v1"), ("k2", "v2")])
    assert await cache.async_get_cache("k1") == "v1"
    await cache.async_set_cache_pipeline([("k3", "v3")], ttl=100)
    assert await cache.async_get_cache("k3") == "v3"


@pytest.mark.asyncio
async def test_async_batch_get(cache):
    cache.set_cache("a", json.dumps({"x": 1}))
    assert await cache.async_batch_get_cache(["a", "missing"]) == [{"x": 1}, None]


@pytest.mark.asyncio
async def test_async_increment(cache):
    assert await cache.async_increment("c", 2) == 2
    assert await cache.async_increment("c", 5) == 7


@pytest.mark.asyncio
async def test_disconnect_is_noop(cache):
    await cache.disconnect()


# --------------------------------------------------------------------------- #
# Sad paths
# --------------------------------------------------------------------------- #


def test_construct_on_path_that_is_a_file_raises(tmp_path):
    file_path = tmp_path / "iam_a_file"
    file_path.write_text("x")
    with pytest.raises(Exception):
        DiskCache(disk_cache_dir=str(file_path))


def test_concurrent_distinct_keys_no_lock_errors(cache):
    def worker(i):
        cache.set_cache(f"k{i}", json.dumps({"i": i}))
        return cache.get_cache(f"k{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, range(200)))
    assert results == [{"i": i} for i in range(200)]


def test_concurrent_same_key_last_write_wins(cache):
    def worker(i):
        cache.set_cache("shared", json.dumps({"i": i}))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, range(200)))
    got = cache.get_cache("shared")
    assert isinstance(got, dict) and 0 <= got["i"] < 200


# --------------------------------------------------------------------------- #
# Property-based round-trip (the json-vs-pickle fidelity surface)
# --------------------------------------------------------------------------- #

_finite_floats = st.floats(allow_nan=False, allow_infinity=False, width=64)
_json_text = st.text(max_size=200)
_hot_shape = st.builds(
    lambda ts, resp: {"timestamp": ts, "response": resp},
    _finite_floats,
    st.builds(
        lambda d: json.dumps(d),
        st.dictionaries(st.text(max_size=20), st.integers(), max_size=5),
    ),
)


@pytest.fixture(scope="module")
def prop_cache():
    d = tempfile.mkdtemp(prefix="dc_prop_")
    try:
        yield DiskCache(disk_cache_dir=d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@settings(max_examples=75, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(key=st.text(min_size=1, max_size=40), value=_hot_shape)
def test_property_hot_shape_roundtrip(prop_cache, key, value):
    prop_cache.set_cache(key, value)
    assert prop_cache.get_cache(key) == value


@settings(max_examples=75, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(key=st.text(min_size=1, max_size=40), n=st.integers().filter(lambda x: x != 0))
def test_property_nonzero_int_roundtrip(prop_cache, key, n):
    prop_cache.set_cache(key, n)
    assert prop_cache.get_cache(key) == n


# --------------------------------------------------------------------------- #
# Stateful: random op interleavings vs a reference model
# --------------------------------------------------------------------------- #


class DiskCacheStateMachine(RuleBasedStateMachine):
    """Values restricted to the round-trippable set so the model is exact."""

    keys = st.sampled_from(["k1", "k2", "k3", "k4"])
    values = st.one_of(
        st.builds(
            lambda n: {"timestamp": 1.0, "response": json.dumps({"n": n})},
            st.integers(),
        ),
        st.integers().filter(lambda x: x != 0),
    )

    def __init__(self):
        super().__init__()
        self._dir = tempfile.mkdtemp(prefix="dc_sm_")
        self.cache = DiskCache(disk_cache_dir=self._dir)
        self.model = {}

    @rule(key=keys, value=values)
    def set_(self, key, value):
        self.cache.set_cache(key, value)
        self.model[key] = value

    @rule(key=keys)
    def get_(self, key):
        assert self.cache.get_cache(key) == self.model.get(key)

    @rule(key=keys)
    def delete_(self, key):
        self.cache.delete_cache(key)
        self.model.pop(key, None)

    @rule()
    def flush_(self):
        self.cache.flush_cache()
        self.model.clear()

    @invariant()
    def all_keys_consistent(self):
        for k in ["k1", "k2", "k3", "k4"]:
            assert self.cache.get_cache(k) == self.model.get(k)

    def teardown(self):
        shutil.rmtree(self._dir, ignore_errors=True)


TestDiskCacheStateMachine = DiskCacheStateMachine.TestCase
TestDiskCacheStateMachine.settings = settings(
    max_examples=40, stateful_step_count=30, deadline=None
)


# --------------------------------------------------------------------------- #
# Value-fidelity snapshot: deterministic JSON, diffed byte-for-byte across
# backends. Writes to $DC_VERIFY_SNAPSHOT if set.
# --------------------------------------------------------------------------- #

_FIDELITY_MATRIX = {
    "hot_shape": {
        "timestamp": 1718600000.123456,
        "response": json.dumps({"id": "x", "choices": []}),
    },
    "plain_string": "plain string",
    "json_object_string": json.dumps({"a": 1}),
    "int_zero": 0,
    "int_five": 5,
    "int_negative": -3,
    "empty_string": "",
    "dumps_zero": json.dumps(0),
    "dumps_false": json.dumps(False),
    "dumps_empty_list": json.dumps([]),
    "dumps_object": json.dumps({"k": "v"}),
    "large_string_in_dict": {"timestamp": 1.0, "response": "A" * 10000},
    "unicode": {"timestamp": 1.0, "response": json.dumps({"t": "héllo 🌍"})},
    "nested": {"timestamp": 2.0, "response": json.dumps({"a": {"b": [1, 2, 3]}})},
}


def test_value_fidelity_snapshot(tmp_path):
    c = DiskCache(disk_cache_dir=str(tmp_path / "fidelity"))
    observed = {}
    for name, value in _FIDELITY_MATRIX.items():
        c.set_cache(f"set::{name}", value)
        got = c.get_cache(f"set::{name}")
        observed[name] = {"type": type(got).__name__, "value": got}

    blob = json.dumps(observed, sort_keys=True, ensure_ascii=True, default=repr)

    out = os.environ.get("DC_VERIFY_SNAPSHOT")
    if out:
        with open(out, "w") as f:
            f.write(blob)

    # Spot assertions so the snapshot is also a real test, not just a recorder.
    assert observed["empty_string"]["value"] is None
    assert observed["int_zero"]["value"] is None
    assert observed["hot_shape"]["value"] == _FIDELITY_MATRIX["hot_shape"]
    assert observed["json_object_string"]["value"] == {"a": 1}
    assert observed["dumps_false"]["value"] is False

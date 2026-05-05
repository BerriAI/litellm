import os
import sys
import tempfile

import pytest

pytest.importorskip("diskcache")

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.caching.disk_cache import DiskCache


@pytest.fixture
def tmp_cache():
    with tempfile.TemporaryDirectory() as d:
        yield DiskCache(disk_cache_dir=d)


def test_disk_cache_uses_jsondisk(tmp_cache):
    """Confirm DiskCache constructs the underlying cache with JSONDisk to
    avoid pickle deserialization (CVE-2025-69872)."""
    import diskcache as dc

    assert tmp_cache.disk_cache.disk.__class__ is dc.JSONDisk


def test_disk_cache_dict_round_trip(tmp_cache):
    payload = {"response": "hello", "timestamp": 1.0}
    tmp_cache.set_cache("k", payload)
    assert tmp_cache.get_cache("k") == payload


def test_disk_cache_string_round_trip_decodes_json(tmp_cache):
    """get_cache decodes JSON strings transparently for callers that stored
    pre-serialized payloads."""
    tmp_cache.set_cache("k", '{"hello": "world"}')
    assert tmp_cache.get_cache("k") == {"hello": "world"}


def test_disk_cache_string_round_trip_returns_raw_when_not_json(tmp_cache):
    tmp_cache.set_cache("k", "not-json")
    assert tmp_cache.get_cache("k") == "not-json"


def test_disk_cache_increment(tmp_cache):
    tmp_cache.set_cache("counter", 42)
    assert tmp_cache.increment_cache("counter", 8) == 50
    assert tmp_cache.get_cache("counter") == 50


def test_disk_cache_ttl_expiry(tmp_cache):
    import time

    tmp_cache.set_cache("k", "v", ttl=1)
    assert tmp_cache.get_cache("k") == "v"
    time.sleep(1.1)
    assert tmp_cache.get_cache("k") is None


def test_disk_cache_delete_and_flush(tmp_cache):
    tmp_cache.set_cache("a", 1)
    tmp_cache.set_cache("b", 2)
    tmp_cache.delete_cache("a")
    assert tmp_cache.get_cache("a") is None
    assert tmp_cache.get_cache("b") == 2
    tmp_cache.flush_cache()
    assert tmp_cache.get_cache("b") is None


def test_disk_cache_missing_key_returns_none(tmp_cache):
    assert tmp_cache.get_cache("never-set") is None

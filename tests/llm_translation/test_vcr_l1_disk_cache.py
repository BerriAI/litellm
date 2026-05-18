"""Unit tests for the local-disk L1 cassette cache.

The L1 layer is meant to absorb the bulk of cassette reads so the
remote Redis/S3 backend only sees first-time misses. CircleCI's
``restore_cache`` / ``save_cache`` mounts the directory across runs,
which is what makes that cheap. These tests verify the read-through,
write-through, TTL, and remote-failure semantics in isolation.
"""

from __future__ import annotations

import os
import sys
import time

import pytest
from vcr.request import Request
from vcr.serializers import yamlserializer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_redis_persister import (  # noqa: E402
    CASSETTE_TTL_SECONDS,
    _LocalDiskL1Cache,
    make_persister,
)


class _RecordingBackend:
    """Minimal in-memory backend that counts get/set calls.

    Using this instead of fakeredis for the L1 tests so we can assert
    *exactly* how many times the L1 layer fell through to the remote.
    """

    name = "test"
    transient_error_types = (RuntimeError,)

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.gets = 0
        self.sets = 0

    def get(self, key):
        self.gets += 1
        return self.store.get(key)

    def set(self, key, payload, ttl_seconds):
        self.sets += 1
        self.store[key] = payload


def _sample_cassette() -> dict:
    request = Request(
        method="POST",
        uri="https://api.anthropic.com/v1/messages",
        body=b'{"model":"claude","messages":[]}',
        headers={"content-type": "application/json"},
    )
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"content-type": ["application/json"]},
        "body": {"string": b'{"id":"msg_1"}'},
    }
    return {"requests": [request], "responses": [response]}


def test_l1_get_returns_payload_after_set(tmp_path):
    inner = _RecordingBackend()
    l1 = _LocalDiskL1Cache(str(tmp_path), inner)
    l1.set("k", b"payload", CASSETTE_TTL_SECONDS)
    assert l1.get("k") == b"payload"


def test_l1_hit_does_not_call_through_to_remote(tmp_path):
    inner = _RecordingBackend()
    l1 = _LocalDiskL1Cache(str(tmp_path), inner)
    l1.set("k", b"payload", CASSETTE_TTL_SECONDS)
    inner.gets = 0
    for _ in range(5):
        assert l1.get("k") == b"payload"
    assert (
        inner.gets == 0
    ), "L1 must absorb repeated reads — that's the whole point of the layer"


def test_l1_miss_falls_through_and_writes_back(tmp_path):
    inner = _RecordingBackend()
    inner.store["k"] = b"remote_payload"
    l1 = _LocalDiskL1Cache(str(tmp_path), inner)

    assert l1.get("k") == b"remote_payload"
    assert inner.gets == 1
    # The first hit must have populated the local copy so the next
    # get bypasses the remote.
    assert l1.get("k") == b"remote_payload"
    assert inner.gets == 1


def test_l1_miss_when_remote_returns_none(tmp_path):
    inner = _RecordingBackend()
    l1 = _LocalDiskL1Cache(str(tmp_path), inner)
    assert l1.get("never-recorded") is None


def test_l1_treats_stale_entry_as_miss(tmp_path):
    inner = _RecordingBackend()
    l1 = _LocalDiskL1Cache(str(tmp_path), inner, ttl_seconds=1)
    l1.set("k", b"v1", ttl_seconds=1)
    inner.store["k"] = b"v2"
    inner.gets = 0

    # Force the on-disk file to look older than the TTL.
    path = l1._path_for("k")
    os.utime(path, (time.time() - 5, time.time() - 5))

    # Stale → fall through → fetch the (newer) remote value and
    # refresh the local copy.
    assert l1.get("k") == b"v2"
    assert inner.gets == 1


def test_l1_set_writes_through_even_when_remote_succeeds(tmp_path):
    inner = _RecordingBackend()
    l1 = _LocalDiskL1Cache(str(tmp_path), inner)
    l1.set("k", b"v", CASSETTE_TTL_SECONDS)
    assert inner.store["k"] == b"v"
    assert os.path.exists(l1._path_for("k"))


def test_l1_writes_local_copy_even_when_remote_set_fails(tmp_path):
    """If the remote backend fails, the L1 still has the bytes for the
    rest of the test session — better than nothing for parallel xdist
    workers running in the same process."""

    class _Failing(_RecordingBackend):
        def set(self, key, payload, ttl_seconds):
            raise RuntimeError("simulated remote outage")

    inner = _Failing()
    l1 = _LocalDiskL1Cache(str(tmp_path), inner)
    with pytest.raises(RuntimeError):
        l1.set("k", b"payload", CASSETTE_TTL_SECONDS)
    # Despite the remote failure, the local file got written.
    assert l1.get("k") == b"payload"


def test_l1_inherits_inner_transient_error_types(tmp_path):
    inner = _RecordingBackend()
    l1 = _LocalDiskL1Cache(str(tmp_path), inner)
    assert RuntimeError in l1.transient_error_types
    assert OSError in l1.transient_error_types


def test_l1_path_layout_is_sharded(tmp_path):
    """Two cassette keys must end up in different directories so a
    single restored CI cache directory doesn't degrade to a flat
    bucket of thousands of files."""
    inner = _RecordingBackend()
    l1 = _LocalDiskL1Cache(str(tmp_path), inner)
    p1 = l1._path_for("a-key")
    p2 = l1._path_for("z-key")
    assert os.path.dirname(p1) != tmp_path
    assert os.path.dirname(p2) != tmp_path
    # Both files live under the base, but the shard prefix differs
    # for nearly all distinct inputs (sha256 of "a-key" vs "z-key").
    assert os.path.dirname(p1) != os.path.dirname(p2)


def test_persister_uses_l1_when_env_var_set(monkeypatch, tmp_path):
    """End-to-end: ``CASSETTE_LOCAL_CACHE_DIR=...`` activates the L1
    layer and a re-load on the same process never touches the inner
    backend after the first miss."""
    inner = _RecordingBackend()
    monkeypatch.setenv("CASSETTE_LOCAL_CACHE_DIR", str(tmp_path / "vcr-l1"))
    persister = make_persister(backend=inner)
    cassette_id = "tests/llm_translation/test_x/test_l1_e2e"
    persister.save_cassette(cassette_id, _sample_cassette(), yamlserializer)

    inner.gets = 0
    for _ in range(3):
        persister.load_cassette(cassette_id, yamlserializer)
    assert inner.gets == 0


def test_persister_without_env_var_does_not_create_l1(monkeypatch, tmp_path):
    """When ``CASSETTE_LOCAL_CACHE_DIR`` is unset, every load goes to
    the inner backend. Avoids surprising local-dev runs that would
    otherwise create a hidden cache directory in the cwd."""
    inner = _RecordingBackend()
    monkeypatch.delenv("CASSETTE_LOCAL_CACHE_DIR", raising=False)
    persister = make_persister(backend=inner)
    cassette_id = "tests/llm_translation/test_x/test_no_l1"
    persister.save_cassette(cassette_id, _sample_cassette(), yamlserializer)

    inner.gets = 0
    for _ in range(3):
        persister.load_cassette(cassette_id, yamlserializer)
    assert inner.gets == 3

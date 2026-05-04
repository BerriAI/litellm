"""Unit tests for the Redis cassette store.

We test against ``fakeredis`` so the suite stays hermetic.
"""

from __future__ import annotations

import os
import sys

import fakeredis
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from tests.e2e_cassette_proxy.redis_store import (  # noqa: E402
    DEFAULT_TTL_SECONDS,
    CachedResponse,
    RedisCassetteStore,
)


def _store(max_payload_bytes=None):
    fake = fakeredis.FakeStrictRedis()
    if max_payload_bytes is None:
        return fake, RedisCassetteStore(client=fake)
    return fake, RedisCassetteStore(client=fake, max_payload_bytes=max_payload_bytes)


def _sample_response(body: bytes = b'{"id":"msg_1"}'):
    return CachedResponse(
        status_code=200,
        headers=(("content-type", "application/json"),),
        body=body,
        reason="OK",
    )


def test_should_round_trip_a_set_then_get():
    _, store = _store()
    store.set("k", _sample_response())
    got = store.get("k")
    assert got is not None
    assert got.status_code == 200
    assert got.body == b'{"id":"msg_1"}'
    assert ("content-type", "application/json") in got.headers


def test_should_return_none_on_miss():
    _, store = _store()
    assert store.get("never-set") is None


def test_should_apply_default_ttl_on_set():
    fake, store = _store()
    store.set("k", _sample_response())
    ttl = fake.ttl("k")
    # fakeredis returns a float; allow a tiny window for clock skew.
    assert DEFAULT_TTL_SECONDS - 5 <= ttl <= DEFAULT_TTL_SECONDS


def test_should_round_trip_binary_response_bodies():
    _, store = _store()
    payload = bytes(range(256))
    store.set("k", _sample_response(body=payload))
    got = store.get("k")
    assert got is not None
    assert got.body == payload


def test_should_skip_oversize_payloads_silently():
    fake, store = _store(max_payload_bytes=128)
    huge = b"x" * 10_000
    persisted = store.set("k", _sample_response(body=huge))
    assert persisted is False
    assert fake.exists("k") == 0


def test_should_persist_payloads_at_the_size_threshold():
    fake, store = _store(max_payload_bytes=10_000)
    fits = b"x" * 100
    persisted = store.set("k", _sample_response(body=fits))
    assert persisted is True
    assert fake.exists("k") == 1


def test_should_treat_corrupt_blob_as_miss_and_evict_it():
    fake, store = _store()
    fake.set("corrupt-key", b"\x00\x01not a valid blob\xff")
    assert store.get("corrupt-key") is None
    assert fake.exists("corrupt-key") == 0


def test_should_return_none_when_get_raises():
    class _ExplodingClient:
        def get(self, key):
            raise ConnectionError("redis is down")

    store = RedisCassetteStore(client=_ExplodingClient())
    assert store.get("k") is None


def test_should_return_false_when_set_raises():
    class _ExplodingClient:
        def set(self, key, value, ex=None):  # noqa: ARG002
            raise ConnectionError("redis is down")

    store = RedisCassetteStore(client=_ExplodingClient())
    assert store.set("k", _sample_response()) is False

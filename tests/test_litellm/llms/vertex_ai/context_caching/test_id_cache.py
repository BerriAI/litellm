import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.vertex_ai.context_caching import id_cache
from litellm.llms.vertex_ai.context_caching.id_cache import (
    SAFETY_MARGIN_SECONDS,
    ResolvedCacheId,
    _rfc3339_to_epoch,
    expire_time_to_ttl,
    lookup_cache_id,
    make_cache_id_key,
    store_cache_id,
)

FUTURE = "2099-01-01T00:00:00Z"
FUTURE_EPOCH = _rfc3339_to_epoch(FUTURE)


@pytest.fixture
def enabled():
    """Enable the feature and start from an empty cache; restore afterwards."""
    prev = litellm.enable_vertex_context_cache_id_caching
    litellm.enable_vertex_context_cache_id_caching = True
    id_cache._EXPLICIT_CACHE_ID_CACHE.flush_cache()
    try:
        yield
    finally:
        litellm.enable_vertex_context_cache_id_caching = prev
        id_cache._EXPLICIT_CACHE_ID_CACHE.flush_cache()


def _key(**overrides):
    base = dict(
        content_key="hash",
        custom_llm_provider="vertex_ai",
        vertex_project="proj",
        vertex_location="us-central1",
        api_base=None,
        api_key="k",
    )
    base.update(overrides)
    return make_cache_id_key(**base)


# --- ttl derivation (spec: entry lifetime bounded by expireTime) ---

def test_ttl_uses_expiretime_minus_margin():
    # expireTime 3600s out -> ttl is 3600 - margin, measured against expireTime, not insertion
    ttl = expire_time_to_ttl(FUTURE, now=FUTURE_EPOCH - 3600)
    assert ttl == pytest.approx(3600 - SAFETY_MARGIN_SECONDS)


def test_ttl_scales_with_expiretime_not_constant():
    # regression guard: a near-expiry cache must get a SMALL ttl, not a fixed span.
    near = expire_time_to_ttl(FUTURE, now=FUTURE_EPOCH - 90)
    far = expire_time_to_ttl(FUTURE, now=FUTURE_EPOCH - 3600)
    assert near == pytest.approx(90 - SAFETY_MARGIN_SECONDS)
    assert far == pytest.approx(3600 - SAFETY_MARGIN_SECONDS)
    assert near < far  # fails if ttl is ever a hardcoded constant


@pytest.mark.parametrize(
    "expire_time",
    [None, "", "not-a-date", "2024-13-45T99:99:99Z"],
)
def test_ttl_none_when_unusable(expire_time):
    assert expire_time_to_ttl(expire_time, now=0) is None


def test_ttl_none_when_within_margin():
    # expiry inside the SAFETY_MARGIN_SECONDS window -> not worth caching
    assert expire_time_to_ttl(FUTURE, now=FUTURE_EPOCH - (SAFETY_MARGIN_SECONDS - 1)) is None


@pytest.mark.parametrize(
    "ts",
    [
        "2024-10-02T15:01:23Z",
        "2024-10-02T15:01:23.045Z",
        "2024-10-02T15:01:23.045123456Z",  # nanoseconds (Vertex form)
        "2024-10-02T15:01:23+00:00",
        "2024-10-02T15:01:23+0000",
    ],
)
def test_rfc3339_parses_provider_shapes(ts):
    assert _rfc3339_to_epoch(ts) is not None


# --- flag gating (spec: off by default) ---

def test_disabled_lookup_and_store_are_noops():
    prev = litellm.enable_vertex_context_cache_id_caching
    litellm.enable_vertex_context_cache_id_caching = False
    id_cache._EXPLICIT_CACHE_ID_CACHE.flush_cache()
    try:
        key = _key()
        store_cache_id(key, "cachedContents/1", FUTURE)
        assert lookup_cache_id(key) is None
        assert id_cache._EXPLICIT_CACHE_ID_CACHE.get_cache(key) is None
    finally:
        litellm.enable_vertex_context_cache_id_caching = prev


def test_flag_defaults_off():
    # guard: nobody flips the default to on without an explicit decision
    fresh = litellm.__dict__.get("enable_vertex_context_cache_id_caching")
    assert fresh is False


# --- round trip + expiry (spec: cache the id; never serve expired) ---

def test_store_then_lookup_round_trip(enabled):
    key = _key()
    store_cache_id(key, "projects/p/locations/l/cachedContents/9", FUTURE)
    assert lookup_cache_id(key) == "projects/p/locations/l/cachedContents/9"


def test_numeric_gemini_id_round_trips_as_string(enabled):
    key = _key(custom_llm_provider="gemini")
    store_cache_id(key, "7096536676058529792", FUTURE)
    got = lookup_cache_id(key)
    assert got == "7096536676058529792"
    assert isinstance(got, str)  # fails without the str() guard (json.loads -> int)


def test_expired_entry_is_a_miss(enabled):
    key = _key()
    # already-expired entry: lookup must treat it as absent
    id_cache._EXPLICIT_CACHE_ID_CACHE.set_cache(key, "cachedContents/1", ttl=-1)
    assert lookup_cache_id(key) is None


def test_within_margin_expiry_is_not_stored(enabled):
    key = _key()
    store_cache_id(key, "cachedContents/1", FUTURE, now=FUTURE_EPOCH - (SAFETY_MARGIN_SECONDS - 1))
    assert lookup_cache_id(key) is None


# --- key scoping (spec: scoped to endpoint/tenant) ---

def test_same_inputs_same_key():
    assert _key() == _key()


@pytest.mark.parametrize(
    "override",
    [
        {"custom_llm_provider": "gemini"},
        {"vertex_project": "other"},
        {"vertex_location": "europe-west1"},
        {"api_base": "https://passthrough.example"},
        {"api_key": "different"},
        {"content_key": "other-hash"},
    ],
)
def test_distinct_scope_distinct_key(override):
    assert _key() != _key(**override)


def test_cross_tenant_entry_is_not_served(enabled):
    # project A stores its id; a byte-identical request under project B must miss
    key_a = _key(vertex_project="A")
    key_b = _key(vertex_project="B")
    store_cache_id(key_a, "projects/A/.../cachedContents/1", FUTURE)
    assert lookup_cache_id(key_b) is None
    assert lookup_cache_id(key_a) == "projects/A/.../cachedContents/1"


def test_key_does_not_leak_raw_api_key():
    key = _key(api_key="super-secret-token")
    assert "super-secret-token" not in key


# --- bounded (spec: bounded) ---

def test_cache_is_bounded(enabled):
    cap = id_cache._MAX_ENTRIES
    for i in range(cap * 2):
        store_cache_id(_key(content_key=f"h{i}"), f"cachedContents/{i}", FUTURE)
    assert len(id_cache._EXPLICIT_CACHE_ID_CACHE.cache_dict) <= cap

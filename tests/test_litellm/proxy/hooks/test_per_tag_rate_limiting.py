"""
Unit tests for per-tag rate limiting on a single key (LIT-3147).

Behavioural test matrix
=======================

| Scenario | Tags on request | tag_rpm_limit on key | key rpm_limit | Expected outcome          |
|----------|-----------------|----------------------|---------------|---------------------------|
| 1        | ["cell-1"]      | {"cell-1": 2}        | 10            | allows ≤2 RPM for cell-1  |
| 2        | ["cell-1"]      | {"cell-1": 2}        | 10            | raises 429 on 3rd request |
| 3        | ["cell-2"]      | {"cell-1": 2}        | 10            | falls back to key limit   |
| 4        | []              | {"cell-1": 2}        | 10            | falls back to key limit   |
| 5        | ["cell-1"]      | {}                   | 2             | uses only key-level limit |
| 6        | ["cell-1","c2"] | {"cell-1":1,"c2":1}  | 10            | enforces each tag limit   |
| 7        | ["cell-1"]      | {"cell-1": 100}      | 2             | tag limit ≠ key limit     |
| 8        | none            | {"cell-1": 1}        | 10            | no tag → key-level only   |
"""

import sys
from datetime import datetime
from typing import List, Optional
import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.proxy.auth.auth_utils import get_key_tag_rpm_limit, get_key_tag_tpm_limit


def _make_handler() -> _PROXY_MaxParallelRequestsHandler:
    local_cache = DualCache()
    return _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )


def _make_user_api_key_dict(
    api_key: str,
    rpm_limit: Optional[int] = None,
    tpm_limit: Optional[int] = None,
    tag_rpm_limit: Optional[dict] = None,
    tag_tpm_limit: Optional[dict] = None,
) -> UserAPIKeyAuth:
    metadata: dict = {}
    if tag_rpm_limit:
        metadata["tag_rpm_limit"] = tag_rpm_limit
    if tag_tpm_limit:
        metadata["tag_tpm_limit"] = tag_tpm_limit
    return UserAPIKeyAuth(
        api_key=api_key,
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        metadata=metadata,
    )


def _make_data(tags: Optional[List[str]] = None) -> dict:
    return {"metadata": {"tags": tags or []}}


# ---------------------------------------------------------------------------
# Helper tests for auth_utils helpers
# ---------------------------------------------------------------------------


def test_get_key_tag_rpm_limit_found():
    """get_key_tag_rpm_limit returns the configured value when tag exists."""
    key_dict = _make_user_api_key_dict("sk-1", tag_rpm_limit={"cell-1": 50})
    assert get_key_tag_rpm_limit(key_dict, "cell-1") == 50


def test_get_key_tag_rpm_limit_missing_tag():
    """get_key_tag_rpm_limit returns None when tag is not in the map."""
    key_dict = _make_user_api_key_dict("sk-1", tag_rpm_limit={"cell-1": 50})
    assert get_key_tag_rpm_limit(key_dict, "cell-2") is None


def test_get_key_tag_rpm_limit_no_metadata():
    """get_key_tag_rpm_limit returns None when the key has no metadata."""
    key_dict = UserAPIKeyAuth(api_key="sk-1", rpm_limit=10)
    assert get_key_tag_rpm_limit(key_dict, "cell-1") is None


def test_get_key_tag_tpm_limit_found():
    key_dict = _make_user_api_key_dict("sk-1", tag_tpm_limit={"cell-1": 5000})
    assert get_key_tag_tpm_limit(key_dict, "cell-1") == 5000


def test_get_key_tag_tpm_limit_missing():
    key_dict = _make_user_api_key_dict("sk-1", tag_tpm_limit={"cell-1": 5000})
    assert get_key_tag_tpm_limit(key_dict, "other") is None


# ---------------------------------------------------------------------------
# Scenario 1 & 2 — tagged request, per-tag RPM limit enforced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_tag_rpm_allows_within_limit():
    """
    Scenario 1: Two requests with tag 'cell-1' and tag_rpm_limit={"cell-1": 2}
    should both succeed.
    """
    _api_key = hash_token("sk-tag-test-1")
    handler = _make_handler()
    user_api_key_dict = _make_user_api_key_dict(
        _api_key, rpm_limit=10, tag_rpm_limit={"cell-1": 2}
    )
    data = _make_data(tags=["cell-1"])

    # First request — should succeed
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    # Second request — should also succeed (rpm_limit=2, current=1)
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data=data,
        call_type="completion",
    )


@pytest.mark.asyncio
async def test_per_tag_rpm_blocks_on_third_request():
    """
    Scenario 2: Third request with tag 'cell-1' and tag_rpm_limit={"cell-1": 2}
    should raise 429.
    """
    _api_key = hash_token("sk-tag-test-2")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = _make_user_api_key_dict(
        _api_key, rpm_limit=100, tag_rpm_limit={"cell-1": 2}
    )
    data = _make_data(tags=["cell-1"])

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )

    # Third request: per-tag limit reached
    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data=data,
            call_type="completion",
        )
    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# Scenario 3 — tag not in map → falls back to key-level limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_untagged_request_uses_key_limit():
    """
    Scenario 3: Request with tag 'cell-2' not in tag_rpm_limit falls back to
    the key-level rpm_limit and should not be blocked by per-tag logic.
    """
    _api_key = hash_token("sk-tag-test-3")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = _make_user_api_key_dict(
        _api_key, rpm_limit=10, tag_rpm_limit={"cell-1": 1}
    )
    # tag "cell-2" is not in tag_rpm_limit, so per-tag check is skipped
    data = _make_data(tags=["cell-2"])

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )
    # Should succeed without hitting the cell-1 limit
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )


# ---------------------------------------------------------------------------
# Scenario 4 — no tags on request → key-level limit applies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tags_uses_key_limit_only():
    """
    Scenario 4: Request without tags and tag_rpm_limit configured; key rpm_limit
    (10) governs — per-tag counters are never touched.
    """
    _api_key = hash_token("sk-tag-test-4")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = _make_user_api_key_dict(
        _api_key, rpm_limit=10, tag_rpm_limit={"cell-1": 1}
    )
    data = _make_data(tags=[])  # empty tags

    # Should not raise even though cell-1 limit is 1
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )


# ---------------------------------------------------------------------------
# Scenario 5 — no tag_rpm_limit map → key-level limit governs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tag_rpm_limit_map_uses_key_limit():
    """
    Scenario 5: Key has no tag_rpm_limit but has rpm_limit=2; tagged request
    should only be blocked by the key-level counter.
    """
    _api_key = hash_token("sk-tag-test-5")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = _make_user_api_key_dict(_api_key, rpm_limit=2)
    data = _make_data(tags=["cell-1"])

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data=data,
            call_type="completion",
        )
    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# Scenario 6 — multiple tags, each with independent limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_tags_independent_limits():
    """
    Scenario 6: Request tagged with both 'cell-1' and 'cell-2'; each has its own
    limit of 1 RPM. First request should pass; second should fail because both
    tag counters are at their limit.
    """
    _api_key = hash_token("sk-tag-test-6")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = _make_user_api_key_dict(
        _api_key,
        rpm_limit=100,
        tag_rpm_limit={"cell-1": 1, "cell-2": 1},
    )
    data = _make_data(tags=["cell-1", "cell-2"])

    # First request succeeds (counters: cell-1=1, cell-2=1)
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )

    # Second request: both cell-1 and cell-2 are at their limit → 429
    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data=data,
            call_type="completion",
        )
    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# Scenario 7 — tag limit higher than key limit → key limit governs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_limit_higher_than_key_limit():
    """
    Scenario 7: tag_rpm_limit for 'cell-1' is 100 but key rpm_limit is 2.
    Key-level limit should still block on the 3rd request.
    """
    _api_key = hash_token("sk-tag-test-7")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = _make_user_api_key_dict(
        _api_key, rpm_limit=2, tag_rpm_limit={"cell-1": 100}
    )
    data = _make_data(tags=["cell-1"])

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )

    # Key-level counter exhausted
    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data=data,
            call_type="completion",
        )
    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# Scenario 8 — metadata absent entirely → key-level limit applies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_metadata_in_data_uses_key_limit():
    """
    Scenario 8: data dict has no 'metadata' key at all — should behave like no tags.
    """
    _api_key = hash_token("sk-tag-test-8")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = _make_user_api_key_dict(
        _api_key, rpm_limit=10, tag_rpm_limit={"cell-1": 1}
    )
    data: dict = {}  # no metadata key

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )


# ---------------------------------------------------------------------------
# TPM-based per-tag limit test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_tag_tpm_limit_blocks_on_success():
    """
    After a successful completion that consumes tokens, the TPM counter for the
    tag should be updated and eventually block further requests.
    """
    _api_key = hash_token("sk-tag-tpm-test")
    local_cache = DualCache()
    usage_cache = InternalUsageCache(local_cache)
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=usage_cache)
    user_api_key_dict = _make_user_api_key_dict(
        _api_key,
        tpm_limit=sys.maxsize,
        tag_tpm_limit={"cell-1": 10},
    )
    data = _make_data(tags=["cell-1"])

    # Pre-seed the per-tag TPM counter as if 10 tokens were already used
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    tag_key = f"{_api_key}::tag::cell-1::{precise_minute}::request_count"
    await local_cache.async_set_cache(
        key=tag_key,
        value={"current_requests": 0, "current_tpm": 10, "current_rpm": 0},
    )

    # Next request should be blocked because tpm 10 >= limit 10
    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data=data,
            call_type="completion",
        )
    assert exc_info.value.status_code == 429

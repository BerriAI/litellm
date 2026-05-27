"""Tests for LIT-3266: skip_batch_input_file_retrieval opt-out flag.

Covers:
- Default behaviour: input-file retrieval still runs (no regression).
- Global flag (`litellm.skip_batch_input_file_retrieval=True`): retrieval is
  skipped, rate-limit counters are NOT touched, no exception is raised.
- Per-deployment flag (`model_info.skip_batch_input_file_retrieval=True`):
  same skip behaviour, on both raw-dict and pydantic-shaped deployments.
- Per-deployment flag set to False does NOT skip.
- Router lookup error: falls back to global flag, never crashes.
- Wrong call_type still short-circuits before the skip check.
- Defensive helper handles missing/None model in data.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://nope")

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.batch_rate_limiter import (
    _PROXY_BatchRateLimiter,
    BatchFileUsage,
)


class _StubRateLimiter:
    """Minimal stand-in for the parallel-rate-limiter v3."""

    window_size = 60

    def _create_rate_limit_descriptors(self, **kwargs):
        return []

    async def atomic_check_and_increment_by_n(self, **kwargs):
        return {"overall_code": "OK", "statuses": []}


def _make_limiter():
    return _PROXY_BatchRateLimiter(
        internal_usage_cache=SimpleNamespace(),
        parallel_request_limiter=_StubRateLimiter(),
    )


def _make_data(model="custom-vllm-model"):
    return {
        "input_file_id": "file-abc123",
        "custom_llm_provider": "openai",
        "model": model,
    }


def _user():
    return UserAPIKeyAuth(api_key="sk-test", user_id="u")


def _install_router(deployment):
    """Install a fake llm_router on proxy_server; return cleanup callable."""
    fake_router = SimpleNamespace(
        model_list=[deployment] if deployment else [],
        get_deployment_by_model_group_name=(
            lambda model_group_name=None, **kw: deployment
        ),
    )
    import litellm.proxy.proxy_server as ps

    prev = getattr(ps, "llm_router", None)
    ps.llm_router = fake_router

    def restore():
        ps.llm_router = prev

    return restore


@pytest.fixture(autouse=True)
def _reset_global_flag():
    prev = getattr(litellm, "skip_batch_input_file_retrieval", False)
    litellm.skip_batch_input_file_retrieval = False
    yield
    litellm.skip_batch_input_file_retrieval = prev


@pytest.mark.asyncio
async def test_default_runs_input_file_retrieval():
    """With neither flag set, the existing fetch happens (no regression)."""
    limiter = _make_limiter()
    restore = _install_router(None)
    try:
        called = {"n": 0}

        async def fake_count(
            file_id, custom_llm_provider="openai", user_api_key_dict=None
        ):
            called["n"] += 1
            return BatchFileUsage(total_tokens=0, request_count=0)

        with patch.object(limiter, "count_input_file_usage", side_effect=fake_count):
            await limiter.async_pre_call_hook(
                user_api_key_dict=_user(),
                cache=None,
                data=_make_data(),
                call_type="acreate_batch",
            )
        assert called["n"] == 1
    finally:
        restore()


@pytest.mark.asyncio
async def test_global_flag_skips_retrieval():
    """Setting litellm.skip_batch_input_file_retrieval=True skips the fetch."""
    litellm.skip_batch_input_file_retrieval = True
    limiter = _make_limiter()
    restore = _install_router(None)
    try:
        with patch.object(limiter, "count_input_file_usage") as m:
            await limiter.async_pre_call_hook(
                user_api_key_dict=_user(),
                cache=None,
                data=_make_data(),
                call_type="acreate_batch",
            )
        assert m.call_count == 0
    finally:
        restore()


@pytest.mark.asyncio
async def test_per_deployment_flag_dict_skips_retrieval():
    """Per-deployment flag on a raw-dict deployment skips the fetch."""
    limiter = _make_limiter()
    deployment = {
        "model_name": "custom-vllm-model",
        "litellm_params": {"model": "openai/custom-vllm-model"},
        "model_info": {
            "id": "dep-1",
            "skip_batch_input_file_retrieval": True,
        },
    }
    restore = _install_router(deployment)
    try:
        with patch.object(limiter, "count_input_file_usage") as m:
            await limiter.async_pre_call_hook(
                user_api_key_dict=_user(),
                cache=None,
                data=_make_data(),
                call_type="acreate_batch",
            )
        assert m.call_count == 0
    finally:
        restore()


@pytest.mark.asyncio
async def test_per_deployment_flag_pydantic_deployment_skips_retrieval():
    """Per-deployment flag on a pydantic-shaped Deployment also works."""
    limiter = _make_limiter()

    class _ModelInfo:
        def __init__(self, **kw):
            self._kw = kw

        def model_dump(self):
            return dict(self._kw)

    class _Deployment:
        model_name = "custom-vllm-model"
        model_info = _ModelInfo(
            id="dep-1",
            skip_batch_input_file_retrieval=True,
        )

    restore = _install_router(_Deployment())
    try:
        with patch.object(limiter, "count_input_file_usage") as m:
            await limiter.async_pre_call_hook(
                user_api_key_dict=_user(),
                cache=None,
                data=_make_data(),
                call_type="acreate_batch",
            )
        assert m.call_count == 0
    finally:
        restore()


@pytest.mark.asyncio
async def test_per_deployment_flag_false_still_runs_retrieval():
    """Per-deployment flag set to False does NOT skip the fetch."""
    limiter = _make_limiter()
    deployment = {
        "model_name": "custom-vllm-model",
        "model_info": {
            "id": "dep-1",
            "skip_batch_input_file_retrieval": False,
        },
    }
    restore = _install_router(deployment)
    try:
        called = {"n": 0}

        async def fake_count(
            file_id, custom_llm_provider="openai", user_api_key_dict=None
        ):
            called["n"] += 1
            return BatchFileUsage(total_tokens=0, request_count=0)

        with patch.object(limiter, "count_input_file_usage", side_effect=fake_count):
            await limiter.async_pre_call_hook(
                user_api_key_dict=_user(),
                cache=None,
                data=_make_data(),
                call_type="acreate_batch",
            )
        assert called["n"] == 1
    finally:
        restore()


@pytest.mark.asyncio
async def test_router_lookup_error_falls_back_to_global_flag():
    """Deployment lookup raises: global flag still wins, no crash."""
    limiter = _make_limiter()
    litellm.skip_batch_input_file_retrieval = True

    class _RaisingRouter:
        model_list = []

        def get_deployment_by_model_group_name(self, model_group_name=None, **kw):
            raise RuntimeError("boom")

    import litellm.proxy.proxy_server as ps

    prev = getattr(ps, "llm_router", None)
    ps.llm_router = _RaisingRouter()
    try:
        with patch.object(limiter, "count_input_file_usage") as m:
            await limiter.async_pre_call_hook(
                user_api_key_dict=_user(),
                cache=None,
                data=_make_data(),
                call_type="acreate_batch",
            )
        # Lookup error is swallowed, then global flag takes effect -> skip.
        assert m.call_count == 0
    finally:
        ps.llm_router = prev


@pytest.mark.asyncio
async def test_non_acreate_batch_call_type_short_circuits():
    """Existing behaviour: any non-acreate_batch call_type returns immediately."""
    limiter = _make_limiter()
    # Set the flag too - proves we never even reach the skip check on
    # non-batch call_types (the call_type guard runs first).
    litellm.skip_batch_input_file_retrieval = True
    restore = _install_router(None)
    try:
        with patch.object(limiter, "count_input_file_usage") as m:
            await limiter.async_pre_call_hook(
                user_api_key_dict=_user(),
                cache=None,
                data=_make_data(),
                call_type="acompletion",
            )
        assert m.call_count == 0
    finally:
        restore()


def test_helper_returns_false_when_data_has_no_model():
    """Defensive: empty/missing model in data falls through to global flag."""
    limiter = _make_limiter()
    restore = _install_router(None)
    try:
        assert limiter._should_skip_input_file_retrieval(data={}) is False
        assert limiter._should_skip_input_file_retrieval(data={"model": None}) is False
        litellm.skip_batch_input_file_retrieval = True
        assert limiter._should_skip_input_file_retrieval(data={}) is True
    finally:
        restore()

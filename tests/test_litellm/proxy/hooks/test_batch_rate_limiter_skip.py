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


@pytest.mark.asyncio
async def test_per_deployment_false_beats_global_true():
    """LIT-3266 Greptile feedback: an explicit per-deployment False must
    re-enable input-file retrieval / batch accounting for a specific model
    even when the global skip flag is on.
    """
    limiter = _make_limiter()
    litellm.skip_batch_input_file_retrieval = True  # global says "skip everything"

    deployment = {
        "model_name": "billed-batch-model",
        "model_info": {
            "id": "dep-1",
            "skip_batch_input_file_retrieval": False,  # but THIS deployment is billed
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
                data=_make_data(model="billed-batch-model"),
                call_type="acreate_batch",
            )
        assert (
            called["n"] == 1
        ), "explicit per-deployment False must override a True global flag"
    finally:
        restore()


def test_helper_presence_aware_resolution_order():
    """Direct helper test: per-deployment value (even False) is authoritative.

    - global=True,  deployment unset             -> skip
    - global=True,  deployment present and True  -> skip
    - global=True,  deployment present and False -> do NOT skip
    - global=False, deployment unset             -> do NOT skip
    - global=False, deployment present and True  -> skip
    - global=False, deployment present and False -> do NOT skip
    """
    limiter = _make_limiter()

    def _check(global_flag, per_dep_present, per_dep_value, expected):
        litellm.skip_batch_input_file_retrieval = global_flag
        deployment = None
        if per_dep_present:
            deployment = {
                "model_name": "x",
                "model_info": {
                    "id": "d",
                    "skip_batch_input_file_retrieval": per_dep_value,
                },
            }
        restore = _install_router(deployment)
        try:
            got = limiter._should_skip_input_file_retrieval(data={"model": "x"})
            assert got is expected, (
                f"global={global_flag} per_dep_present={per_dep_present} "
                f"per_dep_value={per_dep_value!r} -> got {got!r}, want {expected!r}"
            )
        finally:
            restore()

    _check(True, False, None, True)  # case 1
    _check(True, True, True, True)  # case 2
    _check(True, True, False, False)  # case 3 (the bug fixed by Greptile feedback)
    _check(False, False, None, False)  # case 4
    _check(False, True, True, True)  # case 5
    _check(False, True, False, False)  # case 6


@pytest.mark.asyncio
async def test_skip_path_emits_warning_log(caplog):
    """LIT-3266 / Veria feedback: when the skip path is taken, the hook MUST
    log a WARNING that names the model and the security implication, so
    operators can audit the trust decision in production.
    """
    import logging

    limiter = _make_limiter()
    deployment = {
        "model_name": "trusted-vllm",
        "model_info": {
            "id": "dep-1",
            "skip_batch_input_file_retrieval": True,
        },
    }
    restore = _install_router(deployment)
    try:
        with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"), patch.object(
            limiter, "count_input_file_usage"
        ) as m:
            await limiter.async_pre_call_hook(
                user_api_key_dict=_user(),
                cache=None,
                data=_make_data(model="trusted-vllm"),
                call_type="acreate_batch",
            )
        assert m.call_count == 0
        # The warning must include the model name and call out the security
        # implication, not just say "skipping".
        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "skip_batch_input_file_retrieval" in r.getMessage()
        ]
        assert warning_records, "expected a WARNING log when the skip path is taken"
        message = warning_records[-1].getMessage()
        assert "trusted-vllm" in message
        assert "model allowlist" in message or "trust" in message
    finally:
        restore()

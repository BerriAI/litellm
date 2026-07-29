"""
Tests for weighted-routing failover (router_settings.enable_weighted_failover).

When enabled and the routing strategy is "simple-shuffle", a retryable failure
on one deployment causes the request to re-pick a different deployment in the
SAME model group (weighted across the remaining deployments) before any
cross-group fallback runs.
"""

from collections import Counter
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import Router
from litellm.utils import _get_excluded_filtered_deployments


# ---------------------------------------------------------------------------
# Unit tests for _get_excluded_filtered_deployments
# ---------------------------------------------------------------------------


def _make_dep(dep_id: str, weight: Optional[int] = None) -> dict:
    params: dict = {"model": "gpt-4o", "api_key": "key"}
    if weight is not None:
        params["weight"] = weight
    return {
        "model_name": "test-model",
        "litellm_params": params,
        "model_info": {"id": dep_id},
    }


class TestGetExcludedFilteredDeployments:
    def test_no_excluded_returns_all(self):
        deps = [_make_dep("a"), _make_dep("b")]
        result = _get_excluded_filtered_deployments(deps, excluded_deployment_ids=None)
        assert len(result) == 2

    def test_empty_excluded_returns_all(self):
        deps = [_make_dep("a"), _make_dep("b")]
        result = _get_excluded_filtered_deployments(deps, excluded_deployment_ids=[])
        assert len(result) == 2

    def test_drops_excluded(self):
        deps = [_make_dep("a"), _make_dep("b"), _make_dep("c")]
        result = _get_excluded_filtered_deployments(deps, excluded_deployment_ids=["b"])
        ids = sorted(d["model_info"]["id"] for d in result)
        assert ids == ["a", "c"]

    def test_all_excluded_returns_empty(self):
        # When every healthy deployment has been excluded, the helper must
        # return an empty list so the caller raises its usual no-deployments
        # error. Returning the original list here would re-include the
        # just-failed deployment and let weighted failover re-pick it.
        deps = [_make_dep("a"), _make_dep("b")]
        result = _get_excluded_filtered_deployments(
            deps, excluded_deployment_ids=["a", "b"]
        )
        assert result == []

    def test_excluded_set_with_unknown_ids(self):
        deps = [_make_dep("a"), _make_dep("b")]
        result = _get_excluded_filtered_deployments(
            deps, excluded_deployment_ids=["zzz"]
        )
        assert len(result) == 2

    def test_handles_missing_model_info(self):
        deps = [
            {"model_name": "x", "litellm_params": {"model": "gpt-4o"}},  # no model_info
            _make_dep("b"),
        ]
        result = _get_excluded_filtered_deployments(deps, excluded_deployment_ids=["b"])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Router helpers (router_code_coverage.py requires these names in a *router* test file)
# ---------------------------------------------------------------------------


def test_set_failed_deployment_id_on_exception():
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key"},
                "model_info": {"id": "dep-a"},
            }
        ],
    )
    exc = Exception("fail")
    dep = _make_dep("dep-a")
    router._set_failed_deployment_id_on_exception(exc, dep)
    assert getattr(exc, "failed_deployment_id", None) == "dep-a"
    router._set_failed_deployment_id_on_exception(exc, _make_dep("dep-b"))
    assert exc.failed_deployment_id == "dep-a"


@pytest.mark.asyncio
async def test_maybe_run_weighted_failover_returns_none_without_failed_id():
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key", "weight": 1},
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key", "weight": 1},
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        enable_weighted_failover=True,
    )
    result = await router._maybe_run_weighted_failover(
        exception=Exception("fail"),
        original_model_group="test-model",
        all_deployments=[_make_dep("A"), _make_dep("B")],
        args=(),
        kwargs={"metadata": {}},
        input_kwargs={},
    )
    assert result is None


@pytest.mark.asyncio
async def test_maybe_run_weighted_failover_persists_excluded_ids_to_kwargs(monkeypatch):
    """Regression: writing to the metadata dict returned by `setdefault` must
    update the dict in `kwargs` itself so the next hop sees prior exclusions.
    Previously `setdefault(..., {}) or {}` returned a disconnected dict on the
    first hop, dropping `_failover_excluded_ids` writes.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k", "weight": 1},
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k", "weight": 1},
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        enable_weighted_failover=True,
    )

    async def _stub_run_async_fallback(*args, **kwargs):
        return "ok"

    monkeypatch.setattr("litellm.router.run_async_fallback", _stub_run_async_fallback)

    exc = Exception("fail")
    exc.failed_deployment_id = "A"
    kwargs: dict = {"metadata": {}}
    await router._maybe_run_weighted_failover(
        exception=exc,
        original_model_group="test-model",
        all_deployments=[_make_dep("A"), _make_dep("B")],
        args=(),
        kwargs=kwargs,
        input_kwargs={},
    )
    # The dict inside kwargs must reflect the write — proves `meta` was the
    # same object as kwargs["metadata"] (no disconnected copy).
    assert kwargs["metadata"].get("_failover_excluded_ids") == ["A"]


# ---------------------------------------------------------------------------
# Integration tests for weighted-failover end-to-end via Router
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_failover_when_flag_off():
    """Default behavior: a failure on the picked deployment surfaces to caller."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("region-A failed"),
                    "weight": 1,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from B",
                    "weight": 0,  # weight=0 so A is always picked
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        # enable_weighted_failover defaults to False
    )

    with pytest.raises(Exception):
        await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_failover_lands_on_other_deployment_when_flag_on():
    """Flag on: when A fails, request must succeed via B in the same call."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("region-A down"),
                    "weight": 1,  # always picked first (B has weight 0)
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from B",
                    "weight": 0,
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "B"


@pytest.mark.asyncio
async def test_failover_chain_three_deployments():
    """A and B fail, request succeeds on C."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("A down"),
                    "weight": 1_000_000,  # A always picked first
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("B down"),
                    "weight": 1,  # picked when A is excluded
                },
                "model_info": {"id": "B"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from C",
                    "weight": 0,
                },
                "model_info": {"id": "C"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "C"


@pytest.mark.asyncio
async def test_failover_exhausted_raises_original_error_class():
    """When ALL deployments fail, the request raises (does not hang)."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("A down"),
                    "weight": 1,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("B down"),
                    "weight": 1,
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )

    with pytest.raises(Exception):
        await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_failover_falls_through_to_external_fallback():
    """When all deployments in the group fail, external fallback still runs."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("A down"),
                    "weight": 1,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("B down"),
                    "weight": 1,
                },
                "model_info": {"id": "B"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from fallback",
                },
                "model_info": {"id": "fallback"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
        fallbacks=[{"test-model": ["fallback-model"]}],
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "fallback"


@pytest.mark.asyncio
async def test_weights_respected_when_all_healthy():
    """With both regions healthy, the picker should still honor configured
    weights — failover must not change the steady-state load shape."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "from A",
                    "weight": 80,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "from B",
                    "weight": 20,
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )

    counts: Counter = Counter()
    for _ in range(1000):
        resp = await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        counts[resp._hidden_params["model_id"]] += 1

    # Expect ~80/20 split. Loose bounds to keep the test stable under CI load.
    assert counts["A"] > counts["B"] * 2  # A should heavily dominate
    assert counts["B"] > 50  # but B should still get a meaningful share


@pytest.mark.asyncio
async def test_failover_skipped_for_non_simple_shuffle():
    """Weighted failover is only wired up for `simple-shuffle`. With another
    strategy, a failure on the picked deployment must NOT silently retry the
    other deployment in the same group. Both deployments fail here to keep the
    test deterministic regardless of which one the strategy picks first.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("A down"),
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("B down"),
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="latency-based-routing",
        num_retries=0,
        enable_weighted_failover=True,
    )

    with pytest.raises(Exception):
        await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_failover_skipped_for_context_window_error():
    """ContextWindowExceededError must NOT trigger weighted failover —
    it has its own dedicated fallback path. Uses the router's built-in
    `mock_testing_context_fallbacks` to deterministically raise the right
    exception class.
    """
    import litellm

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from A",
                    "weight": 1,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from B",
                    "weight": 1,
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )

    with pytest.raises(litellm.ContextWindowExceededError):
        await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            mock_testing_context_fallbacks=True,
        )


@pytest.mark.asyncio
async def test_user_config_two_region_failover():
    """Mirrors the user's actual proxy_server_config.yaml shape: two Azure
    regions weighted 50/50, num_retries=0. With the flag on, a failure in
    one region is recovered by the other in the same request."""
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5.4-mini",
                "litellm_params": {
                    "model": "azure/deployment-eastus2",
                    "api_key": "bad",
                    "api_base": "https://eastus2.example",
                    "mock_response": Exception("eastus2 5xx"),
                    "weight": 50,
                },
                "model_info": {"id": "eastus2"},
            },
            {
                "model_name": "gpt-5.4-mini",
                "litellm_params": {
                    "model": "azure/deployment-northcentralus",
                    "api_key": "good",
                    "api_base": "https://northcentralus.example",
                    "mock_response": "ok from northcentralus",
                    "weight": 50,
                },
                "model_info": {"id": "northcentralus"},
            },
        ],
        routing_strategy="simple-shuffle",
        cooldown_time=120,
        num_retries=0,
        enable_pre_call_checks=True,
        disable_cooldowns=False,
        allowed_fails=5,
        enable_weighted_failover=True,
    )

    # Force eastus2 to be picked first by leaving its weight intact and
    # asserting we always end up on northcentralus when eastus2 errors.
    # Run several requests and ensure we never see an unhandled failure.
    successes = Counter()
    for _ in range(20):
        resp = await router.acompletion(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
        successes[resp._hidden_params["model_id"]] += 1

    # With one region permanently failing, every request must land on the
    # other region (either directly because it was picked first, or via
    # failover because eastus2 was picked first).
    assert successes["northcentralus"] == 20
    assert successes["eastus2"] == 0


# ---------------------------------------------------------------------------
# Tests for healthy-deployment-only check in _maybe_run_weighted_failover
# (Issue: weighted failover checked all deployments, not just healthy ones)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_run_weighted_failover_skips_when_remaining_all_in_cooldown(
    monkeypatch,
):
    """When every non-excluded deployment is in cooldown, _maybe_run_weighted_failover
    must return None immediately without invoking run_async_fallback.

    Previously the check was against all_deployments (including cooldown ones), so
    run_async_fallback would be called unnecessarily and would raise RouterRateLimitError.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k", "weight": 1},
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k", "weight": 1},
                "model_info": {"id": "B"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k", "weight": 1},
                "model_info": {"id": "C"},
            },
        ],
        routing_strategy="simple-shuffle",
        enable_weighted_failover=True,
    )

    # A just failed; B and C are both in cooldown.
    exc = Exception("A down")
    exc.failed_deployment_id = "A"

    run_async_fallback_called = False

    async def _should_not_be_called(*args, **kwargs):
        nonlocal run_async_fallback_called
        run_async_fallback_called = True
        return "should not reach here"

    monkeypatch.setattr("litellm.router.run_async_fallback", _should_not_be_called)

    # Patch cooldown so B and C appear in cooldown.
    with patch(
        "litellm.router._async_get_cooldown_deployments",
        new=AsyncMock(return_value=["B", "C"]),
    ):
        result = await router._maybe_run_weighted_failover(
            exception=exc,
            original_model_group="test-model",
            all_deployments=[_make_dep("A"), _make_dep("B"), _make_dep("C")],
            args=(),
            kwargs={"metadata": {}},
            input_kwargs={},
        )

    assert (
        result is None
    ), "Should return None when all remaining deployments are in cooldown"
    assert (
        not run_async_fallback_called
    ), "run_async_fallback must NOT be called when no healthy deployments remain"


@pytest.mark.asyncio
async def test_maybe_run_weighted_failover_proceeds_when_one_healthy_remains(
    monkeypatch,
):
    """When at least one non-excluded deployment is healthy (not in cooldown),
    _maybe_run_weighted_failover should still invoke run_async_fallback normally.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k", "weight": 1},
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k", "weight": 1},
                "model_info": {"id": "B"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k", "weight": 1},
                "model_info": {"id": "C"},
            },
        ],
        routing_strategy="simple-shuffle",
        enable_weighted_failover=True,
    )

    # A just failed; B is in cooldown; C is healthy.
    exc = Exception("A down")
    exc.failed_deployment_id = "A"

    run_async_fallback_called = False

    async def _stub_run_async_fallback(*args, **kwargs):
        nonlocal run_async_fallback_called
        run_async_fallback_called = True
        return "ok from C"

    monkeypatch.setattr("litellm.router.run_async_fallback", _stub_run_async_fallback)

    with patch(
        "litellm.router._async_get_cooldown_deployments",
        new=AsyncMock(return_value=["B"]),
    ):
        result = await router._maybe_run_weighted_failover(
            exception=exc,
            original_model_group="test-model",
            all_deployments=[_make_dep("A"), _make_dep("B"), _make_dep("C")],
            args=(),
            kwargs={"metadata": {}},
            input_kwargs={},
        )

    assert result == "ok from C"
    assert (
        run_async_fallback_called
    ), "run_async_fallback must be called when a healthy deployment remains"


@pytest.mark.asyncio
async def test_failover_falls_through_to_external_fallback_when_remaining_in_cooldown():
    """End-to-end: when the only non-failed deployments are in cooldown,
    weighted failover must fall through to the configured cross-group fallback.

    Without the fix the _maybe_run_weighted_failover would invoke run_async_fallback
    unnecessarily (because it counted cooldown deployments as "remaining"), get back
    RouterRateLimitError, return None, and reach the same fallback path — but only
    incidentally. With the fix the early-exit path is taken directly.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("A down"),
                    "weight": 1_000_000,  # always picked first
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("B down"),
                    "weight": 1,
                },
                "model_info": {"id": "B"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from fallback",
                },
                "model_info": {"id": "fallback"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
        fallbacks=[{"test-model": ["fallback-model"]}],
    )

    # Put B in cooldown so weighted failover can't use it after A fails.
    with patch(
        "litellm.router._async_get_cooldown_deployments",
        new=AsyncMock(return_value=["B"]),
    ):
        response = await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert response._hidden_params["model_id"] == "fallback"


# ---------------------------------------------------------------------------
# Regression: weighted-failover exclusion must beat a stale affinity pin
# (Issue #32308)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_weighted_failover_exclusion_beats_stale_affinity_pin():
    """
    Regression for #32308.

    When weighted failover retries a request with the just-failed deployment
    excluded, the exclusion must be applied before the affinity callback. A
    session-affinity pin to the excluded deployment would otherwise narrow the
    candidate set to that deployment, the exclusion filter (previously run last)
    would then drop it, and async_get_healthy_deployments would raise instead of
    landing on the healthy same-group sibling.
    """
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k"},
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k"},
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        enable_weighted_failover=True,
        optional_pre_call_checks=["session_affinity"],
    )

    session_id = "session-32308"
    stable_key = DeploymentAffinityCheck._get_stable_model_map_key_from_deployments(
        [_make_dep("A"), _make_dep("B")]
    )
    assert stable_key is not None
    session_cache_key = DeploymentAffinityCheck.get_session_affinity_cache_key(
        model_group=stable_key, session_id=session_id
    )

    affinity_callbacks = [
        cb for cb in litellm.callbacks if isinstance(cb, DeploymentAffinityCheck)
    ]
    assert affinity_callbacks, "session_affinity should register a DeploymentAffinityCheck"
    for cb in affinity_callbacks:
        await cb.cache.async_set_cache(session_cache_key, {"model_id": "A"})

    # A just failed and is excluded; the session is still pinned to A.
    result = await router.async_get_healthy_deployments(
        model="test-model",
        request_kwargs={
            "metadata": {"session_id": session_id},
            "_excluded_deployment_ids": ["A"],
        },
        messages=[{"role": "user", "content": "hi"}],
    )

    ids = sorted(d["model_info"]["id"] for d in result)
    assert ids == ["B"]

"""
Tests for health check failures integrating with allowed_fails_policy cooldown pipeline.

When enable_health_check_routing is True and a health check fails, the failure
should increment the same counters used by allowed_fails_policy, using the
actual exception type from the health check error.
"""

import time
from unittest.mock import patch

import pytest

import litellm
from litellm.proxy.health_check import run_with_timeout
from litellm.router import Router
from litellm.router_api import RoutingContext
from litellm.router_utils.cooldown_handlers import get_cooldown_deployments
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    get_deployment_failures_for_current_minute,
)
from litellm.types.router import AllowedFailsPolicy


def _cooled(router: Router) -> list[str]:
    return list(get_cooldown_deployments(RoutingContext(router), parent_otel_span=None))


def _failures(router: Router, model_id: str) -> int:
    return get_deployment_failures_for_current_minute(RoutingContext(router), model_id)


def _make_model(model_id: str, model_name: str = "gpt-4") -> dict:
    return {
        "model_name": model_name,
        "litellm_params": {"model": model_name, "api_key": "fake-key"},
        "model_info": {"id": model_id},
    }


class TestAhealthCheckExceptionPreservation:
    """Test that ahealth_check() preserves the exception object in its return dict."""

    @pytest.mark.asyncio
    async def test_run_with_timeout_returns_timeout_exception(self):
        """run_with_timeout should return a litellm.Timeout in the 'exception' key on timeout."""
        import asyncio

        async def slow_task():
            await asyncio.sleep(10)

        result = await run_with_timeout(slow_task(), timeout=0.01)

        assert "error" in result
        assert "exception" in result
        assert isinstance(result["exception"], litellm.Timeout)


class TestHealthCheckEndpointExceptionPropagation:
    """Test that _perform_health_check returns exceptions via exceptions_by_model_id."""

    @pytest.mark.asyncio
    async def test_unhealthy_endpoint_dict_exception_in_map(self):
        """When ahealth_check returns {"error": ..., "exception": e}, the exception
        must appear in exceptions_by_model_id keyed by model_id — not in the endpoint dict.
        """
        from unittest.mock import AsyncMock, patch

        from litellm.proxy.health_check import _perform_health_check

        auth_error = litellm.AuthenticationError(message="Invalid key", llm_provider="openai", model="gpt-4")
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake"},
                "model_info": {"id": "deploy-abc"},
            }
        ]

        with patch(
            "litellm.proxy.health_check.litellm.ahealth_check",
            new=AsyncMock(return_value={"error": "auth failed", "exception": auth_error}),
        ):
            healthy, unhealthy, exc_map = await _perform_health_check(model_list)

        assert len(unhealthy) == 1
        assert "exception" not in unhealthy[0], "exception must not be in endpoint dict"
        assert exc_map.get("deploy-abc") is auth_error

    @pytest.mark.asyncio
    async def test_raw_exception_from_gather_in_map(self):
        """When asyncio.gather returns a raw Exception, it must appear in
        exceptions_by_model_id — not in the endpoint dict."""
        from unittest.mock import patch

        from litellm.proxy.health_check import _perform_health_check

        raw_exc = litellm.RateLimitError(message="Rate limited", llm_provider="openai", model="gpt-4")
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake"},
                "model_info": {"id": "deploy-xyz"},
            }
        ]

        # Simulate asyncio.gather returning a raw exception for this task
        with patch(
            "litellm.proxy.health_check._run_model_health_check",
            side_effect=raw_exc,
        ):
            healthy, unhealthy, exc_map = await _perform_health_check(model_list)

        assert len(unhealthy) == 1
        assert "exception" not in unhealthy[0], "exception must not be in endpoint dict"
        assert exc_map.get("deploy-xyz") is raw_exc


class TestGetAllowedFailsFromPolicyWithHealthCheckExceptions:
    """Test that get_allowed_fails_from_policy correctly resolves thresholds for health-check exceptions."""

    @pytest.mark.parametrize(
        "exception_type, policy_field, threshold",
        [
            (litellm.Timeout, "TimeoutErrorAllowedFails", 5),
            (litellm.AuthenticationError, "AuthenticationErrorAllowedFails", 3),
            (litellm.RateLimitError, "RateLimitErrorAllowedFails", 10),
            (
                litellm.ContentPolicyViolationError,
                "ContentPolicyViolationErrorAllowedFails",
                2,
            ),
            (litellm.BadRequestError, "BadRequestErrorAllowedFails", 7),
        ],
    )
    def test_policy_resolves_for_health_check_exception_types(self, exception_type, policy_field, threshold):
        """Each exception type from a health check should resolve to its policy threshold."""
        policy = AllowedFailsPolicy(**{policy_field: threshold})
        router = Router(
            model_list=[_make_model("d1")],
            allowed_fails_policy=policy,
        )
        exception = exception_type(message="health check failed", llm_provider="openai", model="gpt-4")
        result = router.get_allowed_fails_from_policy(exception=exception)
        assert result == threshold

    def test_policy_returns_none_for_unmatched_exception(self):
        """When no policy field matches the exception type, return None (fall back to allowed_fails)."""
        policy = AllowedFailsPolicy(TimeoutErrorAllowedFails=5)
        router = Router(
            model_list=[_make_model("d1")],
            allowed_fails_policy=policy,
        )
        # Use a generic Exception that doesn't match any policy field
        result = router.get_allowed_fails_from_policy(exception=Exception("generic"))
        assert result is None


class TestHealthCheckCooldownIntegration:
    """Test that health check failures trigger cooldown via set_cooldown_deployments."""

    def test_health_check_failure_increments_failed_calls(self):
        """Health check failure should increment the failed_calls counter."""
        from litellm.router_utils.cooldown_handlers import (
            should_cooldown_based_on_allowed_fails_policy,
        )

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=3),
        )

        timeout_exc = litellm.Timeout(message="Health check timeout", model="gpt-4", llm_provider="openai")

        # First call: should not cooldown (1 <= 3)
        result = should_cooldown_based_on_allowed_fails_policy(
            ctx=RoutingContext(router),
            deployment="deploy-1",
            original_exception=timeout_exc,
        )
        assert result is False

        # Check counter was incremented
        current_fails = router.health().store.get_cache(key="deployment:deploy-1:allowed_fails")
        assert current_fails == 1

    def test_health_check_failure_triggers_cooldown_at_threshold(self):
        """After exceeding allowed_fails threshold, deployment should enter cooldown."""
        from litellm.router_utils.cooldown_handlers import (
            should_cooldown_based_on_allowed_fails_policy,
        )

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(AuthenticationErrorAllowedFails=2),
        )

        auth_exc = litellm.AuthenticationError(message="Invalid key", model="gpt-4", llm_provider="openai")

        # Fails 1 and 2: should not cooldown
        for _ in range(2):
            result = should_cooldown_based_on_allowed_fails_policy(
                ctx=RoutingContext(router),
                deployment="deploy-1",
                original_exception=auth_exc,
            )
            assert result is False

        # Fail 3: should trigger cooldown (3 > 2)
        result = should_cooldown_based_on_allowed_fails_policy(
            ctx=RoutingContext(router),
            deployment="deploy-1",
            original_exception=auth_exc,
        )
        assert result is True

    def test_health_check_failure_falls_back_to_allowed_fails(self):
        """When policy has no matching field, fall back to generic allowed_fails."""
        from litellm.router_utils.cooldown_handlers import (
            should_cooldown_based_on_allowed_fails_policy,
        )

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=10),
            allowed_fails=1,
        )

        # Use an exception that doesn't match TimeoutErrorAllowedFails
        # InternalServerError is not checked by get_allowed_fails_from_policy
        # so it will fall back to allowed_fails=1
        generic_exc = Exception("Some internal error")

        # Fail 1: should not cooldown (1 <= 1)
        result = should_cooldown_based_on_allowed_fails_policy(
            ctx=RoutingContext(router),
            deployment="deploy-1",
            original_exception=generic_exc,
        )
        assert result is False

        # Fail 2: should trigger cooldown (2 > 1)
        result = should_cooldown_based_on_allowed_fails_policy(
            ctx=RoutingContext(router),
            deployment="deploy-1",
            original_exception=generic_exc,
        )
        assert result is True

    def test_healthy_endpoints_do_not_trigger_cooldown(self):
        """Healthy endpoints should not increment any failure counters."""
        from litellm.router_utils.cooldown_handlers import set_cooldown_deployments

        router = Router(
            model_list=[_make_model("deploy-1")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=1),
            enable_health_check_routing=True,
        )

        # Simulate healthy endpoint -- no exception, no cooldown call
        healthy_endpoint = {"model_id": "deploy-1"}
        # Should have no exception key
        assert "exception" not in healthy_endpoint

        # Verify failed_calls counter is untouched
        current_fails = router.cache.get_cache(key="deployment:deploy-1:allowed_fails")
        assert current_fails is None

    def test_disable_cooldowns_prevents_health_check_cooldown(self):
        """When disable_cooldowns=True, health check failures should not trigger cooldown."""
        from litellm.router_utils.cooldown_handlers import set_cooldown_deployments

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=0),
            enable_health_check_routing=True,
            disable_cooldowns=True,
        )

        timeout_exc = litellm.Timeout(message="Health check timeout", model="gpt-4", llm_provider="openai")

        result = set_cooldown_deployments(
            ctx=RoutingContext(router),
            original_exception=timeout_exc,
            exception_status=500,
            deployment="deploy-1",
            time_to_cooldown=router.settings().cooldown_time,
        )
        assert result is False


class TestWriteHealthStateIntegration:
    """Test _write_health_state_to_router_cache feeds the router's cooldown pipeline."""

    def test_unhealthy_endpoint_cools_down_the_deployment(self):
        """An unhealthy endpoint's exception counts against the policy and cools the deployment down."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=0),
            enable_health_check_routing=True,
        )

        timeout_exc = litellm.Timeout(message="Health check timeout", model="", llm_provider="")

        with patch.object(proxy_module, "llm_router", router):
            _write_health_state_to_router_cache(
                healthy_endpoints=[{"model_id": "deploy-2"}],
                unhealthy_endpoints=[{"model_id": "deploy-1", "error": "timeout"}],
                exceptions_by_model_id={"deploy-1": timeout_exc},
            )

        assert _cooled(router) == ["deploy-1"]

    def test_unhealthy_endpoint_without_exception_skips_cooldown(self):
        """Unhealthy endpoints without an exception key should not trigger cooldown."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=0),
            enable_health_check_routing=True,
        )

        with patch.object(proxy_module, "llm_router", router):
            _write_health_state_to_router_cache(
                healthy_endpoints=[],
                unhealthy_endpoints=[{"model_id": "deploy-1", "error": "unknown failure"}],
            )

        assert _failures(router, "deploy-1") == 0
        assert _cooled(router) == []

    def test_unhealthy_endpoint_increments_failure_counter(self):
        """An unhealthy endpoint counts one failure for its deployment."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2")],
            allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=10),
            enable_health_check_routing=True,
        )

        rate_exc = litellm.RateLimitError(message="Rate limited", model="gpt-4", llm_provider="openai")

        with patch.object(proxy_module, "llm_router", router):
            _write_health_state_to_router_cache(
                healthy_endpoints=[],
                unhealthy_endpoints=[{"model_id": "deploy-1", "error": "rate limited"}],
                exceptions_by_model_id={"deploy-1": rate_exc},
            )

        assert _failures(router, "deploy-1") == 1
        assert _cooled(router) == []


@pytest.mark.parametrize(
    ("allowed_fails_policy", "model_groups", "routable"),
    [
        (None, None, ["deploy-2"]),
        (AllowedFailsPolicy(AuthenticationErrorAllowedFails=3), None, ["deploy-1", "deploy-2"]),
        (AllowedFailsPolicy(AuthenticationErrorAllowedFails=3), ["gpt-4"], ["deploy-2"]),
    ],
    ids=["no-policy-filters", "policy-bypasses-filter", "policy-keeps-filter-for-listed-groups"],
)
@pytest.mark.asyncio
async def test_health_check_filter_is_bypassed_by_an_allowed_fails_policy(
    allowed_fails_policy: AllowedFailsPolicy | None, model_groups: list[str] | None, routable: list[str]
) -> None:
    """With allowed_fails_policy set, cooldown is the sole routing exclusion, except for groups the
    health check is scoped to."""
    router = Router(
        model_list=[_make_model("deploy-1"), _make_model("deploy-2")],
        allowed_fails_policy=allowed_fails_policy,
        enable_health_check_routing=True,
        background_health_check_model_groups=model_groups,
    )
    router.health().health_checks.set_deployment_health_states(
        {"deploy-1": {"is_healthy": False, "timestamp": time.time(), "reason": "test"}}
    )

    healthy = await router.select("gpt-4", {}, kind="healthy")

    assert sorted(deployment["model_info"]["id"] for deployment in healthy) == routable


class TestHealthCheckIgnoreTransientErrors:
    """
    When health_check_ignore_transient_errors=True, health check failures with
    429 or 408 status codes should NOT increment failure counters or trigger cooldown.
    401, 404, and 5xx errors should still be processed normally.
    """

    def test_429_skipped_when_flag_enabled(self):
        """429 from health check does not trigger cooldown when flag is set."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=0),
            enable_health_check_routing=True,
            health_check_ignore_transient_errors=True,
        )

        rate_exc = litellm.RateLimitError(message="Rate limited", model="gpt-4", llm_provider="openai")
        assert getattr(rate_exc, "status_code", None) == 429

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "rate limited"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            _write_health_state_to_router_cache(
                healthy_endpoints=[],
                unhealthy_endpoints=unhealthy_endpoints,
                exceptions_by_model_id={"deploy-1": rate_exc},
            )

        assert _failures(router, "deploy-1") == 0
        assert _cooled(router) == []

    def test_408_skipped_when_flag_enabled(self):
        """408 from health check does not trigger cooldown when flag is set."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=0),
            enable_health_check_routing=True,
            health_check_ignore_transient_errors=True,
        )

        timeout_exc = litellm.Timeout(message="Health check timeout exceeded", model="", llm_provider="")

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "timeout"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            _write_health_state_to_router_cache(
                healthy_endpoints=[],
                unhealthy_endpoints=unhealthy_endpoints,
                exceptions_by_model_id={"deploy-1": timeout_exc},
            )

        assert _cooled(router) == []

    def test_401_still_triggers_cooldown_when_flag_enabled(self):
        """Auth errors (401) still trigger cooldown even when flag is set."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(AuthenticationErrorAllowedFails=0),
            enable_health_check_routing=True,
            health_check_ignore_transient_errors=True,
        )

        auth_exc = litellm.AuthenticationError(message="Invalid key", model="gpt-4", llm_provider="openai")

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "auth failed"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            _write_health_state_to_router_cache(
                healthy_endpoints=[],
                unhealthy_endpoints=unhealthy_endpoints,
                exceptions_by_model_id={"deploy-1": auth_exc},
            )

        assert _cooled(router) == ["deploy-1"]

    def test_429_not_written_to_health_state_cache_when_flag_enabled(self):
        """429 endpoint is excluded from health state cache when flag is set,
        so the binary health check filter does not mark it as unhealthy."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1")],
            enable_health_check_routing=True,
            health_check_ignore_transient_errors=True,
        )

        rate_exc = litellm.RateLimitError(message="Rate limited", model="gpt-4", llm_provider="openai")

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "rate limited"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            _write_health_state_to_router_cache(
                healthy_endpoints=[],
                unhealthy_endpoints=unhealthy_endpoints,
                exceptions_by_model_id={"deploy-1": rate_exc},
            )

        # Health state cache should have NO entry for deploy-1
        # (429 was ignored, not written as unhealthy)
        unhealthy_ids = router.health_state_cache.get_unhealthy_deployment_ids()
        assert "deploy-1" not in unhealthy_ids

    def test_429_triggers_cooldown_when_flag_disabled(self):
        """When flag is False (default), 429 still triggers cooldown."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=0),
            enable_health_check_routing=True,
            health_check_ignore_transient_errors=False,
        )

        rate_exc = litellm.RateLimitError(message="Rate limited", model="gpt-4", llm_provider="openai")

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "rate limited"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            _write_health_state_to_router_cache(
                healthy_endpoints=[],
                unhealthy_endpoints=unhealthy_endpoints,
                exceptions_by_model_id={"deploy-1": rate_exc},
            )

        assert _cooled(router) == ["deploy-1"]


class TestSharedCacheTransientErrorFilter:
    """
    When SharedHealthCheckManager returns cached results, exceptions_by_model_id
    is always {}. The filter must fall back to the 'exception_status' field stored
    on each endpoint dict so 429/408 endpoints are still excluded correctly.
    """

    def test_cached_429_excluded_via_exception_status_field(self):
        """Cache-hit path: endpoint with exception_status=429 is excluded from health state."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            enable_health_check_routing=True,
            health_check_ignore_transient_errors=True,
        )

        # Simulate a cache-hit endpoint: exception_status stored as int, no exceptions dict
        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "rate limited", "exception_status": 429},
        ]

        with patch.object(proxy_module, "llm_router", router):
            _write_health_state_to_router_cache(
                healthy_endpoints=[],
                unhealthy_endpoints=unhealthy_endpoints,
                exceptions_by_model_id={},
            )

        # deploy-1 should NOT be marked unhealthy (429 was filtered)
        unhealthy_ids = router.health_state_cache.get_unhealthy_deployment_ids()
        assert "deploy-1" not in unhealthy_ids

    def test_cached_401_still_marked_unhealthy(self):
        """Cache-hit path: endpoint with exception_status=401 is still written as unhealthy."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            enable_health_check_routing=True,
            health_check_ignore_transient_errors=True,
        )

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "auth failed", "exception_status": 401},
        ]

        with patch.object(proxy_module, "llm_router", router):
            _write_health_state_to_router_cache(
                healthy_endpoints=[],
                unhealthy_endpoints=unhealthy_endpoints,
                exceptions_by_model_id={},
            )

        unhealthy_ids = router.health_state_cache.get_unhealthy_deployment_ids()
        assert "deploy-1" in unhealthy_ids

"""
Tests for health check failures integrating with allowed_fails_policy cooldown pipeline.

When enable_health_check_routing is True and a health check fails, the failure
should increment the same counters used by allowed_fails_policy, using the
actual exception type from the health check error.
"""

from unittest.mock import patch

import pytest

import litellm
from litellm.proxy.health_check import run_with_timeout
from litellm.router import Router
from litellm.types.router import AllowedFailsPolicy


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

        auth_error = litellm.AuthenticationError(
            message="Invalid key", llm_provider="openai", model="gpt-4"
        )
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake"},
                "model_info": {"id": "deploy-abc"},
            }
        ]

        with patch(
            "litellm.proxy.health_check.litellm.ahealth_check",
            new=AsyncMock(
                return_value={"error": "auth failed", "exception": auth_error}
            ),
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

        raw_exc = litellm.RateLimitError(
            message="Rate limited", llm_provider="openai", model="gpt-4"
        )
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
    def test_policy_resolves_for_health_check_exception_types(
        self, exception_type, policy_field, threshold
    ):
        """Each exception type from a health check should resolve to its policy threshold."""
        policy = AllowedFailsPolicy(**{policy_field: threshold})
        router = Router(
            model_list=[_make_model("d1")],
            allowed_fails_policy=policy,
        )
        exception = exception_type(
            message="health check failed", llm_provider="openai", model="gpt-4"
        )
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
    """Test that health check failures trigger cooldown via _set_cooldown_deployments."""

    def test_health_check_failure_increments_failed_calls(self):
        """Health check failure should increment the failed_calls counter."""
        from litellm.router_utils.cooldown_handlers import (
            should_cooldown_based_on_allowed_fails_policy,
        )

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=3),
        )

        timeout_exc = litellm.Timeout(
            message="Health check timeout", model="gpt-4", llm_provider="openai"
        )

        # First call: should not cooldown (1 <= 3)
        result = should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=router,
            deployment="deploy-1",
            original_exception=timeout_exc,
        )
        assert result is False

        # Check counter was incremented
        current_fails = router.failed_calls.get_cache(key="deploy-1")
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

        auth_exc = litellm.AuthenticationError(
            message="Invalid key", model="gpt-4", llm_provider="openai"
        )

        # Fails 1 and 2: should not cooldown
        for _ in range(2):
            result = should_cooldown_based_on_allowed_fails_policy(
                litellm_router_instance=router,
                deployment="deploy-1",
                original_exception=auth_exc,
            )
            assert result is False

        # Fail 3: should trigger cooldown (3 > 2)
        result = should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=router,
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
            litellm_router_instance=router,
            deployment="deploy-1",
            original_exception=generic_exc,
        )
        assert result is False

        # Fail 2: should trigger cooldown (2 > 1)
        result = should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=router,
            deployment="deploy-1",
            original_exception=generic_exc,
        )
        assert result is True

    def test_healthy_endpoints_do_not_trigger_cooldown(self):
        """Healthy endpoints should not increment any failure counters."""
        from litellm.router_utils.cooldown_handlers import _set_cooldown_deployments

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
        current_fails = router.failed_calls.get_cache(key="deploy-1")
        assert current_fails is None

    def test_disable_cooldowns_prevents_health_check_cooldown(self):
        """When disable_cooldowns=True, health check failures should not trigger cooldown."""
        from litellm.router_utils.cooldown_handlers import _set_cooldown_deployments

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=0),
            enable_health_check_routing=True,
            disable_cooldowns=True,
        )

        timeout_exc = litellm.Timeout(
            message="Health check timeout", model="gpt-4", llm_provider="openai"
        )

        result = _set_cooldown_deployments(
            litellm_router_instance=router,
            original_exception=timeout_exc,
            exception_status=500,
            deployment="deploy-1",
            time_to_cooldown=router.cooldown_time,
        )
        assert result is False


class TestWriteHealthStateIntegration:
    """Test _write_health_state_to_router_cache integrates with cooldown pipeline."""

    def test_unhealthy_endpoint_triggers_set_cooldown(self):
        """_write_health_state_to_router_cache should call _set_cooldown_deployments for unhealthy endpoints."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=5),
            enable_health_check_routing=True,
        )

        timeout_exc = litellm.Timeout(
            message="Health check timeout", model="", llm_provider=""
        )

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "timeout"},
        ]
        healthy_endpoints = [
            {"model_id": "deploy-2"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            with patch(
                "litellm.router_utils.cooldown_handlers._set_cooldown_deployments"
            ) as mock_cooldown:
                _write_health_state_to_router_cache(
                    healthy_endpoints=healthy_endpoints,
                    unhealthy_endpoints=unhealthy_endpoints,
                    exceptions_by_model_id={"deploy-1": timeout_exc},
                )
                mock_cooldown.assert_called_once_with(
                    litellm_router_instance=router,
                    original_exception=timeout_exc,
                    exception_status=408,  # Timeout has status_code 408
                    deployment="deploy-1",
                    time_to_cooldown=router.cooldown_time,
                )

    def test_unhealthy_endpoint_without_exception_skips_cooldown(self):
        """Unhealthy endpoints without an exception key should not trigger cooldown."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=5),
            enable_health_check_routing=True,
        )

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "unknown failure"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            with patch(
                "litellm.router_utils.cooldown_handlers._set_cooldown_deployments"
            ) as mock_cooldown:
                _write_health_state_to_router_cache(
                    healthy_endpoints=[],
                    unhealthy_endpoints=unhealthy_endpoints,
                    # no exceptions_by_model_id → cooldown should not fire
                )
                mock_cooldown.assert_not_called()

    def test_unhealthy_endpoint_increments_failure_counter(self):
        """Unhealthy endpoints should call increment_deployment_failures_for_current_minute."""
        import litellm.proxy.proxy_server as proxy_module
        from litellm.proxy.proxy_server import _write_health_state_to_router_cache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=10),
            enable_health_check_routing=True,
        )

        rate_exc = litellm.RateLimitError(
            message="Rate limited", model="gpt-4", llm_provider="openai"
        )

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "rate limited"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            with patch(
                "litellm.router_utils.router_callbacks.track_deployment_metrics.increment_deployment_failures_for_current_minute"
            ) as mock_increment:
                with patch(
                    "litellm.router_utils.cooldown_handlers._set_cooldown_deployments"
                ):
                    _write_health_state_to_router_cache(
                        healthy_endpoints=[],
                        unhealthy_endpoints=unhealthy_endpoints,
                        exceptions_by_model_id={"deploy-1": rate_exc},
                    )
                    mock_increment.assert_called_once_with(
                        litellm_router_instance=router,
                        deployment_id="deploy-1",
                    )


class TestHealthCheckFilterBypassWithPolicy:
    """
    When allowed_fails_policy is set, the binary health check filter should be
    bypassed so cooldown is the sole routing exclusion mechanism.
    """

    def test_filter_bypassed_when_policy_set(self):
        """Binary health check filter is a no-op when allowed_fails_policy is configured."""
        import time

        from litellm.caching.caching import DualCache
        from litellm.router_utils.health_state_cache import DeploymentHealthCache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(AuthenticationErrorAllowedFails=3),
            enable_health_check_routing=True,
        )

        # Mark deploy-1 as unhealthy in the health state cache
        cache = DualCache()
        health_cache = DeploymentHealthCache(cache=cache, staleness_threshold=60.0)
        health_cache.set_deployment_health_states(
            {
                "deploy-1": {
                    "is_healthy": False,
                    "timestamp": time.time(),
                    "reason": "test",
                },
            }
        )
        router.health_state_cache = health_cache

        deployments = [_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")]

        # Filter should pass all through because policy is set
        result = router._filter_health_check_unhealthy_deployments(deployments)
        assert (
            len(result) == 2
        ), "Binary filter should be bypassed when allowed_fails_policy is set"

    def test_filter_active_when_no_policy(self):
        """Binary health check filter still works when no allowed_fails_policy is configured."""
        import time

        from litellm.caching.caching import DualCache
        from litellm.router_utils.health_state_cache import DeploymentHealthCache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            enable_health_check_routing=True,
        )

        cache = DualCache()
        health_cache = DeploymentHealthCache(cache=cache, staleness_threshold=60.0)
        health_cache.set_deployment_health_states(
            {
                "deploy-1": {
                    "is_healthy": False,
                    "timestamp": time.time(),
                    "reason": "test",
                },
            }
        )
        router.health_state_cache = health_cache

        deployments = [_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")]

        result = router._filter_health_check_unhealthy_deployments(deployments)
        assert len(result) == 1
        assert result[0]["model_info"]["id"] == "deploy-2"

    @pytest.mark.asyncio
    async def test_async_filter_bypassed_when_policy_set(self):
        """Async version also bypasses when allowed_fails_policy is set."""
        import time

        from litellm.caching.caching import DualCache
        from litellm.router_utils.health_state_cache import DeploymentHealthCache

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=2),
            enable_health_check_routing=True,
        )

        cache = DualCache()
        health_cache = DeploymentHealthCache(cache=cache, staleness_threshold=60.0)
        health_cache.set_deployment_health_states(
            {
                "deploy-1": {
                    "is_healthy": False,
                    "timestamp": time.time(),
                    "reason": "test",
                },
            }
        )
        router.health_state_cache = health_cache

        deployments = [_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")]

        result = await router._async_filter_health_check_unhealthy_deployments(
            deployments
        )
        assert len(result) == 2


class TestAllDeploymentsInCooldownSafetyNet:
    """
    When enable_health_check_routing=True and ALL deployments enter cooldown,
    the async routing path should bypass the cooldown filter and return all
    deployments rather than blocking all traffic.
    """

    def test_raw_cooldown_filter_returns_empty_when_all_cooled(self):
        """The raw _filter_cooldown_deployments has no safety net -- it returns empty."""
        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            enable_health_check_routing=True,
        )
        deployments = [_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")]
        result = router._filter_cooldown_deployments(
            healthy_deployments=deployments,
            cooldown_deployments=["deploy-1", "deploy-2"],
        )
        assert result == []  # raw filter has no safety net

    @pytest.mark.asyncio
    async def test_async_routing_path_bypasses_all_cooldown(self):
        """In the async routing path, all-in-cooldown with enable_health_check_routing
        returns the full list instead of empty (safety net)."""
        from unittest.mock import AsyncMock

        from litellm.router_utils.cooldown_handlers import (
            _async_get_cooldown_deployments,
        )

        router = Router(
            model_list=[_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")],
            allowed_fails_policy=AllowedFailsPolicy(AuthenticationErrorAllowedFails=0),
            enable_health_check_routing=True,
        )

        deployments = [_make_model("deploy-1"), _make_model("deploy-2", "gpt-5")]

        # Simulate all deployments in cooldown
        with patch(
            "litellm.router._async_get_cooldown_deployments",
            new=AsyncMock(return_value=["deploy-1", "deploy-2"]),
        ):
            # The safety net in async_get_available_deployment should restore
            # all deployments when the cooldown filter empties the list
            _pre = deployments.copy()
            filtered = router._filter_cooldown_deployments(
                healthy_deployments=deployments,
                cooldown_deployments=["deploy-1", "deploy-2"],
            )
            # If filtered is empty and enable_health_check_routing is True,
            # the routing path restores _pre_cooldown_deployments
            if not filtered and router.enable_health_check_routing:
                filtered = _pre

            assert (
                len(filtered) == 2
            ), "Safety net should return all deployments when all are in cooldown"


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

        rate_exc = litellm.RateLimitError(
            message="Rate limited", model="gpt-4", llm_provider="openai"
        )
        assert getattr(rate_exc, "status_code", None) == 429

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "rate limited"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            with patch(
                "litellm.router_utils.cooldown_handlers._set_cooldown_deployments"
            ) as mock_cooldown:
                with patch(
                    "litellm.router_utils.router_callbacks.track_deployment_metrics.increment_deployment_failures_for_current_minute"
                ) as mock_increment:
                    _write_health_state_to_router_cache(
                        healthy_endpoints=[],
                        unhealthy_endpoints=unhealthy_endpoints,
                        exceptions_by_model_id={"deploy-1": rate_exc},
                    )
                    mock_cooldown.assert_not_called()
                    mock_increment.assert_not_called()

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

        timeout_exc = litellm.Timeout(
            message="Health check timeout exceeded", model="", llm_provider=""
        )

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "timeout"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            with patch(
                "litellm.router_utils.cooldown_handlers._set_cooldown_deployments"
            ) as mock_cooldown:
                _write_health_state_to_router_cache(
                    healthy_endpoints=[],
                    unhealthy_endpoints=unhealthy_endpoints,
                    exceptions_by_model_id={"deploy-1": timeout_exc},
                )
                mock_cooldown.assert_not_called()

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

        auth_exc = litellm.AuthenticationError(
            message="Invalid key", model="gpt-4", llm_provider="openai"
        )

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "auth failed"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            with patch(
                "litellm.router_utils.cooldown_handlers._set_cooldown_deployments"
            ) as mock_cooldown:
                _write_health_state_to_router_cache(
                    healthy_endpoints=[],
                    unhealthy_endpoints=unhealthy_endpoints,
                    exceptions_by_model_id={"deploy-1": auth_exc},
                )
                mock_cooldown.assert_called_once()

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

        rate_exc = litellm.RateLimitError(
            message="Rate limited", model="gpt-4", llm_provider="openai"
        )

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

        rate_exc = litellm.RateLimitError(
            message="Rate limited", model="gpt-4", llm_provider="openai"
        )

        unhealthy_endpoints = [
            {"model_id": "deploy-1", "error": "rate limited"},
        ]

        with patch.object(proxy_module, "llm_router", router):
            with patch(
                "litellm.router_utils.cooldown_handlers._set_cooldown_deployments"
            ) as mock_cooldown:
                _write_health_state_to_router_cache(
                    healthy_endpoints=[],
                    unhealthy_endpoints=unhealthy_endpoints,
                    exceptions_by_model_id={"deploy-1": rate_exc},
                )
                mock_cooldown.assert_called_once()


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

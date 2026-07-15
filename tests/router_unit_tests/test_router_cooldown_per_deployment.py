"""
Tests for per-deployment cooldown policy overrides, DualCache TTL correction,
and fallback-path cooldown gap fix.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.router_utils.cooldown_cache import CooldownCache, CooldownCacheValue
from litellm.router_utils.cooldown_handlers import (
    _get_deployment_cooldown_policy,
    _resolve_allowed_fails_from_policy,
    _should_cooldown_deployment,
    should_cooldown_based_on_allowed_fails_policy,
)
from litellm.router_utils.fallback_event_handlers import _trigger_cooldown_for_failed_deployment
from litellm.types.router import AllowedFailsPolicy


def _make_router(model_list: list, **kwargs) -> Router:
    return Router(model_list=model_list, **kwargs)


class TestDeploymentLevelAllowedFails:
    def test_deployment_level_allowed_fails_overrides_router_level(self):
        """
        A deployment with model_info.allowed_fails=0 must enter cooldown after 1
        failure even when the router-level allowed_fails=10.
        """
        router = _make_router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {
                        "id": "primary",
                        "allowed_fails": 0,
                    },
                },
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {"id": "secondary"},
                },
            ],
            allowed_fails=10,
        )

        _exception = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        should_cooldown = _should_cooldown_deployment(
            litellm_router_instance=router,
            deployment="primary",
            exception_status=429,
            original_exception=_exception,
        )

        assert should_cooldown is True, "Deployment-level allowed_fails=0 should force cooldown after first failure"

    def test_deployment_level_allowed_fails_does_not_affect_other_deployments(self):
        """
        A deployment without model_info.allowed_fails must still use the router-level
        allowed_fails and not be pulled into cooldown prematurely.
        """
        router = _make_router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {
                        "id": "primary",
                        "allowed_fails": 0,
                    },
                },
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {"id": "secondary"},
                },
            ],
            allowed_fails=10,
        )

        _exception = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        should_cooldown = _should_cooldown_deployment(
            litellm_router_instance=router,
            deployment="secondary",
            exception_status=429,
            original_exception=_exception,
        )

        assert should_cooldown is False, (
            "secondary has no deployment-level policy; with allowed_fails=10 it should not cool down on first failure"
        )


class TestDeploymentLevelAllowedFailsPolicyByExceptionType:
    def test_rate_limit_error_triggers_cooldown_with_zero_threshold(self):
        """
        RateLimitErrorAllowedFails=0 must trigger cooldown after 1 RateLimitError
        even when allowed_fails=5 for other exception types.
        """
        router = _make_router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {
                        "id": "primary",
                        "allowed_fails_policy": {
                            "RateLimitErrorAllowedFails": 0,
                            "InternalServerErrorAllowedFails": 5,
                        },
                    },
                },
            ],
            allowed_fails=10,
        )

        rate_limit_exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        should_cooldown = _should_cooldown_deployment(
            litellm_router_instance=router,
            deployment="primary",
            exception_status=429,
            original_exception=rate_limit_exc,
        )

        assert should_cooldown is True, "RateLimitErrorAllowedFails=0 must trigger cooldown on first rate limit error"

    def test_internal_server_error_respects_per_exception_threshold(self):
        """
        InternalServerErrorAllowedFails=5 must allow 5 InternalServerErrors before cooldown.
        """
        router = _make_router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {
                        "id": "primary",
                        "allowed_fails_policy": {
                            "RateLimitErrorAllowedFails": 0,
                            "InternalServerErrorAllowedFails": 5,
                        },
                    },
                },
            ],
            allowed_fails=10,
        )

        ise = litellm.InternalServerError("Internal error", "openai", "gpt-4")

        for _ in range(5):
            should_cooldown = _should_cooldown_deployment(
                litellm_router_instance=router,
                deployment="primary",
                exception_status=500,
                original_exception=ise,
            )
            assert should_cooldown is False, "Should not cooldown within the allowed_fails threshold"

        should_cooldown = _should_cooldown_deployment(
            litellm_router_instance=router,
            deployment="primary",
            exception_status=500,
            original_exception=ise,
        )
        assert should_cooldown is True, "Should cooldown after exceeding InternalServerErrorAllowedFails=5"


class TestExceptionTypeCountersTrackedIndependently:
    def test_cache_key_suffix_separates_exception_type_counters(self):
        """
        When cache_key_suffix is provided, fail counters for different exception types
        must be independent; RateLimitError fails must not bleed into generic counters.
        """
        router = _make_router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {"id": "primary"},
                },
            ],
            allowed_fails=10,
        )

        rate_limit_exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        ise = litellm.InternalServerError("Internal error", "openai", "gpt-4")

        for _ in range(3):
            should_cooldown_based_on_allowed_fails_policy(
                litellm_router_instance=router,
                deployment="primary",
                original_exception=rate_limit_exc,
                allowed_fails_override=5,
                cache_key_suffix="RateLimitError",
            )

        rl_counter = router.failed_calls.get_cache(key="primary:RateLimitError") or 0
        generic_counter = router.failed_calls.get_cache(key="primary:generic") or 0

        assert rl_counter == 3, "RateLimitError counter should be 3"
        assert generic_counter == 0, "generic counter must be untouched by RateLimitError increments"

        should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=router,
            deployment="primary",
            original_exception=ise,
            allowed_fails_override=5,
            cache_key_suffix="generic",
        )

        generic_counter_after = router.failed_calls.get_cache(key="primary:generic") or 0
        rl_counter_after = router.failed_calls.get_cache(key="primary:RateLimitError") or 0

        assert generic_counter_after == 1, "generic counter should now be 1"
        assert rl_counter_after == 3, "RateLimitError counter must remain unchanged after InternalServerError"


class TestCooldownCacheTTLCorrection:
    def _make_cooldown_cache(self) -> CooldownCache:
        in_memory = InMemoryCache()
        dual_cache = DualCache(in_memory_cache=in_memory)
        return CooldownCache(cache=dual_cache, default_cooldown_time=60.0)

    def test_expired_entry_evicted_and_not_returned(self):
        """
        An entry with timestamp+cooldown_time in the past must be evicted from
        in-memory cache and excluded from the active cooldown list.
        """
        cc = self._make_cooldown_cache()
        model_id = "expired-deployment"
        key = CooldownCache.get_cooldown_cache_key(model_id)

        expired_value: CooldownCacheValue = {
            "exception_received": "Rate limit",
            "status_code": "429",
            "timestamp": time.time() - 120.0,
            "cooldown_time": 60.0,
        }
        cc.cache.in_memory_cache.set_cache(key, expired_value, ttl=600)

        active = cc.get_active_cooldowns(model_ids=[model_id], parent_otel_span=None)

        assert active == [], "Expired cooldown entry must not appear in active cooldowns"
        assert cc.cache.in_memory_cache.get_cache(key) is None, "Expired entry must be evicted from in-memory cache"

    def test_active_entry_is_returned(self):
        """
        An entry whose cooldown window has not elapsed must appear in the active list.
        """
        cc = self._make_cooldown_cache()
        model_id = "active-deployment"
        key = CooldownCache.get_cooldown_cache_key(model_id)

        active_value: CooldownCacheValue = {
            "exception_received": "Rate limit",
            "status_code": "429",
            "timestamp": time.time(),
            "cooldown_time": 60.0,
        }
        cc.cache.in_memory_cache.set_cache(key, active_value, ttl=60)

        active = cc.get_active_cooldowns(model_ids=[model_id], parent_otel_span=None)

        assert len(active) == 1
        assert active[0][0] == model_id

    def test_ttl_corrected_when_in_memory_expiry_far_exceeds_remaining(self):
        """
        When DualCache backfills from Redis using the default 600s TTL, the in-memory
        TTL must be corrected to min(remaining, 60) seconds.
        """
        cc = self._make_cooldown_cache()
        model_id = "backfilled-deployment"
        key = CooldownCache.get_cooldown_cache_key(model_id)

        remaining = 30.0
        value: CooldownCacheValue = {
            "exception_received": "Rate limit",
            "status_code": "429",
            "timestamp": time.time() - (60.0 - remaining),
            "cooldown_time": 60.0,
        }
        cc.cache.in_memory_cache.set_cache(key, value, ttl=600)

        before_expiry = cc.cache.in_memory_cache.ttl_dict.get(key)
        assert before_expiry is not None

        cc.get_active_cooldowns(model_ids=[model_id], parent_otel_span=None)

        after_expiry = cc.cache.in_memory_cache.ttl_dict.get(key)
        assert after_expiry is not None
        corrected_remaining = after_expiry - time.time()
        assert corrected_remaining <= 60.0, "Corrected TTL must not exceed 60s"
        assert corrected_remaining > 0, "Corrected TTL must be positive (cooldown still active)"

    @pytest.mark.asyncio
    async def test_async_expired_entry_evicted(self):
        """
        Async path must also evict expired entries.
        """
        cc = self._make_cooldown_cache()
        model_id = "async-expired"
        key = CooldownCache.get_cooldown_cache_key(model_id)

        expired_value: CooldownCacheValue = {
            "exception_received": "Rate limit",
            "status_code": "429",
            "timestamp": time.time() - 120.0,
            "cooldown_time": 60.0,
        }
        cc.cache.in_memory_cache.set_cache(key, expired_value, ttl=600)

        active = await cc.async_get_active_cooldowns(model_ids=[model_id], parent_otel_span=None)

        assert active == [], "Expired entry must not appear in async active cooldowns"
        assert cc.cache.in_memory_cache.get_cache(key) is None


class TestFallbackDeploymentCooldown:
    def test_trigger_cooldown_for_failed_deployment_calls_set_cooldown(self):
        """
        _trigger_cooldown_for_failed_deployment must call _set_cooldown_deployments
        with the deployment ID extracted from kwargs metadata.
        """
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        kwargs = {
            "litellm_metadata": {
                "model_info": {"id": "fallback-deployment"},
            }
        }
        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs=kwargs,
                exception=exc,
            )

            mock_set_cooldown.assert_called_once()
            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["deployment"] == "fallback-deployment"
            assert call_kwargs["original_exception"] is exc

    def test_trigger_cooldown_no_op_when_deployment_id_missing(self):
        """
        _trigger_cooldown_for_failed_deployment must not raise and must skip
        _set_cooldown_deployments when deployment_id is absent from metadata.
        """
        mock_router = MagicMock()

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={},
                exception=RuntimeError("no metadata"),
            )

            mock_set_cooldown.assert_not_called()

    def test_trigger_cooldown_uses_deployment_cooldown_time_override(self):
        """
        When the deployment has a litellm_params.cooldown_time, that value must be
        passed as time_to_cooldown rather than the router-level cooldown_time.
        """
        mock_router = MagicMock()
        mock_router.cooldown_time = 300.0
        mock_router.get_model_info.return_value = {"litellm_params": {"cooldown_time": 30.0}}

        kwargs = {
            "litellm_metadata": {
                "model_info": {"id": "fallback-deployment"},
            }
        }
        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs=kwargs,
                exception=exc,
            )

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 30.0, (
                "Deployment-level cooldown_time must override router-level value"
            )


class TestNewAllowedFailsPolicyFields:
    def test_service_unavailable_error_matched_by_policy(self):
        """
        ServiceUnavailableError must be matched against ServiceUnavailableErrorAllowedFails.
        """
        policy = {"ServiceUnavailableErrorAllowedFails": 0}
        exc = litellm.ServiceUnavailableError("Service unavailable", "openai", "gpt-4")
        result = _resolve_allowed_fails_from_policy(policy=policy, exception=exc)
        assert result == 0

    def test_bad_gateway_error_matched_by_policy(self):
        """
        BadGatewayError must be matched against BadGatewayErrorAllowedFails.
        """
        policy = {"BadGatewayErrorAllowedFails": 2}
        exc = litellm.BadGatewayError("Bad gateway", "openai", "gpt-4")
        result = _resolve_allowed_fails_from_policy(policy=policy, exception=exc)
        assert result == 2

    def test_not_found_error_matched_by_policy(self):
        """
        NotFoundError must be matched against NotFoundErrorAllowedFails.
        """
        policy = {"NotFoundErrorAllowedFails": 1}
        exc = litellm.NotFoundError("Not found", "openai", "gpt-4")
        result = _resolve_allowed_fails_from_policy(policy=policy, exception=exc)
        assert result == 1

    def test_unknown_exception_type_returns_none(self):
        """
        An exception type not in the policy mapping must return None.
        """
        policy = {"RateLimitErrorAllowedFails": 0}
        exc = ValueError("unexpected error")
        result = _resolve_allowed_fails_from_policy(policy=policy, exception=exc)
        assert result is None

    def test_allowed_fails_policy_model_accepts_new_fields(self):
        """
        AllowedFailsPolicy Pydantic model must accept the three new fields.
        """
        policy = AllowedFailsPolicy(
            ServiceUnavailableErrorAllowedFails=3,
            BadGatewayErrorAllowedFails=2,
            NotFoundErrorAllowedFails=1,
        )
        assert policy.ServiceUnavailableErrorAllowedFails == 3
        assert policy.BadGatewayErrorAllowedFails == 2
        assert policy.NotFoundErrorAllowedFails == 1


class TestRouterLevelGetAllowedFailsFromPolicy:
    """Router.get_allowed_fails_from_policy must handle all AllowedFailsPolicy fields."""

    def _make_router(self, **policy_kwargs):
        return Router(
            model_list=[{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4", "api_key": "fake"}}],
            allowed_fails_policy=AllowedFailsPolicy(**policy_kwargs),
        )

    def test_internal_server_error_returned(self):
        router = self._make_router(InternalServerErrorAllowedFails=7)
        exc = litellm.InternalServerError("500 error", "openai", "gpt-4")
        assert router.get_allowed_fails_from_policy(exc) == 7

    def test_service_unavailable_error_returned(self):
        router = self._make_router(ServiceUnavailableErrorAllowedFails=4)
        exc = litellm.ServiceUnavailableError("503 error", "openai", "gpt-4")
        assert router.get_allowed_fails_from_policy(exc) == 4

    def test_bad_gateway_error_returned(self):
        router = self._make_router(BadGatewayErrorAllowedFails=2)
        exc = litellm.BadGatewayError("502 error", "openai", "gpt-4")
        assert router.get_allowed_fails_from_policy(exc) == 2

    def test_not_found_error_returned(self):
        router = self._make_router(NotFoundErrorAllowedFails=1)
        exc = litellm.NotFoundError("404 error", "openai", "gpt-4")
        assert router.get_allowed_fails_from_policy(exc) == 1

    def test_unmatched_exception_returns_none(self):
        router = self._make_router(InternalServerErrorAllowedFails=5)
        exc = litellm.RateLimitError("429", "openai", "gpt-4")
        assert router.get_allowed_fails_from_policy(exc) is None

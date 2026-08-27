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
    _has_explicit_allowed_fails_policy_for_exception,
    _resolve_allowed_fails_from_policy,
    _should_cooldown_deployment,
    mark_advisor_orchestration_failure,
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
        with the deployment ID stamped on the exception.
        """
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={},
                exception=exc,
            )

            mock_set_cooldown.assert_called_once()
            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["deployment"] == "fallback-deployment"
            assert call_kwargs["original_exception"] is exc

    def test_trigger_cooldown_no_op_when_deployment_id_missing(self):
        """
        _trigger_cooldown_for_failed_deployment must not raise and must skip
        _set_cooldown_deployments when the exception has no failed_deployment_id.
        """
        mock_router = MagicMock()

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={},
                exception=RuntimeError("no stamped deployment id"),
            )

            mock_set_cooldown.assert_not_called()

    def test_trigger_cooldown_does_not_trust_caller_supplied_metadata_bucket(self):
        """
        A metadata bucket can't reliably be told apart from a caller-supplied one
        without knowing the call's function_name, so a client with permission to
        set metadata must not be able to get an arbitrary deployment cooled down
        by forging a deployment_model_name marker.
        """
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        kwargs = {
            "metadata": {
                "model_info": {"id": "attacker-chosen-deployment"},
                "deployment_model_name": "gpt-4",
            }
        }

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs=kwargs,
                exception=exc,
            )

            mock_set_cooldown.assert_not_called()

    def test_trigger_cooldown_increments_failure_counter_before_cooldown_check(self):
        """
        The fallback path must feed the same per-minute failure counter the
        primary path uses, or repeated fallback failures never accumulate toward
        the default percent-fail-rate cooldown threshold.
        """
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with (
            patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown,
            patch(
                "litellm.router_utils.fallback_event_handlers.increment_deployment_failures_for_current_minute"
            ) as mock_increment,
        ):
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_increment.assert_called_once_with(
                litellm_router_instance=mock_router, deployment_id="fallback-deployment"
            )
            mock_set_cooldown.assert_called_once()

    def test_trigger_cooldown_uses_deployment_cooldown_time_override(self):
        """
        When the deployment has a model_info.cooldown_time, that value must be
        passed as time_to_cooldown rather than the router-level cooldown_time.
        """
        mock_router = MagicMock()
        mock_router.cooldown_time = 300.0
        mock_router.get_model_info.return_value = {"model_info": {"cooldown_time": 30.0}}

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={},
                exception=exc,
            )

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 30.0, (
                "Deployment-level cooldown_time must override router-level value"
            )

    def test_trigger_cooldown_skipped_for_advisor_orchestration_failure(self):
        """
        A failure tagged as originating from advisor orchestration (not the selected
        deployment) must not cool down the fallback deployment, matching the same
        guard already applied in Router.deployment_callback_on_failure.
        """
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"
        mark_advisor_orchestration_failure(exc)

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={},
                exception=exc,
            )

            mock_set_cooldown.assert_not_called()

    def test_trigger_cooldown_falls_back_to_litellm_params_cooldown_time(self):
        """
        cooldown_time has pre-existing litellm_params support on the primary
        failure path (Router.deployment_callback_on_failure), so it must still be
        honored as a fallback when model_info doesn't set it, unlike the new
        allowed_fails/allowed_fails_policy fields which are model_info-only.
        """
        mock_router = MagicMock()
        mock_router.cooldown_time = 300.0
        mock_router.get_model_info.return_value = {"litellm_params": {"cooldown_time": 30.0}}

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={},
                exception=exc,
            )

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 30.0, (
                "litellm_params.cooldown_time must still be honored as a fallback"
            )

    def test_trigger_cooldown_prefers_model_info_cooldown_time_over_litellm_params(self):
        mock_router = MagicMock()
        mock_router.cooldown_time = 300.0
        mock_router.get_model_info.return_value = {
            "model_info": {"cooldown_time": 15.0},
            "litellm_params": {"cooldown_time": 30.0},
        }

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={},
                exception=exc,
            )

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 15.0, "model_info.cooldown_time must take priority"


class TestSingleDeploymentModelGroupProtection:
    def test_generic_allowed_fails_does_not_bypass_single_deployment_protection(self):
        """
        Setting only a generic model_info.allowed_fails on a single-deployment model
        group must not disable the "avoid cooldowns on single deployment model groups"
        safety net; before this feature existed the field had no effect at all here,
        so a plain 500 error must behave the same as the no-policy control.
        """
        router = _make_router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {"id": "solo", "allowed_fails": 1},
                },
            ],
        )

        exc = Exception("Internal error")
        for _ in range(2):
            should_cooldown = _should_cooldown_deployment(
                litellm_router_instance=router,
                deployment="solo",
                exception_status=500,
                original_exception=exc,
            )
            assert should_cooldown is False, (
                "single-deployment model group must stay protected from a generic allowed_fails override"
            )

    def test_named_exception_policy_still_overrides_single_deployment_protection(self):
        """
        Unlike a generic allowed_fails, an explicit per-exception-type allowed_fails_policy
        entry is a deliberate, unambiguous opt-in and must still apply even on a
        single-deployment model group.
        """
        router = _make_router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {
                        "id": "solo",
                        "allowed_fails_policy": {"RateLimitErrorAllowedFails": 0},
                    },
                },
            ],
        )

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        should_cooldown = _should_cooldown_deployment(
            litellm_router_instance=router,
            deployment="solo",
            exception_status=429,
            original_exception=exc,
        )
        assert should_cooldown is True, "explicit per-exception-type policy must still cool down a solo deployment"


class TestShouldCooldownBasedOnAllowedFailsPolicyFalsyZero:
    def test_router_level_policy_of_zero_is_not_swallowed_by_allowed_fails(self):
        """
        Router.get_allowed_fails_from_policy returning 0 (a legitimate "cooldown after
        the very first failure" policy) must not be treated as falsy and replaced by
        router.allowed_fails.
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
            allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=0),
        )

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        should_cooldown = should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=router,
            deployment="primary",
            original_exception=exc,
        )
        assert should_cooldown is True, "RateLimitErrorAllowedFails=0 must cool down after the first failure"


class TestResolveAllowedFailsFromPolicyFallsThrough:
    def test_none_value_on_first_match_falls_through_to_next_type(self):
        """
        ContentPolicyViolationError is also a BadRequestError; if the policy names
        ContentPolicyViolationError but leaves its value unset (None) while setting
        BadRequestErrorAllowedFails, resolution must fall through to the
        BadRequestError entry rather than stopping at the first isinstance match.
        """
        policy = {
            "ContentPolicyViolationErrorAllowedFails": None,
            "BadRequestErrorAllowedFails": 3,
        }
        exc = litellm.ContentPolicyViolationError("flagged", "openai", "gpt-4")
        result = _resolve_allowed_fails_from_policy(policy=policy, exception=exc)
        assert result == 3, "must fall through to BadRequestErrorAllowedFails when the more specific field is unset"


class TestDeploymentCallbackOnFailureCooldownTimePrecedence:
    def test_model_info_cooldown_time_used_in_primary_sync_path(self):
        """
        Router.deployment_callback_on_failure (the primary sync failure-callback path,
        as opposed to the fallback path covered by TestFallbackDeploymentCooldown) must
        also honor a model_info.cooldown_time, not just litellm_params.cooldown_time.
        """
        router = _make_router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {"id": "primary", "cooldown_time": 15.0},
                },
            ],
        )

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        kwargs = {
            "exception": exc,
            "litellm_params": {
                "model_info": {"id": "primary", "cooldown_time": 15.0},
            },
        }

        with patch("litellm.router._set_cooldown_deployments") as mock_set_cooldown:
            router.deployment_callback_on_failure(
                kwargs=kwargs,
                completion_response=None,
                start_time=0,
                end_time=1,
            )

            mock_set_cooldown.assert_called_once()
            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 15.0, (
                "model_info.cooldown_time must be honored in the primary sync failure-callback path"
            )

    def test_litellm_params_cooldown_time_still_honored_as_fallback(self):
        """cooldown_time has pre-existing litellm_params support on this primary
        path; it must keep working when model_info doesn't set it."""
        router = _make_router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4", "cooldown_time": 20.0},
                    "model_info": {"id": "primary"},
                },
            ],
        )

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        kwargs = {
            "exception": exc,
            "litellm_params": {
                "model_info": {"id": "primary"},
                "cooldown_time": 20.0,
            },
        }

        with patch("litellm.router._set_cooldown_deployments") as mock_set_cooldown:
            router.deployment_callback_on_failure(
                kwargs=kwargs,
                completion_response=None,
                start_time=0,
                end_time=1,
            )

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 20.0, "litellm_params.cooldown_time must still be honored"


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

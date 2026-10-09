import asyncio
import importlib
import random
import time
from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm import Router
from litellm._internal_context import current_service_target
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.router_utils.cooldown_cache import CooldownCache, CooldownCacheValue
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback
from litellm.router_utils.cooldown_handlers import (
    _get_deployment_cooldown_policy,
    _has_explicit_allowed_fails_policy_for_exception,
    _increment_allowed_fails,
    _is_cooldown_required,
    _resolve_allowed_fails_from_policy,
    _should_cooldown_based_on_deployment_policy,
    _should_cooldown_deployment,
    _should_run_cooldown_logic,
    async_get_cooldown_deployments,
    cast_exception_status_to_int,
    get_cooldown_deployments,
    mark_advisor_orchestration_failure,
    should_cooldown_based_on_allowed_fails_policy,
)
from litellm.router_utils.fallback_event_handlers import (
    _trigger_cooldown_for_failed_deployment,
)
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    get_deployment_failures_for_current_minute,
    increment_deployment_failures_for_current_minute,
    increment_deployment_successes_for_current_minute,
)
from litellm.types.router import (
    AllowedFailsPolicy,
    DeploymentTypedDict,
    LiteLLMParamsTypedDict,
)
from litellm.types.utils import CredentialItem
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


class TestGetDeploymentCooldownPolicy:
    def _make_router(self, deployment_id: str, model_info: dict | None = None):
        router = MagicMock()
        if model_info is None:
            router.get_model_info.return_value = None
        else:
            router.get_model_info.return_value = {"model_info": model_info}
        return router

    def test_deployment_not_found_returns_none_none(self):
        router = self._make_router("dep-1")
        policy, allowed = _get_deployment_cooldown_policy(router, "dep-1")
        assert policy is None
        assert allowed is None

    def test_no_model_info_returns_none_none(self):
        router = MagicMock()
        router.get_model_info.return_value = {"model_info": {}}
        policy, allowed = _get_deployment_cooldown_policy(router, "dep-1")
        assert policy is None
        assert allowed is None

    def test_returns_policy_dict_and_allowed_fails(self):
        router = self._make_router(
            "dep-1",
            {"allowed_fails_policy": {"RateLimitErrorAllowedFails": 2}, "allowed_fails": 3},
        )
        policy, allowed = _get_deployment_cooldown_policy(router, "dep-1")
        assert policy == {"RateLimitErrorAllowedFails": 2}
        assert allowed == 3

    def test_non_dict_policy_treated_as_none(self):
        router = self._make_router("dep-1", {"allowed_fails_policy": "invalid", "allowed_fails": 5})
        policy, allowed = _get_deployment_cooldown_policy(router, "dep-1")
        assert policy is None
        assert allowed == 5

    def test_allowed_fails_only(self):
        router = self._make_router("dep-1", {"allowed_fails": 1})
        policy, allowed = _get_deployment_cooldown_policy(router, "dep-1")
        assert policy is None
        assert allowed == 1


class TestResolveAllowedFailsFromPolicy:
    def test_none_policy_returns_none(self):
        exc = litellm.RateLimitError("429", "openai", "gpt-4")
        assert _resolve_allowed_fails_from_policy(None, exc) is None

    def test_matching_rate_limit_error(self):
        policy = {"RateLimitErrorAllowedFails": 3}
        exc = litellm.RateLimitError("429", "openai", "gpt-4")
        assert _resolve_allowed_fails_from_policy(policy, exc) == 3

    def test_matching_internal_server_error(self):
        policy = {"InternalServerErrorAllowedFails": 5}
        exc = litellm.InternalServerError("500", "openai", "gpt-4")
        assert _resolve_allowed_fails_from_policy(policy, exc) == 5

    def test_matching_service_unavailable_error(self):
        policy = {"ServiceUnavailableErrorAllowedFails": 4}
        exc = litellm.ServiceUnavailableError("503", "openai", "gpt-4")
        assert _resolve_allowed_fails_from_policy(policy, exc) == 4

    def test_matching_bad_gateway_error(self):
        policy = {"BadGatewayErrorAllowedFails": 2}
        exc = litellm.BadGatewayError("502", "openai", "gpt-4")
        assert _resolve_allowed_fails_from_policy(policy, exc) == 2

    def test_matching_not_found_error(self):
        policy = {"NotFoundErrorAllowedFails": 1}
        exc = litellm.NotFoundError("404", "openai", "gpt-4")
        assert _resolve_allowed_fails_from_policy(policy, exc) == 1

    def test_unmatched_exception_returns_none(self):
        policy = {"RateLimitErrorAllowedFails": 3}
        exc = litellm.InternalServerError("500", "openai", "gpt-4")
        assert _resolve_allowed_fails_from_policy(policy, exc) is None

    def test_field_absent_from_policy_returns_none(self):
        policy: dict[str, int] = {}
        exc = litellm.InternalServerError("500", "openai", "gpt-4")
        assert _resolve_allowed_fails_from_policy(policy, exc) is None

    def test_content_policy_violation_not_shadowed_by_bad_request_error(self):
        """ContentPolicyViolationError subclasses BadRequestError, so if
        BadRequestError were checked first, this would incorrectly resolve to
        BadRequestErrorAllowedFails (10) instead of
        ContentPolicyViolationErrorAllowedFails (2)."""
        policy = {"BadRequestErrorAllowedFails": 10, "ContentPolicyViolationErrorAllowedFails": 2}
        exc = litellm.ContentPolicyViolationError("flagged content", "openai", "gpt-4")
        assert _resolve_allowed_fails_from_policy(policy, exc) == 2


class TestShouldCooldownBasedOnDeploymentPolicy:
    def _make_router(self, model_info: dict | None = None):
        router = MagicMock()
        if model_info is None:
            router.get_model_info.return_value = None
        else:
            router.get_model_info.return_value = model_info
        return router

    def test_policy_match_uses_exception_type_as_cache_key_suffix(self):
        policy = {"RateLimitErrorAllowedFails": 0}
        exc = litellm.RateLimitError("429", "openai", "gpt-4")
        router = self._make_router({"litellm_params": {}, "model_info": {}})

        with patch("litellm.router_utils.cooldown_handlers.should_cooldown_based_on_allowed_fails_policy") as mock_sc:
            mock_sc.return_value = True
            result = _should_cooldown_based_on_deployment_policy(
                router, "dep-1", exc, policy, None, is_single_deployment_model_group=False
            )

        assert result is True
        call_kwargs = mock_sc.call_args[1]
        assert call_kwargs["allowed_fails_override"] == 0
        assert call_kwargs["cache_key_suffix"] == "RateLimitError"

    def test_no_policy_match_uses_dep_allowed_fails_and_generic_suffix(self):
        policy: dict[str, int] = {}
        exc = litellm.InternalServerError("500", "openai", "gpt-4")
        router = self._make_router({"litellm_params": {}, "model_info": {}})

        with patch("litellm.router_utils.cooldown_handlers.should_cooldown_based_on_allowed_fails_policy") as mock_sc:
            mock_sc.return_value = False
            result = _should_cooldown_based_on_deployment_policy(
                router, "dep-1", exc, policy, dep_allowed_fails=3, is_single_deployment_model_group=False
            )

        assert result is False
        call_kwargs = mock_sc.call_args[1]
        assert call_kwargs["allowed_fails_override"] == 3
        assert call_kwargs["cache_key_suffix"] == "generic"

    def test_dep_allowed_fails_on_single_deployment_group_does_not_cooldown(self):
        """A generic, deployment-wide allowed_fails predates the per-exception-type
        policy and is a less deliberate opt-in, so on a single-deployment model group
        it must not silently disable the "avoid cooldowns on single deployment model
        groups" safety net."""
        exc = litellm.InternalServerError("500", "openai", "gpt-4")
        router = self._make_router({"litellm_params": {}, "model_info": {}})

        with patch("litellm.router_utils.cooldown_handlers.should_cooldown_based_on_allowed_fails_policy") as mock_sc:
            result = _should_cooldown_based_on_deployment_policy(
                router, "dep-1", exc, None, dep_allowed_fails=3, is_single_deployment_model_group=True
            )

        assert result is False
        mock_sc.assert_not_called()

    def test_named_policy_on_single_deployment_group_still_cools_down(self):
        """Unlike a generic allowed_fails, an explicit per-exception-type policy entry
        is a deliberate opt-in and must still apply on a single-deployment group."""
        policy = {"RateLimitErrorAllowedFails": 0}
        exc = litellm.RateLimitError("429", "openai", "gpt-4")
        router = self._make_router({"litellm_params": {}, "model_info": {}})

        with patch("litellm.router_utils.cooldown_handlers.should_cooldown_based_on_allowed_fails_policy") as mock_sc:
            mock_sc.return_value = True
            result = _should_cooldown_based_on_deployment_policy(
                router, "dep-1", exc, policy, None, is_single_deployment_model_group=True
            )

        assert result is True
        mock_sc.assert_called_once()

    def test_no_policy_and_no_dep_allowed_fails_defers_to_router_level(self):
        """When neither a deployment policy nor a deployment-wide allowed_fails covers
        this exception, defer to router-level behavior instead of forcing an
        immediate cooldown (allowed_fails_override=0 would trip on the first failure
        of any exception type the deployment's config doesn't mention)."""
        exc = litellm.InternalServerError("500", "openai", "gpt-4")
        router = self._make_router({"litellm_params": {}, "model_info": {}})

        with patch("litellm.router_utils.cooldown_handlers.should_cooldown_based_on_allowed_fails_policy") as mock_sc:
            mock_sc.return_value = True
            _should_cooldown_based_on_deployment_policy(
                router, "dep-1", exc, None, None, is_single_deployment_model_group=False
            )

        call_kwargs = mock_sc.call_args[1]
        assert call_kwargs["allowed_fails_override"] is None
        assert call_kwargs["cache_key_suffix"] is None

    def test_partial_policy_without_dep_allowed_fails_defers_for_uncovered_exception(self):
        """A deployment that only sets RateLimitErrorAllowedFails must not force a
        zero-fail threshold on an unrelated TimeoutError; it should defer to
        router-level behavior for exception types its policy doesn't mention."""
        policy = {"RateLimitErrorAllowedFails": 0}
        exc = litellm.Timeout("timed out", "openai", "gpt-4")
        router = self._make_router({"litellm_params": {}, "model_info": {}})

        with patch("litellm.router_utils.cooldown_handlers.should_cooldown_based_on_allowed_fails_policy") as mock_sc:
            mock_sc.return_value = False
            _should_cooldown_based_on_deployment_policy(
                router, "dep-1", exc, policy, dep_allowed_fails=None, is_single_deployment_model_group=False
            )

        call_kwargs = mock_sc.call_args[1]
        assert call_kwargs["allowed_fails_override"] is None
        assert call_kwargs["cache_key_suffix"] is None

    def test_cooldown_time_from_model_info_passed_through(self):
        exc = litellm.RateLimitError("429", "openai", "gpt-4")
        router = self._make_router({"litellm_params": {}, "model_info": {"cooldown_time": 120.0}})

        with patch("litellm.router_utils.cooldown_handlers.should_cooldown_based_on_allowed_fails_policy") as mock_sc:
            mock_sc.return_value = True
            _should_cooldown_based_on_deployment_policy(
                router, "dep-1", exc, None, None, is_single_deployment_model_group=False
            )

        call_kwargs = mock_sc.call_args[1]
        assert call_kwargs["cooldown_time_override"] == 120.0

    def test_cooldown_time_from_litellm_params_used_as_fallback(self):
        """cooldown_time has pre-existing litellm_params support on the primary
        failure path, so it must still be honored here when model_info doesn't
        set it."""
        exc = litellm.RateLimitError("429", "openai", "gpt-4")
        router = self._make_router({"litellm_params": {"cooldown_time": 120.0}, "model_info": {}})

        with patch("litellm.router_utils.cooldown_handlers.should_cooldown_based_on_allowed_fails_policy") as mock_sc:
            mock_sc.return_value = True
            _should_cooldown_based_on_deployment_policy(
                router, "dep-1", exc, None, None, is_single_deployment_model_group=False
            )

        call_kwargs = mock_sc.call_args[1]
        assert call_kwargs["cooldown_time_override"] == 120.0

    def test_cooldown_time_from_model_info_takes_priority_over_litellm_params(self):
        exc = litellm.RateLimitError("429", "openai", "gpt-4")
        router = self._make_router({"litellm_params": {"cooldown_time": 120.0}, "model_info": {"cooldown_time": 15.0}})

        with patch("litellm.router_utils.cooldown_handlers.should_cooldown_based_on_allowed_fails_policy") as mock_sc:
            mock_sc.return_value = True
            _should_cooldown_based_on_deployment_policy(
                router, "dep-1", exc, None, None, is_single_deployment_model_group=False
            )

        call_kwargs = mock_sc.call_args[1]
        assert call_kwargs["cooldown_time_override"] == 15.0

    def test_model_info_none_passes_none_cooldown_time(self):
        exc = litellm.RateLimitError("429", "openai", "gpt-4")
        router = self._make_router(None)

        with patch("litellm.router_utils.cooldown_handlers.should_cooldown_based_on_allowed_fails_policy") as mock_sc:
            mock_sc.return_value = False
            _should_cooldown_based_on_deployment_policy(
                router, "dep-1", exc, None, None, is_single_deployment_model_group=False
            )

        call_kwargs = mock_sc.call_args[1]
        assert call_kwargs["cooldown_time_override"] is None


class TestShouldCooldownBasedOnAllowedFailsPolicy:
    def _make_router(self, cooldown_time: float = 60.0, cache: DualCache | None = None) -> MagicMock:
        router = MagicMock()
        router.cooldown_time = cooldown_time
        router.allowed_fails = 0
        router.allowed_fails_policy = None
        router.get_allowed_fails_from_policy.return_value = None
        router.cache = cache if cache is not None else DualCache(in_memory_cache=InMemoryCache())
        return router

    def test_cooldown_time_override_zero_is_not_falsy(self):
        """cooldown_time_override=0 must be honored; it must not fall through to the router-level value."""
        router = self._make_router(cooldown_time=60.0)
        router.cache = MagicMock()
        router.cache.increment_cache.return_value = 1
        exc = litellm.RateLimitError("429", "openai", "gpt-4")

        should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=router,
            deployment="dep-1",
            original_exception=exc,
            allowed_fails_override=5,
            cooldown_time_override=0.0,
        )

        increment_call = router.cache.increment_cache.call_args
        assert increment_call is not None
        assert increment_call[1]["ttl"] == 0.0, (
            "cooldown_time_override=0 should be used as TTL, not the router-level 60.0"
        )

    def test_fail_counter_is_shared_across_router_instances(self):
        """Two workers (two Router objects over one shared cache) must pool their failures toward allowed_fails."""
        shared_cache = DualCache(in_memory_cache=InMemoryCache())
        workers = (self._make_router(cache=shared_cache), self._make_router(cache=shared_cache))
        exc = litellm.AuthenticationError("401", "openai", "gpt-4")

        results = [
            should_cooldown_based_on_allowed_fails_policy(
                litellm_router_instance=workers[i % 2],
                deployment="dep-1",
                original_exception=exc,
                allowed_fails_override=5,
            )
            for i in range(6)
        ]

        assert results == [False, False, False, False, False, True]
        assert shared_cache.get_cache(key="deployment:dep-1:allowed_fails") == 6

    def test_fleet_wide_count_from_redis_decides_cooldown(self):
        """The Redis (fleet-wide) count decides, even when this process has only seen one failure."""
        redis_cache = MagicMock()
        redis_cache.increment_cache.return_value = 6
        router = self._make_router(cache=DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache))
        exc = litellm.AuthenticationError("401", "openai", "gpt-4")

        result = should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=router,
            deployment="dep-1",
            original_exception=exc,
            allowed_fails_override=5,
        )

        assert result is True
        redis_cache.increment_cache.assert_called_once_with("deployment:dep-1:allowed_fails", 1, ttl=60.0)

    def test_redis_outage_falls_back_to_this_workers_count(self):
        """When every Redis increment fails, the worker's own in-memory count must still cool the deployment down."""
        redis_cache = MagicMock()
        redis_cache.increment_cache.side_effect = ConnectionError("redis down")
        router = self._make_router(cache=DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache))
        exc = litellm.AuthenticationError("401", "openai", "gpt-4")

        results = [
            should_cooldown_based_on_allowed_fails_policy(
                litellm_router_instance=router,
                deployment="dep-1",
                original_exception=exc,
                allowed_fails_override=5,
            )
            for _ in range(6)
        ]

        assert results == [False, False, False, False, False, True]
        assert redis_cache.increment_cache.call_count == 6


class TestRoutingGroupCooldownAlternatives:
    def _router(self, routing_groups=None):
        from litellm import Router

        return Router(
            model_list=[
                {
                    "model_name": "solo-member",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                    "model_info": {"id": "cg-deploy-1"},
                },
                {
                    "model_name": "other-member",
                    "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
                    "model_info": {"id": "cg-deploy-2"},
                },
            ],
            routing_groups=routing_groups,
        )

    def test_group_call_429_cools_down_member_with_alternatives(self):
        from litellm.router_utils.cooldown_handlers import _should_cooldown_deployment

        router = self._router(
            routing_groups=[
                {
                    "group_name": "grouped",
                    "models": ["solo-member", "other-member"],
                    "routing_strategy": "simple-shuffle",
                }
            ]
        )
        assert (
            _should_cooldown_deployment(
                litellm_router_instance=router,
                deployment="cg-deploy-1",
                exception_status=429,
                original_exception=Exception("rate limited"),
                requested_model_group="grouped",
            )
            is True
        )

    def test_direct_member_429_keeps_single_deployment_exemption(self):
        from litellm.router_utils.cooldown_handlers import _should_cooldown_deployment

        router = self._router(
            routing_groups=[
                {
                    "group_name": "grouped",
                    "models": ["solo-member", "other-member"],
                    "routing_strategy": "simple-shuffle",
                }
            ]
        )
        assert (
            _should_cooldown_deployment(
                litellm_router_instance=router,
                deployment="cg-deploy-1",
                exception_status=429,
                original_exception=Exception("rate limited"),
                requested_model_group="solo-member",
            )
            is False
        )

    def test_429_without_request_context_keeps_exemption(self):
        from litellm.router_utils.cooldown_handlers import _should_cooldown_deployment

        router = self._router(routing_groups=None)
        assert (
            _should_cooldown_deployment(
                litellm_router_instance=router,
                deployment="cg-deploy-1",
                exception_status=429,
                original_exception=Exception("rate limited"),
            )
            is False
        )


class TestTeamModelCooldownAlternatives:
    def _router(self, team_deployments: int, blocked_ids: frozenset[str] = frozenset()) -> litellm.Router:
        return litellm.Router(
            model_list=[
                {
                    "model_name": f"model_name_team-1_{i}",
                    "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
                    "model_info": {
                        "id": f"team-deploy-{i}",
                        "team_id": "team-1",
                        "team_public_model_name": "team-gpt-4o-mini",
                        "blocked": f"team-deploy-{i}" in blocked_ids,
                    },
                }
                for i in range(team_deployments)
            ]
        )

    def test_429_on_team_deployment_with_sibling_cools_down(self):
        from litellm.router_utils.cooldown_handlers import _should_cooldown_deployment

        router = self._router(team_deployments=2)
        assert (
            _should_cooldown_deployment(
                litellm_router_instance=router,
                deployment="team-deploy-0",
                exception_status=429,
                original_exception=Exception("rate limited"),
                requested_model_group="team-gpt-4o-mini",
            )
            is True
        )

    def test_429_on_only_team_deployment_keeps_single_deployment_exemption(self):
        from litellm.router_utils.cooldown_handlers import _should_cooldown_deployment

        router = self._router(team_deployments=1)
        assert (
            _should_cooldown_deployment(
                litellm_router_instance=router,
                deployment="team-deploy-0",
                exception_status=429,
                original_exception=Exception("rate limited"),
                requested_model_group="team-gpt-4o-mini",
            )
            is False
        )

    def test_429_with_only_a_blocked_sibling_keeps_single_deployment_exemption(self):
        from litellm.router_utils.cooldown_handlers import _should_cooldown_deployment

        router = self._router(team_deployments=2, blocked_ids=frozenset({"team-deploy-1"}))
        assert (
            _should_cooldown_deployment(
                litellm_router_instance=router,
                deployment="team-deploy-0",
                exception_status=429,
                original_exception=Exception("rate limited"),
                requested_model_group="team-gpt-4o-mini",
            )
            is False
        )


class TestIncrementAllowedFailsServiceTarget:
    def test_fail_counter_bump_declares_the_router_cooldowns_key_family(self):
        """The allowed_fails INCR is cooldown bookkeeping, so its service span must read
        ``redis.incr router_cooldowns`` rather than a bare ``redis.incr``."""
        seen: list[str | None] = []
        cache = MagicMock(spec=DualCache)

        def _increment(**_kwargs):
            seen.append(current_service_target())
            return 2

        cache.increment_cache.side_effect = _increment

        assert _increment_allowed_fails(cache, "deployment:dep-1:fails", ttl=60.0) == 2
        assert seen == ["router_cooldowns"]
        assert current_service_target() is None

    def test_in_memory_fallback_reads_under_the_same_target(self):
        seen: list[str | None] = []
        cache = MagicMock(spec=DualCache)
        cache.increment_cache.side_effect = ConnectionError("redis down")

        def _get(**_kwargs):
            seen.append(current_service_target())
            return 4

        cache.get_cache.side_effect = _get

        assert _increment_allowed_fails(cache, "deployment:dep-1:fails", ttl=60.0) == 4
        assert seen == ["router_cooldowns"]


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function")
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            importlib.reload(litellm.proxy.proxy_server)
    except Exception:
        pass
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)


def _make_router(model_list: list, **kwargs) -> Router:
    return Router(model_list=model_list, **kwargs)


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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

        rl_counter = router.cache.get_cache(key="deployment:primary:allowed_fails:RateLimitError") or 0
        generic_counter = router.cache.get_cache(key="deployment:primary:allowed_fails:generic") or 0

        assert rl_counter == 3, "RateLimitError counter should be 3"
        assert generic_counter == 0, "generic counter must be untouched by RateLimitError increments"

        should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=router,
            deployment="primary",
            original_exception=ise,
            allowed_fails_override=5,
            cache_key_suffix="generic",
        )

        generic_counter_after = router.cache.get_cache(key="deployment:primary:allowed_fails:generic") or 0
        rl_counter_after = router.cache.get_cache(key="deployment:primary:allowed_fails:RateLimitError") or 0

        assert generic_counter_after == 1, "generic counter should now be 1"
        assert rl_counter_after == 3, "RateLimitError counter must remain unchanged after InternalServerError"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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
        cc.in_memory_cache.set_cache(key, expired_value, ttl=600)

        active = cc.get_active_cooldowns(model_ids=[model_id], parent_otel_span=None)

        assert active == [], "Expired cooldown entry must not appear in active cooldowns"
        assert cc.in_memory_cache.get_cache(key) is None, "Expired entry must be evicted from in-memory cache"

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
        cc.in_memory_cache.set_cache(key, active_value, ttl=60)

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
        cc.in_memory_cache.set_cache(key, value, ttl=600)

        before_expiry = cc.in_memory_cache.ttl_dict.get(key)
        assert before_expiry is not None

        cc.get_active_cooldowns(model_ids=[model_id], parent_otel_span=None)

        after_expiry = cc.in_memory_cache.ttl_dict.get(key)
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
        cc.in_memory_cache.set_cache(key, expired_value, ttl=600)

        active = await cc.async_get_active_cooldowns(model_ids=[model_id], parent_otel_span=None)

        assert active == [], "Expired entry must not appear in async active cooldowns"
        assert cc.in_memory_cache.get_cache(key) is None


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestFallbackDeploymentCooldown:
    def test_trigger_cooldown_for_failed_deployment_calls_set_cooldown(self):
        """
        _trigger_cooldown_for_failed_deployment must call set_cooldown_deployments
        with the deployment ID stamped on the exception.
        """
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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
        set_cooldown_deployments when the exception has no failed_deployment_id.
        """
        mock_router = MagicMock()

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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
            patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown,
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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={},
                exception=exc,
            )

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 15.0, "model_info.cooldown_time must take priority"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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

        with patch("litellm.router.set_cooldown_deployments") as mock_set_cooldown:
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

        with patch("litellm.router.set_cooldown_deployments") as mock_set_cooldown:
            router.deployment_callback_on_failure(
                kwargs=kwargs,
                completion_response=None,
                start_time=0,
                end_time=1,
            )

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 20.0, "litellm_params.cooldown_time must still be honored"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestCallerScopedOAuthAuthFailureCooldown:
    @staticmethod
    def _router(
        model_id: str,
        litellm_params: dict[str, object],
        allowed_fails_policy: dict[str, int] | None = None,
    ) -> Router:
        return _make_router(
            model_list=[
                {
                    "model_name": "copilot",
                    "litellm_params": {
                        "model": "microsoft_365_copilot/chat",
                        **litellm_params,
                    },
                    "model_info": {
                        "id": model_id,
                        **({"allowed_fails_policy": allowed_fails_policy} if allowed_fails_policy is not None else {}),
                    },
                }
            ]
        )

    @staticmethod
    def _auth_exception(status: int) -> Exception:
        if status == 401:
            return litellm.AuthenticationError("caller assertion rejected", "microsoft_365_copilot", "copilot")
        return litellm.PermissionDeniedError(
            "caller assertion rejected",
            "microsoft_365_copilot",
            "copilot",
            response=httpx.Response(
                status_code=403,
                request=httpx.Request("GET", "https://litellm.ai"),
            ),
        )

    @staticmethod
    def _callback(router: Router, model_id: str, exception: Exception) -> bool:
        return router.deployment_callback_on_failure(
            kwargs={
                "exception": exception,
                "litellm_params": {"model_info": {"id": model_id}},
            },
            completion_response=None,
            start_time=0,
            end_time=1,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", (401, 403))
    async def test_inline_oauth_caller_auth_failure_does_not_cooldown(self, status: int) -> None:
        model_id: Final = "inline-oauth"
        router: Final = self._router(
            model_id,
            {
                "token_exchange_endpoint": "https://identity.example.com/token",
                "client_id": "copilot-client",
                "client_secret": "copilot-secret",
            },
        )

        result: Final = self._callback(router, model_id, self._auth_exception(status))

        assert result is False
        assert get_deployment_failures_for_current_minute(router, model_id) == 0
        assert get_cooldown_deployments(router, parent_otel_span=None) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", (401, 403))
    async def test_named_oauth_credential_caller_auth_failure_does_not_cooldown(
        self,
        status: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model_id: Final = "named-oauth"
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="copilot-oauth",
                    credential_values={
                        "token_exchange_endpoint": "https://identity.example.com/token",
                        "client_id": "copilot-client",
                        "client_secret": "copilot-secret",
                    },
                    credential_info={"custom_llm_provider": "microsoft_365_copilot"},
                )
            ],
        )
        router: Final = self._router(model_id, {"litellm_credential_name": "copilot-oauth"})

        result: Final = self._callback(router, model_id, self._auth_exception(status))

        assert result is False
        assert get_deployment_failures_for_current_minute(router, model_id) == 0
        assert get_cooldown_deployments(router, parent_otel_span=None) == []

    @pytest.mark.asyncio
    async def test_api_key_deployment_still_cools_down_on_401(self) -> None:
        model_id: Final = "api-key-only"
        router: Final = self._router(model_id, {"api_key": "sk-test"})

        self._callback(
            router,
            model_id,
            litellm.AuthenticationError("upstream rejected the API key", "openai", "gpt-4o-mini"),
        )

        assert get_deployment_failures_for_current_minute(router, model_id) == 1
        assert get_cooldown_deployments(router, parent_otel_span=None) == [model_id]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", (429, 500))
    async def test_oauth_deployment_still_cools_down_on_provider_failures(self, status: int) -> None:
        model_id: Final = f"oauth-provider-{status}"
        router: Final = self._router(
            model_id,
            {
                "token_exchange_endpoint": "https://identity.example.com/token",
                "client_id": "copilot-client",
                "client_secret": "copilot-secret",
            },
            allowed_fails_policy={
                "RateLimitErrorAllowedFails": 0,
                "InternalServerErrorAllowedFails": 0,
            },
        )
        exception: Final[Exception] = (
            litellm.RateLimitError("rate limited", "microsoft_365_copilot", "copilot")
            if status == 429
            else litellm.InternalServerError("provider failed", "microsoft_365_copilot", "copilot")
        )

        self._callback(router, model_id, exception)

        assert get_deployment_failures_for_current_minute(router, model_id) == 1
        assert get_cooldown_deployments(router, parent_otel_span=None) == [model_id]

    @pytest.mark.parametrize("status", (401, 403))
    def test_fallback_oauth_caller_auth_failure_does_not_cooldown(self, status: int) -> None:
        model_id: Final = f"fallback-oauth-{status}"
        router: Final = self._router(
            model_id,
            {
                "token_exchange_endpoint": "https://identity.example.com/token",
                "client_id": "copilot-client",
                "client_secret": "copilot-secret",
            },
        )
        exception: Final = self._auth_exception(status)
        exception.failed_deployment_id = model_id

        _trigger_cooldown_for_failed_deployment(litellm_router=router, kwargs={}, exception=exception)

        assert get_deployment_failures_for_current_minute(router, model_id) == 0
        assert get_cooldown_deployments(router, parent_otel_span=None) == []

    @pytest.mark.asyncio
    async def test_fallback_api_key_auth_failure_still_cools_down(self) -> None:
        model_id: Final = "fallback-api-key"
        router: Final = self._router(model_id, {"api_key": "sk-test"})
        exception: Final = litellm.AuthenticationError("API key rejected", "openai", "gpt-4o-mini")
        exception.failed_deployment_id = model_id

        _trigger_cooldown_for_failed_deployment(litellm_router=router, kwargs={}, exception=exception)

        assert get_deployment_failures_for_current_minute(router, model_id) == 1
        assert get_cooldown_deployments(router, parent_otel_span=None) == [model_id]


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestPerUserCopilotAuthFailureCooldown:
    @staticmethod
    def _router(
        model_id: str,
        litellm_params: Mapping[str, object],
        allowed_fails_policy: Mapping[str, int] | None = None,
    ) -> Router:
        return _make_router(
            model_list=[
                {
                    "model_name": "copilot",
                    "litellm_params": {
                        "model": "github_copilot/gpt-4o",
                        **litellm_params,
                    },
                    "model_info": {
                        "id": model_id,
                        "allowed_fails_policy": dict(allowed_fails_policy)
                        if allowed_fails_policy is not None
                        else None,
                    },
                }
            ]
        )

    @staticmethod
    def _callback(router: Router, model_id: str, exception: Exception) -> bool:
        return router.deployment_callback_on_failure(
            kwargs={
                "exception": exception,
                "litellm_params": {"model_info": {"id": model_id}},
            },
            completion_response=None,
            start_time=0,
            end_time=1,
        )

    @pytest.mark.asyncio
    async def test_per_user_oauth_caller_auth_failure_does_not_cooldown(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from litellm.constants import GITHUB_COPILOT_AUTH_TYPE_KEY, GITHUB_COPILOT_PER_USER_AUTH_TYPE

        model_id: Final = "per-user-oauth"
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="copilot-per-user",
                    credential_values={GITHUB_COPILOT_AUTH_TYPE_KEY: GITHUB_COPILOT_PER_USER_AUTH_TYPE},
                    credential_info={"custom_llm_provider": "github_copilot"},
                )
            ],
        )
        router: Final = self._router(model_id, {"litellm_credential_name": "copilot-per-user"})
        exception: Final = litellm.CallerCredentialAuthenticationError(
            message="reconnect", llm_provider="github_copilot", model=""
        )
        result: Final = self._callback(router, model_id, exception)
        assert result is False
        assert get_cooldown_deployments(router, parent_otel_span=None) == []

    @pytest.mark.asyncio
    async def test_shared_mode_copilot_auth_failure_still_cools_down(self) -> None:
        from litellm.llms.github_copilot.authenticator import Authenticator

        model_id: Final = "shared-mode"
        with (
            patch.object(Authenticator, "get_api_key", return_value="shared-copilot-token"),
            patch.object(Authenticator, "get_api_base", return_value="https://api.githubcopilot.com"),
        ):
            router: Final = self._router(model_id, {"api_key": "sk-shared"})
            self._callback(
                router,
                model_id,
                litellm.AuthenticationError("upstream rejected the shared token", "github_copilot", "gpt-4o"),
            )
        assert get_deployment_failures_for_current_minute(router, model_id) == 1
        assert get_cooldown_deployments(router, parent_otel_span=None) == [model_id]

    @pytest.mark.asyncio
    async def test_fallback_per_user_caller_auth_failure_does_not_cooldown(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from litellm.constants import GITHUB_COPILOT_AUTH_TYPE_KEY, GITHUB_COPILOT_PER_USER_AUTH_TYPE

        model_id: Final = "fallback-per-user"
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="copilot-per-user",
                    credential_values={GITHUB_COPILOT_AUTH_TYPE_KEY: GITHUB_COPILOT_PER_USER_AUTH_TYPE},
                    credential_info={"custom_llm_provider": "github_copilot"},
                )
            ],
        )
        router: Final = self._router(model_id, {"litellm_credential_name": "copilot-per-user"})
        exception: Final = litellm.CallerCredentialAuthenticationError(
            message="reconnect", llm_provider="github_copilot", model=""
        )
        exception.failed_deployment_id = model_id

        _trigger_cooldown_for_failed_deployment(litellm_router=router, kwargs={}, exception=exception)

        assert get_cooldown_deployments(router, parent_otel_span=None) == []

    @pytest.mark.asyncio
    async def test_fallback_shared_mode_auth_failure_still_cools_down(self) -> None:
        from litellm.llms.github_copilot.authenticator import Authenticator

        model_id: Final = "fallback-shared"
        with (
            patch.object(Authenticator, "get_api_key", return_value="shared-copilot-token"),
            patch.object(Authenticator, "get_api_base", return_value="https://api.githubcopilot.com"),
        ):
            router: Final = self._router(model_id, {"api_key": "sk-shared"})
            exception: Final = litellm.AuthenticationError(
                "upstream rejected the shared token", "github_copilot", "gpt-4o"
            )
            exception.failed_deployment_id = model_id

            _trigger_cooldown_for_failed_deployment(litellm_router=router, kwargs={}, exception=exception)

        assert get_cooldown_deployments(router, parent_otel_span=None) == [model_id]

    @pytest.mark.asyncio
    async def test_fallback_rate_limit_on_per_user_deployment_does_not_cooldown(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from litellm.constants import GITHUB_COPILOT_AUTH_TYPE_KEY, GITHUB_COPILOT_PER_USER_AUTH_TYPE

        model_id: Final = "fallback-per-user-429"
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="copilot-per-user",
                    credential_values={GITHUB_COPILOT_AUTH_TYPE_KEY: GITHUB_COPILOT_PER_USER_AUTH_TYPE},
                    credential_info={"custom_llm_provider": "github_copilot"},
                )
            ],
        )
        router: Final = self._router(
            model_id,
            {"litellm_credential_name": "copilot-per-user"},
            allowed_fails_policy={"RateLimitErrorAllowedFails": 0},
        )
        exception: Final = litellm.RateLimitError("copilot 429", "github_copilot", "gpt-4o")
        exception.failed_deployment_id = model_id

        _trigger_cooldown_for_failed_deployment(litellm_router=router, kwargs={}, exception=exception)

        assert get_cooldown_deployments(router, parent_otel_span=None) == []

    @pytest.mark.asyncio
    async def test_fallback_rate_limit_on_shared_deployment_still_cools_down(self) -> None:
        from litellm.llms.github_copilot.authenticator import Authenticator

        model_id: Final = "fallback-shared-429"
        with (
            patch.object(Authenticator, "get_api_key", return_value="shared-copilot-token"),
            patch.object(Authenticator, "get_api_base", return_value="https://api.githubcopilot.com"),
        ):
            router: Final = self._router(
                model_id,
                {"api_key": "sk-shared"},
                allowed_fails_policy={"RateLimitErrorAllowedFails": 0},
            )
            exception: Final = litellm.RateLimitError("copilot 429", "github_copilot", "gpt-4o")
            exception.failed_deployment_id = model_id

            _trigger_cooldown_for_failed_deployment(litellm_router=router, kwargs={}, exception=exception)

        assert get_cooldown_deployments(router, parent_otel_span=None) == [model_id]


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_router_cooldown_event_callback_no_deployment():
    """
    Test the router_cooldown_event_callback function

    Ensures that the router_cooldown_event_callback function does not raise an error when no deployment is found

    In this scenario it should do nothing
    """
    # Mock Router instance
    mock_router = MagicMock()
    mock_router.get_deployment.return_value = None

    await router_cooldown_event_callback(
        litellm_router_instance=mock_router,
        deployment_id="test-deployment",
        exception_status="429",
        cooldown_time=60.0,
    )

    # Assert that the router's get_deployment method was called
    mock_router.get_deployment.assert_called_once_with(model_id="test-deployment")


@pytest.fixture
def testing_litellm_router():
    return Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini"},
                "model_id": "test_deployment",
            },
            {
                "model_name": "test_deployment",
                "litellm_params": {"model": "openai/test_deployment"},
                "model_id": "test_deployment_2",
            },
            {
                "model_name": "test_deployment",
                "litellm_params": {"model": "openai/test_deployment-2"},
                "model_id": "test_deployment_3",
            },
        ]
    )


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_should_run_cooldown_logic(testing_litellm_router):
    testing_litellm_router.disable_cooldowns = True
    # don't run cooldown logic if disable_cooldowns is True
    assert _should_run_cooldown_logic(testing_litellm_router, "test_deployment", 500, Exception("Test")) is False

    # don't cooldown if deployment is None
    testing_litellm_router.disable_cooldowns = False
    assert _should_run_cooldown_logic(testing_litellm_router, None, 500, Exception("Test")) is False

    # don't cooldown if it's a provider default deployment
    testing_litellm_router.provider_default_deployment_ids = ["test_deployment"]
    assert _should_run_cooldown_logic(testing_litellm_router, "test_deployment", 500, Exception("Test")) is False


@pytest.fixture
def single_deployment_router():
    """A router with one deployment whose model_info.id is the lookup-able
    "dep-1" (unlike `testing_litellm_router`'s top-level "model_id" key, which
    is not absorbed into model_info.id and so never resolves via
    get_model_info/get_model_group)."""
    return Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini"},
                "model_info": {"id": "dep-1"},
            },
        ]
    )


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_should_run_cooldown_logic_generic_bad_request_excluded_by_default(
    single_deployment_router,
):
    """A generic BadRequestError/ContentPolicyViolationError (400) is excluded from
    cooldown evaluation by _is_cooldown_required when no allowed_fails_policy is
    configured for that exception type. This is the pre-existing, intentional
    default: a client error is usually not the deployment's fault."""
    exc = litellm.BadRequestError("bad request", "openai", "gpt-5-mini")
    assert _should_run_cooldown_logic(single_deployment_router, "dep-1", 400, exc) is False


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_should_run_cooldown_logic_router_level_policy_does_not_override_bad_request_exclusion(
    single_deployment_router,
):
    """A router-level allowed_fails_policy is a pre-existing, router-wide setting that
    predates the per-deployment override feature, so it must keep its existing behavior
    and stay subject to the generic 4XX exclusion. Only an explicit deployment-level
    policy (an unambiguous per-exception opt-in for that one deployment) overrides it;
    see test_should_run_cooldown_logic_explicit_deployment_level_policy_overrides_content_policy_exclusion."""
    exc = litellm.BadRequestError("bad request", "openai", "gpt-5-mini")
    single_deployment_router.allowed_fails_policy = AllowedFailsPolicy(BadRequestErrorAllowedFails=5)
    assert _should_run_cooldown_logic(single_deployment_router, "dep-1", 400, exc) is False


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_should_run_cooldown_logic_explicit_deployment_level_policy_overrides_content_policy_exclusion(
    single_deployment_router,
):
    """Same as the router-level case, but for a deployment-level allowed_fails_policy
    entry (this PR's per-deployment feature) targeting ContentPolicyViolationError."""
    exc = litellm.ContentPolicyViolationError("flagged content", "openai", "gpt-5-mini")
    deployment_dict = single_deployment_router.get_model_info(id="dep-1")
    deployment_dict["model_info"]["allowed_fails_policy"] = {"ContentPolicyViolationErrorAllowedFails": 0}
    assert _should_run_cooldown_logic(single_deployment_router, "dep-1", 400, exc) is True


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestHasExplicitAllowedFailsPolicyForException:
    def test_no_policy_anywhere_returns_false(self, single_deployment_router):
        exc = litellm.BadRequestError("bad request", "openai", "gpt-5-mini")
        assert _has_explicit_allowed_fails_policy_for_exception(single_deployment_router, "dep-1", exc) is False

    def test_router_level_policy_for_matching_exception_returns_false(self, single_deployment_router):
        """Deliberately scoped to deployment-level only: a router-level policy
        predates this feature and must not be treated as an explicit per-exception
        opt-in for cooldown-gate purposes."""
        exc = litellm.RateLimitError("rate limited", "openai", "gpt-5-mini")
        single_deployment_router.allowed_fails_policy = AllowedFailsPolicy(RateLimitErrorAllowedFails=3)
        assert _has_explicit_allowed_fails_policy_for_exception(single_deployment_router, "dep-1", exc) is False

    def test_router_level_policy_for_different_exception_returns_false(self, single_deployment_router):
        exc = litellm.BadRequestError("bad request", "openai", "gpt-5-mini")
        single_deployment_router.allowed_fails_policy = AllowedFailsPolicy(RateLimitErrorAllowedFails=3)
        assert _has_explicit_allowed_fails_policy_for_exception(single_deployment_router, "dep-1", exc) is False

    def test_deployment_level_policy_for_matching_exception_returns_true(self, single_deployment_router):
        exc = litellm.ContentPolicyViolationError("flagged", "openai", "gpt-5-mini")
        deployment_dict = single_deployment_router.get_model_info(id="dep-1")
        deployment_dict["model_info"]["allowed_fails_policy"] = {"ContentPolicyViolationErrorAllowedFails": 0}
        assert _has_explicit_allowed_fails_policy_for_exception(single_deployment_router, "dep-1", exc) is True

    def test_none_deployment_returns_false(self, single_deployment_router):
        exc = litellm.RateLimitError("rate limited", "openai", "gpt-5-mini")
        single_deployment_router.allowed_fails_policy = AllowedFailsPolicy(RateLimitErrorAllowedFails=3)
        assert _has_explicit_allowed_fails_policy_for_exception(single_deployment_router, None, exc) is False


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_should_cooldown_deployment_rate_limit_error(testing_litellm_router):
    """
    Test the _should_cooldown_deployment function when a rate limit error occurs
    """
    # Test 429 error (rate limit) -> always cooldown a deployment returning 429s
    _exception = litellm.exceptions.RateLimitError("Rate limit", "openai", "gpt-5-mini")
    assert _should_cooldown_deployment(testing_litellm_router, "test_deployment", 429, _exception) is True


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_should_cooldown_deployment_auth_limit_error(testing_litellm_router):
    """
    Test the _should_cooldown_deployment function when an auth limit error occurs
    """
    # Test 401 error (auth limit) -> always cooldown a deployment returning 401s
    _exception = litellm.exceptions.AuthenticationError("Unauthorized", "openai", "gpt-5-mini")
    assert _should_cooldown_deployment(testing_litellm_router, "test_deployment", 401, _exception) is True


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.parametrize("exception_status", (401, 402))
def test_is_cooldown_required_for_account_errors(testing_litellm_router, exception_status):
    assert (
        _is_cooldown_required(
            litellm_router_instance=testing_litellm_router,
            model_id="test_deployment",
            exception_status=exception_status,
        )
        is True
    )


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.parametrize("allowed_fails", (None, 0))
def test_single_deployment_402_does_not_cooldown(
    allowed_fails: int | None,
) -> None:
    assert (
        _should_cooldown_deployment(
            Router(
                model_list=[
                    {
                        "model_name": "gpt-5-mini",
                        "litellm_params": {"model": "gpt-5-mini"},
                        "model_info": {"id": "dep-1"},
                    },
                ],
                allowed_fails=allowed_fails,
            ),
            "dep-1",
            402,
            litellm.PaymentRequiredError(
                message="Insufficient credits",
                model="gpt-5-mini",
                llm_provider="openai",
            ),
        )
        is False
    )


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_single_deployment_402_respects_router_allowed_fails_policy() -> None:
    assert (
        _should_cooldown_deployment(
            Router(
                model_list=[
                    {
                        "model_name": "gpt-5-mini",
                        "litellm_params": {"model": "gpt-5-mini"},
                        "model_info": {"id": "dep-1"},
                    },
                ],
                allowed_fails_policy=AllowedFailsPolicy(BadRequestErrorAllowedFails=0),
            ),
            "dep-1",
            402,
            litellm.PaymentRequiredError(
                message="Insufficient credits",
                model="gpt-5-mini",
                llm_provider="openai",
            ),
        )
        is True
    )


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_single_deployment_402_respects_deployment_allowed_fails_policy() -> None:
    assert (
        _should_cooldown_deployment(
            Router(
                model_list=[
                    {
                        "model_name": "gpt-5-mini",
                        "litellm_params": {"model": "gpt-5-mini"},
                        "model_info": {
                            "id": "dep-1",
                            "allowed_fails_policy": {"BadRequestErrorAllowedFails": 0},
                        },
                    },
                ],
            ),
            "dep-1",
            402,
            litellm.PaymentRequiredError(
                message="Insufficient credits",
                model="gpt-5-mini",
                llm_provider="openai",
            ),
        )
        is True
    )


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_multi_deployment_402_cools_down(testing_litellm_router: Router) -> None:
    assert (
        _should_cooldown_deployment(
            testing_litellm_router,
            "test_deployment",
            402,
            litellm.PaymentRequiredError(
                message="Insufficient credits",
                model="gpt-5-mini",
                llm_provider="openai",
            ),
        )
        is True
    )


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_should_cooldown_deployment(testing_litellm_router):
    """
    Cooldown a deployment if it fails 60% of requests in 1 minute - DEFAULT threshold is 50%
    """
    import logging

    from litellm._logging import verbose_router_logger

    verbose_router_logger.setLevel(logging.DEBUG)

    # Test 429 error (rate limit) -> always cooldown a deployment returning 429s
    _exception = litellm.exceptions.RateLimitError("Rate limit", "openai", "gpt-5-mini")
    assert _should_cooldown_deployment(testing_litellm_router, "test_deployment", 429, _exception) is True

    available_deployment = testing_litellm_router.get_available_deployment(model="test_deployment")
    print("available_deployment", available_deployment)
    assert available_deployment is not None

    deployment_id = available_deployment["model_info"]["id"]
    print("deployment_id", deployment_id)

    # set current success for deployment to 40
    for _ in range(40):
        increment_deployment_successes_for_current_minute(
            litellm_router_instance=testing_litellm_router, deployment_id=deployment_id
        )

    # now we fail 40 requests in a row
    tasks = []
    for _ in range(41):
        tasks.append(
            testing_litellm_router.acompletion(
                model=deployment_id,
                messages=[{"role": "user", "content": "Hello, world!"}],
                max_tokens=100,
                mock_response="litellm.InternalServerError",
            )
        )
    try:
        await asyncio.gather(*tasks)
    except Exception:
        pass

    await asyncio.sleep(1)

    # expect this to fail since it's now 51% of requests are failing
    assert _should_cooldown_deployment(testing_litellm_router, deployment_id, 500, Exception("Test")) is True


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_should_cooldown_deployment_allowed_fails_set_on_router():
    """
    Test the _should_cooldown_deployment function when Router.allowed_fails is set
    """
    # Create a Router instance with a test deployment
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini"},
                "model_id": "test_deployment",
            },
        ]
    )

    # Set up allowed_fails for the test deployment
    router.allowed_fails = 100

    # should not cooldown when fails are below the allowed limit
    for _ in range(100):
        assert _should_cooldown_deployment(router, "test_deployment", 500, Exception("Test")) is False

    assert _should_cooldown_deployment(router, "test_deployment", 500, Exception("Test")) is True


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_increment_deployment_successes_for_current_minute_does_not_write_to_redis(
    testing_litellm_router,
):
    """
    Ensure tracking deployment metrics does not write to redis

    Important - If it writes to redis on every request it will seriously impact performance / latency
    """
    from litellm.caching.dual_cache import DualCache
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.caching.redis_cache import RedisCache
    from litellm.router_utils.router_callbacks.track_deployment_metrics import (
        increment_deployment_successes_for_current_minute,
    )

    # Mock RedisCache
    mock_redis_cache = MagicMock(spec=RedisCache)

    testing_litellm_router.cache = DualCache(redis_cache=mock_redis_cache, in_memory_cache=InMemoryCache())

    # Call the function we're testing
    increment_deployment_successes_for_current_minute(
        litellm_router_instance=testing_litellm_router, deployment_id="test_deployment"
    )

    increment_deployment_failures_for_current_minute(
        litellm_router_instance=testing_litellm_router, deployment_id="test_deployment"
    )

    time.sleep(1)

    # Assert that no methods were called on the mock_redis_cache
    assert not mock_redis_cache.method_calls, "RedisCache methods should not be called"

    print(
        "in memory cache values=",
        testing_litellm_router.cache.in_memory_cache.cache_dict,
    )
    assert testing_litellm_router.cache.in_memory_cache.get_cache("test_deployment:successes") is not None


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_cast_exception_status_to_int():
    assert cast_exception_status_to_int(200) == 200
    assert cast_exception_status_to_int("404") == 404
    assert cast_exception_status_to_int("invalid") == 500


@pytest.fixture
def router():
    return Router(
        model_list=[
            {
                "model_name": "gpt-5.5",
                "litellm_params": {"model": "gpt-5.5"},
                "model_info": {
                    "id": "gpt-4--0",
                },
            }
        ]
    )


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@patch("litellm.router_utils.cooldown_handlers.get_deployment_successes_for_current_minute")
@patch("litellm.router_utils.cooldown_handlers.get_deployment_failures_for_current_minute")
def test_should_cooldown_high_traffic_all_fails(mock_failures, mock_successes, router):
    # Simulate 10 failures, 0 successes
    from litellm.constants import SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD

    mock_failures.return_value = SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD + 1
    mock_successes.return_value = 0

    should_cooldown = _should_cooldown_deployment(
        litellm_router_instance=router,
        deployment="gpt-4--0",
        exception_status=500,
        original_exception=Exception("Test error"),
    )

    assert should_cooldown is True, "Should cooldown when all requests fail with sufficient traffic"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@patch("litellm.router_utils.cooldown_handlers.get_deployment_successes_for_current_minute")
@patch("litellm.router_utils.cooldown_handlers.get_deployment_failures_for_current_minute")
def test_no_cooldown_low_traffic(mock_failures, mock_successes, router):
    # Simulate 3 failures (below MIN_TRAFFIC_THRESHOLD)
    mock_failures.return_value = 3
    mock_successes.return_value = 0

    should_cooldown = _should_cooldown_deployment(
        litellm_router_instance=router,
        deployment="gpt-4--0",
        exception_status=500,
        original_exception=Exception("Test error"),
    )

    assert should_cooldown is False, "Should not cooldown when traffic is below threshold"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@patch("litellm.router_utils.cooldown_handlers.get_deployment_successes_for_current_minute")
@patch("litellm.router_utils.cooldown_handlers.get_deployment_failures_for_current_minute")
def test_cooldown_rate_limit(mock_failures, mock_successes, router):
    """
    Don't cooldown single deployment models, for anything besides traffic
    """
    mock_failures.return_value = 1
    mock_successes.return_value = 0

    should_cooldown = _should_cooldown_deployment(
        litellm_router_instance=router,
        deployment="gpt-4--0",
        exception_status=429,  # Rate limit error
        original_exception=Exception("Rate limit exceeded"),
    )

    assert should_cooldown is False, "Should not cooldown on rate limit error for single deployment models"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@patch("litellm.router_utils.cooldown_handlers.get_deployment_successes_for_current_minute")
@patch("litellm.router_utils.cooldown_handlers.get_deployment_failures_for_current_minute")
def test_mixed_success_failure(mock_failures, mock_successes, router):
    # Simulate 3 failures, 7 successes
    mock_failures.return_value = 3
    mock_successes.return_value = 7

    should_cooldown = _should_cooldown_deployment(
        litellm_router_instance=router,
        deployment="gpt-4--0",
        exception_status=500,
        original_exception=Exception("Test error"),
    )

    assert should_cooldown is False, "Should not cooldown when failure rate is below threshold"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_is_cooldown_required_empty_string_exception_status(testing_litellm_router):
    """
    Test that _is_cooldown_required returns False when exception_status is an empty string
    """
    result = _is_cooldown_required(
        litellm_router_instance=testing_litellm_router,
        model_id="test_deployment",
        exception_status="",
    )

    assert result is False, "Should not require cooldown when exception_status is empty string"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_should_cooldown_deployment_minimum_request_threshold(testing_litellm_router):
    """
    Test that error rate cooldown does NOT trigger on first failure.

    Fixes GitHub issue #17418: Error Rate Cooldown Triggers on First Failed Request

    The problem: With DEFAULT_FAILURE_THRESHOLD_PERCENT=0.5 (50%), a deployment
    gets cooled down after just 1 failed request because 1/1 = 100% > 50%.

    The fix: Add a minimum request threshold (DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS)
    before applying error rate cooldown.
    """
    from litellm.constants import DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS

    # Get a deployment that's not a single-deployment model group
    # (test_deployment_2 and test_deployment_3 are both for "test_deployment" model)
    available_deployment = testing_litellm_router.get_available_deployment(model="test_deployment")
    assert available_deployment is not None
    deployment_id = available_deployment["model_info"]["id"]

    # Simulate only 1 failure (below minimum threshold)
    # This should NOT trigger cooldown even though 100% > 50%
    increment_deployment_failures_for_current_minute(
        litellm_router_instance=testing_litellm_router, deployment_id=deployment_id
    )

    _exception = litellm.exceptions.InternalServerError("Internal error", "openai", "gpt-5-mini")

    # With only 1 request, should NOT cooldown (below minimum threshold)
    should_cooldown = _should_cooldown_deployment(testing_litellm_router, deployment_id, 500, _exception)
    assert should_cooldown is False, (
        f"Should NOT cooldown with only 1 failed request (below minimum threshold of {DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS})"
    )

    # Now add more failures to reach the minimum threshold
    for _ in range(DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS - 1):
        increment_deployment_failures_for_current_minute(
            litellm_router_instance=testing_litellm_router, deployment_id=deployment_id
        )

    # Now with enough requests (all failures), it SHOULD trigger cooldown
    should_cooldown = _should_cooldown_deployment(testing_litellm_router, deployment_id, 500, _exception)
    assert should_cooldown is True, (
        f"Should cooldown when we have {DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS} failed requests (100% failure rate)"
    )


@pytest.mark.asyncio
async def test_dynamic_cooldowns():
    """
    Assert kwargs for completion/embedding have 'cooldown_time' as a litellm_param
    """
    # litellm.set_verbose = True
    tmp_mock = MagicMock()

    litellm.failure_callback = [tmp_mock]

    router = Router(
        model_list=[
            {
                "model_name": "my-fake-model",
                "litellm_params": {
                    "model": "openai/gpt-1",
                    "api_key": "my-key",
                    "mock_response": Exception("this is an error"),
                },
            }
        ],
        cooldown_time=60,
    )

    try:
        _ = router.completion(
            model="my-fake-model",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            cooldown_time=0,
            num_retries=0,
        )
    except Exception:
        pass

    tmp_mock.assert_called_once()

    print(tmp_mock.call_count)

    assert "cooldown_time" in tmp_mock.call_args[0][0]["litellm_params"]
    assert tmp_mock.call_args[0][0]["litellm_params"]["cooldown_time"] == 0


@pytest.mark.asyncio
async def test_cooldown_time_zero_uses_zero_not_default():
    """
    Test that when cooldown_time=0 is passed, it uses 0 instead of the default cooldown time
    AND that the early exit logic prevents cooldown entirely
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "cooldown_time": 0,
                },
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
            },
        ],
        cooldown_time=300,
        num_retries=0,
    )

    with patch.object(router.cooldown_cache, "add_deployment_to_cooldown") as mock_add_cooldown:
        try:
            await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response="litellm.RateLimitError",
            )
        except litellm.RateLimitError:
            pass

        mock_add_cooldown.assert_not_called()

    cooldown_list = await async_get_cooldown_deployments(litellm_router_instance=router, parent_otel_span=None)
    assert len(cooldown_list) == 0

    healthy_deployments, _ = await router._async_get_healthy_deployments(model="gpt-3.5-turbo", parent_otel_span=None)
    assert len(healthy_deployments) == 1


def test_should_run_cooldown_logic_early_exit_on_zero_cooldown():
    """
    Unit test for _should_run_cooldown_logic to verify early exit when time_to_cooldown is 0
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
                "model_info": {
                    "id": "test-deployment-id",
                },
            }
        ],
        cooldown_time=300,
    )

    result = _should_run_cooldown_logic(
        litellm_router_instance=router,
        deployment="test-deployment-id",
        exception_status=429,
        original_exception=litellm.RateLimitError("test error", "openai", "gpt-3.5-turbo"),
        time_to_cooldown=0.0,
    )
    assert result is False, "Should not run cooldown logic when time_to_cooldown is 0"

    result = _should_run_cooldown_logic(
        litellm_router_instance=router,
        deployment="test-deployment-id",
        exception_status=429,
        original_exception=litellm.RateLimitError("test error", "openai", "gpt-3.5-turbo"),
        time_to_cooldown=1e-10,
    )
    assert result is False, "Should not run cooldown logic when time_to_cooldown is effectively 0"

    result = _should_run_cooldown_logic(
        litellm_router_instance=router,
        deployment="test-deployment-id",
        exception_status=429,
        original_exception=litellm.RateLimitError("test error", "openai", "gpt-3.5-turbo"),
        time_to_cooldown=None,
    )
    assert result is True, "Should run cooldown logic when time_to_cooldown is None"

    result = _should_run_cooldown_logic(
        litellm_router_instance=router,
        deployment="test-deployment-id",
        exception_status=429,
        original_exception=litellm.RateLimitError("test error", "openai", "gpt-3.5-turbo"),
        time_to_cooldown=60.0,
    )
    assert result is True, "Should run cooldown logic when time_to_cooldown is positive"


@pytest.mark.parametrize("num_deployments", [1, 2])
def test_single_deployment_no_cooldowns(num_deployments: int):
    """
    Do not cooldown on single deployment.

    Cooldown on multiple deployments.
    """
    model_list = []
    for i in range(num_deployments):
        model = DeploymentTypedDict(
            model_name="gpt-3.5-turbo",
            litellm_params=LiteLLMParamsTypedDict(
                model="gpt-3.5-turbo",
            ),
        )
        model_list.append(model)

    router = Router(model_list=model_list, num_retries=0)

    with patch.object(router.cooldown_cache, "add_deployment_to_cooldown", new=MagicMock()) as mock_client:
        try:
            router.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response="litellm.RateLimitError",
            )
        except litellm.RateLimitError:
            pass

        if num_deployments == 1:
            mock_client.assert_not_called()
        else:
            mock_client.assert_called_once()


@pytest.mark.asyncio
async def test_single_deployment_no_cooldowns_test_prod():
    """
    Do not cooldown on single deployment.

    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-5",
                },
            },
            {
                "model_name": "gpt-12",
                "litellm_params": {
                    "model": "openai/gpt-12",
                },
            },
        ],
        num_retries=0,
    )

    with patch.object(router.cooldown_cache, "add_deployment_to_cooldown", new=MagicMock()) as mock_client:
        try:
            await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response="litellm.RateLimitError",
            )
        except litellm.RateLimitError:
            pass

        await asyncio.sleep(2)

        mock_client.assert_not_called()


@pytest.mark.asyncio()
async def test_high_traffic_cooldowns_all_healthy_deployments():
    """
    PROD TEST - 3 deployments, each deployment fails 25% requests. Assert that no deployments get put into cooldown
    """

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-2",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-3",
                },
            },
        ],
        set_verbose=True,
        debug_level="DEBUG",
    )

    all_deployment_ids = router.get_model_ids()

    from collections import defaultdict

    # Create a defaultdict to track successes and failures for each model ID
    model_stats = defaultdict(lambda: {"successes": 0, "failures": 0})

    litellm.set_verbose = True
    for _ in range(100):
        try:
            model_id = random.choice(all_deployment_ids)

            num_successes = model_stats[model_id]["successes"]
            num_failures = model_stats[model_id]["failures"]
            total_requests = num_failures + num_successes
            if total_requests > 0:
                print(
                    "num failures= ",
                    num_failures,
                    "num successes= ",
                    num_successes,
                    "num_failures/total = ",
                    num_failures / total_requests,
                )

            if total_requests == 0:
                mock_response = "hi"
            elif num_failures / total_requests <= 0.25:
                # Randomly decide between fail and succeed
                if random.random() < 0.5:
                    mock_response = "hi"
                else:
                    mock_response = "litellm.InternalServerError"
            else:
                mock_response = "hi"

            await router.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response=mock_response,
            )
            model_stats[model_id]["successes"] += 1

            await asyncio.sleep(0.0001)
        except litellm.InternalServerError:
            model_stats[model_id]["failures"] += 1
            pass
        except Exception as e:
            print("Failed test model stats=", model_stats)
            raise e
    print("model_stats: ", model_stats)

    cooldown_list = await async_get_cooldown_deployments(litellm_router_instance=router, parent_otel_span=None)
    assert len(cooldown_list) == 0


@pytest.mark.asyncio()
async def test_high_traffic_cooldowns_one_bad_deployment():
    """
    PROD TEST - 3 deployments, 1- deployment fails 6/10 requests, assert that bad deployment gets put into cooldown
    """

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-2",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-3",
                },
            },
        ],
        set_verbose=True,
        debug_level="DEBUG",
    )

    all_deployment_ids = router.get_model_ids()

    from collections import defaultdict

    # Create a defaultdict to track successes and failures for each model ID
    model_stats = defaultdict(lambda: {"successes": 0, "failures": 0})
    bad_deployment_id = random.choice(all_deployment_ids)
    litellm.set_verbose = True
    for _ in range(100):
        try:
            model_id = random.choice(all_deployment_ids)

            num_successes = model_stats[model_id]["successes"]
            num_failures = model_stats[model_id]["failures"]
            total_requests = num_failures + num_successes
            if total_requests > 0:
                print(
                    "num failures= ",
                    num_failures,
                    "num successes= ",
                    num_successes,
                    "num_failures/total = ",
                    num_failures / total_requests,
                )

            if total_requests == 0:
                mock_response = "hi"
            elif bad_deployment_id == model_id:
                if num_failures / total_requests <= 0.6:
                    mock_response = "litellm.InternalServerError"

            elif num_failures / total_requests <= 0.25:
                # Randomly decide between fail and succeed
                if random.random() < 0.5:
                    mock_response = "hi"
                else:
                    mock_response = "litellm.InternalServerError"
            else:
                mock_response = "hi"

            await router.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response=mock_response,
            )
            model_stats[model_id]["successes"] += 1

            await asyncio.sleep(0.0001)
        except litellm.InternalServerError:
            model_stats[model_id]["failures"] += 1
            pass
        except Exception as e:
            print("Failed test model stats=", model_stats)
            raise e
    print("model_stats: ", model_stats)

    cooldown_list = await async_get_cooldown_deployments(litellm_router_instance=router, parent_otel_span=None)
    assert len(cooldown_list) == 1


@pytest.mark.asyncio()
async def test_high_traffic_cooldowns_one_rate_limited_deployment():
    """
    PROD TEST - 3 deployments, 1- deployment fails 6/10 requests, assert that bad deployment gets put into cooldown
    """

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-2",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://api.openai.com-3",
                },
            },
        ],
        set_verbose=True,
        debug_level="DEBUG",
    )

    all_deployment_ids = router.get_model_ids()

    from collections import defaultdict

    # Create a defaultdict to track successes and failures for each model ID
    model_stats = defaultdict(lambda: {"successes": 0, "failures": 0})
    bad_deployment_id = random.choice(all_deployment_ids)
    litellm.set_verbose = True
    for _ in range(100):
        try:
            model_id = random.choice(all_deployment_ids)

            num_successes = model_stats[model_id]["successes"]
            num_failures = model_stats[model_id]["failures"]
            total_requests = num_failures + num_successes
            if total_requests > 0:
                print(
                    "num failures= ",
                    num_failures,
                    "num successes= ",
                    num_successes,
                    "num_failures/total = ",
                    num_failures / total_requests,
                )

            if total_requests == 0:
                mock_response = "hi"
            elif bad_deployment_id == model_id:
                if num_failures / total_requests <= 0.6:
                    mock_response = "litellm.RateLimitError"

            elif num_failures / total_requests <= 0.25:
                # Randomly decide between fail and succeed
                if random.random() < 0.5:
                    mock_response = "hi"
                else:
                    mock_response = "litellm.InternalServerError"
            else:
                mock_response = "hi"

            await router.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response=mock_response,
            )
            model_stats[model_id]["successes"] += 1

            await asyncio.sleep(0.0001)
        except litellm.InternalServerError:
            model_stats[model_id]["failures"] += 1
            pass
        except litellm.RateLimitError:
            model_stats[bad_deployment_id]["failures"] += 1
            pass
        except Exception as e:
            print("Failed test model stats=", model_stats)
            raise e
    print("model_stats: ", model_stats)

    cooldown_list = await async_get_cooldown_deployments(litellm_router_instance=router, parent_otel_span=None)
    assert len(cooldown_list) == 1


def test_router_fallbacks_with_cooldowns_and_model_id():
    """
    Test that after a RateLimitError, the router can still route subsequent
    requests to the same deployment (i.e., mock errors don't permanently
    cool down the deployment).
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {
                    "id": "123",
                },
            }
        ],
        routing_strategy="usage-based-routing-v2",
    )

    try:
        router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="litellm.RateLimitError",
        )
    except litellm.RateLimitError:
        pass

    response = router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
    )
    assert response is not None


@pytest.mark.asyncio()
async def test_router_fallbacks_with_cooldowns_and_dynamic_credentials():
    """
    A 429 answered to a caller-supplied credential cools down none of the shared deployments,
    so the next credential still reaches them, while a 429 owned by a shared deployment does
    """
    from litellm.router_utils.cooldown_handlers import async_get_cooldown_deployments

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {"id": deployment_id},
            }
            for deployment_id in ("123", "456")
        ],
        num_retries=0,
    )
    messages = [{"role": "user", "content": "hi"}]

    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(
            model="gpt-3.5-turbo", messages=messages, api_key="my-bad-key-1", mock_response="litellm.RateLimitError"
        )
    await asyncio.sleep(1)
    assert await async_get_cooldown_deployments(litellm_router_instance=router, parent_otel_span=None) == []

    response = await router.acompletion(
        model="gpt-3.5-turbo", messages=messages, api_key="my-good-key-2", mock_response="served with credential 2"
    )
    assert response.choices[0].message.content == "served with credential 2"

    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(model="gpt-3.5-turbo", messages=messages, mock_response="litellm.RateLimitError")
    await asyncio.sleep(1)
    cooled_down = await async_get_cooldown_deployments(litellm_router_instance=router, parent_otel_span=None)
    assert len(cooled_down) == 1 and cooled_down[0] in {"123", "456"}


@pytest.mark.parametrize(
    "exception,status",
    [
        (
            litellm.CallerCredentialAuthenticationError(message="reconnect", llm_provider="github_copilot", model=""),
            401,
        ),
        (litellm.CallerCredentialRateLimitError(message="slow down", llm_provider="github_copilot", model=""), 429),
    ],
)
def test_caller_credential_errors_never_cool_down_the_shared_deployment(single_deployment_router, exception, status):
    """A per-user credential failure is scoped to one caller's stored token; cooling down the
    shared deployment would punish every other user on the group."""
    assert _should_run_cooldown_logic(single_deployment_router, "dep-1", status, exception) is False


def test_per_user_session_upstream_errors_never_cool_down_the_shared_deployment(single_deployment_router):
    """A 429/401 from the caller's own Copilot seat is scoped to that user; it must
    not cool down the shared deployment for every other caller."""
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    session_kwargs = {
        "github_copilot_user_session": GithubCopilotUserSession(
            token="copilot-token", api_base="https://api.githubcopilot.com"
        )
    }
    for exc, status in (
        (litellm.RateLimitError("copilot 429", "github_copilot", "gpt-4o"), 429),
        (litellm.AuthenticationError("copilot 401", "github_copilot", "gpt-4o"), 401),
        (litellm.InternalServerError("copilot 500", "github_copilot", "gpt-4o"), 500),
    ):
        assert (
            _should_run_cooldown_logic(single_deployment_router, "dep-1", status, exc, request_kwargs=session_kwargs)
            is False
        )


def test_shared_mode_upstream_429_still_cools_down_the_deployment(single_deployment_router):
    """Regression: without the session marker the same 429 is a deployment-health
    signal and cools down exactly as before."""
    exc = litellm.RateLimitError("copilot 429", "github_copilot", "gpt-4o")
    assert (
        _should_run_cooldown_logic(
            single_deployment_router, "dep-1", 429, exc, request_kwargs={"model": "github_copilot/gpt-4o"}
        )
        is True
    )
    assert _should_run_cooldown_logic(single_deployment_router, "dep-1", 429, exc) is True

from unittest.mock import MagicMock, patch

import litellm
from litellm.router_utils.cooldown_handlers import (
    _get_deployment_cooldown_policy,
    _resolve_allowed_fails_from_policy,
    _should_cooldown_based_on_deployment_policy,
    should_cooldown_based_on_allowed_fails_policy,
)


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
    def _make_router(self, cooldown_time: float = 60.0) -> MagicMock:
        router = MagicMock()
        router.cooldown_time = cooldown_time
        router.allowed_fails = 0
        router.allowed_fails_policy = None
        router.get_allowed_fails_from_policy.return_value = None
        router.failed_calls.get_cache.return_value = None
        return router

    def test_cooldown_time_override_zero_is_not_falsy(self):
        """cooldown_time_override=0 must be honored; it must not fall through to the router-level value."""
        router = self._make_router(cooldown_time=60.0)
        exc = litellm.RateLimitError("429", "openai", "gpt-4")

        should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=router,
            deployment="dep-1",
            original_exception=exc,
            allowed_fails_override=5,
            cooldown_time_override=0.0,
        )

        set_cache_call = router.failed_calls.set_cache.call_args
        assert set_cache_call is not None
        assert set_cache_call[1]["ttl"] == 0.0, (
            "cooldown_time_override=0 should be used as TTL, not the router-level 60.0"
        )


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

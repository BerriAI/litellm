"""
Regression tests for Router._pre_call_checks() performance optimization.

Background:
    _pre_call_checks() runs on EVERY request to filter deployments based on
    context window size, rate limits, region constraints, and supported parameters.

Optimization:
    Changed from copy.deepcopy(healthy_deployments) to list(healthy_deployments).
    This is ~1400x faster while maintaining correctness because the function only
    removes items from the list, never modifies the deployment objects themselves.

Critical Requirement:
    The input healthy_deployments list must NEVER be mutated. Callers depend on
    this for retries, fallbacks, and logging.
"""

import copy
from unittest.mock import patch

import pytest

import litellm
from litellm import Router


class TestPreCallChecksOptimization:
    """
    Verify that using list() instead of deepcopy() doesn't break behavior.

    If these tests fail, the optimization should be reverted.
    """

    def test_no_mutation_of_input_list(self):
        """
        Verify the input list is never modified by _pre_call_checks.

        The function uses list() instead of deepcopy for performance.
        This is safe because it only filters items, never modifies them.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-5-mini",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test"},
                    "model_info": {"id": "test-1"},
                },
                {
                    "model_name": "gpt-5-mini",
                    "litellm_params": {"model": "gpt-5.5", "api_key": "sk-test2"},
                    "model_info": {"id": "test-2"},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )

        deployments = router.get_model_list(model_name="gpt-5-mini")
        assert deployments is not None

        # Capture the original state
        original_length = len(deployments)
        original_deployment_ids = [id(d) for d in deployments]
        original_litellm_params_ids = [id(d["litellm_params"]) for d in deployments]
        snapshot = copy.deepcopy(deployments)

        # Call the function under test
        router._pre_call_checks(
            model="gpt-5-mini",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "test"}],
        )

        # Verify nothing changed:
        # 1. Same number of items
        assert len(deployments) == original_length, "List length changed!"
        # 2. Same deployment objects (not replaced with copies)
        assert [id(d) for d in deployments] == original_deployment_ids, "Deployment dicts replaced!"
        # 3. Same nested objects (not replaced with copies)
        assert [id(d["litellm_params"]) for d in deployments] == original_litellm_params_ids, "Nested dicts replaced!"
        # 4. Same values (catches any mutation)
        assert deployments == snapshot, "Values were mutated!"

    def test_filtering_still_works(self):
        """
        Verify that filtering works correctly while preserving the original list.

        Scenario: Send a message too long for one deployment but fine for another.
        Expected: Filtered result excludes the small deployment, but original list is unchanged.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "test",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test"},
                    "model_info": {"id": "small", "max_input_tokens": 50},
                },
                {
                    "model_name": "test",
                    "litellm_params": {"model": "gpt-5.5", "api_key": "sk-test"},
                    "model_info": {"id": "large", "max_input_tokens": 10000},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )

        deployments = router.get_model_list(model_name="test")
        assert deployments is not None

        # Save references to the original deployment objects
        original_small_deployment = deployments[0]  # max_input_tokens=50
        original_large_deployment = deployments[1]  # max_input_tokens=10000

        # Send a long message (100 words) that exceeds 50 tokens but fits in 10000 tokens
        filtered = router._pre_call_checks(
            model="test",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": " ".join(["word"] * 100)}],
        )

        # Verify the filtered result only contains the large deployment
        assert len(filtered) == 1, f"Expected 1 deployment after filtering, got {len(filtered)}"
        assert filtered[0]["model_info"]["id"] == "large", "Wrong deployment kept after filtering"

        # Verify the original list still has both deployments
        assert len(deployments) == 2, f"Original list was modified! Expected 2, got {len(deployments)}"
        assert deployments[0] is original_small_deployment, "First deployment object replaced!"
        assert deployments[1] is original_large_deployment, "Second deployment object replaced!"
        assert deployments[0].get("model_info", {}).get("id") == "small", "First deployment ID changed!"
        assert deployments[1].get("model_info", {}).get("id") == "large", "Second deployment ID changed!"


class TestPreCallChecksLazyGuarding:
    """
    Verify that each pre-call check is only performed when the deployment
    has the relevant configuration variable set.

    These tests ensure expensive operations (tokenization, model-info
    lookups, cache reads) are skipped for deployments that don't need them.
    """

    @pytest.fixture()
    def router_no_limits(self):
        """Router with two deployments, neither has max_input_tokens or rpm."""
        return Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test"},
                    "model_info": {"id": "dep-1"},
                },
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5.5", "api_key": "sk-test"},
                    "model_info": {"id": "dep-2"},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )

    @pytest.fixture()
    def router_with_context_limit(self):
        """Router where one deployment has max_input_tokens, the other doesn't."""
        return Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test"},
                    "model_info": {"id": "no-limit"},
                },
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5.5", "api_key": "sk-test"},
                    "model_info": {"id": "has-limit", "max_input_tokens": 500000},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )

    @pytest.fixture()
    def router_with_rpm(self):
        """Router where one deployment has rpm, the other doesn't."""
        return Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test"},
                    "model_info": {"id": "no-rpm"},
                },
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "gpt-5.5",
                        "api_key": "sk-test",
                        "rpm": 100,
                    },
                    "model_info": {"id": "has-rpm"},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )

    def test_no_tokenization_when_no_deployment_resolves_max_input_tokens(self, router_no_limits):
        """
        token_counter must not be called when no deployment resolves a
        max_input_tokens value (mock get_router_model_info directly, since
        max_input_tokens is commonly derived from litellm's model cost map
        rather than statically declared in model_info).
        """
        deployments = router_no_limits.get_model_list(model_name="test-model")
        assert deployments is not None

        with (
            patch.object(router_no_limits, "get_router_model_info", return_value={}),
            patch("litellm.token_counter") as mock_token_counter,
        ):
            result = router_no_limits._pre_call_checks(
                model="test-model",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": "hello world"}],
            )

        mock_token_counter.assert_not_called()
        assert len(result) == 2, "All deployments should be returned"

    def test_model_info_lookup_always_runs_per_deployment(self, router_no_limits):
        """
        get_router_model_info must be called for every deployment, since
        max_input_tokens can come from litellm's model cost map (not just a
        static model_info override) and there's no cheap way to know in
        advance whether a deployment resolves a limit.
        """
        deployments = router_no_limits.get_model_list(model_name="test-model")
        assert deployments is not None

        with patch.object(
            router_no_limits,
            "get_router_model_info",
            wraps=router_no_limits.get_router_model_info,
        ) as mock_get_info:
            result = router_no_limits._pre_call_checks(
                model="test-model",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": "hello world"}],
            )

        assert mock_get_info.call_count == 2
        assert len(result) == 2

    def test_no_rpm_cache_lookup_when_no_deployment_has_rpm(self, router_no_limits):
        """The RPM cache must not be queried when no deployment has rpm configured."""
        deployments = router_no_limits.get_model_list(model_name="test-model")
        assert deployments is not None

        with patch.object(router_no_limits.cache, "get_cache", wraps=router_no_limits.cache.get_cache) as mock_cache:
            result = router_no_limits._pre_call_checks(
                model="test-model",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": "hello"}],
            )

        mock_cache.assert_not_called()
        assert len(result) == 2

    def test_zero_tokenization_or_rpm_cache_calls_when_nothing_configured(self, router_no_limits):
        """
        A router with zero deployments resolving max_input_tokens and zero
        rpm-configured deployments must make zero calls to
        litellm.token_counter() and the RPM cache get - regardless of message
        size. get_router_model_info() itself still runs per deployment (see
        test_model_info_lookup_always_runs_per_deployment for why it can't be
        statically skipped), it's just mocked here to return no limit.
        """
        deployments = router_no_limits.get_model_list(model_name="test-model")
        assert deployments is not None

        with (
            patch.object(router_no_limits, "get_router_model_info", return_value={}),
            patch("litellm.token_counter") as mock_token_counter,
            patch.object(
                router_no_limits.cache,
                "get_cache",
                wraps=router_no_limits.cache.get_cache,
            ) as mock_cache,
        ):
            result = router_no_limits._pre_call_checks(
                model="test-model",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": " ".join(["word"] * 10000)}],
            )

        mock_token_counter.assert_not_called()
        mock_cache.assert_not_called()
        assert len(result) == 2, "All deployments should pass when no limits are configured"

    def test_tokenization_only_when_deployment_has_max_input_tokens(self, router_with_context_limit):
        """token_counter should be called exactly once when at least one deployment has max_input_tokens."""
        deployments = router_with_context_limit.get_model_list(model_name="test-model")
        assert deployments is not None

        with patch("litellm.token_counter", return_value=100) as mock_token_counter:
            result = router_with_context_limit._pre_call_checks(
                model="test-model",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": "hello world"}],
            )

        mock_token_counter.assert_called_once()
        assert len(result) == 2, "Both deployments should pass (100 < 500000)"

    def test_model_info_lookup_includes_deployment_with_max_input_tokens(self, router_with_context_limit):
        """get_router_model_info should be called for the deployment that has max_input_tokens."""
        deployments = router_with_context_limit.get_model_list(model_name="test-model")
        assert deployments is not None

        original_get_info = router_with_context_limit.get_router_model_info
        call_deployment_ids = []

        def tracking_get_info(deployment, received_model_name, **kwargs):
            dep_id = (deployment.get("model_info") or {}).get("id")
            call_deployment_ids.append(dep_id)
            return original_get_info(deployment=deployment, received_model_name=received_model_name, **kwargs)

        with patch.object(
            router_with_context_limit,
            "get_router_model_info",
            side_effect=tracking_get_info,
        ):
            router_with_context_limit._pre_call_checks(
                model="test-model",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": "hello world"}],
            )

        # get_router_model_info is called for every deployment once any deployment in the
        # group needs the max_input_tokens check, but only "has-limit" declares the limit.
        assert "has-limit" in call_deployment_ids

    def test_token_count_computed_once_for_multiple_limited_deployments(self):
        """When multiple deployments have max_input_tokens, token_counter is called exactly once."""
        router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test"},
                    "model_info": {"id": "limit-a", "max_input_tokens": 1000},
                },
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5.5", "api_key": "sk-test"},
                    "model_info": {"id": "limit-b", "max_input_tokens": 500000},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )
        deployments = router.get_model_list(model_name="test-model")
        assert deployments is not None

        with patch("litellm.token_counter", return_value=50) as mock_token_counter:
            result = router._pre_call_checks(
                model="test-model",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": "hello"}],
            )

        mock_token_counter.assert_called_once()
        assert len(result) == 2

    def test_token_counter_failure_skips_context_checks_gracefully(self, router_with_context_limit):
        """If token_counter raises, all deployments should be returned (fail-open)."""
        deployments = router_with_context_limit.get_model_list(model_name="test-model")
        assert deployments is not None

        with patch("litellm.token_counter", side_effect=Exception("tokenizer error")):
            result = router_with_context_limit._pre_call_checks(
                model="test-model",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": "hello"}],
            )

        assert len(result) == 2, "All deployments should be returned on token count failure"

    def test_rpm_cache_lookup_only_when_deployment_has_rpm(self, router_with_rpm):
        """Cache should be queried when at least one deployment has rpm configured."""
        deployments = router_with_rpm.get_model_list(model_name="test-model")
        assert deployments is not None

        with patch.object(router_with_rpm.cache, "get_cache", return_value={}) as mock_cache:
            result = router_with_rpm._pre_call_checks(
                model="test-model",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": "hello"}],
            )

        assert mock_cache.call_count > 0, "Cache should be queried when rpm is configured"
        assert len(result) == 2

    def test_context_window_filtering_with_mixed_deployments(self):
        """
        In a model group with mixed deployments (some with max_input_tokens,
        some without), only the limited deployment should be filtered when
        input exceeds its limit. Unlimited deployments always pass.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test"},
                    "model_info": {"id": "unlimited"},
                },
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5.5", "api_key": "sk-test"},
                    "model_info": {"id": "limited", "max_input_tokens": 50},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )

        deployments = router.get_model_list(model_name="test-model")
        assert deployments is not None

        # 100 words will exceed 50 tokens
        filtered = router._pre_call_checks(
            model="test-model",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": " ".join(["word"] * 100)}],
        )

        assert len(filtered) == 1, f"Expected 1 deployment, got {len(filtered)}"
        assert filtered[0]["model_info"]["id"] == "unlimited"

    def test_context_window_exceeded_error_when_all_limited_deployments_exceeded(self):
        """
        When ALL deployments have max_input_tokens and all are exceeded,
        ContextWindowExceededError should be raised.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test"},
                    "model_info": {"id": "small-a", "max_input_tokens": 10},
                },
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-5.5", "api_key": "sk-test"},
                    "model_info": {"id": "small-b", "max_input_tokens": 20},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )

        deployments = router.get_model_list(model_name="test-model")
        assert deployments is not None

        with pytest.raises(litellm.ContextWindowExceededError):
            router._pre_call_checks(
                model="test-model",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": " ".join(["word"] * 200)}],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

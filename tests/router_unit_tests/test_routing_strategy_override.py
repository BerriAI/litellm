"""
Unit tests for routing strategy override functionality.

Tests verify:
1. Algorithm correctness - each strategy selects the expected deployment
2. Per-request routing strategy overrides work correctly and deterministically
3. Critical regression fixes (NameError prevention, override preservation)
4. Edge cases (invalid strategies, state preservation)
"""

import sys
import os
import pytest
from unittest.mock import patch, AsyncMock, Mock

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router
from litellm.utils import get_utc_datetime


@pytest.fixture
def base_model_list():
    """Base model list with multiple deployments for testing routing strategies."""
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "fake-key-1",
                "rpm": 100,
                "tpm": 10000,
            },
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "fake-key-2",
                "api_base": "https://example.openai.azure.com/",
                "api_version": "2023-05-15",
                "rpm": 200,
                "tpm": 20000,
            },
            "model_info": {"id": "deployment-2"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "fake-key-3",
                "rpm": 150,
                "tpm": 15000,
            },
            "model_info": {"id": "deployment-3"},
        },
    ]


@pytest.fixture
def pass_through_model_list():
    """Model list with pass-through enabled deployments."""
    return [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "gpt-4",
                "api_key": "fake-key-1",
                "use_in_pass_through": True,
                "rpm": 50,
            },
            "model_info": {"id": "pt-deployment-1"},
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "azure/gpt-4",
                "api_key": "fake-key-2",
                "api_base": "https://example.openai.azure.com/",
                "api_version": "2023-05-15",
                "use_in_pass_through": True,
                "rpm": 100,
            },
            "model_info": {"id": "pt-deployment-2"},
        },
    ]


class TestRoutingAlgorithmCorrectness:
    """Test that each routing strategy selects the correct deployment."""

    @pytest.mark.asyncio
    async def test_cost_based_routing_selects_cheapest(self):
        """Verify cost-based routing picks the cheapest deployment."""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "key1",
                    "input_cost_per_token": 0.002,
                    "output_cost_per_token": 0.002,
                },
                "model_info": {"id": "expensive-1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "key2",
                    "input_cost_per_token": 0.001,  # ← Should win (cheapest)
                    "output_cost_per_token": 0.001,
                },
                "model_info": {"id": "cheap-1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "key3",
                    "input_cost_per_token": 0.0015,
                    "output_cost_per_token": 0.0015,
                },
                "model_info": {"id": "medium-1"},
            },
        ]

        router = Router(
            model_list=model_list,
            routing_strategy="cost-based-routing",
        )

        deployment = await router.async_get_available_deployment(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "test"}],
        )

        # Should select the cheaper deployment
        assert deployment["model_info"]["id"] == "cheap-1"

    @pytest.mark.asyncio
    async def test_latency_based_selects_fastest_deployment(self):
        """Verify latency-based routing picks deployment with lowest latency."""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key1"},
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key2"},
                "model_info": {"id": "deployment-2"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key3"},
                "model_info": {"id": "deployment-3"},
            },
        ]

        router = Router(
            model_list=model_list,
            routing_strategy="latency-based-routing",
        )

        # Pre-populate latency cache:
        # deployment-1: 100ms latency
        # deployment-2: 50ms latency  ← Should win (fastest)
        # deployment-3: 200ms latency
        latency_cache = {
            "deployment-1": {"latency": [0.1, 0.1, 0.1]},  # 100ms avg
            "deployment-2": {"latency": [0.05, 0.05, 0.05]},  # 50ms avg - fastest
            "deployment-3": {"latency": [0.2, 0.2, 0.2]},  # 200ms avg
        }
        router.cache.set_cache(key="test-model_map", value=latency_cache)

        deployment = await router.async_get_available_deployment(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "test"}],
        )

        # Should select deployment with lowest latency
        assert deployment["model_info"]["id"] == "deployment-2"

    @pytest.mark.asyncio
    async def test_usage_based_v2_selects_most_available_tpm(self):
        """Verify usage-based-v2 picks deployment with most available TPM."""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "key1",
                    "tpm": 100,
                },
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "key2",
                    "tpm": 100,
                },
                "model_info": {"id": "deployment-2"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "key3",
                    "tpm": 100,
                },
                "model_info": {"id": "deployment-3"},
            },
        ]

        router = Router(
            model_list=model_list,
            routing_strategy="usage-based-routing-v2",
        )

        # Pre-populate usage cache (current minute format):
        # deployment-1: 50/100 TPM used (50 available)
        # deployment-2: 20/100 TPM used (80 available) ← Should win
        # deployment-3: 90/100 TPM used (10 available)
        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        router.cache.set_cache(
            key=f"deployment-1:gpt-3.5-turbo:tpm:{current_minute}", value=50
        )
        router.cache.set_cache(
            key=f"deployment-2:gpt-3.5-turbo:tpm:{current_minute}", value=20
        )  # Most available
        router.cache.set_cache(
            key=f"deployment-3:gpt-3.5-turbo:tpm:{current_minute}", value=90
        )

        deployment = await router.async_get_available_deployment(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "test"}],
        )

        # Should select deployment with most available TPM
        assert deployment["model_info"]["id"] == "deployment-2"

    @pytest.mark.asyncio
    async def test_least_busy_selects_deployment_with_fewest_requests(self):
        """Verify least-busy picks deployment with fewest active requests."""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key1"},
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key2"},
                "model_info": {"id": "deployment-2"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key3"},
                "model_info": {"id": "deployment-3"},
            },
        ]

        router = Router(
            model_list=model_list,
            routing_strategy="least-busy",
        )

        # Pre-populate request count cache:
        # deployment-1: 5 active requests
        # deployment-2: 2 active requests  ← Should win (least busy)
        # deployment-3: 8 active requests
        request_count_cache = {
            "deployment-1": 5,
            "deployment-2": 2,  # Least busy
            "deployment-3": 8,
        }
        router.cache.set_cache(
            key="test-model_request_count", value=request_count_cache
        )

        deployment = await router.async_get_available_deployment(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "test"}],
        )

        # Should select deployment with fewest active requests
        assert deployment["model_info"]["id"] == "deployment-2"


class TestOverrideDeterminism:
    """Test that overrides produce deterministic results (not random like simple-shuffle)."""

    @pytest.mark.asyncio
    async def test_override_from_shuffle_to_cost_based_is_deterministic(self):
        """
        Verify override from shuffle to cost-based ALWAYS selects cheapest.

        Run 10 times to prove it's deterministic (not random like simple-shuffle).
        """
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "key1",
                    "input_cost_per_token": 0.002,  # Expensive
                    "output_cost_per_token": 0.002,
                },
                "model_info": {"id": "expensive-deployment"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "key2",
                    "input_cost_per_token": 0.001,  # ← Should ALWAYS win
                    "output_cost_per_token": 0.001,
                },
                "model_info": {"id": "cheap-deployment"},
            },
        ]

        # Global: simple-shuffle (random selection)
        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
        )

        # Run 10 times with cost-based override
        # If override works correctly, should ALWAYS select cheap-deployment
        for i in range(10):
            deployment = await router.async_get_available_deployment(
                model="test-model",
                request_kwargs={"routing_strategy": "cost-based-routing"},
                messages=[{"role": "user", "content": "test"}],
            )

            assert deployment["model_info"]["id"] == "cheap-deployment", (
                f"Iteration {i}: Expected cheap-deployment but got {deployment['model_info']['id']}. "
                "Override should deterministically select cheapest, not random."
            )

    @pytest.mark.asyncio
    async def test_override_from_shuffle_to_latency_is_deterministic(self):
        """
        Verify override from shuffle to latency-based ALWAYS selects fastest.

        Run 10 times to prove it's deterministic (not random like simple-shuffle).
        """
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key1"},
                "model_info": {"id": "slow-deployment"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key2"},
                "model_info": {"id": "fast-deployment"},
            },
        ]

        # Global: simple-shuffle (random selection)
        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
        )

        # Pre-populate latency cache
        latency_cache = {
            "slow-deployment": {"latency": [0.2, 0.2]},  # 200ms avg
            "fast-deployment": {"latency": [0.05, 0.05]},  # 50ms avg - fastest
        }
        router.cache.set_cache(key="test-model_map", value=latency_cache)

        # Run 10 times with latency-based override
        # If override works correctly, should ALWAYS select fast-deployment
        for i in range(10):
            deployment = await router.async_get_available_deployment(
                model="test-model",
                request_kwargs={"routing_strategy": "latency-based-routing"},
                messages=[{"role": "user", "content": "test"}],
            )

            assert deployment["model_info"]["id"] == "fast-deployment", (
                f"Iteration {i}: Expected fast-deployment but got {deployment['model_info']['id']}. "
                "Override should deterministically select fastest, not random."
            )


class TestRegressionFixes:
    """Test specific regression fixes for critical bugs."""

    @pytest.mark.asyncio
    async def test_pass_through_no_name_error(self, pass_through_model_list):
        """
        Regression test: deployment = None initialization prevents NameError.

        Bug: Uninitialized deployment variable caused NameError in pass-through routing.
        Fix: Added deployment = None initialization.
        """
        router = Router(
            model_list=pass_through_model_list,
            routing_strategy="simple-shuffle",
        )

        # This should not raise NameError
        deployment = await router.async_get_available_deployment_for_pass_through(
            model="gpt-4",
            request_kwargs={},
            messages=[{"role": "user", "content": "test"}],
        )

        assert deployment is not None
        assert deployment.get("litellm_params", {}).get("use_in_pass_through") is True

    @pytest.mark.asyncio
    async def test_routing_strategy_not_forwarded_to_llm_backend(self, base_model_list):
        """
        Regression test: routing_strategy is not forwarded to LLM backend APIs.

        Security boundary test - ensures routing_strategy override doesn't leak
        to downstream LLM provider calls. We don't want custom parameters leaking
        to third-party APIs.
        """
        router = Router(
            model_list=base_model_list,
            routing_strategy="simple-shuffle",
        )

        request_kwargs = {
            "routing_strategy": "latency-based-routing",
            "temperature": 0.7,
        }

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            from litellm import ModelResponse

            mock_acompletion.return_value = ModelResponse(
                id="test",
                choices=[
                    {"message": {"role": "assistant", "content": "test"}, "index": 0}
                ],
                model="gpt-3.5-turbo",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            )

            # Make actual completion call (not just get_available_deployment)
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                **request_kwargs,
            )

            # Verify completion succeeded
            assert response is not None

            # CRITICAL: Verify routing_strategy was NOT passed to litellm.acompletion
            call_kwargs = mock_acompletion.call_args[1]
            assert (
                "routing_strategy" not in call_kwargs
            ), "routing_strategy should not be forwarded to LLM provider"

            # Verify other params WERE passed correctly
            assert "temperature" in call_kwargs
            assert call_kwargs["temperature"] == 0.7

    def test_routing_strategy_not_forwarded_to_llm_backend_sync(self, base_model_list):
        """
        Regression test: routing_strategy is not forwarded to LLM backend APIs (sync).

        Security boundary test - ensures routing_strategy override doesn't leak
        to downstream LLM provider calls. We don't want custom parameters leaking
        to third-party APIs.
        """
        router = Router(
            model_list=base_model_list,
            routing_strategy="simple-shuffle",
        )

        request_kwargs = {
            "routing_strategy": "latency-based-routing",
            "temperature": 0.7,
        }

        with patch("litellm.completion", new_callable=Mock) as mock_completion:
            from litellm import ModelResponse

            mock_completion.return_value = ModelResponse(
                id="test",
                choices=[
                    {"message": {"role": "assistant", "content": "test"}, "index": 0}
                ],
                model="gpt-3.5-turbo",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            )

            # Make actual completion call
            response = router.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                **request_kwargs,
            )

            # Verify completion succeeded
            assert response is not None

            # CRITICAL: Verify routing_strategy was NOT passed to litellm.completion
            call_kwargs = mock_completion.call_args[1]
            assert (
                "routing_strategy" not in call_kwargs
            ), "routing_strategy should not be forwarded to LLM provider"

            # Verify other params WERE passed correctly
            assert "temperature" in call_kwargs
            assert call_kwargs["temperature"] == 0.7

    def test_sync_latency_uses_override_variable(self, base_model_list):
        """
        Regression test: sync path uses routing_strategy_to_use, not self.routing_strategy.

        Bug: Sync latency-based routing used global strategy instead of per-request override.
        Fix: Changed to use routing_strategy_to_use variable.
        """
        router = Router(
            model_list=base_model_list,
            routing_strategy="cost-based-routing",  # Global is cost-based
            routing_strategy_args={"ttl": 1},
        )

        # Override to latency-based (will trigger lazy init)
        deployment = router.get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={"routing_strategy": "latency-based-routing"},
            messages=[{"role": "user", "content": "test"}],
        )

        # Verify override worked and latency logger was lazily initialized
        assert deployment is not None
        assert (
            hasattr(router, "lowestlatency_logger")
            and router.lowestlatency_logger is not None
        )

    @pytest.mark.asyncio
    async def test_async_to_sync_fallthrough_preserves_override(self, base_model_list):
        """
        Regression test: routing_strategy override is preserved in async→sync fallthrough.

        Bug: Override was lost when async fell through to sync for unsupported strategies.
        Fix: Override is now passed through in fallthrough path.
        """
        router = Router(
            model_list=base_model_list,
            routing_strategy="simple-shuffle",
        )

        # Mock get_available_deployment to capture the routing_strategy passed
        captured_kwargs = {}

        def capture_kwargs(*args, **kwargs):
            captured_kwargs.update(kwargs.get("request_kwargs", {}))
            # Return a valid deployment
            return base_model_list[0]

        with patch.object(
            router, "get_available_deployment", side_effect=capture_kwargs
        ):
            # Trigger async→sync fallthrough with unsupported strategy
            deployment = await router.async_get_available_deployment(
                model="gpt-3.5-turbo",
                request_kwargs={"routing_strategy": "invalid-unknown-strategy"},
                messages=[{"role": "user", "content": "test"}],
            )

            # Verify routing_strategy was preserved in fallthrough
            assert (
                "routing_strategy" in captured_kwargs
            ), "routing_strategy should be preserved in fallthrough"
            assert captured_kwargs["routing_strategy"] == "invalid-unknown-strategy"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_router_state_not_mutated_by_override(self, base_model_list):
        """
        Test that per-request override doesn't mutate router's global strategy.

        Ensures override is truly per-request and doesn't affect router state.
        """
        router = Router(
            model_list=base_model_list,
            routing_strategy="simple-shuffle",
        )

        original_strategy = router.routing_strategy

        # Make request with override (will trigger lazy logger init)
        await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={"routing_strategy": "latency-based-routing"},
            messages=[{"role": "user", "content": "test"}],
        )

        # Router's global strategy should be unchanged
        assert router.routing_strategy == original_strategy
        assert router.routing_strategy == "simple-shuffle"

    @pytest.mark.asyncio
    async def test_override_with_same_strategy_as_global(self, base_model_list):
        """
        Test that overriding with the same strategy as global still works.

        Edge case: override to same strategy shouldn't cause issues.
        """
        router = Router(
            model_list=base_model_list,
            routing_strategy="simple-shuffle",
        )

        # Override to same strategy as global
        deployment = await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={"routing_strategy": "simple-shuffle"},
            messages=[{"role": "user", "content": "test"}],
        )

        # Should still return a deployment
        assert deployment is not None
        assert "model_info" in deployment
        assert "id" in deployment["model_info"]

    @pytest.mark.asyncio
    async def test_latency_routing_with_empty_cache_falls_back(self):
        """
        Test that latency-based routing handles empty cache gracefully.

        Edge case: no latency data available yet - should fall back to available deployments.
        """
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key1"},
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key2"},
                "model_info": {"id": "deployment-2"},
            },
        ]

        router = Router(
            model_list=model_list,
            routing_strategy="latency-based-routing",
        )

        # Don't populate latency cache - it's empty
        # Should still return a deployment (falls back to available deployments)
        deployment = await router.async_get_available_deployment(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "test"}],
        )

        # Should return a deployment even without latency data
        assert deployment is not None
        assert deployment["model_info"]["id"] in ["deployment-1", "deployment-2"]


class TestOverridePrecedence:
    """Test that per-request override takes precedence over global settings."""

    @pytest.mark.asyncio
    async def test_override_takes_precedence_over_global_async(self):
        """Verify request override takes precedence over global routing strategy (async)."""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key1"},
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key2"},
                "model_info": {"id": "deployment-2"},
            },
        ]

        # Global: least-busy
        router = Router(
            model_list=model_list,
            routing_strategy="least-busy",
        )

        # Pre-populate latency cache for override strategy
        latency_cache = {
            "deployment-1": {"latency": [0.1]},  # 100ms
            "deployment-2": {"latency": [0.05]},  # 50ms - faster
        }
        router.cache.set_cache(key="test-model_map", value=latency_cache)

        # Override to latency-based
        deployment = await router.async_get_available_deployment(
            model="test-model",
            request_kwargs={"routing_strategy": "latency-based-routing"},
            messages=[{"role": "user", "content": "test"}],
        )

        # Should use latency-based routing (deployment-2), not least-busy
        assert deployment["model_info"]["id"] == "deployment-2"

    def test_override_takes_precedence_over_global_sync(self):
        """Verify request override takes precedence over global routing strategy (sync)."""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key1"},
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "key2"},
                "model_info": {"id": "deployment-2"},
            },
        ]

        # Global: least-busy
        router = Router(
            model_list=model_list,
            routing_strategy="least-busy",
        )

        # Pre-populate latency cache for override strategy
        latency_cache = {
            "deployment-1": {"latency": [0.1]},  # 100ms
            "deployment-2": {"latency": [0.05]},  # 50ms - faster
        }
        router.cache.set_cache(key="test-model_map", value=latency_cache)

        # Override to latency-based
        deployment = router.get_available_deployment(
            model="test-model",
            request_kwargs={"routing_strategy": "latency-based-routing"},
            messages=[{"role": "user", "content": "test"}],
        )

        # Should use latency-based routing (deployment-2), not least-busy
        assert deployment["model_info"]["id"] == "deployment-2"


class TestRoutingStrategyWithMockTestingFallbacks:
    """Test that per-request routing_strategy is preserved during mock testing fallbacks."""

    @pytest.mark.asyncio
    async def test_routing_strategy_preserved_in_mock_testing_fallbacks(self):
        """
        Test that routing_strategy override is preserved when simulating fallback scenarios.

        This verifies that the fix (using get() instead of pop()) works correctly by
        simulating what happens during fallbacks: multiple calls to get_available_deployment
        with the same request_kwargs.

        Scenario:
        1. Request model="gpt-4" with routing_strategy="cost-based-routing"
        2. First call to get_available_deployment (simulated initial request)
        3. Second call to get_available_deployment (simulated fallback to gpt-3.5-turbo)
        4. KEY: routing_strategy should be preserved in request_kwargs for both calls
        5. Second call selects the CHEAPEST gpt-3.5-turbo installation (proves cost-based routing)
        """
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "test-key-1",
                    "input_cost_per_token": 0.03,
                    "output_cost_per_token": 0.06,
                },
                "model_info": {"id": "gpt4-inst"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key-2",
                    "input_cost_per_token": 0.003,
                    "output_cost_per_token": 0.006,
                },
                "model_info": {"id": "expensive-gpt35-inst"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key-3",
                    "input_cost_per_token": 0.002,
                    "output_cost_per_token": 0.004,
                },
                "model_info": {"id": "medium-gpt35-inst"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key-4",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
                "model_info": {"id": "cheap-gpt35-inst"},
            },
        ]

        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
        )

        # Simulate what happens during fallback: multiple calls with same request_kwargs
        request_kwargs = {
            "routing_strategy": "cost-based-routing",
        }

        # First call: gpt-4 (initial request)
        deployment1 = await router.async_get_available_deployment(
            model="gpt-4",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "test"}],
        )

        # Verify routing_strategy is still in request_kwargs (not popped)
        assert "routing_strategy" in request_kwargs
        assert request_kwargs["routing_strategy"] == "cost-based-routing"

        # Second call: gpt-3.5-turbo (fallback - simulating router switching models)
        deployment2 = await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "test"}],
        )

        # Verify routing_strategy is STILL in request_kwargs after second call
        assert (
            "routing_strategy" in request_kwargs
        ), "routing_strategy should be preserved in request_kwargs for fallbacks"
        assert request_kwargs["routing_strategy"] == "cost-based-routing"

        # KEY ASSERTION: Second call selected the cheapest gpt-3.5-turbo installation
        # This proves cost-based routing was used (not global simple-shuffle)
        assert deployment2["model_info"]["id"] == "cheap-gpt35-inst", (
            f"Expected cheapest gpt-3.5-turbo installation (cheap-gpt35-inst $0.001), "
            f"got {deployment2['model_info']['id']}. "
            "This proves routing_strategy='cost-based-routing' was preserved and worked correctly."
        )

    def test_routing_strategy_preserved_in_mock_testing_fallbacks_sync(self):
        """
        Test that routing_strategy override is preserved during sync fallback scenarios.

        Scenario:
        1. Request model="gpt-4" with routing_strategy="cost-based-routing"
        2. First call to get_available_deployment (simulated initial request)
        3. Second call to get_available_deployment (simulated fallback to gpt-3.5-turbo)
        4. KEY: routing_strategy should be preserved in request_kwargs for both calls
        5. Second call selects the CHEAPEST gpt-3.5-turbo installation (proves cost-based routing)
        """
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "test-key-1",
                    "input_cost_per_token": 0.03,
                    "output_cost_per_token": 0.06,
                },
                "model_info": {"id": "gpt4-inst"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key-2",
                    "input_cost_per_token": 0.003,
                    "output_cost_per_token": 0.006,
                },
                "model_info": {"id": "expensive-gpt35-inst"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key-3",
                    "input_cost_per_token": 0.002,
                    "output_cost_per_token": 0.004,
                },
                "model_info": {"id": "medium-gpt35-inst"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key-4",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
                "model_info": {"id": "cheap-gpt35-inst"},
            },
        ]

        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
        )

        request_kwargs = {
            "routing_strategy": "cost-based-routing",
        }

        deployment1 = router.get_available_deployment(
            model="gpt-4",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "test"}],
        )

        assert "routing_strategy" in request_kwargs
        assert request_kwargs["routing_strategy"] == "cost-based-routing"

        deployment2 = router.get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "test"}],
        )

        assert (
            "routing_strategy" in request_kwargs
        ), "routing_strategy should be preserved in request_kwargs for fallbacks"
        assert request_kwargs["routing_strategy"] == "cost-based-routing"

        assert deployment2["model_info"]["id"] == "cheap-gpt35-inst", (
            f"Expected cheapest gpt-3.5-turbo installation (cheap-gpt35-inst $0.001), "
            f"got {deployment2['model_info']['id']}. "
            "This proves routing_strategy='cost-based-routing' was preserved and worked correctly."
        )

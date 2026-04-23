"""
Unit tests for routing strategy override functionality.

Tests verify:
1. Algorithm correctness - each strategy selects the expected deployment
2. Per-request routing strategy overrides work correctly
3. Critical regression fixes (NameError prevention, override preservation)
"""

import sys
import os
import pytest
from unittest.mock import patch

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
        # deployment-1: 50/100 TPM used
        # deployment-2: 20/100 TPM used  ← Should win (80 available)
        # deployment-3: 90/100 TPM used
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


class TestRequestBodyRoutingStrategyOverride:
    """Test routing_strategy override passed in request body actually uses the correct algorithm."""

    @pytest.mark.asyncio
    async def test_request_body_override_actually_uses_cost_based(self):
        """Verify request body cost-based override ACTUALLY selects cheapest."""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "key1",
                    "input_cost_per_token": 0.002,  # Expensive
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
        ]

        # Global: simple-shuffle
        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
        )

        # Request override: cost-based-routing
        deployment = await router.async_get_available_deployment(
            model="test-model",
            request_kwargs={"routing_strategy": "cost-based-routing"},
            messages=[{"role": "user", "content": "test"}],
        )

        # Verify cheap deployment is selected (not random from simple-shuffle)
        assert deployment["model_info"]["id"] == "cheap-1"

    @pytest.mark.asyncio
    async def test_request_body_override_actually_uses_latency_based(self):
        """Verify request body latency-based override ACTUALLY selects fastest."""
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

        # Global: simple-shuffle
        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
        )

        # Pre-populate latency cache
        latency_cache = {
            "deployment-1": {"latency": [0.2, 0.2]},  # 200ms avg
            "deployment-2": {"latency": [0.05, 0.05]},  # 50ms avg - fastest
        }
        router.cache.set_cache(key="test-model_map", value=latency_cache)

        # Request override: latency-based-routing
        deployment = await router.async_get_available_deployment(
            model="test-model",
            request_kwargs={"routing_strategy": "latency-based-routing"},
            messages=[{"role": "user", "content": "test"}],
        )

        # Verify fastest deployment is selected (not random from simple-shuffle)
        assert deployment["model_info"]["id"] == "deployment-2"

    @pytest.mark.asyncio
    async def test_request_body_routing_strategy_is_popped(self, base_model_list):
        """
        Test that routing_strategy is popped from request_kwargs after use.

        Ensures routing_strategy doesn't leak to downstream LLM provider calls.
        """
        router = Router(
            model_list=base_model_list,
            routing_strategy="simple-shuffle",
        )

        request_kwargs = {
            "routing_strategy": "latency-based-routing",
            "temperature": 0.7,  # Other param that should remain
        }

        # Simulate curl request body
        deployment = await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "test"}],
        )

        # Verify deployment is selected
        assert deployment is not None

        # Verify routing_strategy was popped from request_kwargs
        assert "routing_strategy" not in request_kwargs, (
            "routing_strategy should be popped from request_kwargs "
            "to prevent it from being passed to LLM provider"
        )

        # Verify other params remain
        assert "temperature" in request_kwargs
        assert request_kwargs["temperature"] == 0.7


class TestRegressionFixes:
    """Test specific regression fixes for critical bugs."""

    @pytest.mark.asyncio
    async def test_pass_through_cost_routing_no_name_error(
        self, pass_through_model_list
    ):
        """
        Test that deployment = None initialization prevents NameError.
        Regression test for bug where uninitialized deployment variable caused NameError.
        """
        router = Router(
            model_list=pass_through_model_list,
            routing_strategy="simple-shuffle",
        )

        # This should not raise NameError (fix adds deployment = None initialization)
        deployment = await router.async_get_available_deployment_for_pass_through(
            model="gpt-4",
            request_kwargs={},
            messages=[{"role": "user", "content": "test"}],
        )

        assert deployment is not None
        assert deployment.get("litellm_params", {}).get("use_in_pass_through") is True

    def test_sync_latency_respects_per_request_override(self, base_model_list):
        """
        Test that sync latency-based routing uses per-request override, not global strategy.
        Regression test: verifies routing_strategy_to_use variable is used instead of self.routing_strategy.
        """
        router = Router(
            model_list=base_model_list,
            routing_strategy="cost-based-routing",  # Global is cost-based
            routing_strategy_args={"ttl": 1},
        )

        # Override to latency-based (lazy init will create logger)
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
        Test that routing_strategy override is preserved when async falls through to sync.
        Regression test: verifies the override is not lost in the fallthrough path.
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
            # Trigger the async→sync fallthrough path by using an unsupported routing strategy
            deployment = await router.async_get_available_deployment(
                model="gpt-3.5-turbo",
                request_kwargs={
                    "routing_strategy": "invalid-unknown-strategy"  # Not in async whitelist
                },
                messages=[{"role": "user", "content": "test"}],
            )

            # Verify routing_strategy was preserved in request_kwargs
            assert (
                "routing_strategy" in captured_kwargs
            ), "routing_strategy should be preserved in fallthrough"
            assert captured_kwargs["routing_strategy"] == "invalid-unknown-strategy"

    @pytest.mark.asyncio
    async def test_routing_strategy_not_mutated_across_requests(self, base_model_list):
        """
        Test that routing_strategy override doesn't mutate router's global strategy.
        Regression test: ensures per-request override is truly per-request.
        """
        router = Router(
            model_list=base_model_list,
            routing_strategy="simple-shuffle",
        )

        original_strategy = router.routing_strategy

        # Make request with override (lazy init will create logger)
        await router.async_get_available_deployment(
            model="gpt-3.5-turbo",
            request_kwargs={"routing_strategy": "latency-based-routing"},
            messages=[{"role": "user", "content": "test"}],
        )

        # Router's global strategy should be unchanged
        assert router.routing_strategy == original_strategy
        assert router.routing_strategy == "simple-shuffle"


class TestOverridePrecedence:
    """Test that per-request override takes precedence over global settings."""

    @pytest.mark.asyncio
    async def test_override_takes_precedence_over_global_async(self):
        """Verify request override takes precedence over global routing strategy."""
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
            "deployment-1": {"latency": [0.1]},
            "deployment-2": {"latency": [0.05]},  # Faster
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
        """Test request override takes precedence in sync path."""
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
            "deployment-1": {"latency": [0.1]},
            "deployment-2": {"latency": [0.05]},  # Faster
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

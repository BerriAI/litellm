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
import pytest
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


class TestPreCallChecksSkipsRpmCacheReadWhenUnused:
    """
    _pre_call_checks() must not read the RPM cache at all when no deployment in
    the model group configures `rpm` - that cache read (local + usage-based lookup
    per deployment) runs on every request and is pure overhead when nothing
    actually gates on it.
    """

    def test_no_rpm_cache_read_when_no_deployment_has_rpm(self):
        router = Router(
            model_list=[
                {
                    "model_name": "test",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test"},
                    "model_info": {"id": "no-rpm-1"},
                },
                {
                    "model_name": "test",
                    "litellm_params": {"model": "gpt-5.5", "api_key": "sk-test"},
                    "model_info": {"id": "no-rpm-2"},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )

        get_cache_calls = []
        original_get_cache = router.cache.get_cache

        def _tracking_get_cache(*args, **kwargs):
            get_cache_calls.append(kwargs.get("key", args[0] if args else None))
            return original_get_cache(*args, **kwargs)

        router.cache.get_cache = _tracking_get_cache

        deployments = router.get_model_list(model_name="test")
        assert deployments is not None

        router._pre_call_checks(
            model="test",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert get_cache_calls == [], f"Expected zero cache reads, got {get_cache_calls}"

    def test_rpm_cache_still_read_and_enforced_when_a_deployment_has_rpm(self):
        router = Router(
            model_list=[
                {
                    "model_name": "test",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test", "rpm": 1},
                    "model_info": {"id": "low-rpm"},
                },
                {
                    "model_name": "test",
                    "litellm_params": {"model": "gpt-5.5", "api_key": "sk-test"},
                    "model_info": {"id": "no-rpm"},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )

        get_cache_calls = []
        original_get_cache = router.cache.get_cache

        def _tracking_get_cache(*args, **kwargs):
            key = kwargs.get("key", args[0] if args else None)
            get_cache_calls.append(key)
            return original_get_cache(*args, **kwargs)

        router.cache.get_cache = _tracking_get_cache

        deployments = router.get_model_list(model_name="test")
        assert deployments is not None

        # Simulate the low-rpm deployment already at its limit.
        router.cache.set_cache(key="low-rpm", value=1, local_only=True)

        filtered = router._pre_call_checks(
            model="test",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert get_cache_calls, "Expected the RPM cache to be read when a deployment configures rpm"
        assert [d["model_info"]["id"] for d in filtered] == ["no-rpm"], (
            "Deployment at its rpm limit should have been filtered out"
        )

    def test_no_rpm_cache_read_for_usage_based_routing_v2(self):
        """usage-based-routing-v2 manages its own RPM enforcement; _pre_call_checks
        must not also read the cache even if a deployment configures `rpm`."""
        router = Router(
            model_list=[
                {
                    "model_name": "test",
                    "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test", "rpm": 1},
                    "model_info": {"id": "low-rpm"},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
            routing_strategy="usage-based-routing-v2",
        )

        get_cache_calls = []
        original_get_cache = router.cache.get_cache

        def _tracking_get_cache(*args, **kwargs):
            get_cache_calls.append(kwargs.get("key", args[0] if args else None))
            return original_get_cache(*args, **kwargs)

        router.cache.get_cache = _tracking_get_cache

        deployments = router.get_model_list(model_name="test")
        assert deployments is not None

        router._pre_call_checks(
            model="test",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert get_cache_calls == [], f"Expected zero cache reads under usage-based-routing-v2, got {get_cache_calls}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

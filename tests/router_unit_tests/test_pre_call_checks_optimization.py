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
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-test"},
                    "model_info": {"id": "test-1"},
                },
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {"model": "gpt-4", "api_key": "sk-test2"},
                    "model_info": {"id": "test-2"},
                },
            ],
            set_verbose=False,
            enable_pre_call_checks=True,
        )

        deployments = router.get_model_list(model_name="gpt-3.5-turbo")
        assert deployments is not None
        
        # Capture the original state
        original_length = len(deployments)
        original_deployment_ids = [id(d) for d in deployments]
        original_litellm_params_ids = [id(d["litellm_params"]) for d in deployments]
        snapshot = copy.deepcopy(deployments)

        # Call the function under test
        router._pre_call_checks(
            model="gpt-3.5-turbo",
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
                    "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-test"},
                    "model_info": {"id": "small", "max_input_tokens": 50},
                },
                {
                    "model_name": "test",
                    "litellm_params": {"model": "gpt-4", "api_key": "sk-test"},
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


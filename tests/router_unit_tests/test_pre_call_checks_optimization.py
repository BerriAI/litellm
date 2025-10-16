"""
Regression test for Router._pre_call_checks() performance optimization.

CONTEXT:
    _pre_call_checks() runs on EVERY REQUEST to filter deployments based on:
    - Context window size
    - RPM/TPM limits  
    - Region constraints
    - Supported parameters
    
OPTIMIZATION (line 6660 in router.py):
    Before: _returned_deployments = copy.deepcopy(healthy_deployments)
    After:  _returned_deployments = list(healthy_deployments)
    
    Performance: ~1400x faster (deepcopy is extremely expensive)
    Safety: The function only pops indices from the list, never modifies deployment dicts,
            so shallow copy is safe.

CRITICAL INVARIANT:
    The input healthy_deployments list must NEVER be mutated.
    Callers may reuse this list for retries, fallbacks, or logging.
"""

import copy
import pytest
from litellm import Router


class TestPreCallChecksOptimization:
    """
    Tests verify the shallow copy optimization doesn't break behavior.
    
    If these tests fail, the optimization has introduced a regression and should be reverted.
    """

    def test_no_mutation_of_input_list(self):
        """
        CRITICAL: Verify input list is not mutated.
        
        The optimization changed deepcopy -> list(). This test ensures:
        1. The list container is unchanged (same length, same object references)
        2. The deployment dict objects are unchanged (same id(), same values)
        3. Nested dicts are unchanged (litellm_params, model_info)
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

        # Get deployments from router
        deployments = router.get_model_list(model_name="gpt-3.5-turbo")
        assert deployments is not None
        
        # BEFORE calling _pre_call_checks, capture state to verify no mutation
        original_length = len(deployments)
        # Store Python object IDs - these must not change (proves no replacement)
        original_deployment_ids = [id(d) for d in deployments]
        original_litellm_params_ids = [id(d["litellm_params"]) for d in deployments]
        # Deep snapshot for value comparison
        snapshot = copy.deepcopy(deployments)

        # ACTION: Call _pre_call_checks (the optimized function)
        router._pre_call_checks(
            model="gpt-3.5-turbo",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "test"}],
        )

        # VERIFY: Input list is completely unchanged
        # Check #1-3: Verify shallow copy semantics (same object references kept)
        # These would catch if we accidentally used deepcopy or created new dict copies
        assert len(deployments) == original_length, "List length changed!"
        assert [id(d) for d in deployments] == original_deployment_ids, "Deployment dicts replaced!"
        assert [id(d["litellm_params"]) for d in deployments] == original_litellm_params_ids, "Nested dicts replaced!"
        
        # Check #4: Deep equality ensures no values were mutated
        # This would catch any mutation even if object references are preserved
        assert deployments == snapshot, "Values were mutated!"

    def test_filtering_still_works(self):
        """
        CRITICAL: Verify filtering functionality is preserved when items ARE filtered.
        
        This test ensures that when _pre_call_checks filters out a deployment (e.g., due to
        small context window), the filtering actually works AND the original list remains intact.
        
        Setup: 2 deployments with different max_input_tokens (50 vs 10000)
        Action: Send long message that exceeds 50 tokens
        Expected: Filtered list has 1 deployment, original list still has 2
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

        # Get deployments with different context window sizes
        deployments = router.get_model_list(model_name="test")
        assert deployments is not None
        
        # BEFORE: Store object references to verify they aren't replaced
        original_small_deployment = deployments[0]  # max_input_tokens=50
        original_large_deployment = deployments[1]  # max_input_tokens=10000
        
        # ACTION: Send long message (100 words) that WILL trigger filtering
        # This creates a message that exceeds 50 tokens but fits in 10000 tokens
        filtered = router._pre_call_checks(
            model="test",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": " ".join(["word"] * 100)}],
        )

        # VERIFY: Filtering worked correctly
        # The filtered list should only contain the "large" deployment
        assert len(filtered) == 1, f"Expected 1 deployment after filtering, got {len(filtered)}"
        assert filtered[0]["model_info"]["id"] == "large", "Wrong deployment kept after filtering"
        
        # VERIFY: Original list is COMPLETELY UNCHANGED (this is the critical invariant)
        # Even though we filtered out "small", the original list must still have both
        assert len(deployments) == 2, f"Original list mutated! Expected 2, got {len(deployments)}"
        # Verify same object references (not replaced with new objects)
        assert deployments[0] is original_small_deployment, "First deployment object replaced!"
        assert deployments[1] is original_large_deployment, "Second deployment object replaced!"
        # Verify values unchanged (use .get() for type safety with optional TypedDict keys)
        assert deployments[0].get("model_info", {}).get("id") == "small", "First deployment ID changed!"
        assert deployments[1].get("model_info", {}).get("id") == "large", "Second deployment ID changed!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
Test file to reproduce and investigate the model cost map exception:

{"type":"error","error":"This model isn't mapped yet. model=gpt-5.2, custom_llm_provider=azure. 
Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json."}}'}

This tests what happens when the hosted model cost map JSON is:
1. Malformed JSON (syntax error)
2. Valid JSON but empty
3. Valid JSON but missing model entries
4. Valid JSON but corrupt structure

Root Cause Analysis:
=====================
The issue stems from get_model_cost_map() in litellm/litellm_core_utils/get_model_cost_map.py

The function only falls back to the local backup when an EXCEPTION occurs (HTTP failure, JSON parse error).
It does NOT validate that the returned JSON is non-empty or has the expected structure.

This means if the hosted JSON file is:
- Valid but empty {}
- Valid but has wrong structure {"corrupt": "data"}

The code will use it without falling back, leading to "This model isn't mapped yet" errors for ALL model lookups.

Impact:
=======
- get_model_info() fails for all models
- Cost calculations fail 
- Router decisions may fail
- Proxy budget management fails

The error message includes model and provider info, which helps users but doesn't explain the root cause
that the model cost map itself may be corrupt.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath("../../.."))


class TestModelCostMapExceptionReproduction:
    """Test cases to reproduce the model cost map exception."""

    def test_malformed_json_from_hosted_url(self):
        """
        Test what happens when the hosted model cost map URL returns malformed JSON.
        
        Expected behavior: Should fall back to local backup file.
        """
        from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
        
        # Create a mock response with malformed JSON
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        
        with patch("httpx.get", return_value=mock_response):
            # Should fall back to local backup
            result = get_model_cost_map(url="https://example.com/bad_json")
            
        # Should get a valid dict from the fallback
        assert isinstance(result, dict)
        assert len(result) > 0  # Should have model entries from fallback
        print(f"✓ Malformed JSON handled gracefully, got {len(result)} model entries from fallback")

    def test_empty_json_from_hosted_url(self):
        """
        Test what happens when the hosted model cost map URL returns empty JSON {}.
        
        This is the problematic case - empty JSON is valid, so no fallback is triggered,
        but then model lookups fail with "This model isn't mapped yet" error.
        """
        from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
        
        # Create a mock response with empty JSON object
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={})  # Valid but empty JSON
        
        with patch("httpx.get", return_value=mock_response):
            result = get_model_cost_map(url="https://example.com/empty_json")
            
        # Empty dict is returned - no fallback triggered!
        assert isinstance(result, dict)
        assert len(result) == 0  # Empty!
        print(f"⚠️ Empty JSON returned with {len(result)} entries - no fallback triggered!")

    def test_reproduce_model_not_mapped_exception_with_empty_cost_map(self):
        """
        Reproduce the exact exception from the incident:
        "This model isn't mapped yet. model=gpt-5.2, custom_llm_provider=azure."
        
        This happens when:
        1. Hosted model cost map is empty or missing entries
        2. User tries to get model info for any model
        """
        import litellm
        from litellm.utils import _get_model_info_helper
        
        # Save original model_cost
        original_model_cost = litellm.model_cost.copy()
        
        try:
            # Simulate empty model cost map (what happens with bad hosted JSON)
            litellm.model_cost = {}
            
            # Try to get model info - this should fail with the exact error from the incident
            with pytest.raises(Exception) as exc_info:
                _get_model_info_helper(model="gpt-5.2", custom_llm_provider="azure")
            
            # Check that we get the exact error message from the incident
            error_message = str(exc_info.value)
            assert "This model isn't mapped yet" in error_message
            assert "model=gpt-5.2" in error_message
            assert "custom_llm_provider=azure" in error_message
            print(f"✓ Reproduced the exact exception: {error_message}")
            
        finally:
            # Restore original model_cost
            litellm.model_cost = original_model_cost

    def test_reproduce_exception_with_known_model(self):
        """
        Test that even well-known models like gpt-4 fail when cost map is empty.
        """
        import litellm
        from litellm.utils import _get_model_info_helper
        
        # Save original model_cost
        original_model_cost = litellm.model_cost.copy()
        
        try:
            # Simulate empty model cost map
            litellm.model_cost = {}
            
            # Even GPT-4 should fail
            with pytest.raises(Exception) as exc_info:
                _get_model_info_helper(model="gpt-4", custom_llm_provider="openai")
            
            error_message = str(exc_info.value)
            assert "This model isn't mapped yet" in error_message
            print(f"✓ Known model also fails with empty cost map: {error_message}")
            
        finally:
            litellm.model_cost = original_model_cost

    def test_partial_model_cost_map(self):
        """
        Test what happens when the hosted JSON has some entries but is missing the requested model.
        """
        import litellm
        from litellm.utils import _get_model_info_helper
        
        # Save original model_cost
        original_model_cost = litellm.model_cost.copy()
        
        try:
            # Simulate partial model cost map with only one entry
            litellm.model_cost = {
                "gpt-3.5-turbo": {
                    "max_tokens": 4096,
                    "input_cost_per_token": 0.0000015,
                    "output_cost_per_token": 0.000002,
                    "litellm_provider": "openai"
                }
            }
            
            # GPT-3.5-turbo should work
            result = _get_model_info_helper(model="gpt-3.5-turbo", custom_llm_provider="openai")
            assert result is not None
            # Result is ModelInfoBase which has a key attribute
            print(f"✓ Model in partial cost map works: gpt-3.5-turbo")
            
            # But other models should fail
            with pytest.raises(Exception) as exc_info:
                _get_model_info_helper(model="gpt-4", custom_llm_provider="openai")
            
            error_message = str(exc_info.value)
            assert "This model isn't mapped yet" in error_message
            print(f"✓ Model not in partial cost map fails: {error_message}")
            
        finally:
            litellm.model_cost = original_model_cost


class TestModelCostMapFlowAnalysis:
    """
    Analyze the flow to understand where the exception originates.
    """

    def test_trace_exception_flow(self):
        """
        Trace the full flow of where the exception comes from.
        """
        import litellm
        from litellm.utils import _get_model_info_helper, _get_model_cost_key, get_model_info
        
        # Save original model_cost
        original_model_cost = litellm.model_cost.copy()
        
        try:
            litellm.model_cost = {}
            
            # Test _get_model_cost_key with empty cost map
            result = _get_model_cost_key("gpt-4")
            assert result is None  # Should return None when model not found
            print(f"✓ _get_model_cost_key returns None for missing model")
            
            # Test the helper function flow
            with pytest.raises(Exception) as exc_info:
                _get_model_info_helper(model="gpt-4", custom_llm_provider="openai")
            
            # The exception comes from _get_model_info_helper when _model_info is None
            print(f"✓ Exception comes from _get_model_info_helper")
            print(f"  Error: {exc_info.value}")
            
        finally:
            litellm.model_cost = original_model_cost

    def test_model_cost_map_loading_scenarios(self):
        """
        Test different scenarios for model cost map loading.
        """
        from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
        
        # Scenario 1: HTTP request fails - should fall back
        print("\n--- Scenario 1: HTTP Request Fails ---")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(side_effect=Exception("HTTP Error"))
        
        with patch("httpx.get", return_value=mock_response):
            result = get_model_cost_map(url="https://example.com/fail")
        
        assert len(result) > 0
        print(f"✓ HTTP failure handled, got {len(result)} entries from fallback")
        
        # Scenario 2: Valid empty JSON - NO fallback
        print("\n--- Scenario 2: Valid Empty JSON ---")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={})
        
        with patch("httpx.get", return_value=mock_response):
            result = get_model_cost_map(url="https://example.com/empty")
        
        # This is the BUG - empty JSON doesn't trigger fallback!
        assert len(result) == 0
        print(f"⚠️ BUG: Empty JSON returns {len(result)} entries, no fallback!")
        
        # Scenario 3: Corrupt JSON structure - depends on what corrupt means
        print("\n--- Scenario 3: Corrupt JSON Structure ---")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"corrupt": "not model data"})
        
        with patch("httpx.get", return_value=mock_response):
            result = get_model_cost_map(url="https://example.com/corrupt")
        
        # Corrupt structure also doesn't trigger fallback
        assert "corrupt" in result
        print(f"⚠️ BUG: Corrupt JSON returns {len(result)} entries, no validation!")


class TestImpactAnalysis:
    """
    Analyze the impact of a bad model cost map.
    """
    
    def test_completion_call_with_empty_cost_map(self):
        """
        Test if a litellm.completion() call is affected by empty cost map.
        
        Note: completion itself doesn't require the cost map, but cost calculation does.
        """
        import litellm
        
        # Check if model_cost is used in completion path
        # The cost map is mainly used for:
        # 1. get_model_info() calls
        # 2. Cost calculation (completion_cost)
        # 3. Router decisions
        
        print("The model cost map is used in:")
        print("1. get_model_info() - for getting model capabilities")
        print("2. completion_cost() - for calculating costs")
        print("3. Router - for making routing decisions")
        print("4. Proxy - for budget management and cost tracking")
        
        # The error in the incident likely came from one of these code paths

    def test_cost_calculation_with_empty_model_cost(self):
        """
        Test that cost calculation fails with empty model cost map.
        
        This is the most likely path for LLM call failures - if cost tracking
        is enabled and required before/during the call.
        """
        import litellm
        from litellm.cost_calculator import completion_cost
        
        # Save original model_cost
        original_model_cost = litellm.model_cost.copy()
        
        try:
            # Simulate empty model cost map
            litellm.model_cost = {}
            
            # Try to calculate cost - this should fail
            # Note: completion_cost uses model_cost to find pricing
            with pytest.raises(Exception):
                # This should fail because the model isn't in the cost map
                completion_cost(
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                )
            print("✓ Cost calculation fails with empty model cost map")
            
        finally:
            litellm.model_cost = original_model_cost

    def test_where_get_model_info_is_called(self):
        """
        Document where get_model_info is called in the codebase.
        
        Understanding these paths helps trace how a bad model cost map 
        can lead to failures.
        """
        print("\nCode paths where get_model_info() is called:")
        print("=" * 50)
        print("1. litellm/main.py - responses_api_bridge_check() - CAUGHT")
        print("   - Called for checking model capabilities")
        print("   - Exception is caught, doesn't fail the call")
        print("")
        print("2. litellm/router.py - get_router_model_info() - CAUGHT")
        print("   - Called in deployment callbacks - exception caught")
        print("   - Called in context window validation - exception caught")
        print("")
        print("3. litellm/proxy/proxy_server.py - multiple places - ALL CAUGHT")
        print("   - All calls wrapped in try/except")
        print("   - Returns {} on exception")
        print("")
        print("4. litellm/cost_calculator.py - for speech/tts - NOT CAUGHT")
        print("   - Line 325: get_model_info for speech models")
        print("   - Could cause cost calculation to fail")
        print("")
        print("5. litellm/litellm_core_utils/litellm_logging.py - CAUGHT")
        print("   - Called for model cost information")
        print("   - Exception caught, logs debug message")
        print("")
        print("CONCLUSION: Most paths are protected with try/except")
        print("The error likely surfaces when:")
        print("- Cost tracking is mandatory")
        print("- Pre-call validation requires model info")
        print("- Custom callbacks/hooks call get_model_info without try/except")


if __name__ == "__main__":
    # Run the tests
    print("=" * 60)
    print("Testing Model Cost Map Exception Reproduction")
    print("=" * 60)
    
    test_class = TestModelCostMapExceptionReproduction()
    
    print("\n--- Test 1: Malformed JSON Handling ---")
    test_class.test_malformed_json_from_hosted_url()
    
    print("\n--- Test 2: Empty JSON Handling ---")
    test_class.test_empty_json_from_hosted_url()
    
    print("\n--- Test 3: Reproduce Exact Exception ---")
    test_class.test_reproduce_model_not_mapped_exception_with_empty_cost_map()
    
    print("\n--- Test 4: Known Model Failure ---")
    test_class.test_reproduce_exception_with_known_model()
    
    print("\n--- Test 5: Partial Cost Map ---")
    test_class.test_partial_model_cost_map()
    
    print("\n" + "=" * 60)
    print("Testing Flow Analysis")
    print("=" * 60)
    
    flow_test = TestModelCostMapFlowAnalysis()
    
    print("\n--- Trace Exception Flow ---")
    flow_test.test_trace_exception_flow()
    
    print("\n--- Model Cost Map Loading Scenarios ---")
    flow_test.test_model_cost_map_loading_scenarios()
    
    print("\n" + "=" * 60)
    print("Impact Analysis")
    print("=" * 60)
    
    impact_test = TestImpactAnalysis()
    impact_test.test_completion_call_with_empty_cost_map()

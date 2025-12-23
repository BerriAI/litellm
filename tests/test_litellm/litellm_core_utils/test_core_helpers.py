"""Tests for litellm_core_utils.core_helpers module."""

import pytest

from litellm.litellm_core_utils.core_helpers import (
    get_litellm_metadata_from_kwargs,
    get_tokens_for_tpm,
    reconstruct_model_name,
    safe_deep_copy,
    safe_divide,
)


def test_reconstruct_model_name_prefers_deployment_value():
    """Ensure deployment metadata wins when reconstructing the model name."""

    metadata = {"deployment": "vertex_ai/gemini-1.5-flash"}

    result = reconstruct_model_name(
        model_name="gemini-1.5-flash",
        custom_llm_provider="vertex_ai",
        metadata=metadata,
    )

    assert result == "vertex_ai/gemini-1.5-flash"


def test_reconstruct_model_name_adds_bedrock_prefix_when_missing():
    """Bedrock model names without prefixes should gain the provider prefix."""

    metadata = {}

    result = reconstruct_model_name(
        model_name="us.anthropic.claude-3-sonnet",
        custom_llm_provider="bedrock",
        metadata=metadata,
    )

    assert result == "bedrock/us.anthropic.claude-3-sonnet"


def test_reconstruct_model_name_returns_original_for_other_providers():
    """Non-Bedrock providers should not prepend anything."""

    metadata = {}

    result = reconstruct_model_name(
        model_name="claude-3-sonnet",
        custom_llm_provider="anthropic",
        metadata=metadata,
    )

    assert result == "claude-3-sonnet"


def test_safe_divide_weight_scenario():
    """Test safe division in the context of weight calculations"""
    # Simulate weight calculation scenario
    weights = [3, 7, 0, 2]
    total_weight = sum(weights)  # 12
    
    # Normal case
    normalized_weights = [safe_divide(w, total_weight) for w in weights]
    expected = [0.25, 7/12, 0.0, 1/6]
    
    for i, (actual, exp) in enumerate(zip(normalized_weights, expected)):
        assert abs(actual - exp) < 1e-10, f"Weight {i}: Expected {exp}, got {actual}"
    
    # Zero total weight scenario (division by zero)
    zero_weights = [0, 0, 0]
    zero_total = sum(zero_weights)  # 0
    
    # Should return default values (0) for all weights
    normalized_zero_weights = [safe_divide(w, zero_total) for w in zero_weights]
    expected_zero = [0, 0, 0]
    
    assert normalized_zero_weights == expected_zero, f"Expected {expected_zero}, got {normalized_zero_weights}"


def test_safe_deep_copy_with_non_pickleables_and_span():
    """
    Verify safe_deep_copy:
    - does not crash when non-pickleables are present,
    - preserves structure/keys,
    - deep-copies JSON-y payloads (e.g., messages),
    - keeps non-pickleables by reference,
    - redacts OTEL span in the copy and restores it in the original.
    """
    import threading
    rlock = threading.RLock()
    data = {
        "metadata": {"litellm_parent_otel_span": rlock, "x": 1},
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {"handle": rlock},
        "ok": True,
    }

    copied = safe_deep_copy(data)

    # Structure preserved
    assert set(copied.keys()) == set(data.keys())

    # Messages are deep-copied (new object, same content)
    assert copied["messages"] is not data["messages"]
    assert copied["messages"][0] == data["messages"][0]

    # Non-pickleable subtree kept by reference (no crash)
    assert copied["optional_params"] is data["optional_params"]
    assert copied["optional_params"]["handle"] is rlock

    # OTEL span: redacted in the copy, restored in original
    assert copied["metadata"]["litellm_parent_otel_span"] == "placeholder"
    assert data["metadata"]["litellm_parent_otel_span"] is rlock

    # Other simple fields unchanged
    assert copied["ok"] is True
    assert copied["metadata"]["x"] == 1


# ============================================================================
# Tests for get_tokens_for_tpm
# ============================================================================

class TestGetTokensForTpm:
    """Tests for the get_tokens_for_tpm function."""

    def test_flag_disabled_returns_total_tokens(self):
        """When exclude_cached_tokens_from_tpm is False, should return total_tokens unchanged."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = False
            
            # With usage object containing cached tokens
            usage = {"prompt_tokens_details": {"cached_tokens": 100}}
            result = get_tokens_for_tpm(500, usage)
            assert result == 500, f"Expected 500 when flag is disabled, got {result}"
            
            # With None usage
            result = get_tokens_for_tpm(500, None)
            assert result == 500, f"Expected 500 with None usage, got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_flag_enabled_with_none_usage(self):
        """When flag is enabled but usage is None, should return total_tokens."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            result = get_tokens_for_tpm(500, None)
            assert result == 500, f"Expected 500 with None usage, got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_dict_usage_with_prompt_tokens_details(self):
        """Test with dict usage containing prompt_tokens_details (Chat Completions API)."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            usage = {
                "prompt_tokens": 400,
                "completion_tokens": 100,
                "total_tokens": 500,
                "prompt_tokens_details": {
                    "cached_tokens": 200
                }
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 300, f"Expected 500-200=300, got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_dict_usage_with_input_tokens_details(self):
        """Test with dict usage containing input_tokens_details (Responses API)."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            usage = {
                "input_tokens": 400,
                "output_tokens": 100,
                "total_tokens": 500,
                "input_tokens_details": {
                    "cached_tokens": 150
                }
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 350, f"Expected 500-150=350, got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_dict_usage_input_tokens_details_takes_priority(self):
        """Test that input_tokens_details is checked before prompt_tokens_details."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            # Both present - input_tokens_details should be used
            usage = {
                "total_tokens": 500,
                "input_tokens_details": {
                    "cached_tokens": 100
                },
                "prompt_tokens_details": {
                    "cached_tokens": 200
                }
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 400, f"Expected 500-100=400 (input_tokens_details), got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_usage_object_with_prompt_tokens_details(self):
        """Test with Usage object containing prompt_tokens_details."""
        import litellm
        from litellm.types.utils import Usage, PromptTokensDetailsWrapper
        
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            usage = Usage(
                prompt_tokens=400,
                completion_tokens=100,
                total_tokens=500,
                prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=250)
            )
            result = get_tokens_for_tpm(500, usage)
            assert result == 250, f"Expected 500-250=250, got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_no_cached_tokens_returns_total(self):
        """Test when there are no cached tokens in usage."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            # Dict without cached_tokens
            usage = {
                "prompt_tokens": 400,
                "completion_tokens": 100,
                "total_tokens": 500,
                "prompt_tokens_details": {}
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 500, f"Expected 500 (no cached tokens), got {result}"
            
            # Dict with cached_tokens = 0
            usage = {
                "total_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 0}
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 500, f"Expected 500 (cached_tokens=0), got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_cached_tokens_greater_than_total_returns_zero(self):
        """Test that result is never negative (uses max(0, ...))."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            # Edge case: cached_tokens > total_tokens (shouldn't happen, but handle gracefully)
            usage = {
                "total_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 200}
            }
            result = get_tokens_for_tpm(100, usage)
            assert result == 0, f"Expected 0 (not negative), got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_cached_tokens_none_treated_as_zero(self):
        """Test that None cached_tokens is treated as 0."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            usage = {
                "total_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": None}
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 500, f"Expected 500 (None treated as 0), got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_fallback_to_prompt_tokens_details_when_input_is_zero(self):
        """Test fallback to prompt_tokens_details when input_tokens_details has 0 cached."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            usage = {
                "total_tokens": 500,
                "input_tokens_details": {"cached_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 150}
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 350, f"Expected 500-150=350 (fallback to prompt_tokens_details), got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

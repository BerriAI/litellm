"""
Test for _is_prompt_management_model early exit optimization.

Verifies that the early return for models without "/" doesn't break
prompt management model detection.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router


def test_is_prompt_management_model_optimization():
    """
    Test early exit optimization works correctly for all cases.
    
    Optimization: Check if "/" in model name before calling expensive
    get_model_list(). This short-circuits 99% of requests that use
    standard model names like "gpt-4", "claude-3", etc.
    
    Tests both negative (early exit) and positive (actual detection) cases.
    """
    import litellm
    
    # Test 1: Standard models without "/" -> early exit returns False
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
            },
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "anthropic/claude-3-sonnet-20240229"},
            },
        ]
    )
    
    assert router._is_prompt_management_model("gpt-4") is False
    assert router._is_prompt_management_model("claude-3") is False
    
    # Test 2: Models with "/" but not in model_list -> False after check
    assert router._is_prompt_management_model("unknown/model") is False
    
    # Test 3: Actual prompt management models ARE detected (critical positive case)
    original_callbacks = litellm._known_custom_logger_compatible_callbacks.copy()
    if "langfuse_prompt" not in litellm._known_custom_logger_compatible_callbacks:
        litellm._known_custom_logger_compatible_callbacks.append("langfuse_prompt")
    
    try:
        router_with_prompt = Router(
            model_list=[
                {
                    "model_name": "my-langfuse-prompt/test_id",
                    "litellm_params": {"model": "langfuse_prompt/actual_prompt_id"},
                },
            ]
        )
        
        # Critical: Must still detect prompt management models correctly
        assert router_with_prompt._is_prompt_management_model("my-langfuse-prompt/test_id") is True
        
    finally:
        litellm._known_custom_logger_compatible_callbacks = original_callbacks


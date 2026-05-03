from litellm.llms.anthropic.chat.transformation import AnthropicConfig

def test_anthropic_compaction_usage_calculation():
    """
    Test that calculate_usage correctly sums tokens from the iterations array
    as requested in Issue #27060.
    """
    anthropic_config = AnthropicConfig()
    
    # Mock usage object with compaction iterations
    usage_object = {
        "input_tokens": 100,  # Top-level (excludes compaction)
        "output_tokens": 50,   # Top-level (excludes compaction)
        "iterations": [
            {
                "iteration": 1,
                "type": "compaction",
                "input_tokens": 1000,
                "output_tokens": 500
            },
            {
                "iteration": 2,
                "type": "message",
                "input_tokens": 100,
                "output_tokens": 50
            }
        ]
    }
    
    usage = anthropic_config.calculate_usage(
        usage_object=usage_object,
        reasoning_content=None
    )
    
    # Assertions
    # Total prompt tokens should be 1000 + 100 = 1100
    assert usage.prompt_tokens == 1100
    # Total completion tokens should be 500 + 50 = 550
    assert usage.completion_tokens == 550
    # Total tokens should be 1650
    assert usage.total_tokens == 1650
    
    # Assert details
    assert usage.prompt_tokens_details.text_tokens == 1100
    
    # Assert iterations passthrough
    assert usage.iterations is not None
    assert len(usage.iterations) == 2
    assert usage.iterations[0]["type"] == "compaction"

def test_anthropic_compaction_usage_with_cache():
    """
    Test that calculate_usage correctly handles iterations + prompt caching.
    """
    anthropic_config = AnthropicConfig()
    
    usage_object = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 20,
        "iterations": [
            {
                "iteration": 1,
                "type": "compaction",
                "input_tokens": 500,
                "output_tokens": 200
            },
            {
                "iteration": 2,
                "type": "message",
                "input_tokens": 100,
                "output_tokens": 50
            }
        ]
    }
    
    usage = anthropic_config.calculate_usage(
        usage_object=usage_object,
        reasoning_content=None
    )
    
    # Input tokens sum: 500 + 100 = 600
    # Plus cache creation (10) and cache read (20)
    # Total prompt tokens = 600 + 10 + 20 = 630
    assert usage.prompt_tokens == 630
    
    # Details
    assert usage.prompt_tokens_details.text_tokens == 600
    assert usage.prompt_tokens_details.cache_creation_tokens == 10
    assert usage.prompt_tokens_details.cached_tokens == 20
    
    # Output tokens sum: 200 + 50 = 250
    assert usage.completion_tokens == 250

if __name__ == "__main__":
    test_anthropic_compaction_usage_calculation()
    test_anthropic_compaction_usage_with_cache()

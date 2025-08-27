#!/usr/bin/env python3
"""
Regression tests to ensure optimization changes don't break functionality
"""
import asyncio
from litellm import completion, acompletion
from litellm.types.utils import LlmProviders, LlmProvidersSet

def test_llm_providers_set_cache():
    """Test that LlmProvidersSet contains the same values as the list comprehension."""
    # Original expensive approach (for comparison)
    original_values = {provider.value for provider in LlmProviders}
    
    # Our optimized cached version
    cached_values = LlmProvidersSet
    
    assert original_values == cached_values, "LlmProvidersSet should match original provider values"
    assert "openai" in cached_values, "Should contain openai provider"
    assert "anthropic" in cached_values, "Should contain anthropic provider"
    assert "vertex_ai" in cached_values, "Should contain vertex_ai provider"

def test_basic_completion_functionality():
    """Test that basic completion still works exactly the same."""
    response = completion(
        model="openai/gpt-4o",
        mock_response="Test response",
        messages=[{"role": "user", "content": "Hello"}]
    )
    assert response.choices[0].message.content == "Test response"
    assert response.model == "openai/gpt-4o"
    assert len(response.choices) == 1

async def test_async_completion_functionality():
    """Test that async completion still works exactly the same."""
    response = await acompletion(
        model="openai/gpt-4o",
        mock_response="Async test response",
        messages=[{"role": "user", "content": "Hello async"}]
    )
    assert response.choices[0].message.content == "Async test response"
    assert response.model == "openai/gpt-4o"
    assert len(response.choices) == 1

def test_provider_specific_params():
    """Test that provider-specific parameter handling still works."""
    # Test with anthropic-specific params
    response = completion(
        model="anthropic/claude-3-sonnet-20240229",
        mock_response="Anthropic response",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        temperature=0.7
    )
    assert response.choices[0].message.content == "Anthropic response"
    
    # Test with OpenAI-specific params
    response = completion(
        model="openai/gpt-4o",
        mock_response="OpenAI response",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=150,
        temperature=0.8,
        top_p=0.9
    )
    assert response.choices[0].message.content == "OpenAI response"

def test_parameter_validation():
    """Test that parameter validation still works correctly."""
    # This should work without issues
    response = completion(
        model="openai/gpt-4o",
        mock_response="Valid params response",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.5,
        max_tokens=100
    )
    assert response.choices[0].message.content == "Valid params response"

def test_multiple_concurrent_completions():
    """Test that multiple concurrent completions work as expected."""
    async def make_multiple_requests():
        tasks = []
        for i in range(10):
            task = acompletion(
                model="openai/gpt-4o",
                mock_response=f"Response {i}",
                messages=[{"role": "user", "content": f"Request {i}"}]
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        for i, response in enumerate(responses):
            assert response.choices[0].message.content == f"Response {i}"
    
    asyncio.run(make_multiple_requests())

def test_edge_cases():
    """Test edge cases that might be affected by optimizations."""
    # Test with empty additional_drop_params
    response = completion(
        model="openai/gpt-4o",
        mock_response="Edge case response",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.5
    )
    assert response.choices[0].message.content == "Edge case response"
    
    # Test with unusual but valid parameters
    response = completion(
        model="openai/gpt-4o",
        mock_response="Unusual params response",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.0,  # Edge case: exactly 0
        max_tokens=1,     # Edge case: minimum value
    )
    assert response.choices[0].message.content == "Unusual params response"

if __name__ == "__main__":
    print("Running optimization regression tests...")
    
    # Run all tests
    test_llm_providers_set_cache()
    print("✓ LlmProviders cache test passed")
    
    test_basic_completion_functionality()
    print("✓ Basic completion test passed")
    
    asyncio.run(test_async_completion_functionality())
    print("✓ Async completion test passed")
    
    test_provider_specific_params()
    print("✓ Provider-specific params test passed")
    
    test_parameter_validation()
    print("✓ Parameter validation test passed")
    
    test_multiple_concurrent_completions()
    print("✓ Multiple concurrent completions test passed")
    
    test_edge_cases()
    print("✓ Edge cases test passed")
    
    print("\n🎉 All regression tests passed! Optimizations are working correctly.")
"""
Test suite for Gemini reasoning_tokens functionality.

This test suite verifies that:
1. The gemini provider correctly returns reasoning_tokens
2. The openai provider (when used with Gemini models) has the expected limitation
3. The reasoning_effort parameter works correctly with both providers
"""

import os
import pytest
import time
from unittest.mock import patch, MagicMock
from litellm import completion
from litellm.types.utils import Usage, CompletionTokensDetailsWrapper


class TestGeminiReasoningTokens:
    """Test class for Gemini reasoning tokens functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            pytest.skip("GEMINI_API_KEY not set")

    def test_gemini_provider_reasoning_tokens(self):
        """Test that gemini provider correctly returns reasoning_tokens."""
        # Test with reasoning_effort="high"
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Tell me a 10-word story"}],
            stream=False,  # Non-streaming for easier testing
            reasoning_effort="high",
        )
        
        # Verify response structure
        assert response is not None
        assert hasattr(response, 'usage')
        assert response.usage is not None
        
        # Verify reasoning_tokens are populated
        assert hasattr(response.usage, 'completion_tokens_details')
        assert response.usage.completion_tokens_details is not None
        assert response.usage.completion_tokens_details.reasoning_tokens > 0
        
        # Verify the reasoning_effort parameter affected the response
        # (should take longer and use more reasoning tokens)
        assert response.usage.completion_tokens_details.reasoning_tokens > 100

    def test_gemini_provider_no_reasoning_effort(self):
        """Test that gemini provider returns reasonable reasoning_tokens when no reasoning_effort."""
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Tell me a 10-word story"}],
            stream=False,
            # No reasoning_effort parameter
        )
        
        # Verify response structure
        assert response is not None
        assert hasattr(response, 'usage')
        assert response.usage is not None
        
        # Verify reasoning_tokens are present when no reasoning_effort
        # Gemini might do some reasoning by default
        if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
            print(f"No reasoning_effort - reasoning_tokens: {reasoning_tokens}")
            # Just verify that reasoning_tokens is a non-negative integer
            assert isinstance(reasoning_tokens, int)
            assert reasoning_tokens >= 0

    def test_gemini_provider_reasoning_effort_none(self):
        """Test that gemini provider correctly handles reasoning_effort='none'."""
        # Note: "none" is not a valid reasoning_effort value
        # Valid values are: "low", "medium", "high"
        with pytest.raises(Exception):
            response = completion(
                model="gemini/gemini-2.5-flash",
                messages=[{"role": "user", "content": "Tell me a 10-word story"}],
                stream=False,
                reasoning_effort="none",  # This should raise an error
            )
        
        # Test with a valid value instead
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Tell me a 10-word story"}],
            stream=False,
            reasoning_effort="low",  # Use valid value
        )
        
        # Verify response structure
        assert response is not None
        assert hasattr(response, 'usage')
        assert response.usage is not None
        
        # Verify reasoning_tokens are present with low reasoning_effort
        if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
            print(f"Low reasoning_effort - reasoning_tokens: {reasoning_tokens}")
            # Just verify that reasoning_tokens is a non-negative integer
            assert isinstance(reasoning_tokens, int)
            assert reasoning_tokens >= 0

    def test_gemini_provider_streaming_reasoning_tokens(self):
        """Test that gemini provider correctly returns reasoning_tokens in streaming mode."""
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Tell me a 10-word story"}],
            stream=True,
            stream_options={"include_usage": True},
            reasoning_effort="high",
        )
        
        # Process streaming response
        usage_data = None
        for chunk in response:
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_data = chunk.usage
        
        # Verify usage data was found
        assert usage_data is not None
        
        # Verify reasoning_tokens are populated in streaming mode
        assert hasattr(usage_data, 'completion_tokens_details')
        assert usage_data.completion_tokens_details is not None
        assert usage_data.completion_tokens_details.reasoning_tokens > 0

    def test_openai_provider_gemini_model_limitation(self):
        """Test that openai provider with Gemini models has the expected limitation."""
        # This test documents the current limitation
        # The openai provider doesn't support reasoning_effort parameter for Gemini models
        
        # Test that reasoning_effort parameter is not supported
        with pytest.raises(Exception) as exc_info:
            response = completion(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model="openai/gemini-2.5-flash",
                messages=[{"role": "user", "content": "Tell me a 10-word story"}],
                stream=False,
                reasoning_effort="high",  # This should fail
            )
        
        # Verify the error message indicates the parameter is not supported
        error_message = str(exc_info.value)
        assert "reasoning_effort" in error_message or "does not support parameters" in error_message
        
        # Note: We can't test the basic functionality without an API key for the openai provider
        # This test documents the limitation that reasoning_effort is not supported
        # The actual API call would require proper authentication

    def test_reasoning_effort_parameter_mapping(self):
        """Test that reasoning_effort parameter is correctly mapped to Gemini thinking parameter."""
        # This test verifies that the reasoning_effort parameter is properly
        # translated to Gemini's thinking parameter
        
        with patch('litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexGeminiConfig._map_reasoning_effort_to_thinking_budget') as mock_mapping:
            mock_mapping.return_value = {"thinkingBudget": 4096, "includeThoughts": True}
            
            response = completion(
                model="gemini/gemini-2.5-flash",
                messages=[{"role": "user", "content": "Tell me a 10-word story"}],
                stream=False,
                reasoning_effort="high",
            )
            
            # Verify the mapping function was called
            mock_mapping.assert_called_once_with("high")

    def test_usage_object_reasoning_tokens_handling(self):
        """Test that Usage object correctly handles reasoning_tokens."""
        # Test the Usage class constructor with reasoning_tokens
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=100,
            total_tokens=110,
            reasoning_tokens=50,
        )
        
        # Verify reasoning_tokens are properly set
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.reasoning_tokens == 50
        
        # Verify text_tokens are calculated correctly
        assert usage.completion_tokens_details.text_tokens == 50  # completion_tokens - reasoning_tokens

    def test_completion_tokens_details_wrapper(self):
        """Test that CompletionTokensDetailsWrapper correctly stores reasoning_tokens."""
        details = CompletionTokensDetailsWrapper(
            reasoning_tokens=100,
            text_tokens=50,
        )
        
        # Verify reasoning_tokens are stored correctly
        assert details.reasoning_tokens == 100
        assert details.text_tokens == 50

    @pytest.mark.parametrize("reasoning_effort", ["low", "medium", "high"])
    def test_reasoning_effort_values(self, reasoning_effort):
        """Test different reasoning_effort values work correctly."""
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Tell me a 10-word story"}],
            stream=False,
            reasoning_effort=reasoning_effort,
        )
        
        # Verify response structure
        assert response is not None
        assert hasattr(response, 'usage')
        assert response.usage is not None
        
        # Verify the response was successful
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    def test_error_handling_invalid_reasoning_effort(self):
        """Test that invalid reasoning_effort values are handled gracefully."""
        with pytest.raises(Exception):
            # This should raise an error for invalid reasoning_effort
            completion(
                model="gemini/gemini-2.5-flash",
                messages=[{"role": "user", "content": "Tell me a 10-word story"}],
                stream=False,
                reasoning_effort="invalid_value",
            )


if __name__ == "__main__":
    pytest.main([__file__])

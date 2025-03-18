import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class TestReasoningToThinking:
    """Tests for translating OpenAI reasoning_effort to Anthropic thinking parameter"""

    @pytest.mark.parametrize(
        "reasoning_effort, expected_thinking_type, expected_max_tokens",
        [
            ("low", "enabled", 12000),
            ("medium", "enabled", 24000),
            ("high", "enabled", 36000),
            ("research_effort", "enabled", 24000),
        ],
    )
    def test_anthropic_reasoning_effort_parameter(self, reasoning_effort, expected_thinking_type, expected_max_tokens):
        """Test that reasoning_effort is correctly mapped to thinking parameter in Anthropic requests"""
        # Create an instance of AnthropicConfig
        config = AnthropicConfig()
        
        # Verify reasoning_effort is supported for Claude 3.7 Sonnet
        supported_params = config.get_supported_openai_params("claude-3-7-sonnet")
        assert "reasoning_effort" in supported_params, "reasoning_effort should be supported for Claude 3.7 Sonnet"
        
        # Test that mapping reasoning_effort to optional_params works correctly
        optional_params = {}
        non_default_params = {"reasoning_effort": reasoning_effort}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-3-7-sonnet",
            drop_params=False
        )
        
        # Verify the thinking parameter was correctly set
        assert "thinking" in result
        assert result["thinking"]["type"] == expected_thinking_type
        
        # Verify max_tokens was adjusted based on the token budget
        assert result["max_tokens"] == expected_max_tokens
    
    @pytest.mark.parametrize(
        "reasoning_level, expected_budget, expected_max_tokens",
        [
            ("medium", 16000, 24000),
            ("research_effort", 16000, 24000)
        ]
    )
    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_reasoning_effort_e2e(self, mock_post, reasoning_level, expected_budget, expected_max_tokens):
        """Test end-to-end flow using reasoning_effort with Claude 3.7 Sonnet"""
        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_012345",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "This is a test response"}],
            "model": "claude-3-7-sonnet",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }
        mock_response.headers = {}
        mock_post.return_value = mock_response
        
        # Make completion call with reasoning_effort
        try:
            response = litellm.completion(
                model="claude-3-7-sonnet",
                messages=[{"role": "user", "content": f"Testing {reasoning_level} to thinking mapping"}],
                reasoning_effort=reasoning_level,
                api_key="fake-key"  # Use a fake key for testing
            )
            
            # Verify the request was made with correct parameters
            called_args = mock_post.call_args
            request_body = called_args.kwargs["data"]
            
            # Check that thinking parameter was set in the request
            assert "thinking" in request_body
            assert request_body["thinking"]["type"] == "enabled"
            assert request_body["thinking"]["budget_tokens"] == expected_budget
            
            # Check that max_tokens was adjusted based on the token budget
            assert "max_tokens" in request_body
            assert request_body["max_tokens"] == expected_max_tokens
            
        except Exception as e:
            pytest.fail(f"Test failed with exception: {e}")
    
    def test_reasoning_effort_not_sent_to_non_supported_models(self):
        """Test that reasoning_effort is not sent to models that don't support it"""
        config = AnthropicConfig()
        
        # Verify reasoning_effort is not supported for Claude 3.5 Sonnet
        supported_params = config.get_supported_openai_params("claude-3-5-sonnet")
        assert "reasoning_effort" not in supported_params, "reasoning_effort should not be supported for Claude 3.5 Sonnet"
        
        # Test that mapping reasoning_effort to optional_params doesn't include it for unsupported models
        optional_params = {}
        non_default_params = {"reasoning_effort": "medium"}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-3-5-sonnet",
            drop_params=True
        )
        
        # Verify the thinking parameter was not set for unsupported models
        assert "thinking" not in result
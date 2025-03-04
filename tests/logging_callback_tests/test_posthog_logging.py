import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path

import litellm
from litellm.integrations.posthog import PosthogLogger
from tests.logging_callback_tests.base_test import BaseLoggingCallbackTest
from litellm.types.utils import ModelResponse


class TestPosthogLogger(BaseLoggingCallbackTest):
    """
    Test class for PosthogLogger
    """
    
    @pytest.fixture
    def mock_posthog(self):
        """Mock the posthog module"""
        with patch("posthog.capture") as mock_capture:
            with patch.dict(os.environ, {"POSTHOG_API_KEY": "test_key"}):
                yield mock_capture
    
    @pytest.fixture
    def posthog_logger(self, mock_posthog):
        """Create a PosthogLogger instance with mocked posthog"""
        with patch("posthog.api_key", "test_key"):
            with patch("posthog.host", "https://us.i.posthog.com"):
                logger = PosthogLogger()
                logger.posthog.capture = mock_posthog
                return logger
    
    def test_parallel_tool_calls(self, mock_response_obj: ModelResponse, posthog_logger):
        """
        Check if parallel tool calls are correctly logged by PosthogLogger
        
        Relevant issue - https://github.com/BerriAI/litellm/issues/6677
        """
        # Setup
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What's the weather and news in New York?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather in a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            },
                            "required": ["city"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_news",
                        "description": "Get the latest news for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            },
                            "required": ["city"]
                        }
                    }
                }
            ],
            "litellm_call_id": "test-call-id",
            "litellm_params": {"metadata": {"user_id": "test-user"}}
        }
        
        # Execute
        posthog_logger.log_success_event(kwargs, mock_response_obj, start_time, end_time)
        
        # Verify
        posthog_logger.posthog.capture.assert_called_once()
        
        # Get the call arguments
        call_args = posthog_logger.posthog.capture.call_args[1]
        
        # Verify the event name
        assert call_args["event"] == "$ai_generation"
        
        # Verify the user ID
        assert call_args["distinct_id"] == "test-user"
        
        # Verify properties
        properties = call_args["properties"]
        assert properties["$ai_trace_id"] == "test-call-id"
        assert properties["$ai_model"] == "gpt-4o-mini"
        assert properties["$ai_provider"] == "openai"
        assert properties["$ai_is_error"] is False
        assert properties["$ai_http_status"] == 200
        
        # Verify tool calls are included
        assert "$ai_tools" in properties
        assert len(properties["$ai_tools"]) == 2
        
        # Verify output choices contain tool calls
        assert "$ai_output_choices" in properties
        output_choices = properties["$ai_output_choices"]
        assert len(output_choices) == 1
        
        # Verify tool calls in the output
        tool_calls = output_choices[0]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "get_news"
    
    def test_embedding_logging(self, posthog_logger):
        """Test that embedding events are correctly logged"""
        # Setup
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        
        kwargs = {
            "model": "text-embedding-ada-002",
            "input": ["Embed this text"],
            "litellm_call_id": "test-embedding-id",
            "litellm_params": {"metadata": {"user_id": "test-user"}},
            "call_type": "embeddings"
        }
        
        # Mock response object for embeddings
        response_obj = MagicMock()
        response_obj.usage.prompt_tokens = 5
        response_obj.data = [MagicMock()]
        response_obj.data[0].embedding = [0.1, 0.2, 0.3, 0.4]  # 4-dimensional embedding
        
        # Execute
        posthog_logger.log_success_event(kwargs, response_obj, start_time, end_time)
        
        # Verify
        posthog_logger.posthog.capture.assert_called_once()
        
        # Get the call arguments
        call_args = posthog_logger.posthog.capture.call_args[1]
        
        # Verify the event name
        assert call_args["event"] == "$ai_embedding"
        
        # Verify properties
        properties = call_args["properties"]
        assert properties["$ai_trace_id"] == "test-embedding-id"
        assert properties["$ai_model"] == "text-embedding-ada-002"
        assert properties["$ai_input"] == ["Embed this text"]
    
    def test_failure_logging(self, posthog_logger):
        """Test that failure events are correctly logged"""
        # Setup
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Generate text that will fail"}],
            "litellm_call_id": "test-failure-id",
            "call_type": "completion",
            "litellm_params": {"metadata": {"user_id": "test-user"}}
        }
        
        # Error response
        error_response = Exception("Rate limit exceeded")
        
        # Execute
        posthog_logger.log_failure_event(kwargs, error_response, start_time, end_time)
        
        # Verify
        posthog_logger.posthog.capture.assert_called_once()
        
        # Get the call arguments
        call_args = posthog_logger.posthog.capture.call_args[1]
        
        # Verify the event name
        assert call_args["event"] == "$ai_generation"
        
        # Verify properties
        properties = call_args["properties"]
        assert properties["$ai_trace_id"] == "test-failure-id"
        assert properties["$ai_model"] == "gpt-4"
        assert properties["$ai_provider"] == "openai"
        assert properties["$ai_is_error"] is True
        assert properties["$ai_error"] == "Rate limit exceeded"
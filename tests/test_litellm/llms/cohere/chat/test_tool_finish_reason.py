"""Test that Cohere tool calls set finish_reason correctly"""
import pytest
from unittest.mock import Mock
from litellm.llms.cohere.chat.transformation import CohereChatConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from litellm.types.utils import ModelResponse


class TestCohereToolCallFinishReason:
    def test_cohere_v1_tool_calls_finish_reason(self):
        """Test that v1 transformation sets finish_reason to 'tool_calls' when tool calls are present"""
        # Create response with tool calls
        response_json = {
            "text": "",
            "tool_calls": [
                {
                    "name": "get_weather",
                    "generation_id": "test123",
                    "parameters": {
                        "location": "San Francisco",
                        "unit": "celsius"
                    }
                }
            ],
            "meta": {
                "billed_units": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            }
        }
        
        # Create mock response
        mock_response = Mock()
        mock_response.json.return_value = response_json
        mock_response.status_code = 200
        
        # Test transformation
        config = CohereChatConfig()
        model_response = ModelResponse()
        
        result = config.transform_response(
            model="command-r",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None
        )
        
        # Verify finish_reason is 'tool_calls'
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0].function.name == "get_weather"
        
    def test_cohere_v2_tool_calls_finish_reason(self):
        """Test that v2 transformation sets finish_reason to 'tool_calls' when tool calls are present"""
        # Create response with tool calls in v2 format
        response_json = {
            "message": {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {
                        "id": "test123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco", "unit": "celsius"}'
                        }
                    }
                ]
            },
            "usage": {
                "tokens": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            }
        }
        
        # Create mock response
        mock_response = Mock()
        mock_response.json.return_value = response_json
        mock_response.status_code = 200
        
        # Test transformation
        config = CohereV2ChatConfig()
        model_response = ModelResponse()
        
        result = config.transform_response(
            model="command-r-08-2024",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None
        )
        
        # Verify finish_reason is 'tool_calls'
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0].function.name == "get_weather"
        
    def test_cohere_v1_no_tool_calls_finish_reason(self):
        """Test that v1 transformation sets finish_reason to 'stop' when no tool calls are present"""
        # Create response without tool calls
        response_json = {
            "text": "The weather in San Francisco is nice today.",
            "meta": {
                "billed_units": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            }
        }
        
        # Create mock response
        mock_response = Mock()
        mock_response.json.return_value = response_json
        mock_response.status_code = 200
        
        # Test transformation
        config = CohereChatConfig()
        model_response = ModelResponse()
        
        result = config.transform_response(
            model="command-r",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None
        )
        
        # Verify finish_reason is 'stop' (default)
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message.content == "The weather in San Francisco is nice today."
        assert result.choices[0].message.tool_calls is None
        
    def test_cohere_v2_no_tool_calls_finish_reason(self):
        """Test that v2 transformation sets finish_reason to 'stop' when no tool calls are present"""
        # Create response without tool calls in v2 format
        response_json = {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The weather in San Francisco is nice today."
                    }
                ]
            },
            "usage": {
                "tokens": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            }
        }
        
        # Create mock response
        mock_response = Mock()
        mock_response.json.return_value = response_json
        mock_response.status_code = 200
        
        # Test transformation
        config = CohereV2ChatConfig()
        model_response = ModelResponse()
        
        result = config.transform_response(
            model="command-r-08-2024",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None
        )
        
        # Verify finish_reason is 'stop' (default)
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message.content == "The weather in San Francisco is nice today."
        assert result.choices[0].message.tool_calls is None
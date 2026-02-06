import datetime
import httpx
import pytest
import json
from unittest.mock import patch, MagicMock

from litellm import ModelResponse
from litellm.llms.oci.chat.transformation import (
    OCIChatConfig,
    get_vendor_from_model,
    OCIStreamWrapper,
)
from litellm.types.llms.oci import OCIVendors

# Test constants
TEST_COMPARTMENT_ID = "ocid1.compartment.oc1..xxxxxx"
BASE_OCI_PARAMS = {
    "oci_region": "us-ashburn-1",
    "oci_user": "ocid1.user.oc1..xxxxxxEXAMPLExxxxxx",
    "oci_fingerprint": "4f:29:77:cc:b1:3e:55:ab:61:2a:de:47:f1:38:4c:90",
    "oci_tenancy": "ocid1.tenancy.oc1..xxxxxxEXAMPLExxxxxx",
    "oci_compartment_id": TEST_COMPARTMENT_ID,
    "oci_key_file": "/path/to/private_key.pem",
}


class TestOCICohereToolCalls:
    """Test Cohere tool calling functionality"""

    def test_cohere_tool_definition_transformation(self):
        """Test that OpenAI tool definitions are correctly transformed to Cohere format"""
        config = OCIChatConfig()
        
        # OpenAI format tools
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city or location to get weather for"
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit (celsius or fahrenheit)",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
        
        # Transform tools
        cohere_tools = config.adapt_tool_definitions_to_cohere_standard(openai_tools)
        
        # Verify transformation
        assert len(cohere_tools) == 2
        
        # Check first tool
        weather_tool = cohere_tools[0]
        assert weather_tool.name == "get_weather"
        assert weather_tool.description == "Get current weather for a location"
        assert "location" in weather_tool.parameterDefinitions
        assert "unit" in weather_tool.parameterDefinitions
        
        # Check location parameter
        location_param = weather_tool.parameterDefinitions["location"]
        assert location_param.description == "The city or location to get weather for"
        assert location_param.type == "string"
        assert location_param.isRequired == True
        
        # Check unit parameter
        unit_param = weather_tool.parameterDefinitions["unit"]
        assert unit_param.description == "Temperature unit (celsius or fahrenheit)"
        assert unit_param.type == "string"
        assert unit_param.isRequired == False
        
        # Check second tool
        calc_tool = cohere_tools[1]
        assert calc_tool.name == "calculate"
        assert calc_tool.description == "Perform mathematical calculations"
        assert "expression" in calc_tool.parameterDefinitions
        
        expression_param = calc_tool.parameterDefinitions["expression"]
        assert expression_param.description == "Mathematical expression to evaluate"
        assert expression_param.type == "string"
        assert expression_param.isRequired == True

    def test_cohere_request_with_tools(self):
        """Test request transformation for Cohere models with tools"""
        config = OCIChatConfig()
        messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city or location to get weather for"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "tools": tools,
        }

        transformed_request = config.transform_request(
            model="cohere.command-latest",
            messages=messages,  # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Verify basic structure
        assert transformed_request["compartmentId"] == TEST_COMPARTMENT_ID
        assert transformed_request["servingMode"]["servingType"] == "ON_DEMAND"
        assert transformed_request["servingMode"]["modelId"] == "cohere.command-latest"
        
        # Verify Cohere-specific structure
        chat_request = transformed_request["chatRequest"]
        assert chat_request["apiFormat"] == "COHERE"
        assert chat_request["message"] == "What's the weather like in Tokyo?"
        assert chat_request["chatHistory"] == []
        
        # Verify default parameters are included
        assert chat_request["maxTokens"] == 600
        assert chat_request["temperature"] == 1
        assert chat_request["topK"] == 0
        assert chat_request["topP"] == 0.75
        assert chat_request["frequencyPenalty"] == 0
        
        # Verify tools are transformed correctly
        assert "tools" in chat_request
        assert len(chat_request["tools"]) == 1
        tool = chat_request["tools"][0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get current weather for a location"
        assert "parameterDefinitions" in tool
        assert "location" in tool["parameterDefinitions"]

    def test_cohere_response_with_tool_calls(self):
        """Test response transformation for Cohere models with tool calls"""
        config = OCIChatConfig()
        
        # Mock Cohere response with tool calls
        mock_cohere_response = {
            "modelId": "cohere.command-latest",
            "modelVersion": "1.0",
            "chatResponse": {
                "apiFormat": "COHERE",
                "text": "I will look up the weather in Tokyo.",
                "finishReason": "COMPLETE",
                "toolCalls": [
                    {
                        "name": "get_weather",
                        "parameters": {
                            "location": "Tokyo"
                        }
                    }
                ],
                "usage": {
                    "promptTokens": 26,
                    "completionTokens": 22,
                    "totalTokens": 48
                }
            }
        }

        response = httpx.Response(
            status_code=200, 
            json=mock_cohere_response, 
            headers={"Content-Type": "application/json"}
        )
        
        result = config.transform_response(
            model="cohere.command-latest",
            raw_response=response,
            model_response=ModelResponse(),
            logging_obj={},  # type: ignore
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding={},
        )

        # Verify response structure
        assert isinstance(result, ModelResponse)
        assert result.model == "cohere.command-latest"
        assert result.choices[0].message.content == "I will look up the weather in Tokyo."
        
        # Verify tool calls are present
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        
        tool_call = result.choices[0].message.tool_calls[0]
        assert tool_call.id == "call_0"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments == '{"location": "Tokyo"}'
        
        # Verify usage
        assert result.usage.prompt_tokens == 26
        assert result.usage.completion_tokens == 22
        assert result.usage.total_tokens == 48

    def test_cohere_chat_history_with_tool_calls(self):
        """Test chat history transformation with tool calls"""
        config = OCIChatConfig()
        
        messages = [
            {"role": "user", "content": "What's the weather like in Tokyo?"},
            {
                "role": "assistant", 
                "content": "I will look up the weather in Tokyo.",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": "The weather in Tokyo is 22°C with partly cloudy skies.",
                "tool_call_id": "call_0"
            }
        ]
        
        chat_history = config.adapt_messages_to_cohere_standard(messages)
        
        # Verify chat history structure (excludes last message)
        assert len(chat_history) == 2
        
        # Check user message
        user_msg = chat_history[0]
        assert user_msg.role == "USER"
        assert user_msg.message == "What's the weather like in Tokyo?"
        
        # Check assistant message with tool calls
        assistant_msg = chat_history[1]
        assert assistant_msg.role == "CHATBOT"
        assert assistant_msg.message == "I will look up the weather in Tokyo."
        assert assistant_msg.toolCalls is not None
        assert len(assistant_msg.toolCalls) == 1
        assert assistant_msg.toolCalls[0].name == "get_weather"
        # The parameters should be parsed as JSON
        assert assistant_msg.toolCalls[0].parameters == {"location": "Tokyo"}
        
        # Note: The tool message (last message) is excluded from chat history
        # This is the expected behavior for Cohere models

    def test_cohere_streaming_chunk_handling(self):
        """Test Cohere streaming chunk handling"""
        # Mock the required parameters
        mock_stream = MagicMock()
        mock_model = "cohere.command-latest"
        mock_logging = MagicMock()
        
        stream_wrapper = OCIStreamWrapper(
            completion_stream=mock_stream,
            model=mock_model,
            logging_obj=mock_logging
        )
        
        # Mock Cohere streaming chunk
        cohere_chunk = {
            "apiFormat": "COHERE",
            "text": "I will look up the weather",
            "index": 0
        }
        
        chunk_data = f"data: {json.dumps(cohere_chunk)}"
        result = stream_wrapper.chunk_creator(chunk_data)
        
        # Verify streaming chunk structure
        assert result.choices[0].delta.content == "I will look up the weather"
        assert result.choices[0].index == 0
        assert result.choices[0].finish_reason is None

    def test_cohere_streaming_finish_chunk(self):
        """Test Cohere streaming finish chunk handling"""
        # Mock the required parameters
        mock_stream = MagicMock()
        mock_model = "cohere.command-latest"
        mock_logging = MagicMock()
        
        stream_wrapper = OCIStreamWrapper(
            completion_stream=mock_stream,
            model=mock_model,
            logging_obj=mock_logging
        )
        
        # Mock Cohere finish chunk
        cohere_finish_chunk = {
            "apiFormat": "COHERE",
            "text": ".",
            "index": 0,
            "finishReason": "COMPLETE"
        }
        
        chunk_data = f"data: {json.dumps(cohere_finish_chunk)}"
        result = stream_wrapper.chunk_creator(chunk_data)
        
        # Verify finish chunk structure
        assert result.choices[0].delta.content == "."
        assert result.choices[0].index == 0
        assert result.choices[0].finish_reason == "stop"  # COMPLETE is mapped to stop

    def test_cohere_parameter_mapping_excludes_tool_choice(self):
        """Test that tool_choice is excluded from Cohere parameter mapping"""
        config = OCIChatConfig()
        supported_params = config.get_supported_openai_params("cohere.command-latest")
        
        # Should support standard parameters
        assert "stream" in supported_params
        assert "max_tokens" in supported_params
        assert "temperature" in supported_params
        assert "tools" in supported_params
        assert "top_p" in supported_params
        
        # Should NOT support tool_choice (removed for Cohere)
        assert "tool_choice" not in supported_params

    def test_cohere_default_parameters(self):
        """Test that Cohere requests include required default parameters"""
        config = OCIChatConfig()
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"oci_compartment_id": TEST_COMPARTMENT_ID}

        transformed_request = config.transform_request(
            model="cohere.command-latest",
            messages=messages,  # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        chat_request = transformed_request["chatRequest"]
        
        # Verify all required default parameters are present
        assert chat_request["maxTokens"] == 600
        assert chat_request["temperature"] == 1
        assert chat_request["topK"] == 0
        assert chat_request["topP"] == 0.75
        assert chat_request["frequencyPenalty"] == 0

    def test_cohere_parameter_override(self):
        """Test that user-provided parameters override defaults"""
        config = OCIChatConfig()
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "temperature": 0.5,
            "max_tokens": 1000,
        }

        transformed_request = config.transform_request(
            model="cohere.command-latest",
            messages=messages,  # type: ignore
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        chat_request = transformed_request["chatRequest"]
        
        # Verify user parameters override defaults
        assert chat_request["temperature"] == 0.5
        assert chat_request["maxTokens"] == 1000
        
        # Verify other defaults are still present
        assert chat_request["topK"] == 0
        assert chat_request["topP"] == 0.75
        assert chat_request["frequencyPenalty"] == 0

    def test_cohere_vendor_detection(self):
        """Test that Cohere models are correctly identified"""
        assert get_vendor_from_model("cohere.command-latest") == OCIVendors.COHERE
        assert get_vendor_from_model("cohere.command-a-03-2025") == OCIVendors.COHERE
        assert get_vendor_from_model("cohere.command-plus-latest") == OCIVendors.COHERE
        assert get_vendor_from_model("cohere.command-r-plus-08-2024") == OCIVendors.COHERE
        assert get_vendor_from_model("cohere.command-r-08-2024") == OCIVendors.COHERE

    def test_cohere_error_handling_invalid_tool_format(self):
        """Test error handling for invalid tool format"""
        config = OCIChatConfig()
        
        # Invalid tool format (missing function key)
        invalid_tools = [
            {
                "type": "function",
                "name": "get_weather",  # Missing "function" wrapper
                "description": "Get weather"
            }
        ]
        
        # The function should handle missing function key gracefully
        cohere_tools = config.adapt_tool_definitions_to_cohere_standard(invalid_tools)
        
        # Should create a tool with empty name and description
        assert len(cohere_tools) == 1
        assert cohere_tools[0].name == ""
        assert cohere_tools[0].description == ""

    def test_cohere_response_without_tool_calls(self):
        """Test response transformation without tool calls"""
        config = OCIChatConfig()
        
        mock_cohere_response = {
            "modelId": "cohere.command-latest",
            "modelVersion": "1.0",
            "chatResponse": {
                "apiFormat": "COHERE",
                "text": "Hello! How can I help you today?",
                "finishReason": "COMPLETE",
                "usage": {
                    "promptTokens": 10,
                    "completionTokens": 15,
                    "totalTokens": 25
                }
            }
        }

        response = httpx.Response(
            status_code=200, 
            json=mock_cohere_response, 
            headers={"Content-Type": "application/json"}
        )
        
        result = config.transform_response(
            model="cohere.command-latest",
            raw_response=response,
            model_response=ModelResponse(),
            logging_obj={},  # type: ignore
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding={},
        )

        # Verify response structure
        assert isinstance(result, ModelResponse)
        assert result.model == "cohere.command-latest"
        assert result.choices[0].message.content == "Hello! How can I help you today?"
        assert result.choices[0].message.tool_calls is None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 15
        assert result.usage.total_tokens == 25


class TestOCICohereStreaming:
    """Test Cohere streaming functionality"""
    
    def _create_stream_wrapper(self):
        """Helper to create OCIStreamWrapper with required parameters"""
        mock_stream = MagicMock()
        mock_model = "cohere.command-latest"
        mock_logging = MagicMock()
        
        return OCIStreamWrapper(
            completion_stream=mock_stream,
            model=mock_model,
            logging_obj=mock_logging
        )

    def test_cohere_streaming_wrapper_initialization(self):
        """Test OCIStreamWrapper initialization"""
        stream_wrapper = self._create_stream_wrapper()
        
        assert hasattr(stream_wrapper, 'chunk_creator')
        assert hasattr(stream_wrapper, '_handle_cohere_stream_chunk')
        assert hasattr(stream_wrapper, '_handle_generic_stream_chunk')

    def test_cohere_streaming_chunk_parsing(self):
        """Test parsing of Cohere streaming chunks"""
        stream_wrapper = self._create_stream_wrapper()
        
        # Test valid Cohere chunk
        cohere_chunk = {
            "apiFormat": "COHERE",
            "text": "Hello",
            "index": 0
        }
        chunk_data = f"data: {json.dumps(cohere_chunk)}"
        
        result = stream_wrapper.chunk_creator(chunk_data)
        assert result.choices[0].delta.content == "Hello"
        assert result.choices[0].index == 0

    def test_cohere_streaming_invalid_chunk_format(self):
        """Test error handling for invalid chunk format"""
        stream_wrapper = self._create_stream_wrapper()
        
        # Test invalid chunk (not starting with "data:")
        with pytest.raises(ValueError, match="Chunk does not start with 'data:'"):
            stream_wrapper.chunk_creator("invalid chunk")

    def test_cohere_streaming_non_json_chunk(self):
        """Test error handling for non-JSON chunk"""
        stream_wrapper = self._create_stream_wrapper()
        
        # Test non-JSON chunk
        with pytest.raises(json.JSONDecodeError):
            stream_wrapper.chunk_creator("data: invalid json")

    def test_cohere_streaming_generic_chunk_fallback(self):
        """Test fallback to generic chunk handling for non-Cohere chunks"""
        stream_wrapper = self._create_stream_wrapper()
        
        # Test generic chunk (no apiFormat or different apiFormat)
        generic_chunk = {
            "apiFormat": "GEMINI",
            "text": "Hello from Gemini"
        }
        chunk_data = f"data: {json.dumps(generic_chunk)}"
        
        # This should fall back to generic handling
        result = stream_wrapper.chunk_creator(chunk_data)
        # The exact structure depends on the generic handler implementation
        assert hasattr(result, 'choices')

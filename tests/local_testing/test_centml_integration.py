"""
Integration tests for CentML provider functionality.

Tests both completion and chat completion endpoints with proper parameter validation
for function calling and JSON schema support.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(
    0,
    os.path.abspath("../../"))  # Adds the parent directory to the system path

import litellm
from litellm import completion, text_completion
from litellm.exceptions import UnsupportedParamsError, AuthenticationError


class TestCentmlIntegration:
    """Integration tests for CentML provider"""

    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        # Set required environment variables for testing
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        # Use a test API key - tests will mock actual API calls
        os.environ["CENTML_API_KEY"] = "test_key_for_integration_tests"

    def get_test_tools(self):
        """Helper method to get test tools for function calling"""
        return [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state/country"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]

    def get_test_json_schema(self):
        """Helper method to get test JSON schema"""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
                "age": {
                    "type": "integer"
                },
                "city": {
                    "type": "string"
                }
            },
            "required": ["name", "age", "city"]
        }

    # Parameter Validation Tests (don't require mocking HTTP responses)

    def test_chat_completion_function_calling_blocked_on_json_only_model(self):
        """Test that function calling is blocked on models that only support JSON schema"""
        # This should raise UnsupportedParamsError
        with pytest.raises(UnsupportedParamsError) as exc_info:
            completion(model="centml/deepseek-ai/DeepSeek-R1",
                       messages=[{
                           "role": "user",
                           "content": "What's the weather in Paris?"
                       }],
                       tools=self.get_test_tools(),
                       tool_choice="auto")

        assert "tools" in str(exc_info.value)
        assert "centml does not support parameters" in str(exc_info.value)

    def test_chat_completion_function_calling_blocked_on_unsupported_model(
            self):
        """Test that function calling is blocked on models that don't support it"""
        with pytest.raises(UnsupportedParamsError) as exc_info:
            completion(model="centml/Qwen/QwQ-32B",
                       messages=[{
                           "role": "user",
                           "content": "What's the weather in Paris?"
                       }],
                       tools=self.get_test_tools(),
                       tool_choice="auto")

        assert "tools" in str(exc_info.value)

    def test_chat_completion_json_schema_blocked_on_unsupported_model(self):
        """Test that JSON schema is blocked on models that don't support it"""
        with pytest.raises(UnsupportedParamsError) as exc_info:
            completion(model="centml/Qwen/QwQ-32B",
                       messages=[{
                           "role": "user",
                           "content": "Generate a person's info"
                       }],
                       response_format={
                           "type": "json_schema",
                           "json_schema": {
                               "name": "person",
                               "schema": self.get_test_json_schema()
                           }
                       })

        assert "response_format" in str(exc_info.value)

    def test_text_completion_prompt_validation(self):
        """Test that text completion validates prompts correctly"""
        # Multiple prompts should be rejected
        with pytest.raises(Exception):  # Should raise during transformation
            text_completion(
                model="centml/meta-llama/Llama-3.3-70B-Instruct",
                prompt=["Prompt 1",
                        "Prompt 2"],  # Multiple prompts not supported
                max_tokens=50)

    # Simplified Parameter Matrix Tests - One Model Per Category

    @pytest.mark.parametrize(
        "model,supports_function_calling,supports_json_schema,description",
        [
            # One representative model per category
            ("centml/meta-llama/Llama-3.3-70B-Instruct", True, True,
             "JSON ✅ / Tool ✅"),
            ("centml/deepseek-ai/DeepSeek-R1", False, True, "JSON ✅ / Tool ❌"),
            ("centml/Qwen/QwQ-32B", False, False, "JSON ❌ / Tool ❌"),
        ])
    def test_parameter_support_matrix_simplified(self, model,
                                                 supports_function_calling,
                                                 supports_json_schema,
                                                 description):
        """Test the parameter support matrix with one representative model per category"""

        # Test function calling - we only test the validation, not actual API calls
        if not supports_function_calling:
            # Should be blocked
            with pytest.raises(UnsupportedParamsError) as exc_info:
                completion(model=model,
                           messages=[{
                               "role": "user",
                               "content": "Test"
                           }],
                           tools=self.get_test_tools())
            assert "tools" in str(
                exc_info.value
            ), f"Function calling should be blocked for {description} model {model}"

        # Test JSON schema - we only test the validation, not actual API calls
        if not supports_json_schema:
            # Should be blocked
            with pytest.raises(UnsupportedParamsError) as exc_info:
                completion(model=model,
                           messages=[{
                               "role": "user",
                               "content": "Generate JSON"
                           }],
                           response_format={
                               "type": "json_schema",
                               "json_schema": {
                                   "name": "test",
                                   "schema": self.get_test_json_schema()
                               }
                           })
            assert "response_format" in str(
                exc_info.value
            ), f"JSON schema should be blocked for {description} model {model}"

    def test_authentication_error_handling(self):
        """Test that authentication errors are properly handled"""
        # Remove API key to trigger auth error
        old_key = os.environ.get("CENTML_API_KEY")
        if "CENTML_API_KEY" in os.environ:
            del os.environ["CENTML_API_KEY"]

        try:
            with pytest.raises(
                (AuthenticationError,
                 Exception)):  # Could be various auth-related errors
                completion(model="centml/meta-llama/Llama-3.3-70B-Instruct",
                           messages=[{
                               "role": "user",
                               "content": "Hello"
                           }])
        finally:
            # Restore API key
            if old_key:
                os.environ["CENTML_API_KEY"] = old_key

    def test_drop_params_functionality(self):
        """Test that drop_params can bypass parameter validation"""
        # Enable drop_params to bypass unsupported parameter validation
        original_drop_params = getattr(litellm, 'drop_params', False)
        litellm.drop_params = True

        try:
            # Mock the actual HTTP call to avoid real API requests
            with patch(
                    'litellm.llms.openai.openai.OpenAIChatCompletion.completion'
            ) as mock_completion:
                mock_completion.return_value = litellm.ModelResponse(
                    id="test_id",
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Test"
                        },
                        "finish_reason": "stop"
                    }],
                    model="centml/Qwen/QwQ-32B",
                    usage={
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6
                    })

                # This should work even though QwQ doesn't support tools
                response = completion(
                    model="centml/Qwen/QwQ-32B",
                    messages=[{
                        "role": "user",
                        "content": "Test"
                    }],
                    tools=self.get_test_tools(
                    )  # This would normally be blocked
                )
                assert response is not None
        finally:
            # Restore original drop_params setting
            litellm.drop_params = original_drop_params

    # Tests that verify parameter passing works correctly for supported models

    def test_function_calling_params_passed_correctly(self):
        """Test that function calling parameters are passed correctly for supported models"""
        with patch('litellm.llms.openai.openai.OpenAIChatCompletion.completion'
                   ) as mock_completion:
            mock_completion.return_value = litellm.ModelResponse(
                id="test_id",
                choices=[{
                    "index": 0,
                    "message": {
                        "role":
                        "assistant",
                        "content":
                        None,
                        "tool_calls": [{
                            "id": "test_tool_call",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Paris"}'
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                model="centml/meta-llama/Llama-3.3-70B-Instruct",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                })

            # This should work - model supports function calling
            response = completion(
                model="centml/meta-llama/Llama-3.3-70B-Instruct",
                messages=[{
                    "role": "user",
                    "content": "What's the weather in Paris?"
                }],
                tools=self.get_test_tools(),
                tool_choice="auto")

            assert response is not None
            mock_completion.assert_called_once()

    def test_json_schema_params_passed_correctly(self):
        """Test that JSON schema parameters are passed correctly for supported models"""
        with patch('litellm.llms.openai.openai.OpenAIChatCompletion.completion'
                   ) as mock_completion:
            mock_completion.return_value = litellm.ModelResponse(
                id="test_id",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content":
                        '{"name": "John", "age": 30, "city": "Paris"}'
                    },
                    "finish_reason": "stop"
                }],
                model="centml/meta-llama/Llama-3.3-70B-Instruct",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                })

            # This should work - model supports JSON schema
            response = completion(
                model="centml/meta-llama/Llama-3.3-70B-Instruct",
                messages=[{
                    "role": "user",
                    "content": "Generate a person's info"
                }],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "person",
                        "schema": self.get_test_json_schema()
                    }
                })

            assert response is not None
            mock_completion.assert_called_once()

    def test_json_schema_allowed_on_json_only_model(self):
        """Test that JSON schema works on models that support it but not function calling"""
        with patch('litellm.llms.openai.openai.OpenAIChatCompletion.completion'
                   ) as mock_completion:
            mock_completion.return_value = litellm.ModelResponse(
                id="test_id",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content":
                        '{"name": "Jane", "age": 25, "city": "London"}'
                    },
                    "finish_reason": "stop"
                }],
                model="centml/deepseek-ai/DeepSeek-R1",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                })

            # This should work - model supports JSON schema
            response = completion(model="centml/deepseek-ai/DeepSeek-R1",
                                  messages=[{
                                      "role":
                                      "user",
                                      "content":
                                      "Generate a person's info"
                                  }],
                                  response_format={
                                      "type": "json_schema",
                                      "json_schema": {
                                          "name": "person",
                                          "schema":
                                          self.get_test_json_schema()
                                      }
                                  })

            assert response is not None
            mock_completion.assert_called_once()

    def test_basic_chat_completion_works(self):
        """Test that basic chat completion works on all models (no advanced parameters)"""
        test_models = [
            "centml/meta-llama/Llama-3.3-70B-Instruct",  # Supports both
            "centml/deepseek-ai/DeepSeek-R1",  # JSON only
            "centml/Qwen/QwQ-32B"  # Neither
        ]

        for model in test_models:
            with patch(
                    'litellm.llms.openai.openai.OpenAIChatCompletion.completion'
            ) as mock_completion:
                mock_completion.return_value = litellm.ModelResponse(
                    id="test_id",
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help you today?"
                        },
                        "finish_reason": "stop"
                    }],
                    model=model,
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15
                    })

                # Basic request should work for all models
                response = completion(model=model,
                                      messages=[{
                                          "role": "user",
                                          "content": "Hello"
                                      }],
                                      max_tokens=100,
                                      temperature=0.7)

                assert response is not None
                mock_completion.assert_called_once()

    def test_text_completion_basic_request(self):
        """Test basic text completion functionality"""
        with patch('litellm.llms.openai.openai.OpenAIChatCompletion.completion'
                   ) as mock_completion:
            mock_completion.return_value = litellm.ModelResponse(
                id="test_id",
                choices=[{
                    "index": 0,
                    "text": "The weather today is sunny and warm.",
                    "finish_reason": "stop"
                }],
                model="centml/meta-llama/Llama-3.3-70B-Instruct",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18
                })

            response = text_completion(
                model="centml/meta-llama/Llama-3.3-70B-Instruct",
                prompt="The weather today is",
                max_tokens=50,
                temperature=0.7)

            assert response is not None
            mock_completion.assert_called_once()

    # CentML Documentation Example Tests

    def test_centml_json_schema_documentation_example(self):
        """Test using the exact JSON schema example from CentML documentation"""
        with patch('litellm.llms.openai.openai.OpenAIChatCompletion.completion'
                   ) as mock_completion:
            mock_completion.return_value = litellm.ModelResponse(
                id="test_id",
                choices=[{
                    "index": 0,
                    "message": {
                        "role":
                        "assistant",
                        "content":
                        '{"ssid": "OfficeNetSecure", "securityProtocol": "WPA2-Enterprise", "bandwidth": "1300 Mbps"}'
                    },
                    "finish_reason": "stop"
                }],
                model="centml/deepseek-ai/DeepSeek-R1",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120
                })

            # CentML documentation JSON schema example
            schema_json = {
                "title": "WirelessAccessPoint",
                "type": "object",
                "properties": {
                    "ssid": {
                        "title": "SSID",
                        "type": "string"
                    },
                    "securityProtocol": {
                        "title": "SecurityProtocol",
                        "type": "string"
                    },
                    "bandwidth": {
                        "title": "Bandwidth",
                        "type": "string"
                    }
                },
                "required": ["ssid", "securityProtocol", "bandwidth"]
            }

            response = completion(
                model="centml/deepseek-ai/DeepSeek-R1",
                messages=[{
                    "role":
                    "system",
                    "content":
                    "You are a helpful assistant that answers in JSON."
                }, {
                    "role":
                    "user",
                    "content":
                    "The access point's SSID should be 'OfficeNetSecure', it uses WPA2-Enterprise as its security protocol, and it's capable of a bandwidth of up to 1300 Mbps on the 5 GHz band."
                }],
                max_tokens=5000,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "schema",
                        "schema": schema_json
                    }
                })

            assert response is not None
            mock_completion.assert_called_once()

    def test_centml_tool_calling_documentation_example(self):
        """Test using the exact tool calling example from CentML documentation"""
        with patch('litellm.llms.openai.openai.OpenAIChatCompletion.completion'
                   ) as mock_completion:
            mock_completion.return_value = litellm.ModelResponse(
                id="test_id",
                choices=[{
                    "index": 0,
                    "message": {
                        "role":
                        "assistant",
                        "content":
                        None,
                        "tool_calls": [{
                            "id": "tool_call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Paris, France"}'
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                model="centml/meta-llama/Llama-3.3-70B-Instruct",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120
                })

            # CentML documentation tool calling example
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description":
                    "Get current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type":
                                "string",
                                "description":
                                "City and country e.g. Bogotá, Colombia"
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }]

            response = completion(
                model="centml/meta-llama/Llama-3.3-70B-Instruct",
                messages=[{
                    "role":
                    "system",
                    "content":
                    "You are a helpful assistant. You have access to ONLY get_weather function that provides temperature information for locations."
                }, {
                    "role": "user",
                    "content": "What is the weather like in Paris today?"
                }],
                max_tokens=4096,
                tools=tools,
                tool_choice="auto")

            assert response is not None
            mock_completion.assert_called_once()

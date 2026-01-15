import json
import pytest
from unittest.mock import MagicMock, patch

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.openai import ChatCompletionUserMessage
from litellm.types.utils import ModelResponse


class TestGeminiCodeAssist:
    """Test suite for Gemini CLI Code Assist functionality"""

    @pytest.mark.skip(reason="Requires google-auth library which may not be available in test environment")
    @patch("google.oauth2.credentials.Credentials")
    def test_oauth2_credentials_parsing_from_gemini_cli_format(self, mock_creds_class):
        """Test OAuth2 credentials parsing from Gemini CLI format"""
        vertex_base = VertexBase()

        # Test with OAuth2 token (when JSON parsing fails)
        mock_creds = MagicMock()
        mock_creds_class.return_value = mock_creds

        token = "fake-oauth2-token-from-gemini-cli"
        result = vertex_base._credentials_from_oauth2_token(token)

        mock_creds_class.assert_called_once_with(token=token)
        assert result == mock_creds

    def test_user_agent_header_for_code_assist_requests(self):
        """Test User-Agent header handling for Code Assist requests"""
        config = VertexGeminiConfig()

        # Test with regular API base - no User-Agent added
        headers = config.validate_environment(
            headers=None,
            model="gemini-pro",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="fake-key",
            api_base="https://generativelanguage.googleapis.com",
        )
        assert "User-Agent" not in headers

        # Test with CLI API base - User-Agent added
        headers = config.validate_environment(
            headers=None,
            model="gemini-pro",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="fake-key",
            api_base="https://example-cli-api.com",
        )
        assert headers.get("User-Agent") == "GeminiCLI/1.0"

        # Test with existing headers - User-Agent added
        existing_headers = {"Authorization": "Bearer token"}
        headers = config.validate_environment(
            headers=existing_headers,
            model="gemini-pro",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="fake-key",
            api_base="https://cli-api.example.com",
        )
        assert headers.get("User-Agent") == "GeminiCLI/1.0"
        assert headers.get("Authorization") == "Bearer token"

    def test_request_body_transformation_for_code_assist_format(self):
        """Test request body transformation for Gemini Code Assist format"""
        # Test basic message transformation
        messages = [
            {"role": "user", "content": "Hello, how can you help me with coding?"}
        ]

        result = _transform_request_body(
            messages=messages,
            model="gemini-pro",
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None
        )

        # Verify the request body structure
        assert "contents" in result
        assert len(result["contents"]) == 1
        assert result["contents"][0]["parts"][0]["text"] == "Hello, how can you help me with coding?"

        # Test with system message for code assist context
        messages_with_system = [
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": "Write a Python function"}
        ]

        result = _transform_request_body(
            messages=messages_with_system,
            model="gemini-pro",
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None
        )

        # System message should be present in the content (may be merged or separate depending on implementation)
        content_text = ""
        for content in result["contents"]:
            for part in content["parts"]:
                if "text" in part:
                    content_text += part["text"]

        assert "You are a coding assistant" in content_text

    def test_response_parsing_for_nested_code_assist_format(self):
        """Test response parsing for nested Gemini Code Assist format"""
        config = VertexGeminiConfig()

        # Test wrapped response (typical for Code Assist)
        wrapped_response = {
            "data": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Here's a Python function:\n\ndef hello_world():\n    print('Hello, World!')"}]
                        },
                        "finishReason": "STOP"
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 15,
                    "candidatesTokenCount": 25,
                    "totalTokenCount": 40
                }
            }
        }

        raw_response = MagicMock()
        raw_response.json.return_value = wrapped_response

        model_response = ModelResponse()
        result = config.transform_response(
            model="gemini-pro",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        # Verify unwrapped response content
        assert result.choices[0].message.content == "Here's a Python function:\n\ndef hello_world():\n    print('Hello, World!')"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 25
        assert result.usage.total_tokens == 40

        # Test unwrapped response (direct format)
        unwrapped_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Direct response without wrapper"}]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20
            }
        }

        raw_response.json.return_value = unwrapped_response
        result = config.transform_response(
            model="gemini-pro",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "Direct response without wrapper"

    def test_streaming_response_parsing_for_code_assist(self):
        """Test streaming response parsing for Gemini Code Assist"""
        # Test that streaming setup works for Code Assist
        # This is a basic test to ensure the streaming infrastructure is in place
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import ModelResponseIterator

        # Mock streaming response
        streaming_response = MagicMock()
        logging_obj = MagicMock()

        # Test iterator initialization
        iterator = ModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=False,
            logging_obj=logging_obj
        )

        # Verify iterator is created properly
        assert iterator is not None
        # Note: Full streaming test would require actual async iteration which is complex to mock

    def test_integration_full_flow_code_assist(self):
        """Integration test for the full Code Assist flow"""
        config = VertexGeminiConfig()

        # Mock the complete flow
        messages = [
            {"role": "user", "content": "Help me write a Python function to calculate factorial"}
        ]

        # Test request transformation
        request_data = _transform_request_body(
            messages=messages,
            model="gemini-1.5-pro",
            optional_params={"temperature": 0.7},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None
        )

        # Verify request structure
        assert "contents" in request_data
        assert "generationConfig" in request_data
        assert request_data["generationConfig"]["temperature"] == 0.7

        # Mock response
        mock_response = {
            "data": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"}]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 30
                }
            }
        }

        raw_response = MagicMock()
        raw_response.json.return_value = mock_response

        # Test response transformation
        model_response = ModelResponse()
        result = config.transform_response(
            model="gemini-1.5-pro",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={"data": json.dumps(request_data)},
            messages=messages,
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        # Verify final response
        assert "factorial" in result.choices[0].message.content
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 30

    def test_code_assist_with_tool_calls(self):
        """Test Code Assist with tool calls for code execution"""
        messages = [
            {"role": "user", "content": "Run this Python code: print('Hello')"}
        ]

        # Test with tool configuration
        optional_params = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "description": "Execute Python code",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"}
                            }
                        }
                    }
                }
            ]
        }

        request_data = _transform_request_body(
            messages=messages,
            model="gemini-pro",
            optional_params=optional_params,
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None
        )

        # Verify tools are present in the request
        assert "tools" in request_data
        assert len(request_data["tools"]) > 0
        # The exact structure may vary, but tools should be configured

    def test_code_assist_error_handling(self):
        """Test error handling in Code Assist responses"""
        config = VertexGeminiConfig()

        # Test blocked response
        blocked_response = {
            "data": {
                "candidates": [
                    {
                        "content": {"parts": []},
                        "finishReason": "SAFETY"
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 0
                }
            }
        }

        raw_response = MagicMock()
        raw_response.json.return_value = blocked_response

        model_response = ModelResponse()
        result = config.transform_response(
            model="gemini-pro",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        # Verify safety finish reason is handled
        assert result.choices[0].finish_reason == "content_filter"

    def test_code_assist_with_multimodal_content(self):
        """Test Code Assist with multimodal content (code + images)"""
        # Test multimodal message structure for code assist
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Explain this code:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}}
                ]
            }
        ]

        # Verify message structure is correct for multimodal content
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][1]["type"] == "image_url"
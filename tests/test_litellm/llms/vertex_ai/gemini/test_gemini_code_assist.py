"""
Test cases for Gemini Code Assist integration.

Tests the new functionality added in commit a15b21a6c149d98a4a01a914250e8bf0dcc3b117:
- OAuth2 credentials support
- User-Agent header for Code Assist requests
- Nested response format handling
- GeminiCodeAssistRequestBody format
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from typing import List, cast

import litellm
from litellm import completion, ModelResponse
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body
from litellm.types.llms.vertex_ai import GeminiCodeAssistRequestBody
from litellm.types.llms.openai import AllMessageValues


class TestGeminiCodeAssistOAuth2:
    """Test OAuth2 credentials handling for Gemini CLI."""

    def test_oauth2_credentials_parsing(self):
        """Test that OAuth2 credentials from Gemini CLI are parsed correctly."""
        vertex_base = VertexBase()
        
        # Mock OAuth2 credentials similar to Gemini CLI format
        oauth2_creds = {
            "access_token": "ya29.a0AS3H6Nx_test_token",
            "refresh_token": "1//09FtpJYpxOd_test_refresh",
            "scope": "openid https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email",
            "token_type": "Bearer",
            "id_token": "eyJhbGciOiJSUzI1NiIs_test_id_token",
            "expiry_date": 1753710424846
        }
        
        # Test OAuth2 credentials creation
        creds = vertex_base._credentials_from_oauth2_token(
            oauth2_creds, 
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        assert creds is not None
        assert creds.token == "ya29.a0AS3H6Nx_test_token"
        assert creds.refresh_token == "1//09FtpJYpxOd_test_refresh"
        assert creds.client_id == "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
        assert creds.client_secret == "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"
        assert creds.token_uri == "https://oauth2.googleapis.com/token"
        
        # Test expiry date conversion
        expected_expiry = datetime.fromtimestamp(1753710424846 / 1000, tz=timezone.utc)
        assert creds.expiry == expected_expiry

    def test_oauth2_credentials_missing_tokens(self):
        """Test that missing access_token or refresh_token raises ValueError."""
        vertex_base = VertexBase()
        
        # Missing access_token
        with pytest.raises(ValueError, match="OAuth2 credentials must contain both access_token and refresh_token"):
            vertex_base._credentials_from_oauth2_token(
                {"refresh_token": "test"}, 
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        
        # Missing refresh_token
        with pytest.raises(ValueError, match="OAuth2 credentials must contain both access_token and refresh_token"):
            vertex_base._credentials_from_oauth2_token(
                {"access_token": "test"}, 
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

    def test_oauth2_credentials_detection_in_load_credentials(self):
        """Test that OAuth2 credentials are detected correctly."""
        vertex_base = VertexBase()

        oauth2_creds = {
            "access_token": "ya29.test",
            "refresh_token": "1//test",
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "token_type": "Bearer",
            "expiry_date": 1753710424846
        }

        # Test that OAuth2 credentials can be created from the dict
        creds = vertex_base._credentials_from_oauth2_token(
            oauth2_creds,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # Verify the credentials were created correctly
        assert creds is not None
        assert creds.token == "ya29.test"
        assert creds.refresh_token == "1//test"


class TestGeminiCodeAssistHeaders:
    """Test User-Agent header handling for Gemini Code Assist."""

    def test_user_agent_header_for_code_assist(self):
        """Test that User-Agent header is added for Code Assist requests."""
        config = VertexGeminiConfig()
        
        headers = config.validate_environment(
            headers=None,
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "test"}],
            optional_params={},
            litellm_params={},
            api_key="test-key",
            api_base="https://cloudcode-pa.googleapis.com/v1internal"
        )
        
        assert "User-Agent" in headers
        assert headers["User-Agent"] == "GeminiCLI/0.1.17 (darwin; arm64)"

    def test_no_user_agent_for_regular_vertex(self):
        """Test that User-Agent header is not added for regular Vertex AI requests."""
        config = VertexGeminiConfig()
        
        headers = config.validate_environment(
            headers=None,
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "test"}],
            optional_params={},
            litellm_params={},
            api_key="test-key",
            api_base="https://us-central1-aiplatform.googleapis.com"
        )
        
        assert "User-Agent" not in headers

    def test_user_agent_with_existing_headers(self):
        """Test that User-Agent is added alongside existing headers."""
        config = VertexGeminiConfig()
        
        existing_headers = {"Custom-Header": "custom-value"}
        
        headers = config.validate_environment(
            headers=existing_headers,
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "test"}],
            optional_params={},
            litellm_params={},
            api_key="test-key",
            api_base="https://cloudcode-pa.googleapis.com/v1internal"
        )
        
        assert headers["User-Agent"] == "GeminiCLI/0.1.17 (darwin; arm64)"
        assert headers["Custom-Header"] == "custom-value"
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"


class TestGeminiCodeAssistRequestFormat:
    """Test request body transformation for Gemini Code Assist."""

    def test_code_assist_request_body_format(self):
        """Test that Code Assist requests use the special nested format."""
        messages: List[AllMessageValues] = [{"role": "user", "content": "Hello"}]
        
        result = _transform_request_body(
            messages=messages,
            model="gemini-2.5-pro",
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None,
            api_base="https://cloudcode-pa.googleapis.com/v1internal",
            vertex_project="test-project-123"
        )
        
        # Should return GeminiCodeAssistRequestBody format
        assert isinstance(result, dict)
        assert "model" in result
        assert "project" in result
        assert "request" in result
        
        assert result["model"] == "gemini-2.5-pro"
        assert result["project"] == "test-project-123"
        assert "contents" in result["request"]

    def test_regular_request_body_format(self):
        """Test that regular requests use standard format."""
        messages: List[AllMessageValues] = [{"role": "user", "content": "Hello"}]
        
        result = _transform_request_body(
            messages=messages,
            model="gemini-2.5-pro",
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None,
            api_base="https://us-central1-aiplatform.googleapis.com",
            vertex_project="test-project-123"
        )
        
        # Should return standard RequestBody format
        assert isinstance(result, dict)
        assert "contents" in result
        # Should not have the nested Code Assist format
        assert "model" not in result or "project" not in result

    def test_code_assist_without_vertex_project(self):
        """Test that Code Assist format is not used without vertex_project."""
        messages: List[AllMessageValues] = [{"role": "user", "content": "Hello"}]
        
        result = _transform_request_body(
            messages=messages,
            model="gemini-2.5-pro",
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None,
            api_base="https://cloudcode-pa.googleapis.com/v1internal",
            vertex_project=None
        )
        
        # Should return standard format even with Code Assist API base
        assert isinstance(result, dict)
        assert "contents" in result
        assert "model" not in result or "project" not in result


class TestGeminiCodeAssistResponseParsing:
    """Test response parsing for nested Gemini Code Assist format."""

    def test_nested_response_unwrapping(self):
        """Test that nested responses are unwrapped correctly."""
        config = VertexGeminiConfig()
        
        # Mock nested response format
        nested_response_data = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello from Gemini!"}]
                        },
                        "finishReason": "STOP"
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15
                }
            }
        }
        
        # Mock raw response
        mock_raw_response = MagicMock()
        mock_raw_response.json.return_value = nested_response_data
        mock_raw_response.text = json.dumps(nested_response_data)
        mock_raw_response.headers = {}
        
        # Mock logging object
        mock_logging_obj = MagicMock()
        
        # Test response transformation
        model_response = config.transform_response(
            model="gemini-2.5-pro",
            raw_response=mock_raw_response,
            model_response=ModelResponse(),
            logging_obj=mock_logging_obj,
            request_data={},
            messages=cast(List[AllMessageValues], [{"role": "user", "content": "Hello"}]),
            optional_params={},
            litellm_params={},
            encoding=None
        )

        # Verify response was parsed correctly
        assert len(model_response.choices) > 0
        assert model_response.choices[0].message.content == "Hello from Gemini!"  # type: ignore
        assert model_response.usage.prompt_tokens == 10  # type: ignore
        assert model_response.usage.completion_tokens == 5  # type: ignore
        assert model_response.usage.total_tokens == 15  # type: ignore

    def test_standard_response_parsing_still_works(self):
        """Test that standard response format still works."""
        config = VertexGeminiConfig()
        
        # Standard response format (not nested)
        standard_response_data = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello from standard Gemini!"}]
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 7,
                "totalTokenCount": 19
            }
        }
        
        # Mock raw response
        mock_raw_response = MagicMock()
        mock_raw_response.json.return_value = standard_response_data
        mock_raw_response.text = json.dumps(standard_response_data)
        mock_raw_response.headers = {}
        
        # Mock logging object
        mock_logging_obj = MagicMock()
        
        # Test response transformation
        model_response = config.transform_response(
            model="gemini-2.5-pro",
            raw_response=mock_raw_response,
            model_response=ModelResponse(),
            logging_obj=mock_logging_obj,
            request_data={},
            messages=cast(List[AllMessageValues], [{"role": "user", "content": "Hello"}]),
            optional_params={},
            litellm_params={},
            encoding=None
        )

        # Verify response was parsed correctly
        assert len(model_response.choices) > 0
        assert model_response.choices[0].message.content == "Hello from standard Gemini!"  # type: ignore
        assert model_response.usage.prompt_tokens == 12  # type: ignore
        assert model_response.usage.completion_tokens == 7  # type: ignore
        assert model_response.usage.total_tokens == 19  # type: ignore


class TestGeminiCodeAssistStreamingResponse:
    """Test streaming response parsing for Gemini Code Assist."""

    def test_streaming_response_unwrapping(self):
        """Test that streaming responses are unwrapped correctly."""
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import ModelResponseIterator

        # Mock streaming chunk with nested format
        nested_chunk = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Streaming text"}]
                        },
                        "finishReason": "STOP"
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 8,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 11
                }
            }
        }

        # Mock logging object
        mock_logging_obj = MagicMock()

        # Create iterator
        iterator = ModelResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            logging_obj=mock_logging_obj
        )

        # Test chunk parsing
        result = iterator.chunk_parser(nested_chunk)

        assert result is not None
        assert len(result.choices) > 0
        assert result.choices[0].delta.content == "Streaming text"

    def test_streaming_standard_format_still_works(self):
        """Test that standard streaming format still works."""
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import ModelResponseIterator

        # Standard streaming chunk (not nested)
        standard_chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Standard streaming"}]
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 6,
                "candidatesTokenCount": 4,
                "totalTokenCount": 10
            }
        }

        # Mock logging object
        mock_logging_obj = MagicMock()

        # Create iterator
        iterator = ModelResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            logging_obj=mock_logging_obj
        )

        # Test chunk parsing
        result = iterator.chunk_parser(standard_chunk)

        assert result is not None
        assert len(result.choices) > 0
        assert result.choices[0].delta.content == "Standard streaming"


class TestGeminiCodeAssistIntegration:
    """Integration tests for Gemini Code Assist functionality."""

    @pytest.mark.parametrize("model", ["gemini-2.5-pro", "gemini-2.5-flash"])
    def test_code_assist_api_base_detection(self, model):
        """Test that Code Assist API base is detected correctly."""
        # Test that the special API base triggers Code Assist behavior
        api_base = "https://cloudcode-pa.googleapis.com/v1internal"

        # Mock request transformation
        messages: List[AllMessageValues] = [{"role": "user", "content": "Test message"}]

        result = _transform_request_body(
            messages=messages,
            model=model,
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None,
            api_base=api_base,
            vertex_project="test-project"
        )

        # Should use Code Assist format
        assert "model" in result
        assert "project" in result
        assert "request" in result
        assert result["model"] == model
        assert result["project"] == "test-project"

    def test_code_assist_with_system_message(self):
        """Test Code Assist request with system message."""
        messages: List[AllMessageValues] = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function"}
        ]

        result = _transform_request_body(
            messages=messages,
            model="gemini-2.5-pro",
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None,
            api_base="https://cloudcode-pa.googleapis.com/v1internal",
            vertex_project="test-project"
        )

        # Should have system instruction in the nested request
        assert "system_instruction" in result["request"]  # type: ignore
        assert result["request"]["system_instruction"]["parts"][0]["text"] == "You are a helpful coding assistant."  # type: ignore

    def test_vertex_project_required_for_code_assist(self):
        """Test that vertex_project is required for Code Assist format."""
        messages: List[AllMessageValues] = [{"role": "user", "content": "Test"}]

        # Without vertex_project, should use standard format
        result_without_project = _transform_request_body(
            messages=messages,
            model="gemini-2.5-pro",
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None,
            api_base="https://cloudcode-pa.googleapis.com/v1internal",
            vertex_project=None
        )

        # Should not use Code Assist format
        assert "contents" in result_without_project
        assert "model" not in result_without_project or "project" not in result_without_project

        # With vertex_project, should use Code Assist format
        result_with_project = _transform_request_body(
            messages=messages,
            model="gemini-2.5-pro",
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None,
            api_base="https://cloudcode-pa.googleapis.com/v1internal",
            vertex_project="test-project"
        )

        # Should use Code Assist format
        assert "model" in result_with_project
        assert "project" in result_with_project
        assert "request" in result_with_project

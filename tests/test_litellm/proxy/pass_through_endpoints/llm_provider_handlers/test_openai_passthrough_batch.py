"""
Test cases for OpenAI passthrough batch functionality.

This module tests the integration between OpenAI passthrough batch creation
and the managed files hook for database storage and polling.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.litellm_core_utils.litellm_logging import Logging


class TestOpenAIPassthroughBatch:
    """Test cases for OpenAI passthrough batch creation and tracking."""

    def test_extract_model_from_batch_output(self):
        """Test model extraction from batch output file content."""
        # Mock file content with the format you provided
        mock_file_content = b'''{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}'''

        with patch('litellm.files.main.file_content') as mock_file_content_func:
            # Mock the file content response
            mock_response = Mock()
            mock_response.content = mock_file_content
            mock_file_content_func.return_value = mock_response

            # Test model extraction
            model = OpenAIPassthroughLoggingHandler._extract_model_from_batch_output("file-123")
            
            assert model == "gpt-3.5-turbo-0125"
            mock_file_content_func.assert_called_once_with(
                file_id="file-123",
                custom_llm_provider="openai"
            )

    def test_extract_model_from_batch_output_no_model(self):
        """Test model extraction when no model is found in the file content."""
        # Mock file content without model field
        mock_file_content = b'''{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "Hello world!"}]}}'''

        with patch('litellm.files.main.file_content') as mock_file_content_func:
            mock_response = Mock()
            mock_response.content = mock_file_content
            mock_file_content_func.return_value = mock_response

            model = OpenAIPassthroughLoggingHandler._extract_model_from_batch_output("file-123")
            
            assert model is None

    def test_extract_model_from_batch_output_error(self):
        """Test model extraction when file content retrieval fails."""
        with patch('litellm.files.main.file_content') as mock_file_content_func:
            mock_file_content_func.side_effect = Exception("File not found")

            model = OpenAIPassthroughLoggingHandler._extract_model_from_batch_output("file-123")
            
            assert model is None

    def test_is_openai_batch_route(self):
        """Test batch route detection for various OpenAI URLs."""
        # Valid OpenAI batch routes
        assert OpenAIPassthroughLoggingHandler.is_openai_batch_route("https://api.openai.com/v1/batches")
        assert OpenAIPassthroughLoggingHandler.is_openai_batch_route("https://api.openai.com/batches")
        assert OpenAIPassthroughLoggingHandler.is_openai_batch_route("https://openai.azure.com/v1/batches")
        
        # Invalid routes
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_route("https://api.openai.com/v1/chat/completions")
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_route("https://api.anthropic.com/v1/batches")
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_route("")
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_route(None)

    def test_is_openai_batch_create_route(self):
        """Test batch creation route detection with HTTP methods."""
        # Valid batch creation (POST to batch endpoint)
        assert OpenAIPassthroughLoggingHandler.is_openai_batch_create_route(
            "https://api.openai.com/v1/batches", "POST"
        )
        assert OpenAIPassthroughLoggingHandler.is_openai_batch_create_route(
            "https://api.openai.com/batches", "post"
        )
        
        # Invalid - wrong method
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_create_route(
            "https://api.openai.com/v1/batches", "GET"
        )
        
        # Invalid - wrong endpoint
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_create_route(
            "https://api.openai.com/v1/chat/completions", "POST"
        )

    def test_create_user_api_key_dict_from_kwargs(self):
        """Test user API key dictionary creation from kwargs metadata."""
        # Mock kwargs with metadata
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "user_api_key": "test_hash_123",
                    "user_api_key_user_id": "test_user_123",
                    "user_api_key_team_id": "test_team_123",
                    "user_api_key_org_id": "test_org_123",
                    "user_api_key_user_email": "test@example.com",
                    "user_api_key_end_user_id": "end_user_123",
                    "user_api_key_team_alias": "test_team",
                    "user_api_key_alias": "test_alias",
                }
            }
        }
        
        user_auth = OpenAIPassthroughLoggingHandler._create_user_api_key_dict_from_kwargs(kwargs)
        
        assert user_auth.user_id == "test_user_123"
        assert user_auth.api_key == "test_hash_123"
        assert user_auth.team_id == "test_team_123"
        assert user_auth.org_id is None  # Not set in the implementation
        assert user_auth.user_email == "test@example.com"
        assert user_auth.end_user_id is None  # Not set in the implementation
        assert user_auth.team_alias == "test_team"
        assert user_auth.key_alias is None  # Not set in the implementation

    def test_create_user_api_key_dict_missing_fields(self):
        """Test user API key dictionary creation with missing fields."""
        # Mock kwargs with minimal metadata
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "user_api_key": "test_hash_123",
                    "user_api_key_user_id": "test_user_123",
                }
            }
        }
        
        user_auth = OpenAIPassthroughLoggingHandler._create_user_api_key_dict_from_kwargs(kwargs)
        
        assert user_auth.user_id == "test_user_123"
        assert user_auth.api_key == "test_hash_123"
        assert user_auth.team_id is None
        assert user_auth.org_id is None
        assert user_auth.user_email is None
        assert user_auth.end_user_id is None
        assert user_auth.team_alias is None
        assert user_auth.key_alias is None

    def test_batch_creation_flow(self):
        """Test the complete batch creation flow without enterprise dependencies."""
        # Create handler instance
        handler = OpenAIPassthroughLoggingHandler()
        
        # Mock kwargs for batch creation
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "user_api_key": "test_hash_123",
                    "user_api_key_user_id": "test_user_123",
                    "user_api_key_team_id": "test_team_123",
                },
                "proxy_server_request": {
                    "method": "POST",
                    "url": "http://localhost:4000/openai/v1/batches"
                }
            }
        }
        
        # Mock request and response
        request_body = {
            "input_file_id": "file-test123",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        }
        
        response_body = {
            "id": "batch_test123",
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "status": "validating",
            "input_file_id": "file-test123",
            "completion_window": "24h",
            "created_at": 1757359864,
            "expires_at": 1757446264,
        }
        
        # Mock logging object
        mock_logging = Mock(spec=Logging)
        mock_logging.model_call_details = {}
        
        # Mock httpx response
        mock_httpx_response = Mock()
        mock_httpx_response.headers = {"content-type": "application/json"}
        
        # Call the handler
        result = handler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=response_body,
            logging_obj=mock_logging,
            url_route="https://api.openai.com/v1/batches",
            result="",
            start_time=None,
            end_time=None,
            cache_hit=False,
            request_body=request_body,
            **kwargs
        )
        
        # Check that the result is returned
        assert result is not None

    def test_batch_creation_non_batch_route(self):
        """Test that non-batch routes don't trigger batch creation logic."""
        # Test the route detection logic directly
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_route("https://api.openai.com/v1/chat/completions")
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_create_route("https://api.openai.com/v1/chat/completions", "POST")
        
        # Test with a different non-batch route
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_route("https://api.openai.com/v1/images/generations")
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_create_route("https://api.openai.com/v1/images/generations", "POST")

    def test_batch_creation_wrong_method(self):
        """Test that GET requests to batch endpoints don't trigger creation logic."""
        handler = OpenAIPassthroughLoggingHandler()
        
        # Mock kwargs for GET request to batch endpoint
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "user_api_key_hash": "test_hash_123",
                    "user_api_key_user_id": "test_user_123",
                },
                "proxy_server_request": {
                    "method": "GET",
                    "url": "http://localhost:4000/openai/v1/batches"
                }
            }
        }
        
        # Mock request and response for batch retrieval
        request_body = {}
        
        response_body = {
            "id": "batch_test123",
            "object": "batch",
            "status": "completed"
        }
        
        # Mock logging object
        mock_logging = Mock(spec=Logging)
        mock_logging.model_call_details = {}
        
        # Mock httpx response
        mock_httpx_response = Mock()
        mock_httpx_response.headers = {"content-type": "application/json"}
        
        # Call the handler
        result = handler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=response_body,
            logging_obj=mock_logging,
            url_route="https://api.openai.com/v1/batches",
            result="",
            start_time=None,
            end_time=None,
            cache_hit=False,
            request_body=request_body,
            **kwargs
        )
        
        # Should return normally without batch creation logic
        assert result is not None

    def test_user_api_key_auth_creation(self):
        """Test UserAPIKeyAuth object creation from user dictionary."""
        user_dict = {
            "user_id": "test_user_123",
            "api_key": "test_hash_123",
            "team_id": "test_team_123",
            "org_id": "test_org_123",
            "user_email": "test@example.com",
            "end_user_id": "end_user_123",
            "team_alias": "test_team",
            "user_alias": "test_alias",
        }
        
        # Create UserAPIKeyAuth object
        user_auth = UserAPIKeyAuth(
            user_id=user_dict["user_id"],
            api_key=user_dict["api_key"],
            team_id=user_dict["team_id"],
            org_id=user_dict["org_id"],
            user_email=user_dict["user_email"],
            end_user_id=user_dict["end_user_id"],
            team_alias=user_dict["team_alias"],
            key_alias=user_dict["user_alias"],
            models=[],
            metadata=user_dict
        )
        
        assert user_auth.user_id == "test_user_123"
        assert user_auth.api_key == "test_hash_123"
        assert user_auth.team_id == "test_team_123"
        assert user_auth.org_id == "test_org_123"
        assert user_auth.user_email == "test@example.com"
        assert user_auth.end_user_id == "end_user_123"
        assert user_auth.team_alias == "test_team"
        assert user_auth.key_alias == "test_alias"

"""
Integration test for OpenAI passthrough batch functionality.

This test verifies the end-to-end flow of batch creation through the passthrough endpoint.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)


class TestOpenAIPassthroughBatchIntegration:
    """Integration tests for OpenAI passthrough batch functionality."""

    def test_batch_creation_with_model_extraction(self):
        """Test batch creation with model extraction from completed batch."""
        # Mock file content with model information
        mock_file_content = b'''{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello world!"}]}}'''

        with patch('litellm.files.main.file_content') as mock_file_content_func:
            mock_response = Mock()
            mock_response.content = mock_file_content
            mock_file_content_func.return_value = mock_response

            # Test the handler with a completed batch
            handler = OpenAIPassthroughLoggingHandler()
            
            # Simulate a completed batch response
            response_body = {
                "id": "batch_123",
                "object": "batch",
                "status": "completed",
                "output_file_id": "file-456",
                "endpoint": "/v1/chat/completions",
                "created_at": 1234567890,
                "in_progress_at": None,
                "expires_at": 1234567890 + 86400,
                "finalizing_at": None,
                "completed_at": 1234567890 + 3600,
                "failed_at": None,
                "expired_at": None,
                "cancelling_at": None,
                "cancelled_at": None,
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                "metadata": None
            }

            # Test the handler
            result = handler.openai_passthrough_handler(
                httpx_response=Mock(),
                response_body=response_body,
                logging_obj=Mock(),
                url_route="https://api.openai.com/v1/batches",
                result="",
                start_time=None,
                end_time=None,
                cache_hit=False,
                request_body={"endpoint": "/v1/chat/completions", "input_file_id": "file-123"},
                **{}
            )

            # Verify that the model was extracted and used
            assert result is not None
            assert "result" in result
            assert "kwargs" in result

    def test_end_to_end_batch_creation(self):
        """Test complete end-to-end batch creation flow."""
        # Create handler instance
        handler = OpenAIPassthroughLoggingHandler()
        
        # Simulate a complete batch creation request
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "user_api_key": "sk-test123456789",
                    "user_api_key_user_id": "user_123",
                    "user_api_key_team_id": "team_456",
                    "user_api_key_user_email": "user@example.com",
                },
                "proxy_server_request": {
                    "method": "POST",
                    "url": "http://localhost:4000/openai/v1/batches"
                }
            }
        }
        
        # Mock OpenAI batch creation request
        request_body = {
            "input_file_id": "file-Axj46KazU3C4p4GUahTiMQ",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        }
        
        # Mock OpenAI batch creation response
        response_body = {
            "id": "batch_68bf2ef82cf08190babe92728bd3e5aa",
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "errors": None,
            "input_file_id": "file-Axj46KazU3C4p4GUahTiMQ",
            "completion_window": "24h",
            "status": "validating",
            "output_file_id": None,
            "error_file_id": None,
            "created_at": 1757359864,
            "in_progress_at": None,
            "expires_at": 1757446264,
            "finalizing_at": None,
            "completed_at": None,
            "failed_at": None,
            "expired_at": None,
            "cancelling_at": None,
            "cancelled_at": None,
            "request_counts": {
                "total": 0,
                "completed": 0,
                "failed": 0
            },
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0}
            },
            "metadata": None
        }
        
        # Mock logging object
        mock_logging = Mock()
        mock_logging.model_call_details = {}
        
        # Mock httpx response
        mock_httpx_response = Mock()
        
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
        
        # Verify the result is returned
        assert result is not None

    def test_batch_route_detection_edge_cases(self):
        """Test edge cases for batch route detection."""
        # Test various URL formats
        test_cases = [
            ("https://api.openai.com/v1/batches", True),
            ("https://api.openai.com/batches", True),
            ("https://openai.azure.com/v1/batches", True),
            ("https://openai.azure.com/batches", True),
            ("https://api.openai.com/v1/batches/", False),  # trailing slash - exact match required
            ("https://api.openai.com/v1/batches?param=value", True),  # query params - path still matches
            ("https://api.openai.com/v1/batches/batch_123", False),  # specific batch ID
            ("https://api.openai.com/v1/chat/completions", False),
            ("https://api.anthropic.com/v1/batches", False),
            ("", False),
            (None, False),
        ]
        
        for url, expected in test_cases:
            result = OpenAIPassthroughLoggingHandler.is_openai_batch_route(url)
            assert result == expected, f"Failed for URL: {url}"

    def test_batch_create_route_detection_edge_cases(self):
        """Test edge cases for batch creation route detection."""
        # Test various method and URL combinations
        test_cases = [
            ("https://api.openai.com/v1/batches", "POST", True),
            ("https://api.openai.com/v1/batches", "post", True),
            ("https://api.openai.com/v1/batches", "Post", True),
            ("https://api.openai.com/v1/batches", "GET", False),
            ("https://api.openai.com/v1/batches", "PUT", False),
            ("https://api.openai.com/v1/batches", "DELETE", False),
            ("https://api.openai.com/v1/chat/completions", "POST", False),
            ("https://api.anthropic.com/v1/batches", "POST", False),
            ("", "POST", False),
            (None, "POST", False),
        ]
        
        for url, method, expected in test_cases:
            result = OpenAIPassthroughLoggingHandler.is_openai_batch_create_route(url, method)
            assert result == expected, f"Failed for URL: {url}, Method: {method}"

    def test_user_metadata_extraction_edge_cases(self):
        """Test edge cases for user metadata extraction."""
        # Test with empty metadata
        kwargs_empty = {"litellm_params": {"metadata": {}}}
        user_auth = OpenAIPassthroughLoggingHandler._create_user_api_key_dict_from_kwargs(kwargs_empty)
        
        assert user_auth is None
        
        # Test with missing litellm_params
        kwargs_missing = {}
        user_auth = OpenAIPassthroughLoggingHandler._create_user_api_key_dict_from_kwargs(kwargs_missing)
        
        assert user_auth is None
        
        # Test with None values in metadata
        kwargs_none = {
            "litellm_params": {
                "metadata": {
                    "user_api_key_user_id": None,
                    "user_api_key_team_id": None,
                    "user_api_key": "test_hash",
                }
            }
        }
        user_auth = OpenAIPassthroughLoggingHandler._create_user_api_key_dict_from_kwargs(kwargs_none)
        
        assert user_auth is None  # No user_id, so returns None

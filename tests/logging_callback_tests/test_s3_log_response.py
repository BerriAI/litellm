"""
Tests for S3 s3_log_response parameter.

When s3_log_response=False, the S3 logger should log only prompts (no responses).
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

import litellm
from litellm.integrations.s3 import S3Logger


class TestS3LogResponse:
    """Test the s3_log_response parameter functionality."""

    def test_filter_payload_excludes_response_when_false(self):
        """Test that response is excluded from payload when s3_log_response=False."""
        with patch("boto3.client"):
            # Set up S3 callback params with s3_log_response=False
            litellm.s3_callback_params = {
                "s3_bucket_name": "test-bucket",
                "s3_log_response": False,
            }
            
            logger = S3Logger()
            
            # Create a mock payload with both messages and response
            mock_payload = {
                "id": "test-id",
                "messages": [{"role": "user", "content": "Hello"}],
                "response": "Hi there!",
                "model": "gpt-4",
                "metadata": {},
            }
            
            filtered = logger._filter_payload_fields(mock_payload)
            
            # Response should be removed
            assert "response" not in filtered
            # Messages should still be present
            assert "messages" in filtered
            assert filtered["messages"] == [{"role": "user", "content": "Hello"}]

    def test_filter_payload_includes_response_when_true(self):
        """Test that response is included in payload when s3_log_response=True (default)."""
        with patch("boto3.client"):
            # Set up S3 callback params with s3_log_response=True (or not set)
            litellm.s3_callback_params = {
                "s3_bucket_name": "test-bucket",
                "s3_log_response": True,
            }
            
            logger = S3Logger()
            
            # Create a mock payload with both messages and response
            mock_payload = {
                "id": "test-id",
                "messages": [{"role": "user", "content": "Hello"}],
                "response": "Hi there!",
                "model": "gpt-4",
                "metadata": {},
            }
            
            filtered = logger._filter_payload_fields(mock_payload)
            
            # Both should be present
            assert "response" in filtered
            assert "messages" in filtered
            assert filtered["response"] == "Hi there!"

    def test_filter_payload_default_includes_response(self):
        """Test that response is included by default when s3_log_response is not set."""
        with patch("boto3.client"):
            # Set up S3 callback params without s3_log_response
            litellm.s3_callback_params = {
                "s3_bucket_name": "test-bucket",
            }
            
            logger = S3Logger()
            
            # Default should be True
            assert logger.s3_log_response is True
            
            mock_payload = {
                "id": "test-id",
                "messages": [{"role": "user", "content": "Hello"}],
                "response": "Hi there!",
                "model": "gpt-4",
                "metadata": {},
            }
            
            filtered = logger._filter_payload_fields(mock_payload)
            
            # Response should be present (default behavior)
            assert "response" in filtered

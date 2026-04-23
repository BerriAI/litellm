"""
Tests for OpenAI Containers API regional api_base support.

Validates that litellm.create_container and litellm.upload_container_file
correctly use regional endpoints like https://us.api.openai.com/v1 for
US Data Residency instead of defaulting to https://api.openai.com/v1.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm


class TestContainerRegionalApiBase:
    """Test suite for container API regional api_base support."""

    def setup_method(self):
        """Set up test fixtures."""
        os.environ["OPENAI_API_KEY"] = "sk-test123"

    def teardown_method(self):
        """Clean up after tests."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]
        if "OPENAI_API_BASE" in os.environ:
            del os.environ["OPENAI_API_BASE"]
        litellm.api_base = None

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_create_container_uses_regional_api_base(self, mock_post):
        """
        Test that litellm.create_container uses the regional api_base when provided.
        
        This validates the fix for US Data Residency support where requests should
        go to https://us.api.openai.com/v1 instead of https://api.openai.com/v1.
        """
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "cntr_123456",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Test Container"
        }
        mock_post.return_value = mock_response

        litellm.create_container(
            name="Test Container",
            custom_llm_provider="openai",
            api_base="https://us.api.openai.com/v1",
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        called_url = call_args[1]["url"]
        
        assert "us.api.openai.com" in called_url, f"Expected US regional URL, got: {called_url}"
        assert called_url == "https://us.api.openai.com/v1/containers"

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_create_container_uses_env_var_openai_base_url(self, mock_post):
        """
        Test that litellm.create_container uses OPENAI_BASE_URL env var.
        """
        os.environ["OPENAI_BASE_URL"] = "https://us.api.openai.com/v1"
        
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "cntr_123456",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Test Container"
        }
        mock_post.return_value = mock_response

        litellm.create_container(
            name="Test Container",
            custom_llm_provider="openai",
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        called_url = call_args[1]["url"]
        
        assert "us.api.openai.com" in called_url, f"Expected US regional URL, got: {called_url}"

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_create_container_defaults_to_standard_openai(self, mock_post):
        """
        Test that litellm.create_container defaults to standard OpenAI URL
        when no regional api_base is configured.
        """
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "cntr_123456",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "last_active_at": 1747857508,
            "name": "Test Container"
        }
        mock_post.return_value = mock_response

        litellm.create_container(
            name="Test Container",
            custom_llm_provider="openai",
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        called_url = call_args[1]["url"]
        
        assert called_url == "https://api.openai.com/v1/containers"

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_upload_container_file_uses_regional_api_base(self, mock_post):
        """
        Test that litellm.upload_container_file uses the regional api_base when provided.
        """
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "file_123456",
            "object": "container.file",
            "created_at": 1747857508,
            "container_id": "cntr_123456",
            "path": "/mnt/user/data.csv",
            "source": "user",
        }
        mock_post.return_value = mock_response

        litellm.upload_container_file(
            container_id="cntr_123456",
            file=("data.csv", b"col1,col2\n1,2", "text/csv"),
            custom_llm_provider="openai",
            api_base="https://us.api.openai.com/v1",
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        called_url = call_args[1]["url"]
        
        assert "us.api.openai.com" in called_url, f"Expected US regional URL, got: {called_url}"
        assert "cntr_123456/files" in called_url


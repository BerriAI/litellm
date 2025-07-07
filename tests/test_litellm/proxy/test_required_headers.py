"""
Test required headers functionality in LiteLLM Proxy.

This tests the header-based request blocking feature that allows blocking requests
unless they have specific required headers.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import Request
from starlette.datastructures import Headers

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request


class TestRequiredHeaders:
    """Test the required headers functionality."""

    def create_mock_request(self, headers_dict=None):
        """Create a mock FastAPI Request object."""
        if headers_dict is None:
            headers_dict = {}
        
        # Convert to lowercase keys as FastAPI does
        headers_dict = {k.lower(): v for k, v in headers_dict.items()}
        
        mock_request = Mock(spec=Request)
        mock_request.headers = Headers(headers_dict)
        
        # Mock the URL object properly
        mock_url = Mock()
        mock_url.path = "/v1/chat/completions"
        mock_url.__str__ = Mock(return_value="http://localhost:4000/v1/chat/completions")
        mock_request.url = mock_url
        
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.client = None
        
        return mock_request

    def create_user_api_key_dict(self, key_metadata=None, team_metadata=None):
        """Create a UserAPIKeyAuth object for testing."""
        return UserAPIKeyAuth(
            api_key="test-key-hash",
            user_id="test-user",
            team_id="test-team",
            metadata=key_metadata or {},
            team_metadata=team_metadata or {},
            parent_otel_span=None,
        )

    @pytest.mark.asyncio
    async def test_no_required_headers_allows_request(self):
        """Test that requests pass when no required headers are configured."""
        request = self.create_mock_request({"user-agent": "TestBot/1.0"})
        user_api_key_dict = self.create_user_api_key_dict()
        data = {"model": "gpt-3.5-turbo", "messages": []}
        
        # Should not raise any exception
        result = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=Mock(),
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_key_level_required_header_exact_match(self):
        """Test key-level required header with exact string match."""
        request = self.create_mock_request({
            "user-agent": "MyApp/1.0"
        })
        
        user_api_key_dict = self.create_user_api_key_dict(
            key_metadata={
                "required_headers": {
                    "User-Agent": "MyApp/1.0"
                }
            }
        )
        
        data = {"model": "gpt-3.5-turbo", "messages": []}
        
        # Should not raise any exception
        result = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=Mock(),
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_key_level_required_header_wrong_value_blocks(self):
        """Test that wrong header value blocks the request."""
        request = self.create_mock_request({
            "user-agent": "BadBot/1.0"
        })
        
        user_api_key_dict = self.create_user_api_key_dict(
            key_metadata={
                "required_headers": {
                    "User-Agent": "MyApp/1.0"
                }
            }
        )
        
        data = {"model": "gpt-3.5-turbo", "messages": []}
        
        with pytest.raises(ProxyException) as exc_info:
            await add_litellm_data_to_request(
                data=data,
                request=request,
                user_api_key_dict=user_api_key_dict,
                proxy_config=Mock(),
            )
        
        assert "Request blocked" in str(exc_info.value)
        assert "User-Agent" in str(exc_info.value)
        assert exc_info.value.code == 403

    @pytest.mark.asyncio
    async def test_key_level_required_header_wildcard_match(self):
        """Test wildcard pattern matching for required headers."""
        request = self.create_mock_request({
            "user-agent": "MyApp/2.0 (+https://example.com)"
        })
        
        user_api_key_dict = self.create_user_api_key_dict(
            key_metadata={
                "required_headers": {
                    "User-Agent": "MyApp*"
                }
            }
        )
        
        data = {"model": "gpt-3.5-turbo", "messages": []}
        
        # Should not raise any exception
        result = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=Mock(),
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_key_level_required_header_list_of_values(self):
        """Test required header with list of allowed values."""
        request = self.create_mock_request({
            "user-agent": "MyApp/1.0"
        })
        
        user_api_key_dict = self.create_user_api_key_dict(
            key_metadata={
                "required_headers": {
                    "User-Agent": [
                        "MyApp/1.0",
                        "MyApp/2.0"
                    ]
                }
            }
        )
        
        data = {"model": "gpt-3.5-turbo", "messages": []}
        
        # Should not raise any exception
        result = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=Mock(),
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_team_level_required_header(self):
        """Test team-level required header validation."""
        request = self.create_mock_request({
            "user-agent": "MyApp/1.0"
        })
        
        user_api_key_dict = self.create_user_api_key_dict(
            team_metadata={
                "required_headers": {
                    "User-Agent": "MyApp/1.0"
                }
            }
        )
        
        data = {"model": "gpt-3.5-turbo", "messages": []}
        
        # Should not raise any exception
        result = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=Mock(),
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_multiple_required_headers(self):
        """Test multiple required headers."""
        request = self.create_mock_request({
            "user-agent": "MyApp/1.0",
            "origin": "https://example.com"
        })
        
        user_api_key_dict = self.create_user_api_key_dict(
            key_metadata={
                "required_headers": {
                    "User-Agent": "MyApp*",
                    "Origin": "https://example.com"
                }
            }
        )
        
        data = {"model": "gpt-3.5-turbo", "messages": []}
        
        # Should not raise any exception
        result = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=Mock(),
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_case_insensitive_header_matching(self):
        """Test that header matching is case insensitive."""
        request = self.create_mock_request({
            "user-agent": "MyApp/1.0"  # lowercase in request
        })
        
        user_api_key_dict = self.create_user_api_key_dict(
            key_metadata={
                "required_headers": {
                    "User-Agent": "MyApp/1.0"  # mixed case in config
                }
            }
        )
        
        data = {"model": "gpt-3.5-turbo", "messages": []}
        
        # Should not raise any exception
        result = await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=Mock(),
        )
        
        assert result is not None

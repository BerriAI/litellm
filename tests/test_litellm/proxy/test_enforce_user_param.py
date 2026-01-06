"""
Tests for enforce_user_param feature with POST/GET method filtering and MCP route exclusion.

Tests verify that:
1. enforce_user_param only applies to POST requests
2. GET requests like /v1/models are not affected
3. MCP routes are excluded from enforcement
4. POST requests to completion endpoints still require user param when enforced
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import Request

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import common_checks
from litellm.proxy.auth.route_checks import RouteChecks


class MockRequest:
    """Mock FastAPI Request object"""
    def __init__(self, method: str = "POST"):
        self.method = method


def get_mock_user_token():
    """Create a mock UserAPIKeyAuth token for testing"""
    return UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="test-team",
        org_id="test-org",
        models=["*"],
        metadata={}
    )


class TestEnforceUserParamPostGetFiltering:
    """Test POST/GET method filtering for enforce_user_param"""

    @pytest.mark.asyncio
    async def test_post_completion_without_user_param_should_fail(self):
        """POST to /v1/chat/completions without user param should raise error"""
        request = MockRequest(method="POST")
        general_settings = {"enforce_user_param": True}
        request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            with pytest.raises(Exception) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings=general_settings,
                    route="/v1/chat/completions",
                    llm_router=None,
                    proxy_logging_obj=MagicMock(),
                    valid_token=get_mock_user_token(),
                    request=request,
                )
            
            assert "user" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_post_completion_with_user_param_should_pass(self):
        """POST to /v1/chat/completions with user param should pass"""
        request = MockRequest(method="POST")
        general_settings = {"enforce_user_param": True}
        request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "user": "user123"
        }
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            # Should not raise exception
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/v1/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_get_models_without_user_param_should_pass(self):
        """GET to /v1/models without user param should NOT raise error"""
        request = MockRequest(method="GET")
        general_settings = {"enforce_user_param": True}
        request_body = {}  # GET requests typically don't have body
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            # Should not raise exception
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/v1/models",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_get_files_without_user_param_should_pass(self):
        """GET to /v1/files without user param should NOT raise error"""
        request = MockRequest(method="GET")
        general_settings = {"enforce_user_param": True}
        request_body = {}
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/v1/files",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_post_embeddings_without_user_param_should_fail(self):
        """POST to /v1/embeddings without user param should raise error"""
        request = MockRequest(method="POST")
        general_settings = {"enforce_user_param": True}
        request_body = {
            "model": "text-embedding-ada-002",
            "input": "test"
        }
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            with pytest.raises(Exception) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings=general_settings,
                    route="/v1/embeddings",
                    llm_router=None,
                    proxy_logging_obj=MagicMock(),
                    valid_token=get_mock_user_token(),
                    request=request,
                )
            
            assert "user" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_post_embeddings_with_user_param_should_pass(self):
        """POST to /v1/embeddings with user param should pass"""
        request = MockRequest(method="POST")
        general_settings = {"enforce_user_param": True}
        request_body = {
            "model": "text-embedding-ada-002",
            "input": "test",
            "user": "user123"
        }
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/v1/embeddings",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True


class TestEnforceUserParamMCPExclusion:
    """Test MCP route exclusion from enforce_user_param"""

    @pytest.mark.asyncio
    async def test_mcp_route_without_user_param_should_pass(self):
        """POST to MCP route without user param should NOT raise error"""
        request = MockRequest(method="POST")
        general_settings = {"enforce_user_param": True}
        request_body = {"action": "list_tools"}
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            # Should not raise exception for MCP routes
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/mcp/tools/list",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_mcp_root_route_without_user_param_should_pass(self):
        """POST to /mcp without user param should NOT raise error"""
        request = MockRequest(method="POST")
        general_settings = {"enforce_user_param": True}
        request_body = {"data": "test"}
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/mcp",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True


class TestEnforceUserParamDisabled:
    """Test behavior when enforce_user_param is disabled"""

    @pytest.mark.asyncio
    async def test_post_without_user_param_when_disabled_should_pass(self):
        """POST without user param when enforce_user_param=False should pass"""
        request = MockRequest(method="POST")
        general_settings = {"enforce_user_param": False}
        request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/v1/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_post_without_user_param_when_not_set_should_pass(self):
        """POST without user param when enforce_user_param not set should pass"""
        request = MockRequest(method="POST")
        general_settings = {}  # enforce_user_param not set
        request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/v1/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True


class TestEnforceUserParamEdgeCases:
    """Test edge cases for enforce_user_param"""

    @pytest.mark.asyncio
    async def test_request_without_method_attribute_should_pass(self):
        """Request without method attribute should not raise error"""
        request = MagicMock()
        del request.method  # Remove method attribute
        request.__hasattr__ = MagicMock(return_value=False)
        
        general_settings = {"enforce_user_param": True}
        request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            # Should not raise error even without method
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/v1/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_case_insensitive_http_method(self):
        """HTTP method comparison should be case-insensitive"""
        request = MockRequest(method="post")  # lowercase
        general_settings = {"enforce_user_param": True}
        request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            with pytest.raises(Exception) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings=general_settings,
                    route="/v1/chat/completions",
                    llm_router=None,
                    proxy_logging_obj=MagicMock(),
                    valid_token=get_mock_user_token(),
                    request=request,
                )
            
            assert "user" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_put_method_should_not_enforce_user_param(self):
        """PUT method should not enforce user param (only POST)"""
        request = MockRequest(method="PUT")
        general_settings = {"enforce_user_param": True}
        request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            # Should not raise for PUT method
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/v1/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_patch_method_should_not_enforce_user_param(self):
        """PATCH method should not enforce user param (only POST)"""
        request = MockRequest(method="PATCH")
        general_settings = {"enforce_user_param": True}
        request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        with patch('litellm.proxy.auth.auth_checks._is_api_route_allowed', new_callable=AsyncMock, return_value=True):
            # Should not raise for PATCH method
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings=general_settings,
                route="/v1/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=get_mock_user_token(),
                request=request,
            )
            
            assert result is True


if __name__ == "__main__":
    # Run tests with: pytest tests/test_litellm/proxy/test_enforce_user_param.py -v
    pytest.main([__file__, "-v"])

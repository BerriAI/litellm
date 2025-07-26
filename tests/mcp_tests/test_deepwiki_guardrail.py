"""
Test DeepWiki MCP Guardrail functionality
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.guardrails.guardrail_hooks.mcp import MCPGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache

# Configure pytest-asyncio to use a new event loop for each test
# Only mark specific tests as async, not all tests


class TestDeepWikiGuardrail:
    """Test DeepWiki MCP Guardrail functionality"""

    def test_deepwiki_guardrail_init(self):
        """Test DeepWiki MCPGuardrail initialization"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-content-guard",
            mcp_server_name="deepwiki",
            tool_name="search_wikipedia",
            block_on_error=True,
            timeout=30.0,
            validation_rules={
                "max_query_length": 1000,
                "forbidden_keywords": ["password", "secret"]
            }
        )
        
        assert guardrail.guardrail_name == "deepwiki-content-guard"
        assert guardrail.mcp_server_name == "deepwiki"
        assert guardrail.tool_name == "search_wikipedia"
        assert guardrail.block_on_error is True
        assert guardrail.timeout == 30.0

    def test_deepwiki_should_apply_to_tool(self):
        """Test DeepWiki tool filtering logic"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            mcp_server_name="deepwiki",
            tool_name="search_wikipedia"
        )
        
        # Should apply to specific tool
        assert guardrail._should_apply_to_tool("search_wikipedia", "deepwiki") is True
        
        # Should not apply to different tool
        assert guardrail._should_apply_to_tool("get_article", "deepwiki") is False
        
        # Should not apply to different server
        assert guardrail._should_apply_to_tool("search_wikipedia", "github") is False

    def test_deepwiki_validate_tool_arguments(self):
        """Test DeepWiki argument validation"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            validation_rules={
                "max_query_length": 10,
                "forbidden_keywords": ["secret"]
            }
        )
        
        # Valid arguments
        valid_args = {"query": "Python", "language": "en"}
        result = guardrail._validate_tool_arguments("search_wikipedia", valid_args)
        assert result == valid_args
        
        # Invalid arguments (query too long)
        invalid_args = {"query": "This query is too long for the validation", "language": "en"}
        with pytest.raises(ValueError, match="Query exceeds maximum length"):
            guardrail._validate_tool_arguments("search_wikipedia", invalid_args)

    @pytest.mark.asyncio
    async def test_deepwiki_pre_call_hook(self):
        """Test DeepWiki pre-call hook"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            mcp_server_name="deepwiki",
            tool_name="search_wikipedia",
            validation_rules={
                "max_query_length": 10
            }
        )
        
        data = {
            "name": "search_wikipedia",
            "mcp_server_name": "deepwiki",
            "arguments": {"query": "Python", "language": "en"}
        }
        
        user_api_key = UserAPIKeyAuth()
        cache = DualCache()
        
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=cache,
            data=data,
            call_type="completion"
        )
        
        assert result == data

    @pytest.mark.asyncio
    async def test_deepwiki_pre_call_hook_validation_failure(self):
        """Test DeepWiki pre-call hook with validation failure"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            mcp_server_name="deepwiki",
            tool_name="search_wikipedia",
            block_on_error=True,
            validation_rules={
                "max_query_length": 5
            }
        )
        
        data = {
            "name": "search_wikipedia",
            "mcp_server_name": "deepwiki",
            "arguments": {"query": "This query is too long", "language": "en"}
        }
        
        user_api_key = UserAPIKeyAuth()
        cache = DualCache()
        
        with pytest.raises(ValueError, match="MCP Guardrail pre-call validation failed"):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key,
                cache=cache,
                data=data,
                call_type="completion"
            )

    @pytest.mark.asyncio
    async def test_deepwiki_pre_call_hook_skip_validation(self):
        """Test DeepWiki pre-call hook skipping validation"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            mcp_server_name="deepwiki",
            tool_name="search_wikipedia"
        )
        
        # Different tool - should skip validation
        data = {
            "name": "get_article",
            "mcp_server_name": "deepwiki",
            "arguments": {"query": "This should not be validated"}
        }
        
        user_api_key = UserAPIKeyAuth()
        cache = DualCache()
        
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=cache,
            data=data,
            call_type="completion"
        )
        
        assert result == data

    @pytest.mark.asyncio
    async def test_deepwiki_during_call_hook(self):
        """Test DeepWiki during-call hook"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            mcp_server_name="deepwiki"
        )
        
        data = {
            "name": "search_wikipedia",
            "mcp_server_name": "deepwiki",
            "arguments": {"query": "Python", "language": "en"}
        }
        
        user_api_key = UserAPIKeyAuth()
        
        result = await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=user_api_key,
            call_type="completion"
        )
        
        assert result == data

    @pytest.mark.asyncio
    async def test_deepwiki_post_call_hook(self):
        """Test DeepWiki post-call hook"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            mcp_server_name="deepwiki",
            tool_name="search_wikipedia"
        )
        
        data = {
            "name": "search_wikipedia",
            "mcp_server_name": "deepwiki",
            "arguments": {"query": "Python", "language": "en"}
        }
        
        response = {
            "results": [
                {"title": "Python (programming language)", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"}
            ],
            "total_count": 1
        }
        user_api_key = UserAPIKeyAuth()
        
        result = await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key,
            response=response
        )
        
        assert result == response

    def test_deepwiki_custom_validation_function(self):
        """Test DeepWiki custom validation function loading"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            custom_validation_function="cookbook.deepwiki_validation:validate_deepwiki_request"
        )
        
        # Test that the function is loaded correctly
        assert guardrail.custom_validation_function is not None
        assert callable(guardrail.custom_validation_function)

    def test_deepwiki_apply_validation_rules(self):
        """Test DeepWiki validation rules application"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            validation_rules={
                "max_query_length": 10,
                "forbidden_keywords": ["secret", "password"]
            }
        )
        
        arguments = {"query": "Python programming", "description": "This contains a secret"}
        
        with pytest.raises(ValueError, match="Forbidden keyword"):
            guardrail._apply_validation_rules(arguments)

    def test_deepwiki_apply_custom_validation(self):
        """Test DeepWiki custom validation application"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            custom_validation_function="cookbook.deepwiki_validation:validate_deepwiki_request"
        )
        
        arguments = {"query": "Python", "language": "en"}
        validation_rules = {"max_query_length": 10}
        
        result = guardrail._apply_custom_validation("search_wikipedia", arguments)
        assert result == arguments

    def test_deepwiki_validate_tool_result(self):
        """Test DeepWiki tool result validation"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-guard",
            validation_rules={
                "max_result_size": 100
            }
        )
        
        result = {"results": [{"title": "Python", "url": "https://example.com"}], "total_count": 1}
        validated_result = guardrail._validate_tool_result("search_wikipedia", result)
        assert validated_result == result


class TestDeepWikiGuardrailIntegration:
    """Test DeepWiki MCP Guardrail integration scenarios"""

    @pytest.mark.asyncio
    async def test_deepwiki_search_guardrail(self):
        """Test DeepWiki search with guardrails"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-search-guard",
            mcp_server_name="deepwiki",
            tool_name="search_wikipedia",
            block_on_error=True,
            validation_rules={
                "min_query_length": 3,
                "max_query_length": 50,
                "forbidden_search_terms": ["admin", "internal", "private"]
            }
        )
        
        # Valid request
        valid_data = {
            "name": "search_wikipedia",
            "mcp_server_name": "deepwiki",
            "arguments": {
                "query": "Python programming language",
                "language": "en",
                "max_results": 10
            }
        }
        
        user_api_key = UserAPIKeyAuth()
        cache = DualCache()
        
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=cache,
            data=valid_data,
            call_type="completion"
        )
        
        assert result == valid_data
        
        # Invalid request - forbidden search term
        invalid_data = {
            "name": "search_wikipedia",
            "mcp_server_name": "deepwiki",
            "arguments": {
                "query": "admin panel access",
                "language": "en"
            }
        }
        
        with pytest.raises(ValueError, match="Forbidden search term"):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key,
                cache=cache,
                data=invalid_data,
                call_type="completion"
            )

    @pytest.mark.asyncio
    async def test_deepwiki_content_guardrail(self):
        """Test DeepWiki content guardrails"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-content-guard",
            mcp_server_name="deepwiki",
            block_on_error=True,
            validation_rules={
                "max_query_length": 100,
                "forbidden_keywords": ["password", "secret", "api_key"],
                "allowed_search_types": ["article", "topic", "concept"]
            }
        )
        
        # Valid request
        valid_data = {
            "name": "get_article",
            "mcp_server_name": "deepwiki",
            "arguments": {
                "title": "Python programming",
                "search_type": "article"
            }
        }
        
        user_api_key = UserAPIKeyAuth()
        cache = DualCache()
        
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=cache,
            data=valid_data,
            call_type="completion"
        )
        
        assert result == valid_data
        
        # Invalid request - forbidden keyword
        invalid_data = {
            "name": "get_article",
            "mcp_server_name": "deepwiki",
            "arguments": {
                "title": "How to manage api_key secrets",
                "search_type": "article"
            }
        }
        
        with pytest.raises(ValueError, match="Forbidden keyword"):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key,
                cache=cache,
                data=invalid_data,
                call_type="completion"
            )

    @pytest.mark.asyncio
    async def test_deepwiki_general_guardrail(self):
        """Test DeepWiki general guardrails"""
        guardrail = MCPGuardrail(
            guardrail_name="deepwiki-general-guard",
            mcp_server_name="deepwiki",
            block_on_error=False,  # Don't block, just log warnings
            validation_rules={
                "max_argument_size": 1024,
                "allowed_tool_patterns": ["search_*", "get_*", "list_*"],
                "forbidden_tool_patterns": ["delete_*", "modify_*", "admin_*"]
            }
        )
        
        # Valid request
        valid_data = {
            "name": "search_wikipedia",
            "mcp_server_name": "deepwiki",
            "arguments": {
                "query": "Python"
            }
        }
        
        user_api_key = UserAPIKeyAuth()
        cache = DualCache()
        
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=cache,
            data=valid_data,
            call_type="completion"
        )
        
        assert result == valid_data
        
        # Invalid request - forbidden tool pattern
        invalid_data = {
            "name": "delete_article",
            "mcp_server_name": "deepwiki",
            "arguments": {
                "title": "Some article"
            }
        }
        
        with pytest.raises(ValueError, match="matches forbidden pattern"):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key,
                cache=cache,
                data=invalid_data,
                call_type="completion"
            )


if __name__ == "__main__":
    pytest.main([__file__]) 
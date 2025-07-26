"""
Test MCP Guardrail functionality
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


class TestMCPGuardrail:
    """Test MCP Guardrail functionality"""

    def test_mcp_guardrail_init(self):
        """Test MCPGuardrail initialization"""
        guardrail = MCPGuardrail(
            guardrail_name="test-mcp-guard",
            mcp_server_name="github",
            tool_name="create_issue",
            block_on_error=True,
            timeout=30.0,
            validation_rules={
                "max_title_length": 256,
                "forbidden_keywords": ["password", "secret"]
            }
        )
        
        assert guardrail.guardrail_name == "test-mcp-guard"
        assert guardrail.mcp_server_name == "github"
        assert guardrail.tool_name == "create_issue"
        assert guardrail.block_on_error is True
        assert guardrail.timeout == 30.0

    def test_should_apply_to_tool(self):
        """Test tool filtering logic"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            mcp_server_name="github",
            tool_name="create_issue"
        )
        
        # Should apply to specific tool
        assert guardrail._should_apply_to_tool("create_issue", "github") is True
        
        # Should not apply to different tool
        assert guardrail._should_apply_to_tool("list_issues", "github") is False
        
        # Should not apply to different server
        assert guardrail._should_apply_to_tool("create_issue", "zapier") is False

    def test_should_apply_to_tool_server_only(self):
        """Test server-only filtering"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            mcp_server_name="github"
        )
        
        # Should apply to any tool in the server
        assert guardrail._should_apply_to_tool("create_issue", "github") is True
        assert guardrail._should_apply_to_tool("list_issues", "github") is True
        
        # Should not apply to different server
        assert guardrail._should_apply_to_tool("create_issue", "zapier") is False

    def test_should_apply_to_tool_general(self):
        """Test general guardrail (no specific server/tool)"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard"
        )
        
        # Should apply to any tool/server
        assert guardrail._should_apply_to_tool("create_issue", "github") is True
        assert guardrail._should_apply_to_tool("list_issues", "zapier") is True

    def test_validate_tool_arguments(self):
        """Test argument validation"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            validation_rules={
                "max_title_length": 10,
                "forbidden_keywords": ["secret"]
            }
        )
        
        # Valid arguments
        valid_args = {"title": "Test Issue", "body": "Test body"}
        result = guardrail._validate_tool_arguments("create_issue", valid_args)
        assert result == valid_args
        
        # Invalid arguments (title too long)
        invalid_args = {"title": "This title is too long", "body": "Test body"}
        with pytest.raises(ValueError, match="Title exceeds maximum length"):
            guardrail._validate_tool_arguments("create_issue", invalid_args)

    @pytest.mark.asyncio
    async def test_pre_call_hook(self):
        """Test pre-call hook"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            mcp_server_name="github",
            tool_name="create_issue",
            validation_rules={
                "max_title_length": 10
            }
        )
        
        data = {
            "name": "create_issue",
            "mcp_server_name": "github",
            "arguments": {"title": "Test", "body": "Test body"}
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
    async def test_pre_call_hook_validation_failure(self):
        """Test pre-call hook with validation failure"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            mcp_server_name="github",
            tool_name="create_issue",
            block_on_error=True,
            validation_rules={
                "max_title_length": 5
            }
        )
        
        data = {
            "name": "create_issue",
            "mcp_server_name": "github",
            "arguments": {"title": "This title is too long", "body": "Test body"}
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
    async def test_pre_call_hook_skip_validation(self):
        """Test pre-call hook skipping validation"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            mcp_server_name="github",
            tool_name="create_issue"
        )
        
        # Different tool - should skip validation
        data = {
            "name": "list_issues",
            "mcp_server_name": "github",
            "arguments": {"title": "This should not be validated"}
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
    async def test_during_call_hook(self):
        """Test during-call hook"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            mcp_server_name="github"
        )
        
        data = {
            "name": "create_issue",
            "mcp_server_name": "github",
            "arguments": {"title": "Test", "body": "Test body"}
        }
        
        user_api_key = UserAPIKeyAuth()
        
        result = await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=user_api_key,
            call_type="completion"
        )
        
        assert result == data

    @pytest.mark.asyncio
    async def test_post_call_hook(self):
        """Test post-call hook"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            mcp_server_name="github",
            tool_name="create_issue"
        )
        
        data = {
            "name": "create_issue",
            "mcp_server_name": "github",
            "arguments": {"title": "Test", "body": "Test body"}
        }
        
        response = {"id": 123, "number": 1, "html_url": "https://github.com/issue/123"}
        user_api_key = UserAPIKeyAuth()
        
        result = await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key,
            response=response
        )
        
        assert result == response

    def test_custom_validation_function(self):
        """Test custom validation function loading"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            custom_validation_function="cookbook.mcp_validation:validate_github_request"
        )
        
        # Test that the function is loaded correctly
        assert guardrail.custom_validation_function is not None
        assert callable(guardrail.custom_validation_function)

    def test_apply_validation_rules(self):
        """Test validation rules application"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            validation_rules={
                "max_title_length": 10,
                "forbidden_keywords": ["secret", "password"]
            }
        )
        
        arguments = {"title": "Test Issue", "body": "This contains a secret"}
        
        with pytest.raises(ValueError, match="Forbidden keyword"):
            guardrail._apply_validation_rules(arguments)

    def test_apply_custom_validation(self):
        """Test custom validation application"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            custom_validation_function="cookbook.mcp_validation:validate_github_request"
        )
        
        arguments = {"title": "Test", "body": "Test body"}
        validation_rules = {"max_title_length": 10}
        
        result = guardrail._apply_custom_validation("create_issue", arguments)
        assert result == arguments

    def test_validate_tool_result(self):
        """Test tool result validation"""
        guardrail = MCPGuardrail(
            guardrail_name="test-guard",
            validation_rules={
                "max_result_size": 100
            }
        )
        
        result = {"id": 123, "data": "small result"}
        validated_result = guardrail._validate_tool_result("create_issue", result)
        assert validated_result == result

    def test_get_config_model(self):
        """Test config model retrieval"""
        config_model = MCPGuardrail.get_config_model()
        assert config_model is not None


class TestMCPGuardrailIntegration:
    """Test MCP Guardrail integration scenarios"""

    @pytest.mark.asyncio
    async def test_github_issue_creation_guardrail(self):
        """Test GitHub issue creation with guardrails"""
        guardrail = MCPGuardrail(
            guardrail_name="github-security",
            mcp_server_name="github",
            tool_name="create_issue",
            block_on_error=True,
            validation_rules={
                "max_title_length": 50,
                "max_body_length": 1000,
                "forbidden_keywords": ["password", "secret", "api_key"],
                "allowed_repositories": ["my-org/test-repo"]
            }
        )
        
        # Valid request
        valid_data = {
            "name": "create_issue",
            "mcp_server_name": "github",
            "arguments": {
                "owner": "my-org",
                "repo": "test-repo",
                "title": "Bug report",
                "body": "Found a bug in the code"
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
            "name": "create_issue",
            "mcp_server_name": "github",
            "arguments": {
                "owner": "my-org",
                "repo": "test-repo",
                "title": "Bug report",
                "body": "Found a bug and the password is 123456"
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
    async def test_zapier_webhook_guardrail(self):
        """Test Zapier webhook creation with guardrails"""
        guardrail = MCPGuardrail(
            guardrail_name="zapier-safety",
            mcp_server_name="zapier",
            tool_name="create_webhook",
            block_on_error=True,
            validation_rules={
                "allowed_webhook_urls": ["https://my-app.com/webhook"],
                "max_webhook_count": 5
            }
        )
        
        # Valid request
        valid_data = {
            "name": "create_webhook",
            "mcp_server_name": "zapier",
            "arguments": {
                "url": "https://my-app.com/webhook",
                "events": ["push", "pull_request"]
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
        
        # Invalid request - unauthorized URL
        invalid_data = {
            "name": "create_webhook",
            "mcp_server_name": "zapier",
            "arguments": {
                "url": "https://malicious-site.com/webhook",
                "events": ["push"]
            }
        }
        
        with pytest.raises(ValueError, match="Webhook URL.*not in the allowed list"):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key,
                cache=cache,
                data=invalid_data,
                call_type="completion"
            )


if __name__ == "__main__":
    pytest.main([__file__]) 
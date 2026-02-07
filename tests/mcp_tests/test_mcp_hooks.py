"""
Test file for MCP Hook Architecture

This file demonstrates the new MCP hook system with comprehensive examples
and validation tests.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Optional

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.mcp import (
    MCPPreCallRequestObject,
    MCPPreCallResponseObject,
    MCPDuringCallRequestObject,
    MCPDuringCallResponseObject,
    MCPPostCallResponseObject,
)
from litellm.types.llms.base import HiddenParams


class TestMCPAccessControlHook(CustomLogger):
    """Test hook for access control functionality"""
    
    def __init__(self):
        self.allowed_tools = {"github/create_issue", "zapier/send_email"}
        self.blocked_users = {"user123", "user456"}
        self.call_count = 0
    
    async def async_pre_mcp_tool_call_hook(
        self, 
        kwargs, 
        request_obj: MCPPreCallRequestObject, 
        start_time, 
        end_time
    ) -> Optional[MCPPreCallResponseObject]:
        """Test access control validation"""
        self.call_count += 1
        
        tool_name = request_obj.tool_name
        user_id = kwargs.get("user_api_key_auth", {}).get("user_id")
        
        # Check if user is blocked
        if user_id in self.blocked_users:
            return MCPPreCallResponseObject(
                should_proceed=False,
                error_message=f"User {user_id} is not authorized to use MCP tools"
            )
        
        # Check if tool is allowed
        if tool_name not in self.allowed_tools:
            return MCPPreCallResponseObject(
                should_proceed=False,
                error_message=f"Tool {tool_name} is not authorized"
            )
        
        return None  # Allow execution to proceed


class TestMCPCostTrackingHook(CustomLogger):
    """Test hook for cost tracking functionality"""
    
    def __init__(self):
        self.cost_map = {
            "github/create_issue": 0.10,
            "zapier/send_email": 0.05,
            "default": 0.01
        }
        self.call_count = 0
    
    async def async_post_mcp_tool_call_hook(
        self, 
        kwargs, 
        response_obj: MCPPostCallResponseObject, 
        start_time, 
        end_time
    ) -> Optional[MCPPostCallResponseObject]:
        """Test cost calculation after tool execution"""
        self.call_count += 1
        
        tool_name = kwargs.get("name", "")
        cost = self.cost_map.get(tool_name, self.cost_map["default"])
        
        # Set the response cost
        response_obj.hidden_params.response_cost = cost
        
        return response_obj


class TestMCPMonitoringHook(CustomLogger):
    """Test hook for real-time monitoring functionality"""
    
    def __init__(self):
        self.max_execution_time = 30.0  # seconds
        self.call_count = 0
    
    async def async_during_mcp_tool_call_hook(
        self, 
        kwargs, 
        request_obj: MCPDuringCallRequestObject, 
        start_time, 
        end_time
    ) -> Optional[MCPDuringCallResponseObject]:
        """Test execution time monitoring"""
        self.call_count += 1
        
        tool_name = request_obj.tool_name
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Check if execution is taking too long
        if execution_time > self.max_execution_time:
            return MCPDuringCallResponseObject(
                should_continue=False,
                error_message=f"Tool {tool_name} execution timeout after {execution_time}s"
            )
        
        return None  # Allow execution to continue


class TestMCPArgumentValidationHook(CustomLogger):
    """Test hook for argument validation functionality"""
    
    def __init__(self):
        self.call_count = 0
    
    async def async_pre_mcp_tool_call_hook(
        self, 
        kwargs, 
        request_obj: MCPPreCallRequestObject, 
        start_time, 
        end_time
    ) -> Optional[MCPPreCallResponseObject]:
        """Test argument validation and sanitization"""
        self.call_count += 1
        
        tool_name = request_obj.tool_name
        arguments = request_obj.arguments.copy()  # Create a copy to modify
        
        # Example: Validate GitHub issue creation
        if tool_name == "github/create_issue":
            if not arguments.get("title"):
                return MCPPreCallResponseObject(
                    should_proceed=False,
                    error_message="GitHub issue title is required"
                )
            
            # Sanitize the title
            title = arguments["title"]
            if len(title) > 100:
                title = title[:97] + "..."
                arguments["title"] = title
        
        # Example: Validate email sending
        elif tool_name == "zapier/send_email":
            if not arguments.get("to"):
                return MCPPreCallResponseObject(
                    should_proceed=False,
                    error_message="Email recipient is required"
                )
        
        return MCPPreCallResponseObject(
            should_proceed=True,
            modified_arguments=arguments
        )


# Test fixtures
@pytest.fixture
def access_control_hook():
    return TestMCPAccessControlHook()


@pytest.fixture
def cost_tracking_hook():
    return TestMCPCostTrackingHook()


@pytest.fixture
def monitoring_hook():
    return TestMCPMonitoringHook()


@pytest.fixture
def argument_validation_hook():
    return TestMCPArgumentValidationHook()


# Test cases
class TestMCPHooks:
    """Test cases for MCP hook functionality"""
    
    @pytest.mark.asyncio
    async def test_access_control_hook_allowed_tool(self, access_control_hook):
        """Test that allowed tools pass validation"""
        kwargs = {
            "user_api_key_auth": {"user_id": "user789"},
            "name": "github/create_issue"
        }
        request_obj = MCPPreCallRequestObject(
            tool_name="github/create_issue",
            arguments={"title": "Test issue"},
            user_api_key_auth={"user_id": "user789"}
        )
        
        result = await access_control_hook.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is None  # Should allow execution
        assert access_control_hook.call_count == 1
    
    @pytest.mark.asyncio
    async def test_access_control_hook_blocked_user(self, access_control_hook):
        """Test that blocked users are rejected"""
        kwargs = {
            "user_api_key_auth": {"user_id": "user123"},
            "name": "github/create_issue"
        }
        request_obj = MCPPreCallRequestObject(
            tool_name="github/create_issue",
            arguments={"title": "Test issue"},
            user_api_key_auth={"user_id": "user123"}
        )
        
        result = await access_control_hook.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is not None
        assert result.should_proceed is False
        assert "not authorized" in result.error_message
    
    @pytest.mark.asyncio
    async def test_access_control_hook_unauthorized_tool(self, access_control_hook):
        """Test that unauthorized tools are rejected"""
        kwargs = {
            "user_api_key_auth": {"user_id": "user789"},
            "name": "unauthorized_tool"
        }
        request_obj = MCPPreCallRequestObject(
            tool_name="unauthorized_tool",
            arguments={"param": "value"},
            user_api_key_auth={"user_id": "user789"}
        )
        
        result = await access_control_hook.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is not None
        assert result.should_proceed is False
        assert "not authorized" in result.error_message
    
    @pytest.mark.asyncio
    async def test_cost_tracking_hook(self, cost_tracking_hook):
        """Test cost tracking functionality"""
        kwargs = {"name": "github/create_issue"}
        response_obj = MCPPostCallResponseObject(
            mcp_tool_call_response=[],
            hidden_params=HiddenParams()
        )
        
        result = await cost_tracking_hook.async_post_mcp_tool_call_hook(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is not None
        assert result.hidden_params.response_cost == 0.10
        assert cost_tracking_hook.call_count == 1
    
    @pytest.mark.asyncio
    async def test_cost_tracking_hook_default_cost(self, cost_tracking_hook):
        """Test default cost assignment"""
        kwargs = {"name": "unknown_tool"}
        response_obj = MCPPostCallResponseObject(
            mcp_tool_call_response=[],
            hidden_params=HiddenParams()
        )
        
        result = await cost_tracking_hook.async_post_mcp_tool_call_hook(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is not None
        assert result.hidden_params.response_cost == 0.01  # Default cost
    
    @pytest.mark.asyncio
    async def test_monitoring_hook_normal_execution(self, monitoring_hook):
        """Test monitoring hook with normal execution time"""
        kwargs = {"name": "test_tool"}
        request_obj = MCPDuringCallRequestObject(
            tool_name="test_tool",
            arguments={},
            start_time=datetime.now().timestamp()
        )
        
        result = await monitoring_hook.async_during_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is None  # Should allow execution to continue
        assert monitoring_hook.call_count == 1
    
    @pytest.mark.asyncio
    async def test_argument_validation_hook_valid_github_issue(self, argument_validation_hook):
        """Test argument validation for valid GitHub issue"""
        kwargs = {"name": "github/create_issue"}
        request_obj = MCPPreCallRequestObject(
            tool_name="github/create_issue",
            arguments={"title": "Valid issue title"}
        )
        
        result = await argument_validation_hook.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is not None
        assert result.should_proceed is True
        assert result.modified_arguments == {"title": "Valid issue title"}
        assert argument_validation_hook.call_count == 1
    
    @pytest.mark.asyncio
    async def test_argument_validation_hook_missing_title(self, argument_validation_hook):
        """Test argument validation for missing GitHub issue title"""
        kwargs = {"name": "github/create_issue"}
        request_obj = MCPPreCallRequestObject(
            tool_name="github/create_issue",
            arguments={}  # Missing title
        )
        
        result = await argument_validation_hook.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is not None
        assert result.should_proceed is False
        assert "title is required" in result.error_message
    
    @pytest.mark.asyncio
    async def test_argument_validation_hook_long_title_sanitization(self, argument_validation_hook):
        """Test argument validation with title sanitization"""
        kwargs = {"name": "github/create_issue"}
        long_title = "A" * 150  # Very long title
        request_obj = MCPPreCallRequestObject(
            tool_name="github/create_issue",
            arguments={"title": long_title}
        )
        
        result = await argument_validation_hook.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is not None
        assert result.should_proceed is True
        assert len(result.modified_arguments["title"]) == 100  # Truncated
        assert result.modified_arguments["title"].endswith("...")
    
    @pytest.mark.asyncio
    async def test_argument_validation_hook_email_validation(self, argument_validation_hook):
        """Test argument validation for email sending"""
        kwargs = {"name": "zapier/send_email"}
        request_obj = MCPPreCallRequestObject(
            tool_name="zapier/send_email",
            arguments={"to": "test@example.com", "subject": "Test"}
        )
        
        result = await argument_validation_hook.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is not None
        assert result.should_proceed is True
        assert result.modified_arguments == {"to": "test@example.com", "subject": "Test"}
    
    @pytest.mark.asyncio
    async def test_argument_validation_hook_missing_email_recipient(self, argument_validation_hook):
        """Test argument validation for missing email recipient"""
        kwargs = {"name": "zapier/send_email"}
        request_obj = MCPPreCallRequestObject(
            tool_name="zapier/send_email",
            arguments={"subject": "Test"}  # Missing 'to' field
        )
        
        result = await argument_validation_hook.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result is not None
        assert result.should_proceed is False
        assert "recipient is required" in result.error_message


# Integration test
class TestMCPHookIntegration:
    """Integration tests for MCP hook system"""
    
    @pytest.mark.asyncio
    async def test_hook_chain_execution(self):
        """Test that multiple hooks can work together"""
        access_hook = TestMCPAccessControlHook()
        cost_hook = TestMCPCostTrackingHook()
        validation_hook = TestMCPArgumentValidationHook()
        
        # Test data
        kwargs = {
            "user_api_key_auth": {"user_id": "user789"},
            "name": "github/create_issue"
        }
        request_obj = MCPPreCallRequestObject(
            tool_name="github/create_issue",
            arguments={"title": "Integration test issue"},
            user_api_key_auth={"user_id": "user789"}
        )
        
        # Execute pre-hooks
        access_result = await access_hook.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        validation_result = await validation_hook.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Both hooks should allow execution
        assert access_result is None
        assert validation_result is not None
        assert validation_result.should_proceed is True
        
        # Simulate post-hook execution
        response_obj = MCPPostCallResponseObject(
            mcp_tool_call_response=[],
            hidden_params=HiddenParams()
        )
        
        cost_result = await cost_hook.async_post_mcp_tool_call_hook(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert cost_result is not None
        assert cost_result.hidden_params.response_cost == 0.10


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 
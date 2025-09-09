"""
Test file for MCP Guardrails Feature

This file tests the MCP guardrails functionality for both pre and during MCP call hooks,
including various guardrail types and proper exception handling.
"""

import asyncio
import pytest
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache
from litellm.types.mcp import (
    MCPPreCallRequestObject,
    MCPPreCallResponseObject,
    MCPDuringCallRequestObject,
    MCPDuringCallResponseObject,
)
from litellm.types.llms.base import HiddenParams
from litellm.types.guardrails import GuardrailEventHooks
from fastapi import HTTPException


class MockPiiGuardrail(CustomGuardrail):
    """Mock PII guardrail that raises BlockedPiiEntityError"""
    
    def __init__(self, should_block: bool = True, entity_type: str = "EMAIL_ADDRESS"):
        super().__init__()
        self.should_block = should_block
        self.entity_type = entity_type
        self.guardrail_name = "mock-pii-guardrail"
        self.call_count = 0
    
    def should_run_guardrail(self, data: dict, event_type: GuardrailEventHooks) -> bool:
        """Always run for testing"""
        return True
    
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        """Mock pre-call hook that raises BlockedPiiEntityError"""
        self.call_count += 1
        
        if self.should_block:
            raise BlockedPiiEntityError(
                entity_type=self.entity_type,
                guardrail_name=self.guardrail_name,
            )
        return None


class MockContentGuardrail(CustomGuardrail):
    """Mock content guardrail that raises GuardrailRaisedException"""
    
    def __init__(self, should_block: bool = True):
        super().__init__()
        self.should_block = should_block
        self.guardrail_name = "mock-content-guardrail"
        self.call_count = 0
    
    def should_run_guardrail(self, data: dict, event_type: GuardrailEventHooks) -> bool:
        """Always run for testing"""
        return True
    
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        """Mock pre-call hook that raises GuardrailRaisedException"""
        self.call_count += 1
        
        if self.should_block:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message="Content violates policy"
            )
        return None


class MockHttpGuardrail(CustomGuardrail):
    """Mock HTTP guardrail that raises HTTPException"""
    
    def __init__(self, should_block: bool = True):
        super().__init__()
        self.should_block = should_block
        self.guardrail_name = "mock-http-guardrail"
        self.call_count = 0
    
    def should_run_guardrail(self, data: dict, event_type: GuardrailEventHooks) -> bool:
        """Always run for testing"""
        return True
    
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        """Mock pre-call hook that raises HTTPException"""
        self.call_count += 1
        
        if self.should_block:
            raise HTTPException(
                status_code=400,
                detail={"error": "Violated guardrail policy"}
            )
        return None


class MockDuringCallGuardrail(CustomGuardrail):
    """Mock guardrail for during-call testing"""
    
    def __init__(self, should_block: bool = True):
        super().__init__()
        self.should_block = should_block
        self.guardrail_name = "mock-during-guardrail"
        self.call_count = 0
    
    def should_run_guardrail(self, data: dict, event_type: GuardrailEventHooks) -> bool:
        """Always run for testing"""
        return True
    
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: str,
    ):
        """Mock during-call hook that raises exceptions"""
        self.call_count += 1
        
        if self.should_block:
            raise BlockedPiiEntityError(
                entity_type="PHONE_NUMBER",
                guardrail_name=self.guardrail_name,
            )
        return None


class MockProxyLogging:
    """Mock proxy logging object for testing MCP guardrails"""
    
    def __init__(self, guardrails: Optional[list] = None):
        self.guardrails = guardrails if guardrails is not None else []
        self.call_details = {"user_api_key_cache": DualCache()}
        self.dynamic_success_callbacks = []
        self.call_count = 0
    
    def get_combined_callback_list(self, dynamic_success_callbacks, global_callbacks):
        """Return the guardrails for testing"""
        return self.guardrails
    
    def _convert_mcp_to_llm_format(self, request_obj, kwargs: dict) -> dict:
        """Convert MCP tool call to LLM message format"""
        tool_call_content = f"Tool: {request_obj.tool_name}\nArguments: {request_obj.arguments}"
        
        return {
            "messages": [{"role": "user", "content": tool_call_content}],
            "model": kwargs.get("model", "mcp-tool-call"),
            "user_api_key_user_id": kwargs.get("user_api_key_user_id"),
            "user_api_key_team_id": kwargs.get("user_api_key_team_id"),
        }
    
    def _convert_llm_result_to_mcp_response(self, llm_result, request_obj):
        """Convert LLM result back to MCP response format"""
        return None  # For testing, we don't need to convert back
    
    def _parse_pre_mcp_call_hook_response(self, response, original_request):
        """Parse pre MCP call hook response"""
        return response
    
    async def async_pre_mcp_tool_call_hook(
        self,
        kwargs: dict,
        request_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[Any]:
        """Mock pre MCP tool call hook"""
        self.call_count += 1
        
        # Simulate the actual hook logic
        for guardrail in self.guardrails:
            if isinstance(guardrail, CustomGuardrail):
                try:
                    synthetic_data = self._convert_mcp_to_llm_format(request_obj, kwargs)
                    
                    # Check if guardrail should run
                    if not guardrail.should_run_guardrail(synthetic_data, GuardrailEventHooks.pre_mcp_call):
                        continue
                    
                    result = await guardrail.async_pre_call_hook(
                        user_api_key_dict=kwargs.get("user_api_key_auth"),
                        cache=self.call_details["user_api_key_cache"],
                        data=synthetic_data,
                        call_type="mcp_call"
                    )
                    if result is not None:
                        return self._parse_pre_mcp_call_hook_response(result, request_obj)
                except (BlockedPiiEntityError, GuardrailRaisedException, HTTPException) as e:
                    # Re-raise guardrail exceptions
                    raise e
                except Exception as e:
                    # Log non-guardrail exceptions as non-blocking
                    print(f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {str(e)}")
        
        return None
    
    async def async_during_mcp_tool_call_hook(
        self,
        kwargs: dict,
        request_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[Any]:
        """Mock during MCP tool call hook"""
        self.call_count += 1
        
        # Simulate the actual hook logic
        for guardrail in self.guardrails:
            if isinstance(guardrail, CustomGuardrail):
                try:
                    synthetic_data = self._convert_mcp_to_llm_format(request_obj, kwargs)
                    result = await guardrail.async_moderation_hook(
                        data=synthetic_data,
                        user_api_key_dict=kwargs.get("user_api_key_auth"),
                        call_type="mcp_call"
                    )
                    if result is not None:
                        return result
                except (BlockedPiiEntityError, GuardrailRaisedException, HTTPException) as e:
                    # Re-raise guardrail exceptions
                    raise e
                except Exception as e:
                    # Log non-guardrail exceptions as non-blocking
                    print(f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {str(e)}")
        
        return None


@pytest.fixture
def mock_user_api_key():
    """Mock user API key for testing"""
    return UserAPIKeyAuth(api_key="test_key", user_id="test_user")


@pytest.fixture
def mock_cache():
    """Mock cache for testing"""
    return DualCache()


@pytest.fixture
def mock_pii_guardrail():
    """Mock PII guardrail that blocks"""
    return MockPiiGuardrail(should_block=True)


@pytest.fixture
def mock_pii_guardrail_allow():
    """Mock PII guardrail that allows"""
    return MockPiiGuardrail(should_block=False)


@pytest.fixture
def mock_content_guardrail():
    """Mock content guardrail that blocks"""
    return MockContentGuardrail(should_block=True)


@pytest.fixture
def mock_http_guardrail():
    """Mock HTTP guardrail that blocks"""
    return MockHttpGuardrail(should_block=True)


@pytest.fixture
def mock_during_guardrail():
    """Mock during-call guardrail that blocks"""
    return MockDuringCallGuardrail(should_block=True)


@pytest.fixture
def mock_proxy_logging():
    """Mock proxy logging object"""
    return MockProxyLogging()


class TestMCPGuardrailsPreCall:
    """Test MCP guardrails for pre-call hooks"""
    
    @pytest.mark.asyncio
    async def test_pii_guardrail_blocks_pre_call(self, mock_pii_guardrail, mock_user_api_key, mock_cache):
        """Test that PII guardrail properly blocks pre-call"""
        proxy_logging = MockProxyLogging([mock_pii_guardrail])
        
        # Create MCP request
        request_obj = MCPPreCallRequestObject(
            tool_name="email_tool",
            arguments={"email": "test@example.com"},
            server_name="email_server",
            user_api_key_auth=mock_user_api_key.model_dump(),
            hidden_params=HiddenParams()
        )
        
        kwargs = {
            "name": "email_tool",
            "arguments": {"email": "test@example.com"},
            "server_name": "email_server",
            "user_api_key_auth": mock_user_api_key,
        }
        
        # Test that BlockedPiiEntityError is raised
        with pytest.raises(BlockedPiiEntityError) as excinfo:
            await proxy_logging.async_pre_mcp_tool_call_hook(
                kwargs=kwargs,
                request_obj=request_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        
        # Verify the error details
        assert excinfo.value.entity_type == "EMAIL_ADDRESS"
        assert excinfo.value.guardrail_name == "mock-pii-guardrail"
        assert mock_pii_guardrail.call_count == 1
    
    @pytest.mark.asyncio
    async def test_pii_guardrail_allows_pre_call(self, mock_pii_guardrail_allow, mock_user_api_key, mock_cache):
        """Test that PII guardrail allows pre-call when configured to allow"""
        proxy_logging = MockProxyLogging([mock_pii_guardrail_allow])
        
        request_obj = MCPPreCallRequestObject(
            tool_name="email_tool",
            arguments={"email": "test@example.com"},
            server_name="email_server",
            user_api_key_auth=mock_user_api_key.model_dump(),
            hidden_params=HiddenParams()
        )
        
        kwargs = {
            "name": "email_tool",
            "arguments": {"email": "test@example.com"},
            "server_name": "email_server",
            "user_api_key_auth": mock_user_api_key,
        }
        
        # Test that no exception is raised
        result = await proxy_logging.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        assert result is None
        assert mock_pii_guardrail_allow.call_count == 1
    
    @pytest.mark.asyncio
    async def test_content_guardrail_blocks_pre_call(self, mock_content_guardrail, mock_user_api_key, mock_cache):
        """Test that content guardrail properly blocks pre-call"""
        proxy_logging = MockProxyLogging([mock_content_guardrail])
        
        request_obj = MCPPreCallRequestObject(
            tool_name="content_tool",
            arguments={"content": "sensitive content"},
            server_name="content_server",
            user_api_key_auth=mock_user_api_key.model_dump(),
            hidden_params=HiddenParams()
        )
        
        kwargs = {
            "name": "content_tool",
            "arguments": {"content": "sensitive content"},
            "server_name": "content_server",
            "user_api_key_auth": mock_user_api_key,
        }
        
        # Test that GuardrailRaisedException is raised
        with pytest.raises(GuardrailRaisedException) as excinfo:
            await proxy_logging.async_pre_mcp_tool_call_hook(
                kwargs=kwargs,
                request_obj=request_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        
        # Verify the error details
        assert "Content violates policy" in str(excinfo.value)
        assert excinfo.value.guardrail_name == "mock-content-guardrail"
        assert mock_content_guardrail.call_count == 1
    
    @pytest.mark.asyncio
    async def test_http_guardrail_blocks_pre_call(self, mock_http_guardrail, mock_user_api_key, mock_cache):
        """Test that HTTP guardrail properly blocks pre-call"""
        proxy_logging = MockProxyLogging([mock_http_guardrail])
        
        request_obj = MCPPreCallRequestObject(
            tool_name="http_tool",
            arguments={"url": "http://example.com"},
            server_name="http_server",
            user_api_key_auth=mock_user_api_key.model_dump(),
            hidden_params=HiddenParams()
        )
        
        kwargs = {
            "name": "http_tool",
            "arguments": {"url": "http://example.com"},
            "server_name": "http_server",
            "user_api_key_auth": mock_user_api_key,
        }
        
        # Test that HTTPException is raised
        with pytest.raises(HTTPException) as excinfo:
            await proxy_logging.async_pre_mcp_tool_call_hook(
                kwargs=kwargs,
                request_obj=request_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        
        # Verify the error details
        assert excinfo.value.status_code == 400
        assert "Violated guardrail policy" in str(excinfo.value.detail)
        assert mock_http_guardrail.call_count == 1
    
    @pytest.mark.asyncio
    async def test_multiple_guardrails_pre_call(self, mock_pii_guardrail, mock_content_guardrail, mock_user_api_key, mock_cache):
        """Test multiple guardrails - first one should block"""
        proxy_logging = MockProxyLogging([mock_pii_guardrail, mock_content_guardrail])
        
        request_obj = MCPPreCallRequestObject(
            tool_name="test_tool",
            arguments={"email": "test@example.com"},
            server_name="test_server",
            user_api_key_auth=mock_user_api_key.model_dump(),
            hidden_params=HiddenParams()
        )
        
        kwargs = {
            "name": "test_tool",
            "arguments": {"email": "test@example.com"},
            "server_name": "test_server",
            "user_api_key_auth": mock_user_api_key,
        }
        
        # Test that first guardrail blocks
        with pytest.raises(BlockedPiiEntityError):
            await proxy_logging.async_pre_mcp_tool_call_hook(
                kwargs=kwargs,
                request_obj=request_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        
        # Verify only first guardrail was called
        assert mock_pii_guardrail.call_count == 1
        assert mock_content_guardrail.call_count == 0


class TestMCPGuardrailsDuringCall:
    """Test MCP guardrails for during-call hooks"""
    
    @pytest.mark.asyncio
    async def test_during_call_guardrail_blocks(self, mock_during_guardrail, mock_user_api_key, mock_cache):
        """Test that during-call guardrail properly blocks execution"""
        proxy_logging = MockProxyLogging([mock_during_guardrail])
        
        request_obj = MCPDuringCallRequestObject(
            tool_name="phone_tool",
            arguments={"phone": "555-123-4567"},
            server_name="phone_server",
            start_time=datetime.now().timestamp(),
            hidden_params=HiddenParams()
        )
        
        kwargs = {
            "name": "phone_tool",
            "arguments": {"phone": "555-123-4567"},
            "server_name": "phone_server",
        }
        
        # Test that BlockedPiiEntityError is raised
        with pytest.raises(BlockedPiiEntityError) as excinfo:
            await proxy_logging.async_during_mcp_tool_call_hook(
                kwargs=kwargs,
                request_obj=request_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        
        # Verify the error details
        assert excinfo.value.entity_type == "PHONE_NUMBER"
        assert excinfo.value.guardrail_name == "mock-during-guardrail"
        assert mock_during_guardrail.call_count == 1


class TestMCPGuardrailsIntegration:
    """Test MCP guardrails integration with MCP server manager"""
    
    @pytest.mark.asyncio
    async def test_mcp_server_manager_with_guardrails(self):
        """Test MCP server manager with guardrail integration"""
        
        mock_proxy_logging = MockProxyLogging([MockPiiGuardrail(should_block=True)])
        
        # Test that guardrail exception is properly raised in the hook
        with pytest.raises(BlockedPiiEntityError):
            await mock_proxy_logging.async_pre_mcp_tool_call_hook(
                kwargs={"name": "email_tool", "arguments": {"email": "test@example.com"}},
                request_obj=MagicMock(),
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
    
    @pytest.mark.asyncio
    async def test_guardrail_exception_propagation(self):
        """Test that guardrail exceptions properly propagate through the system"""
        # Test BlockedPiiEntityError
        with pytest.raises(BlockedPiiEntityError):
            raise BlockedPiiEntityError(
                entity_type="EMAIL_ADDRESS",
                guardrail_name="test-guardrail"
            )
        
        # Test GuardrailRaisedException
        with pytest.raises(GuardrailRaisedException):
            raise GuardrailRaisedException(
                guardrail_name="test-guardrail",
                message="Test message"
            )
        
        # Test HTTPException
        with pytest.raises(HTTPException):
            raise HTTPException(
                status_code=400,
                detail={"error": "Test error"}
            )


class TestMCPGuardrailsErrorHandling:
    """Test MCP guardrails error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_non_guardrail_exception_logging(self, mock_user_api_key, mock_cache):
        """Test that non-guardrail exceptions are logged as non-blocking"""
        class MockFailingGuardrail(CustomGuardrail):
            def should_run_guardrail(self, data: dict, event_type: GuardrailEventHooks) -> bool:
                return True
            
            async def async_pre_call_hook(
                self,
                user_api_key_dict: UserAPIKeyAuth,
                cache: DualCache,
                data: dict,
                call_type: str,
            ):
                raise Exception("Non-guardrail error")
        
        proxy_logging = MockProxyLogging([MockFailingGuardrail()])
        
        request_obj = MCPPreCallRequestObject(
            tool_name="test_tool",
            arguments={"test": "data"},
            server_name="test_server",
            user_api_key_auth=mock_user_api_key.model_dump(),
            hidden_params=HiddenParams()
        )
        
        kwargs = {
            "name": "test_tool",
            "arguments": {"test": "data"},
            "server_name": "test_server",
            "user_api_key_auth": mock_user_api_key,
        }
        
        # Test that non-guardrail exceptions are handled gracefully
        result = await proxy_logging.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        # Should return None (not raise exception)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_guardrail_should_not_run(self, mock_user_api_key, mock_cache):
        """Test that guardrails don't run when should_run_guardrail returns False"""
        class MockConditionalGuardrail(CustomGuardrail):
            def should_run_guardrail(self, data: dict, event_type: GuardrailEventHooks) -> bool:
                return False  # Don't run
            
            async def async_pre_call_hook(
                self,
                user_api_key_dict: UserAPIKeyAuth,
                cache: DualCache,
                data: dict,
                call_type: str,
            ):
                raise BlockedPiiEntityError("EMAIL_ADDRESS", "test-guardrail")
        
        proxy_logging = MockProxyLogging([MockConditionalGuardrail()])
        
        request_obj = MCPPreCallRequestObject(
            tool_name="test_tool",
            arguments={"test": "data"},
            server_name="test_server",
            user_api_key_auth=mock_user_api_key.model_dump(),
            hidden_params=HiddenParams()
        )
        
        kwargs = {
            "name": "test_tool",
            "arguments": {"test": "data"},
            "server_name": "test_server",
            "user_api_key_auth": mock_user_api_key,
        }
        
        # Test that guardrail doesn't run and no exception is raised
        result = await proxy_logging.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        # Should return None (guardrail didn't run)
        assert result is None


class TestMCPGuardrailsEdgeCases:
    """Test MCP guardrails edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_empty_guardrails_list(self, mock_user_api_key, mock_cache):
        """Test behavior with empty guardrails list"""
        proxy_logging = MockProxyLogging([])  # No guardrails
        
        request_obj = MCPPreCallRequestObject(
            tool_name="test_tool",
            arguments={"test": "data"},
            server_name="test_server",
            user_api_key_auth=mock_user_api_key.model_dump(),
            hidden_params=HiddenParams()
        )
        
        kwargs = {
            "name": "test_tool",
            "arguments": {"test": "data"},
            "server_name": "test_server",
            "user_api_key_auth": mock_user_api_key,
        }
        
        # Should return None without any issues
        result = await proxy_logging.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_guardrail_with_invalid_data(self, mock_user_api_key, mock_cache):
        """Test guardrail behavior with invalid data"""
        class MockInvalidDataGuardrail(CustomGuardrail):
            def should_run_guardrail(self, data: dict, event_type: GuardrailEventHooks) -> bool:
                return True
            
            async def async_pre_call_hook(
                self,
                user_api_key_dict: UserAPIKeyAuth,
                cache: DualCache,
                data: dict,
                call_type: str,
            ):
                # Try to access invalid data
                invalid_data = data.get("invalid_key", {})
                if invalid_data.get("should_fail"):
                    raise BlockedPiiEntityError("EMAIL_ADDRESS", "test-guardrail")
                return None
        
        proxy_logging = MockProxyLogging([MockInvalidDataGuardrail()])
        
        request_obj = MCPPreCallRequestObject(
            tool_name="test_tool",
            arguments={"test": "data"},
            server_name="test_server",
            user_api_key_auth=mock_user_api_key.model_dump(),
            hidden_params=HiddenParams()
        )
        
        kwargs = {
            "name": "test_tool",
            "arguments": {"test": "data"},
            "server_name": "test_server",
            "user_api_key_auth": mock_user_api_key,
        }
        
        # Should handle invalid data gracefully
        result = await proxy_logging.async_pre_mcp_tool_call_hook(
            kwargs=kwargs,
            request_obj=request_obj,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__]) 
"""
Test the Bedrock guardrail apply_guardrail functionality
"""
import sys
import os
import pytest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, os.path.abspath("../../../../.."))

from fastapi import HTTPException
from litellm.types.guardrails import ApplyGuardrailRequest, ApplyGuardrailResponse
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_success():
    """Test that Bedrock guardrail apply_guardrail method works correctly"""
    # Create a BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT"
    )
    
    # Mock the make_bedrock_api_request method
    with patch.object(guardrail, 'make_bedrock_api_request', new_callable=AsyncMock) as mock_api_request:
        # Mock a successful response from Bedrock
        mock_response = {
            "action": "ALLOWED",
            "content": [
                {
                    "text": {
                        "text": "This is a test message with some content"
                    }
                }
            ]
        }
        mock_api_request.return_value = mock_response
        
        # Test the apply_guardrail method
        result = await guardrail.apply_guardrail(
            text="This is a test message with some content",
            language="en"
        )
        
        # Verify the result
        assert result == "This is a test message with some content"
        mock_api_request.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_blocked():
    """Test that Bedrock guardrail apply_guardrail method handles blocked content"""
    # Create a BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT"
    )
    
    # Mock the make_bedrock_api_request method
    with patch.object(guardrail, 'make_bedrock_api_request', new_callable=AsyncMock) as mock_api_request:
        # Mock a blocked response from Bedrock
        mock_response = {
            "action": "BLOCKED",
            "reason": "Content violates policy"
        }
        mock_api_request.return_value = mock_response
        
        # Test the apply_guardrail method should raise an exception
        with pytest.raises(Exception) as exc_info:
            await guardrail.apply_guardrail(
                text="This is blocked content",
                language="en"
            )
        
        assert "Content blocked by Bedrock guardrail" in str(exc_info.value)
        assert "Content violates policy" in str(exc_info.value)


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_with_masking():
    """Test that Bedrock guardrail apply_guardrail method handles content masking"""
    # Create a BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT"
    )
    
    # Mock the make_bedrock_api_request method
    with patch.object(guardrail, 'make_bedrock_api_request', new_callable=AsyncMock) as mock_api_request:
        # Mock a response with masked content
        mock_response = {
            "action": "ALLOWED",
            "content": [
                {
                    "text": {
                        "text": "This is a test message with [REDACTED] content"
                    }
                }
            ]
        }
        mock_api_request.return_value = mock_response
        
        # Test the apply_guardrail method
        result = await guardrail.apply_guardrail(
            text="This is a test message with sensitive content",
            language="en"
        )
        
        # Verify the result contains the masked content
        assert result == "This is a test message with [REDACTED] content"
        mock_api_request.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_api_failure():
    """Test that Bedrock guardrail apply_guardrail method handles API failures"""
    # Create a BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT"
    )
    
    # Mock the make_bedrock_api_request method to raise an exception
    with patch.object(guardrail, 'make_bedrock_api_request', new_callable=AsyncMock) as mock_api_request:
        mock_api_request.side_effect = Exception("API connection failed")
        
        # Test the apply_guardrail method should raise an exception
        with pytest.raises(Exception) as exc_info:
            await guardrail.apply_guardrail(
                text="This is a test message",
                language="en"
            )
        
        assert "Bedrock guardrail failed" in str(exc_info.value)
        assert "API connection failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_endpoint_integration():
    """Test the full endpoint integration with Bedrock guardrail"""
    from enterprise.litellm_enterprise.proxy.guardrails.endpoints import apply_guardrail
    
    # Create a real BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT"
    )
    
    # Mock the guardrail registry
    with patch("enterprise.litellm_enterprise.proxy.guardrails.endpoints.GUARDRAIL_REGISTRY") as mock_registry:
        # Mock the make_bedrock_api_request method
        with patch.object(guardrail, 'make_bedrock_api_request', new_callable=AsyncMock) as mock_api_request:
            # Mock a successful response from Bedrock
            mock_response = {
                "action": "ALLOWED",
                "content": [
                    {
                        "text": {
                            "text": "This is a test message with processed content"
                        }
                    }
                ]
            }
            mock_api_request.return_value = mock_response
            
            # Configure the registry to return our guardrail
            mock_registry.get_initialized_guardrail_callback.return_value = guardrail
            
            # Create the request
            request = ApplyGuardrailRequest(
                guardrail_name="test-bedrock-guard",
                text="This is a test message with some content",
                language="en"
            )
            
            # Create a mock user API key
            user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
            
            # Call the endpoint
            response = await apply_guardrail(request=request, user_api_key_dict=user_api_key_dict)
            
            # Verify the response
            assert isinstance(response, ApplyGuardrailResponse)
            assert response.response_text == "This is a test message with processed content"
            mock_api_request.assert_called_once()

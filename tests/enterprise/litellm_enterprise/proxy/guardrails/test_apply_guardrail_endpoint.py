"""
Test the /guardrails/apply_guardrail endpoint
"""
import sys
import os
import pytest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, os.path.abspath("../../../../.."))

from fastapi import HTTPException
from litellm.types.guardrails import ApplyGuardrailRequest, ApplyGuardrailResponse
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_guardrail import CustomGuardrail


@pytest.mark.asyncio
async def test_apply_guardrail_endpoint_returns_correct_response():
    """Test that apply_guardrail endpoint returns ApplyGuardrailResponse object"""
    from enterprise.litellm_enterprise.proxy.guardrails.endpoints import apply_guardrail
    
    # Mock the guardrail registry
    with patch("enterprise.litellm_enterprise.proxy.guardrails.endpoints.GUARDRAIL_REGISTRY") as mock_registry:
        # Create a mock guardrail
        mock_guardrail = Mock(spec=CustomGuardrail)
        mock_guardrail.apply_guardrail = AsyncMock(return_value="Redacted text: [REDACTED] and [REDACTED]")
        
        # Configure the registry to return our mock guardrail
        mock_registry.get_initialized_guardrail_callback.return_value = mock_guardrail
        
        # Create the request
        request = ApplyGuardrailRequest(
            guardrail_name="test-guardrail",
            text="Test text with PII",
            language="en",
            entities=["EMAIL_ADDRESS", "PERSON"]
        )
        
        # Create a mock user API key
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
        
        # Call the endpoint
        response = await apply_guardrail(request=request, user_api_key_dict=user_api_key_dict)
        
        # Verify the response is of the correct type
        assert isinstance(response, ApplyGuardrailResponse)
        assert response.response_text == "Redacted text: [REDACTED] and [REDACTED]"
        
        # Verify the guardrail was called with correct parameters
        mock_guardrail.apply_guardrail.assert_called_once_with(
            text="Test text with PII",
            language="en",
            entities=["EMAIL_ADDRESS", "PERSON"]
        )


@pytest.mark.asyncio
async def test_apply_guardrail_endpoint_guardrail_not_found():
    """Test that apply_guardrail endpoint raises exception when guardrail not found"""
    from enterprise.litellm_enterprise.proxy.guardrails.endpoints import apply_guardrail
    
    # Mock the guardrail registry to return None
    with patch("enterprise.litellm_enterprise.proxy.guardrails.endpoints.GUARDRAIL_REGISTRY") as mock_registry:
        mock_registry.get_initialized_guardrail_callback.return_value = None
        
        # Create the request
        request = ApplyGuardrailRequest(
            guardrail_name="non-existent-guardrail",
            text="Test text",
            language="en"
        )
        
        # Create a mock user API key
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
        
        # Verify exception is raised
        with pytest.raises(Exception) as exc_info:
            await apply_guardrail(request=request, user_api_key_dict=user_api_key_dict)
        
        assert "Guardrail non-existent-guardrail not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_apply_guardrail_endpoint_with_presidio_guardrail():
    """Test apply_guardrail endpoint with a Presidio-like guardrail"""
    from enterprise.litellm_enterprise.proxy.guardrails.endpoints import apply_guardrail
    
    # Mock the guardrail registry
    with patch("enterprise.litellm_enterprise.proxy.guardrails.endpoints.GUARDRAIL_REGISTRY") as mock_registry:
        # Create a mock guardrail that simulates Presidio behavior
        mock_guardrail = Mock(spec=CustomGuardrail)
        # Simulate masking PII entities
        mock_guardrail.apply_guardrail = AsyncMock(
            return_value="My name is [PERSON] and my email is [EMAIL_ADDRESS]"
        )
        
        # Configure the registry to return our mock guardrail
        mock_registry.get_initialized_guardrail_callback.return_value = mock_guardrail
        
        # Create the request
        request = ApplyGuardrailRequest(
            guardrail_name="pii-detection-guard",
            text="My name is John Doe and my email is john@example.com",
            language="en",
            entities=["EMAIL_ADDRESS", "PERSON"]
        )
        
        # Create a mock user API key
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
        
        # Call the endpoint
        response = await apply_guardrail(request=request, user_api_key_dict=user_api_key_dict)
        
        # Verify the response is of the correct type
        assert isinstance(response, ApplyGuardrailResponse)
        assert response.response_text == "My name is [PERSON] and my email is [EMAIL_ADDRESS]"
        assert "john@example.com" not in response.response_text
        assert "John Doe" not in response.response_text


@pytest.mark.asyncio
async def test_apply_guardrail_endpoint_without_optional_params():
    """Test apply_guardrail endpoint without optional language and entities parameters"""
    from enterprise.litellm_enterprise.proxy.guardrails.endpoints import apply_guardrail
    
    # Mock the guardrail registry
    with patch("enterprise.litellm_enterprise.proxy.guardrails.endpoints.GUARDRAIL_REGISTRY") as mock_registry:
        # Create a mock guardrail
        mock_guardrail = Mock(spec=CustomGuardrail)
        mock_guardrail.apply_guardrail = AsyncMock(return_value="Processed text")
        
        # Configure the registry to return our mock guardrail
        mock_registry.get_initialized_guardrail_callback.return_value = mock_guardrail
        
        # Create the request without optional parameters
        request = ApplyGuardrailRequest(
            guardrail_name="test-guardrail",
            text="Test text"
        )
        
        # Create a mock user API key
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
        
        # Call the endpoint
        response = await apply_guardrail(request=request, user_api_key_dict=user_api_key_dict)
        
        # Verify the response is of the correct type
        assert isinstance(response, ApplyGuardrailResponse)
        assert response.response_text == "Processed text"
        
        # Verify the guardrail was called with None for optional parameters
        mock_guardrail.apply_guardrail.assert_called_once_with(
            text="Test text",
            language=None,
            entities=None
        )

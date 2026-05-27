"""
Test the /guardrails/apply_guardrail endpoint
"""

import os
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import ApplyGuardrailRequest, ApplyGuardrailResponse


@contextmanager
def _mock_proxy_logging():
    """Patch the proxy-server globals that apply_guardrail imports at call time."""
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.post_call_success_hook = AsyncMock(return_value=None)
    mock_proxy_logging.post_call_failure_hook = AsyncMock(return_value=None)
    mock_logging_obj = MagicMock()
    mock_logging_obj.async_success_handler = AsyncMock(return_value=None)
    mock_logging_obj.async_failure_handler = AsyncMock(return_value=None)
    mock_logging_obj.success_handler = MagicMock(return_value=None)
    mock_logging_obj.failure_handler = MagicMock(return_value=None)
    mock_logging_obj.model_call_details = {}

    with patch(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing"
    ) as mock_proc_cls, patch(
        "litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging
    ), patch(
        "litellm.proxy.proxy_server.general_settings", {}
    ), patch(
        "litellm.proxy.proxy_server.proxy_config", MagicMock()
    ), patch(
        "litellm.proxy.proxy_server.version", "0.0.0"
    ):
        mock_proc = MagicMock()
        mock_proc.common_processing_pre_call_logic = AsyncMock(
            return_value=({}, mock_logging_obj)
        )
        mock_proc_cls.return_value = mock_proc
        yield mock_proxy_logging


@pytest.mark.asyncio
async def test_apply_guardrail_endpoint_returns_correct_response():
    """Test that apply_guardrail endpoint returns ApplyGuardrailResponse object"""
    from litellm.proxy.guardrails.guardrail_endpoints import apply_guardrail

    # Mock the guardrail registry
    with patch(
        "litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY"
    ) as mock_registry, _mock_proxy_logging():
        # Create a mock guardrail
        mock_guardrail = Mock(spec=CustomGuardrail)
        # Apply guardrail returns GenericGuardrailAPIInputs (dict with texts key)
        mock_guardrail.apply_guardrail = AsyncMock(
            return_value={"texts": ["Redacted text: [REDACTED] and [REDACTED]"]}
        )

        # Configure the registry to return our mock guardrail
        mock_registry.get_initialized_guardrail_callback.return_value = mock_guardrail

        # Create the request
        request = ApplyGuardrailRequest(
            guardrail_name="test-guardrail",
            text="Test text with PII",
            language="en",
            entities=["EMAIL_ADDRESS", "PERSON"],
        )

        # Create a mock user API key
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        # Call the endpoint
        response = await apply_guardrail(
            fastapi_request=Mock(),
            request=request,
            user_api_key_dict=user_api_key_dict,
        )

        # Verify the response is of the correct type
        assert isinstance(response, ApplyGuardrailResponse)
        assert response.response_text == "Redacted text: [REDACTED] and [REDACTED]"

        # Verify the guardrail was called with correct parameters
        mock_guardrail.apply_guardrail.assert_called_once_with(
            inputs={"texts": ["Test text with PII"]},
            request_data={},
            input_type="request",
        )


@pytest.mark.asyncio
async def test_apply_guardrail_endpoint_guardrail_not_found():
    """Test that apply_guardrail endpoint raises exception when guardrail not found"""
    from litellm.proxy._types import ProxyException
    from litellm.proxy.guardrails.guardrail_endpoints import apply_guardrail

    # Mock the guardrail registry to return None
    with patch(
        "litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY"
    ) as mock_registry, _mock_proxy_logging():
        mock_registry.get_initialized_guardrail_callback.return_value = None

        # Create the request
        request = ApplyGuardrailRequest(
            guardrail_name="non-existent-guardrail", text="Test text", language="en"
        )

        # Create a mock user API key
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        # Verify exception is raised
        with pytest.raises(ProxyException) as exc_info:
            await apply_guardrail(
                fastapi_request=Mock(),
                request=request,
                user_api_key_dict=user_api_key_dict,
            )

        assert "non-existent-guardrail" in exc_info.value.message
        assert "not found" in exc_info.value.message


@pytest.mark.asyncio
async def test_apply_guardrail_endpoint_with_presidio_guardrail():
    """Test apply_guardrail endpoint with a Presidio-like guardrail"""
    from litellm.proxy.guardrails.guardrail_endpoints import apply_guardrail

    # Mock the guardrail registry
    with patch(
        "litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY"
    ) as mock_registry, _mock_proxy_logging():
        # Create a mock guardrail that simulates Presidio behavior
        mock_guardrail = Mock(spec=CustomGuardrail)
        # Simulate masking PII entities - returns GenericGuardrailAPIInputs (dict with texts key)
        mock_guardrail.apply_guardrail = AsyncMock(
            return_value={"texts": ["My name is [PERSON] and my email is [EMAIL_ADDRESS]"]}
        )

        # Configure the registry to return our mock guardrail
        mock_registry.get_initialized_guardrail_callback.return_value = mock_guardrail

        # Create the request
        request = ApplyGuardrailRequest(
            guardrail_name="pii-detection-guard",
            text="My name is John Doe and my email is john@example.com",
            language="en",
            entities=["EMAIL_ADDRESS", "PERSON"],
        )

        # Create a mock user API key
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        # Call the endpoint
        response = await apply_guardrail(
            fastapi_request=Mock(),
            request=request,
            user_api_key_dict=user_api_key_dict,
        )

        # Verify the response is of the correct type
        assert isinstance(response, ApplyGuardrailResponse)
        assert (
            response.response_text
            == "My name is [PERSON] and my email is [EMAIL_ADDRESS]"
        )
        assert "john@example.com" not in response.response_text
        assert "John Doe" not in response.response_text


@pytest.mark.asyncio
async def test_apply_guardrail_endpoint_without_optional_params():
    """Test apply_guardrail endpoint without optional language and entities parameters"""
    from litellm.proxy.guardrails.guardrail_endpoints import apply_guardrail

    # Mock the guardrail registry
    with patch(
        "litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY"
    ) as mock_registry, _mock_proxy_logging():
        # Create a mock guardrail
        mock_guardrail = Mock(spec=CustomGuardrail)
        # Returns GenericGuardrailAPIInputs (dict with texts key)
        mock_guardrail.apply_guardrail = AsyncMock(
            return_value={"texts": ["Processed text"]}
        )

        # Configure the registry to return our mock guardrail
        mock_registry.get_initialized_guardrail_callback.return_value = mock_guardrail

        # Create the request without optional parameters
        request = ApplyGuardrailRequest(
            guardrail_name="test-guardrail", text="Test text"
        )

        # Create a mock user API key
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        # Call the endpoint
        response = await apply_guardrail(
            fastapi_request=Mock(),
            request=request,
            user_api_key_dict=user_api_key_dict,
        )

        # Verify the response is of the correct type
        assert isinstance(response, ApplyGuardrailResponse)
        assert response.response_text == "Processed text"

        # Verify the guardrail was called with correct parameters
        mock_guardrail.apply_guardrail.assert_called_once_with(
            inputs={"texts": ["Test text"]}, request_data={}, input_type="request"
        )

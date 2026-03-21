"""Test health check helper functions"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.constants import LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
from litellm.litellm_core_utils.health_check_helpers import HealthCheckHelpers
from litellm.main import ahealth_check
from litellm.proxy._types import UserAPIKeyAuth


def test_update_model_params_with_health_check_tracking_information():
    """Test _update_model_params_with_health_check_tracking_information adds required tracking info."""
    initial_model_params = {
        "model": "gpt-3.5-turbo",
        "api_key": "test_key"
    }
    
    with patch(
        "litellm.proxy._types.UserAPIKeyAuth.get_litellm_internal_health_check_user_api_key_auth"
    ) as mock_get_auth:
        mock_auth = MagicMock()
        mock_get_auth.return_value = mock_auth
        
        with patch(
            "litellm.proxy.litellm_pre_call_utils.LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata"
        ) as mock_add_auth:
            mock_add_auth.return_value = {
                **initial_model_params,
                "litellm_metadata": {
                    "tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME],
                    "user_api_key_auth": mock_auth
                }
            }
            
            result = HealthCheckHelpers._update_model_params_with_health_check_tracking_information(
                initial_model_params
            )
            
            # Verify that litellm_metadata was added
            assert "litellm_metadata" in result
            assert result["litellm_metadata"]["tags"] == [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME]
            
            # Verify the auth setup was called
            mock_add_auth.assert_called_once()
            call_args = mock_add_auth.call_args
            assert call_args[1]["user_api_key_dict"] == mock_auth
            assert call_args[1]["_metadata_variable_name"] == "litellm_metadata"


def test_get_metadata_for_health_check_call():
    """Test _get_metadata_for_health_check_call returns correct metadata structure."""
    result = HealthCheckHelpers._get_metadata_for_health_check_call()
    
    expected_metadata = {
        "tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME],
    }
    
    assert result == expected_metadata
    assert isinstance(result["tags"], list)
    assert len(result["tags"]) == 1
    assert result["tags"][0] == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME


def test_get_litellm_internal_health_check_user_api_key_auth():
    """Test get_litellm_internal_health_check_user_api_key_auth returns properly configured UserAPIKeyAuth object."""
    result = UserAPIKeyAuth.get_litellm_internal_health_check_user_api_key_auth()
    
    # Verify the returned object is of correct type
    assert isinstance(result, UserAPIKeyAuth)
    
    # Verify all fields are set to the expected constant value
    assert result.api_key == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
    assert result.team_id == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
    assert result.key_alias == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
    assert result.team_alias == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME


@pytest.mark.asyncio
async def test_ahealth_check_failure_masks_raw_request_headers():
    """
    Security test: Verify that when ahealth_check() fails, the raw_request_headers
    in raw_request_typed_dict are properly masked to prevent API key leaks.
    
    This tests the fix for the security vulnerability where Authorization headers
    were being exposed in health check error responses.
    """
    # Use a model configuration that will fail (invalid endpoint)
    test_api_key = "dapi-test-key-1234567890abcdef"
    test_headers = {
        "Authorization": f"Bearer {test_api_key}",
        "Content-Type": "application/json",
    }
    
    response = await ahealth_check(
        model_params={
            "model": "databricks/dbrx-instruct",
            "api_base": "https://invalid-endpoint-that-will-fail.com/",
            "api_key": test_api_key,
            "headers": test_headers,
        },
        mode="chat",
    )
    
    # Should have error and raw_request_typed_dict
    assert "error" in response
    assert "raw_request_typed_dict" in response
    
    raw_request_dict = response["raw_request_typed_dict"]
    assert raw_request_dict is not None
    assert isinstance(raw_request_dict, dict)
    assert "raw_request_headers" in raw_request_dict
    
    headers = raw_request_dict["raw_request_headers"]
    assert headers is not None
    
    # Security check: Authorization header should be masked, not show full key
    if "Authorization" in headers:
        auth_header = headers["Authorization"]
        # Should be masked (e.g., "Be****90" or similar)
        assert auth_header != f"Bearer {test_api_key}", "Authorization header must be masked"
        assert auth_header != test_api_key, "API key must not appear in Authorization header"
        # Masked headers typically have asterisks or are truncated
        assert "*" in auth_header or len(auth_header) < len(f"Bearer {test_api_key}"), \
            f"Authorization header should be masked but got: {auth_header}"
    
    # Content-Type should remain unmasked (not sensitive)
    if "Content-Type" in headers:
        assert headers["Content-Type"] == "application/json"
    
    print(f"Masked Authorization header: {headers.get('Authorization', 'NOT FOUND')}") 
"""
Tests for model metrics validation functionality.

Tests the validate_and_normalize_model_group helper and ensures
validation doesn't break existing endpoints.
"""
import os
import sys
import pytest
from unittest.mock import Mock
from fastapi import status

# Add parent directory to path to import from local code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import from local litellm module
import litellm.proxy.proxy_server as proxy_server
from litellm.proxy.utils import ProxyException
from litellm.router import Router, Deployment

# Get the function from the module
validate_and_normalize_model_group = proxy_server.validate_and_normalize_model_group


def test_validate_and_normalize_model_group():
    """Test validate_and_normalize_model_group helper function with various inputs."""
    # Test 1: Valid model_group passes through
    mock_router = Mock(spec=Router)
    mock_router.model_names = ["gpt-4", "gpt-3.5-turbo"]
    mock_router.has_model_id.return_value = False
    
    result = validate_and_normalize_model_group(mock_router, "gpt-4")
    assert result == "gpt-4"
    
    # Test 2: Converts model_id to model_group name
    mock_router.has_model_id.return_value = True
    mock_deployment = Mock(spec=Deployment)
    mock_deployment.model_name = "gpt-4"
    mock_router.get_deployment.return_value = mock_deployment
    
    model_id = "model_name_65cc21c4-8797-45ef-b450-a5f1fca107a2"
    result = validate_and_normalize_model_group(mock_router, model_id)
    assert result == "gpt-4"
    
    # Test 3: Invalid model_group raises 404
    mock_router.has_model_id.return_value = False
    mock_router.model_names = ["gpt-4"]
    
    with pytest.raises(ProxyException) as exc_info:
        validate_and_normalize_model_group(mock_router, "invalid-model")
    assert str(exc_info.value.code) == str(status.HTTP_404_NOT_FOUND)
    
    # Test 4: Graceful fallback when router unavailable
    result = validate_and_normalize_model_group(None, "any-model")
    assert result == "any-model"
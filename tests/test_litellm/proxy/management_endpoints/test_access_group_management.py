"""
Test access group management endpoints with wildcard model support
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm import Router
from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
    validate_models_exist,
)
from litellm.router import Deployment, LiteLLM_Params
from litellm.types.router import ModelInfo


def test_validate_models_exist_with_wildcard():
    """
    Test that validate_models_exist correctly validates model names against wildcard patterns.
    
    Scenario: Router has openai/* wildcard configured, user tries to create access group
    with openai/gpt-4 which should match the wildcard pattern.
    """
    # Create a router with a wildcard model
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",  # Wildcard pattern
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "fake-key-for-test",
                },
            }
        ]
    )
    
    # Test 1: Model that matches wildcard pattern should be valid
    model_names = ["openai/gpt-4"]
    all_valid, missing = validate_models_exist(model_names, router)
    
    assert all_valid is True, f"Expected openai/gpt-4 to match openai/* pattern"
    assert len(missing) == 0, f"Expected no missing models, got: {missing}"
    
    # Test 2: Model that doesn't match pattern should be invalid
    model_names = ["anthropic/claude-3"]
    all_valid, missing = validate_models_exist(model_names, router)
    
    assert all_valid is False, "Expected anthropic/claude-3 to not match"
    assert "anthropic/claude-3" in missing, f"Expected anthropic/claude-3 in missing, got: {missing}"
    
    # Test 3: Mix of valid wildcard match and invalid model
    model_names = ["openai/gpt-4", "nonexistent/model"]
    all_valid, missing = validate_models_exist(model_names, router)
    
    assert all_valid is False, "Expected validation to fail with mixed models"
    assert "openai/gpt-4" not in missing, "openai/gpt-4 should not be in missing"
    assert "nonexistent/model" in missing, f"Expected nonexistent/model in missing, got: {missing}"


def test_validate_models_exist_with_exact_and_wildcard():
    """
    Test validation with both exact model names and wildcard patterns.
    """
    # Create a router with both exact models and wildcard pattern
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",  # Exact model
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "fake-key-for-test",
                },
            },
            {
                "model_name": "openai/*",  # Wildcard pattern
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "fake-key-for-test",
                },
            },
        ]
    )
    
    # Test: Both exact and wildcard matches should be valid
    model_names = ["gpt-4", "openai/gpt-3.5-turbo"]
    all_valid, missing = validate_models_exist(model_names, router)
    
    assert all_valid is True, "Expected both exact and wildcard matches to be valid"
    assert len(missing) == 0, f"Expected no missing models, got: {missing}"


def test_validate_models_exist_no_router():
    """
    Test that validation fails gracefully when router is None.
    """
    model_names = ["gpt-4"]
    all_valid, missing = validate_models_exist(model_names, llm_router=None)
    
    assert all_valid is False, "Expected validation to fail with no router"
    assert missing == model_names, "Expected all models to be missing when router is None"


@pytest.mark.asyncio
async def test_create_duplicate_access_group_fails():
    """
    Test that creating an access group with a name that already exists returns 409 error.
    
    Scenario: User creates "production-models" access group, then tries to create it again.
    Should fail with 409 Conflict.
    """
    from fastapi import HTTPException

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    # Mock dependencies
    mock_router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "fake-key",
                },
            }
        ]
    )
    
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
        return_value=[
            MagicMock(
                model_id="1",
                model_name="openai/gpt-4",
                model_info={"access_groups": ["production-models"]},  # Already exists
            )
        ]
    )
    
    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    
    request_data = NewModelGroupRequest(
        access_group="production-models",
        model_names=["openai/gpt-4"],
    )
    
    # Mock the imported dependencies from proxy_server (where they're actually imported from)
    with patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        
        # Should raise 409 Conflict
        with pytest.raises(HTTPException) as exc_info:
            await create_model_group(data=request_data, user_api_key_dict=mock_user)
        
        assert exc_info.value.status_code == 409
        assert "already exists" in str(exc_info.value.detail)


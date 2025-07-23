"""
Tests for the all-org-models feature in API endpoints
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request
from fastapi.testclient import TestClient

from litellm.proxy._types import (
    SpecialModelNames,
    UserAPIKeyAuth,
)


@pytest.mark.asyncio
async def test_models_endpoint_with_all_org_models():
    """Test /v1/models endpoint resolves all-org-models correctly"""
    
    # Mock dependencies
    mock_prisma = MagicMock()
    mock_org = MagicMock()
    mock_org.models = ["gpt-4", "claude-3-opus", "gemini-pro"]
    mock_prisma.db.litellm_organizationtable.find_unique = AsyncMock(return_value=mock_org)
    
    # Mock router
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus", "gemini-pro"]
    mock_router.get_model_access_groups.return_value = {}
    mock_router.get_model_group_info.return_value = []
    
    # Mock user auth with all-org-models
    mock_user_auth = UserAPIKeyAuth(
        token="test-key",
        models=[SpecialModelNames.all_org_models.value],
        org_id="test-org-123",
        team_models=[],
        user_id="test-user",
    )
    
    # Import after mocking to ensure mocks are in place
    with patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), \
         patch("litellm.proxy.proxy_server.general_settings", {"infer_model_from_keys": False}), \
         patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()), \
         patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()):
        
        from litellm.proxy.proxy_server import model_list
        
        # Call the endpoint function directly
        result = await model_list(
            user_api_key_dict=mock_user_auth,
            return_wildcard_routes=False,
            team_id=None,
            include_model_access_groups=False,
            only_model_access_groups=False,
            include_metadata=False,
            fallback_type=None,
        )
        
        # Check that the models returned match the org models
        assert "data" in result
        model_ids = [model["id"] for model in result["data"]]
        
        # Should contain the org models
        assert "gpt-4" in model_ids
        assert "claude-3-opus" in model_ids  
        assert "gemini-pro" in model_ids
        
        # Verify org was queried
        mock_prisma.db.litellm_organizationtable.find_unique.assert_called_with(
            where={"organization_id": "test-org-123"}
        )


@pytest.mark.asyncio
async def test_models_endpoint_no_org_models():
    """Test /v1/models endpoint when org has no models"""
    
    # Mock dependencies
    mock_prisma = MagicMock()
    mock_org = MagicMock()
    mock_org.models = []  # Empty models
    mock_prisma.db.litellm_organizationtable.find_unique = AsyncMock(return_value=mock_org)
    
    # Mock router
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["gpt-3.5-turbo", "gpt-4"]
    mock_router.get_model_access_groups.return_value = {}
    mock_router.get_model_group_info.return_value = []
    
    # Mock user auth with all-org-models
    mock_user_auth = UserAPIKeyAuth(
        token="test-key",
        models=[SpecialModelNames.all_org_models.value],
        org_id="test-org-123",
        team_models=[],
        user_id="test-user",
    )
    
    with patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), \
         patch("litellm.proxy.proxy_server.general_settings", {"infer_model_from_keys": False}), \
         patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()), \
         patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()):
        
        from litellm.proxy.proxy_server import model_list
        
        # Call the endpoint function directly  
        result = await model_list(
            user_api_key_dict=mock_user_auth,
            return_wildcard_routes=False,
            team_id=None,
            include_model_access_groups=False,
            only_model_access_groups=False,
            include_metadata=False,
            fallback_type=None,
        )
        
        # Check that no models are returned when org has no models
        assert "data" in result
        assert len(result["data"]) == 0


@pytest.mark.asyncio
async def test_model_info_v2_endpoint_with_all_org_models():
    """Test /v2/model/info endpoint resolves all-org-models correctly"""
    
    # Mock dependencies
    mock_prisma = MagicMock()
    mock_org = MagicMock()
    mock_org.models = ["gpt-4", "claude-3-opus"]
    mock_prisma.db.litellm_organizationtable.find_unique = AsyncMock(return_value=mock_org)
    
    # Mock router and model list
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]
    mock_router.get_model_access_groups.return_value = {}
    
    mock_model_list = [
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "gpt-4", "mode": "chat"}
        },
        {
            "model_name": "claude-3-opus", 
            "litellm_params": {"model": "claude-3-opus-20240229"},
            "model_info": {"id": "claude-3-opus", "mode": "chat"}
        }
    ]
    
    # Mock user auth with all-org-models
    mock_user_auth = UserAPIKeyAuth(
        token="test-key",
        models=[SpecialModelNames.all_org_models.value],
        org_id="test-org-123",
        team_models=[],
        user_id="test-user",
    )
    
    with patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.llm_model_list", mock_model_list), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), \
         patch("litellm.proxy.proxy_server.general_settings", {}), \
         patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()), \
         patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()):
        
        from litellm.proxy.proxy_server import model_info_v2
        
        # Call the endpoint function directly
        result = await model_info_v2(
            user_api_key_dict=mock_user_auth,
            model=None,
            user_models_only=False,
        )
        
        # Check that the models returned match the org models
        assert "data" in result
        assert len(result["data"]) == 2  # Should have 2 org models
        
        model_names = [model["model_name"] for model in result["data"]]
        assert "gpt-4" in model_names
        assert "claude-3-opus" in model_names


@pytest.mark.asyncio
async def test_chat_completion_with_all_org_models():
    """Test that chat completion validates model access for all-org-models"""
    from litellm.proxy.auth.auth_checks import can_key_call_model
    
    # Mock user auth with all-org-models
    valid_token = UserAPIKeyAuth(
        token="test-key",
        models=[SpecialModelNames.all_org_models.value],
        org_id="test-org-123",
        team_models=[],
        team_model_aliases={},
        user_id="test-user",
    )
    
    # Mock router
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["gpt-4", "claude-3-opus"]
    mock_router.get_model_access_groups.return_value = {}
    
    # Mock get_key_models_async to return org models
    with patch("litellm.proxy.auth.auth_checks.get_key_models_async") as mock_get_models:
        mock_get_models.return_value = ["gpt-4", "claude-3-opus"]
        
        # Test allowed model
        result = await can_key_call_model(
            model="gpt-4",
            llm_model_list=None,
            valid_token=valid_token,
            llm_router=mock_router,
        )
        assert result is True
        
        # Test with model not in org
        mock_get_models.return_value = ["gpt-4", "claude-3-opus"]
        
        # This should raise an exception for unauthorized model
        from litellm.proxy._types import ProxyException
        with pytest.raises(ProxyException) as exc_info:
            await can_key_call_model(
                model="gpt-3.5-turbo",  # Not in org models
                llm_model_list=None,
                valid_token=valid_token,
                llm_router=mock_router,
            )
        
        assert "Key not allowed to access model" in str(exc_info.value)
"""
Tests for the "all-org-models" feature
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import (
    LiteLLM_OrganizationTable,
    SpecialModelNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.model_checks import (
    get_key_models,
    get_key_models_async,
    get_org_models,
)


@pytest.mark.asyncio
async def test_get_org_models():
    """Test get_org_models function fetches organization models correctly"""
    # Mock organization with models
    mock_org = MagicMock()
    mock_org.models = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]
    
    # Mock prisma client
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_organizationtable.find_unique = AsyncMock(return_value=mock_org)
    
    with patch("litellm.proxy.auth.model_checks.prisma_client", mock_prisma):
        # Test successful fetch
        result = await get_org_models("test-org-id")
        assert result == ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]
        
        # Verify the query was made correctly
        mock_prisma.db.litellm_organizationtable.find_unique.assert_called_once_with(
            where={"organization_id": "test-org-id"}
        )


@pytest.mark.asyncio
async def test_get_org_models_not_found():
    """Test get_org_models returns None when organization not found"""
    # Mock prisma client returning None
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_organizationtable.find_unique = AsyncMock(return_value=None)
    
    with patch("litellm.proxy.auth.model_checks.prisma_client", mock_prisma):
        result = await get_org_models("non-existent-org")
        assert result is None


@pytest.mark.asyncio
async def test_get_org_models_no_prisma_client():
    """Test get_org_models returns None when prisma client is not available"""
    with patch("litellm.proxy.auth.model_checks.prisma_client", None):
        result = await get_org_models("test-org-id")
        assert result is None


def test_get_key_models_with_all_org_models():
    """Test get_key_models recognizes all-org-models but doesn't resolve it (sync)"""
    user_api_key_dict = UserAPIKeyAuth(
        models=[SpecialModelNames.all_org_models.value],
        org_id="test-org-id",
        team_models=[],
    )
    proxy_model_list = ["model1", "model2"]
    model_access_groups = {}
    
    result = get_key_models(
        user_api_key_dict,
        proxy_model_list,
        model_access_groups,
    )
    
    # In sync version, all-org-models should remain unresolved
    assert result == [SpecialModelNames.all_org_models.value]


@pytest.mark.asyncio
async def test_get_key_models_async_with_all_org_models():
    """Test get_key_models_async resolves all-org-models correctly"""
    user_api_key_dict = UserAPIKeyAuth(
        models=[SpecialModelNames.all_org_models.value],
        org_id="test-org-id",
        team_models=[],
    )
    proxy_model_list = ["model1", "model2"]
    model_access_groups = {}
    
    # Mock get_org_models to return organization models
    with patch("litellm.proxy.auth.model_checks.get_org_models") as mock_get_org_models:
        mock_get_org_models.return_value = ["org-model-1", "org-model-2", "org-model-3"]
        
        result = await get_key_models_async(
            user_api_key_dict,
            proxy_model_list,
            model_access_groups,
        )
        
        # Should resolve to organization models
        assert result == ["org-model-1", "org-model-2", "org-model-3"]
        mock_get_org_models.assert_called_once_with("test-org-id")


@pytest.mark.asyncio
async def test_get_key_models_async_no_org_id():
    """Test get_key_models_async when key has no org_id"""
    user_api_key_dict = UserAPIKeyAuth(
        models=[SpecialModelNames.all_org_models.value],
        org_id=None,  # No org_id
        team_models=[],
    )
    proxy_model_list = ["model1", "model2"]
    model_access_groups = {}
    
    result = await get_key_models_async(
        user_api_key_dict,
        proxy_model_list,
        model_access_groups,
    )
    
    # Should remain as all-org-models since no org_id
    assert result == [SpecialModelNames.all_org_models.value]


@pytest.mark.asyncio
async def test_get_key_models_async_with_all_team_models():
    """Test get_key_models_async correctly handles all-team-models"""
    user_api_key_dict = UserAPIKeyAuth(
        models=[SpecialModelNames.all_team_models.value],
        org_id="test-org-id",
        team_models=["team-model-1", "team-model-2"],
    )
    proxy_model_list = ["model1", "model2"]
    model_access_groups = {}
    
    result = await get_key_models_async(
        user_api_key_dict,
        proxy_model_list,
        model_access_groups,
    )
    
    # Should resolve to team models
    assert result == ["team-model-1", "team-model-2"]


@pytest.mark.asyncio
async def test_get_key_models_async_with_all_proxy_models():
    """Test get_key_models_async correctly handles all-proxy-models"""
    user_api_key_dict = UserAPIKeyAuth(
        models=[SpecialModelNames.all_proxy_models.value],
        org_id="test-org-id",
        team_models=[],
    )
    proxy_model_list = ["proxy-model-1", "proxy-model-2", "proxy-model-3"]
    model_access_groups = {}
    
    result = await get_key_models_async(
        user_api_key_dict,
        proxy_model_list,
        model_access_groups,
    )
    
    # Should resolve to proxy models
    assert result == ["proxy-model-1", "proxy-model-2", "proxy-model-3"]


@pytest.mark.asyncio
async def test_can_key_call_model_with_all_org_models():
    """Test can_key_call_model resolves all-org-models during auth check"""
    from litellm.proxy.auth.auth_checks import can_key_call_model
    
    valid_token = UserAPIKeyAuth(
        models=[SpecialModelNames.all_org_models.value],
        org_id="test-org-id",
        team_models=[],
        team_model_aliases={},
    )
    
    # Mock the router
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["gpt-3.5-turbo", "gpt-4"]
    mock_router.get_model_access_groups.return_value = {}
    
    # Mock get_org_models
    with patch("litellm.proxy.auth.auth_checks.get_key_models_async") as mock_get_key_models:
        mock_get_key_models.return_value = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]
        
        # Test that the key can call a model that's in the org
        result = await can_key_call_model(
            model="gpt-3.5-turbo",
            llm_model_list=None,
            valid_token=valid_token,
            llm_router=mock_router,
        )
        
        assert result is True
        
        # Verify async resolution was called
        mock_get_key_models.assert_called_once()
        

@pytest.mark.asyncio 
async def test_integration_all_org_models_flow():
    """Integration test for the full all-org-models flow"""
    # Setup org with models
    org_models = ["gpt-4", "claude-3-opus", "gemini-pro"]
    
    # Create API key with all-org-models
    user_api_key_dict = UserAPIKeyAuth(
        models=[SpecialModelNames.all_org_models.value],
        org_id="test-org-123",
        team_models=[],
        team_model_aliases={},
    )
    
    # Mock dependencies
    mock_prisma = MagicMock()
    mock_org = MagicMock()
    mock_org.models = org_models
    mock_prisma.db.litellm_organizationtable.find_unique = AsyncMock(return_value=mock_org)
    
    with patch("litellm.proxy.auth.model_checks.prisma_client", mock_prisma):
        # Test get_key_models_async resolves correctly
        result = await get_key_models_async(
            user_api_key_dict,
            proxy_model_list=["model1", "model2"],
            model_access_groups={},
        )
        
        assert result == org_models
        
        # Verify org was queried
        mock_prisma.db.litellm_organizationtable.find_unique.assert_called_with(
            where={"organization_id": "test-org-123"}
        )
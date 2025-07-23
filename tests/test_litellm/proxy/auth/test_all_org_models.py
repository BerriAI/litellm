"""
Simplified tests for the "all-org-models" feature focusing on core functionality
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import (
    SpecialModelNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.model_checks import (
    get_key_models,
    get_key_models_async,
)


def test_special_model_name_exists():
    """Test that all_org_models is defined in SpecialModelNames enum"""
    assert hasattr(SpecialModelNames, 'all_org_models')
    assert SpecialModelNames.all_org_models.value == "all-org-models"


def test_get_key_models_recognizes_all_org_models():
    """Test that sync get_key_models recognizes but doesn't resolve all-org-models"""
    user_api_key_dict = UserAPIKeyAuth(
        models=["all-org-models"],
        org_id="test-org",
        team_models=[],
    )
    
    result = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=["model1", "model2"],
        model_access_groups={},
    )
    
    # Should not resolve in sync version
    assert result == ["all-org-models"]


def test_get_key_models_with_regular_models():
    """Test get_key_models with regular model names"""
    user_api_key_dict = UserAPIKeyAuth(
        models=["gpt-4", "claude-3"],
        org_id="test-org",
        team_models=[],
    )
    
    result = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=["model1", "model2"],
        model_access_groups={},
    )
    
    assert result == ["gpt-4", "claude-3"]


def test_get_key_models_with_all_team_models():
    """Test get_key_models resolves all-team-models"""
    user_api_key_dict = UserAPIKeyAuth(
        models=["all-team-models"],
        org_id="test-org",
        team_models=["team-model-1", "team-model-2"],
    )
    
    result = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=["model1", "model2"],
        model_access_groups={},
    )
    
    assert result == ["team-model-1", "team-model-2"]


def test_get_key_models_with_all_proxy_models():
    """Test get_key_models resolves all-proxy-models"""
    user_api_key_dict = UserAPIKeyAuth(
        models=["all-proxy-models"],
        org_id="test-org",
        team_models=[],
    )
    
    proxy_models = ["proxy-1", "proxy-2", "proxy-3"]
    
    result = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=proxy_models,
        model_access_groups={},
    )
    
    assert result == proxy_models


@pytest.mark.asyncio
async def test_get_key_models_async_basic():
    """Test async version maintains compatibility with sync version"""
    user_api_key_dict = UserAPIKeyAuth(
        models=["gpt-4"],
        org_id=None,
        team_models=[],
    )
    
    result = await get_key_models_async(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=["model1"],
        model_access_groups={},
    )
    
    assert result == ["gpt-4"]


@pytest.mark.asyncio 
async def test_get_key_models_async_with_org_models():
    """Test get_key_models_async resolves all-org-models when org_id present"""
    user_api_key_dict = UserAPIKeyAuth(
        models=["all-org-models"],
        org_id="test-org-123",
        team_models=[],
    )
    
    # Mock get_org_models
    with patch("litellm.proxy.auth.model_checks.get_org_models") as mock_get_org:
        mock_get_org.return_value = ["org-model-1", "org-model-2"]
        
        result = await get_key_models_async(
            user_api_key_dict=user_api_key_dict,
            proxy_model_list=["proxy-1"],
            model_access_groups={},
        )
        
        assert result == ["org-model-1", "org-model-2"]
        mock_get_org.assert_called_once_with("test-org-123")


@pytest.mark.asyncio
async def test_get_key_models_async_no_org_id():
    """Test get_key_models_async doesn't resolve when no org_id"""
    user_api_key_dict = UserAPIKeyAuth(
        models=["all-org-models"],
        org_id=None,
        team_models=[],
    )
    
    result = await get_key_models_async(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=["proxy-1"],
        model_access_groups={},
    )
    
    # Should not resolve without org_id
    assert result == ["all-org-models"]


@pytest.mark.asyncio
async def test_auth_checks_integration():
    """Test that can_key_call_model works with all-org-models"""
    from litellm.proxy.auth.auth_checks import can_key_call_model
    from litellm.proxy._types import ProxyException
    
    valid_token = UserAPIKeyAuth(
        models=["all-org-models"],
        org_id="test-org",
        team_models=[],
        team_model_aliases={},
    )
    
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["gpt-4", "claude-3"]
    mock_router.get_model_access_groups.return_value = {}
    
    # Mock successful resolution
    with patch("litellm.proxy.auth.model_checks.get_key_models_async") as mock_get_models:
        mock_get_models.return_value = ["gpt-4", "claude-3"]
        
        # Should allow access to org model
        result = await can_key_call_model(
            model="gpt-4",
            llm_model_list=None,
            valid_token=valid_token,
            llm_router=mock_router,
        )
        assert result is True
        
        # Reset mock for unauthorized test
        mock_get_models.return_value = ["gpt-4"]  # Only gpt-4 in org
        
        # Should deny access to non-org model
        with pytest.raises(ProxyException):
            await can_key_call_model(
                model="claude-3",
                llm_model_list=None,
                valid_token=valid_token,
                llm_router=mock_router,
            )
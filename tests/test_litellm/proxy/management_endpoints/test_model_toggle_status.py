"""
Test model enable/disable (toggle status) functionality
"""
import json
import os
import sys
import uuid
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
from litellm.proxy._types import (
    LiteLLM_ProxyModelTable,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.model_management_endpoints import (
    toggle_model_status,
    get_db_model,
)
from litellm.types.router import Deployment, LiteLLM_Params


class MockPrismaClient:
    def __init__(self, model_exists=True, is_active=True):
        self.model_exists = model_exists
        self.is_active = is_active
        self.update_called = False
        self.db = self
        self.litellm_proxymodeltable = self

    async def find_unique(self, where):
        if self.model_exists:
            return LiteLLM_ProxyModelTable(
                model_id=where.get("model_id"),
                model_name="test-model",
                litellm_params={"model": "gpt-3.5-turbo"},
                model_info={"db_model": True},
                is_active=self.is_active,
                created_by="test_user",
                updated_by="test_user"
            )
        return None

    async def update(self, where, data):
        self.update_called = True
        self.update_data = data
        return LiteLLM_ProxyModelTable(
            model_id=where.get("model_id"),
            model_name="test-model",
            litellm_params={"model": "gpt-3.5-turbo"},
            model_info={"db_model": True},
            is_active=data.get("is_active", self.is_active),
            created_by="test_user",
            updated_by=data.get("updated_by", "test_user")
        )


class TestModelToggleStatus:
    @pytest.mark.asyncio
    async def test_toggle_model_status_enable_to_disable(self):
        """Test toggling a model from active to inactive"""
        model_id = str(uuid.uuid4())
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_admin", 
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        
        # Mock dependencies
        mock_prisma = MockPrismaClient(model_exists=True, is_active=True)
        
        with patch("litellm.proxy.management_endpoints.model_management_endpoints.prisma_client", mock_prisma):
            with patch("litellm.proxy.management_endpoints.model_management_endpoints.litellm_proxy_admin_name", "admin"):
                with patch("litellm.proxy.management_endpoints.model_management_endpoints.store_model_in_db", True):
                    with patch("litellm.proxy.management_endpoints.model_management_endpoints.premium_user", False):
                        with patch("litellm.proxy.management_endpoints.model_management_endpoints.get_db_model", mock_prisma.find_unique):
                            with patch("litellm.proxy.management_endpoints.model_management_endpoints.clear_cache", AsyncMock()):
                                with patch("litellm.proxy.management_endpoints.model_management_endpoints.get_utc_datetime", lambda: "2024-01-01T00:00:00"):
                                    with patch("litellm.proxy.management_endpoints.model_management_endpoints.ModelManagementAuthChecks.can_user_make_model_call", AsyncMock()):
                                        result = await toggle_model_status(
                                            model_id=model_id,
                                            user_api_key_dict=user_api_key_dict
                                        )
        
        assert result["is_active"] == False
        assert result["message"] == "Model disabled successfully"
        assert mock_prisma.update_called == True
        assert mock_prisma.update_data["is_active"] == False

    @pytest.mark.asyncio
    async def test_toggle_model_status_disable_to_enable(self):
        """Test toggling a model from inactive to active"""
        model_id = str(uuid.uuid4())
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_admin",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        
        # Mock dependencies - model starts as inactive
        mock_prisma = MockPrismaClient(model_exists=True, is_active=False)
        
        with patch("litellm.proxy.management_endpoints.model_management_endpoints.prisma_client", mock_prisma):
            with patch("litellm.proxy.management_endpoints.model_management_endpoints.litellm_proxy_admin_name", "admin"):
                with patch("litellm.proxy.management_endpoints.model_management_endpoints.store_model_in_db", True):
                    with patch("litellm.proxy.management_endpoints.model_management_endpoints.premium_user", False):
                        with patch("litellm.proxy.management_endpoints.model_management_endpoints.get_db_model", mock_prisma.find_unique):
                            with patch("litellm.proxy.management_endpoints.model_management_endpoints.clear_cache", AsyncMock()):
                                with patch("litellm.proxy.management_endpoints.model_management_endpoints.get_utc_datetime", lambda: "2024-01-01T00:00:00"):
                                    with patch("litellm.proxy.management_endpoints.model_management_endpoints.ModelManagementAuthChecks.can_user_make_model_call", AsyncMock()):
                                        result = await toggle_model_status(
                                            model_id=model_id,
                                            user_api_key_dict=user_api_key_dict
                                        )
        
        assert result["is_active"] == True
        assert result["message"] == "Model enabled successfully"
        assert mock_prisma.update_called == True
        assert mock_prisma.update_data["is_active"] == True

    @pytest.mark.asyncio
    async def test_toggle_model_status_model_not_found(self):
        """Test toggling status for a non-existent model"""
        model_id = str(uuid.uuid4())
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_admin",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        
        # Mock dependencies - model doesn't exist
        mock_prisma = MockPrismaClient(model_exists=False)
        
        with patch("litellm.proxy.management_endpoints.model_management_endpoints.prisma_client", mock_prisma):
            with patch("litellm.proxy.management_endpoints.model_management_endpoints.store_model_in_db", True):
                with patch("litellm.proxy.management_endpoints.model_management_endpoints.get_db_model", mock_prisma.find_unique):
                    with pytest.raises(ProxyException) as exc_info:
                        await toggle_model_status(
                            model_id=model_id,
                            user_api_key_dict=user_api_key_dict
                        )
                    
                    assert f"Model {model_id} not found" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_toggle_model_status_no_db_connection(self):
        """Test toggling status when database is not connected"""
        model_id = str(uuid.uuid4())
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_admin",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        
        with patch("litellm.proxy.management_endpoints.model_management_endpoints.prisma_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await toggle_model_status(
                    model_id=model_id,
                    user_api_key_dict=user_api_key_dict
                )
            
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_toggle_model_status_non_db_model(self):
        """Test toggling status when store_model_in_db is False"""
        model_id = str(uuid.uuid4())
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test_admin",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        
        mock_prisma = MockPrismaClient()
        
        with patch("litellm.proxy.management_endpoints.model_management_endpoints.prisma_client", mock_prisma):
            with patch("litellm.proxy.management_endpoints.model_management_endpoints.store_model_in_db", False):
                with pytest.raises(ProxyException) as exc_info:
                    await toggle_model_status(
                        model_id=model_id,
                        user_api_key_dict=user_api_key_dict
                    )
                
                assert "Model toggle only supported for DB-stored models" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_update_deployment_with_is_active(self):
        """Test that updateDeployment properly handles is_active field"""
        from litellm.types.router import updateDeployment
        
        # Test that is_active field is accepted
        update_data = updateDeployment(
            model_name="test-model",
            is_active=False
        )
        
        assert update_data.is_active == False
        
        # Test with is_active=True
        update_data = updateDeployment(
            model_name="test-model",
            is_active=True
        )
        
        assert update_data.is_active == True
        
        # Test with is_active=None (not provided)
        update_data = updateDeployment(
            model_name="test-model"
        )
        
        assert update_data.is_active is None
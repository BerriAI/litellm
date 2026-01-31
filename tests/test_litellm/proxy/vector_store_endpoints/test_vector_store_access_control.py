"""
Test vector store access control based on team membership.

Core tests:
1. Access control logic works correctly for different team scenarios
2. Delete endpoint enforces team access control
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.vector_store_endpoints.management_endpoints import (
    _check_vector_store_access,
)
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore


def test_check_vector_store_access():
    """Test core access control logic for team-based vector store access"""
    
    # Test 1: Legacy vector stores (no team_id) are accessible to all
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_legacy",
        "custom_llm_provider": "openai",
        "team_id": None,
    }
    user = UserAPIKeyAuth(team_id="team_456")
    assert _check_vector_store_access(vector_store, user) is True
    
    # Test 2: User can access their team's vector stores
    vector_store = {
        "vector_store_id": "vs_team",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(team_id="team_456")
    assert _check_vector_store_access(vector_store, user) is True
    
    # Test 3: User cannot access other teams' vector stores
    vector_store = {
        "vector_store_id": "vs_team",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(team_id="team_789")
    assert _check_vector_store_access(vector_store, user) is False


@pytest.mark.asyncio
async def test_delete_vector_store_checks_access():
    """Test that delete endpoint enforces team access control"""
    from litellm.proxy.vector_store_endpoints.management_endpoints import (
        delete_vector_store,
    )
    from litellm.types.vector_stores import VectorStoreDeleteRequest

    mock_prisma = MagicMock()
    mock_vector_store = MagicMock(
        model_dump=lambda: {
            "vector_store_id": "vs_123",
            "custom_llm_provider": "openai",
            "team_id": "team_456",
        }
    )
    mock_prisma.db.litellm_managedvectorstorestable.find_unique = AsyncMock(
        return_value=mock_vector_store
    )

    # User from different team should get 403
    user_api_key_dict = UserAPIKeyAuth(team_id="team_789")
    request = VectorStoreDeleteRequest(vector_store_id="vs_123")

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma,
    ):
        with patch("litellm.vector_store_registry", None):
            with pytest.raises(HTTPException) as exc_info:
                await delete_vector_store(
                    data=request, user_api_key_dict=user_api_key_dict
                )

            assert exc_info.value.status_code == 403
            assert "Access denied" in exc_info.value.detail

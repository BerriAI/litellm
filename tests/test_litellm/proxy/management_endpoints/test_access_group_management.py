"""
Test access group management endpoints
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm import Router


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

    # Mock dependencies - use exact model name (not wildcard)
    mock_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",  # Exact model name
                "litellm_params": {
                    "model": "gpt-4",
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
                model_name="gpt-4",
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
        model_names=["gpt-4"],
    )

    # Mock the imported dependencies from proxy_server (where they're actually imported from)
    with patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):

        # Should raise 409 Conflict
        with pytest.raises(HTTPException) as exc_info:
            await create_model_group(data=request_data, user_api_key_dict=mock_user)

        assert exc_info.value.status_code == 409
        assert "already exists" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_create_access_group_with_model_ids_tags_only_specific_deployments():
    """
    Test that using model_ids only tags the specific deployments, not all
    deployments sharing the same model_name.

    Fixes: https://github.com/BerriAI/litellm/issues/21544
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    deploy_a = MagicMock(model_id="deploy-A", model_name="gpt-4o", model_info={})

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=deploy_a)
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(
        access_group="production-models",
        model_ids=["deploy-A"],
    )

    with patch("litellm.proxy.proxy_server.llm_router", MagicMock()), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), \
         patch(
             "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
             new_callable=AsyncMock,
         ):
        response = await create_model_group(data=request_data, user_api_key_dict=mock_user)

    assert response.models_updated == 1
    assert response.model_ids == ["deploy-A"]
    mock_prisma.db.litellm_proxymodeltable.find_unique.assert_called_once_with(
        where={"model_id": "deploy-A"}
    )
    assert mock_prisma.db.litellm_proxymodeltable.update.call_count == 1
    update_call = mock_prisma.db.litellm_proxymodeltable.update.call_args
    assert update_call.kwargs["where"] == {"model_id": "deploy-A"}


@pytest.mark.asyncio
async def test_create_access_group_with_model_names_tags_all_deployments():
    """
    Test backward compat: model_names still tags ALL deployments sharing that model_name.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    deploy_a = MagicMock(model_id="deploy-A", model_name="gpt-4o", model_info={})
    deploy_b = MagicMock(model_id="deploy-B", model_name="gpt-4o", model_info={})
    deploy_c = MagicMock(model_id="deploy-C", model_name="gpt-4o", model_info={})

    mock_router = Router(
        model_list=[{"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o", "api_key": "fake-key"}}]
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
        side_effect=[[], [deploy_a, deploy_b, deploy_c]]
    )
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(access_group="production-models", model_names=["gpt-4o"])

    with patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), \
         patch(
             "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
             new_callable=AsyncMock,
         ):
        response = await create_model_group(data=request_data, user_api_key_dict=mock_user)

    assert response.models_updated == 3
    assert response.model_names == ["gpt-4o"]
    assert mock_prisma.db.litellm_proxymodeltable.update.call_count == 3


@pytest.mark.asyncio
async def test_create_access_group_model_ids_takes_priority_over_model_names():
    """
    Test that when both model_ids and model_names are provided, model_ids is used.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    deploy_a = MagicMock(model_id="deploy-A", model_name="gpt-4o", model_info={})

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=deploy_a)
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(
        access_group="production-models",
        model_names=["gpt-4o"],
        model_ids=["deploy-A"],
    )

    with patch("litellm.proxy.proxy_server.llm_router", MagicMock()), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), \
         patch(
             "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
             new_callable=AsyncMock,
         ):
        response = await create_model_group(data=request_data, user_api_key_dict=mock_user)

    assert response.models_updated == 1
    mock_prisma.db.litellm_proxymodeltable.find_unique.assert_called_once_with(
        where={"model_id": "deploy-A"}
    )


@pytest.mark.asyncio
async def test_create_access_group_requires_model_names_or_model_ids():
    """
    Test that creating an access group without model_names or model_ids fails.
    """
    from fastapi import HTTPException
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(access_group="production-models")

    with patch("litellm.proxy.proxy_server.llm_router", MagicMock()), \
         patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
        with pytest.raises(HTTPException) as exc_info:
            await create_model_group(data=request_data, user_api_key_dict=mock_user)
        assert exc_info.value.status_code == 400
        assert "model_names or model_ids" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_create_access_group_invalid_model_id_returns_400():
    """
    Test that passing a non-existent model_id returns 400 error.
    """
    from fastapi import HTTPException
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=None)

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(
        access_group="production-models",
        model_ids=["non-existent-id"],
    )

    with patch("litellm.proxy.proxy_server.llm_router", MagicMock()), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), \
         patch(
             "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
             new_callable=AsyncMock,
         ):
        with pytest.raises(HTTPException) as exc_info:
            await create_model_group(data=request_data, user_api_key_dict=mock_user)
        assert exc_info.value.status_code == 400
        assert "non-existent-id" in str(exc_info.value.detail)

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


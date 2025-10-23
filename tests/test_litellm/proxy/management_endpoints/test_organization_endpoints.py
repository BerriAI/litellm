import asyncio
import json
import os
import sys
from litellm._uuid import uuid
from typing import Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path


@pytest.mark.asyncio
async def test_organization_update_object_permissions_existing_permission(monkeypatch):
    """
    Test updating object permissions when an organization already has an existing object_permission_id.

    This test verifies that when updating vector stores for an organization that already has an
    object_permission_id, the existing LiteLLM_ObjectPermissionTable record is updated
    with the new permissions and the object_permission_id remains the same.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_OrganizationTable,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock existing organization with object_permission_id
    existing_organization_row = LiteLLM_OrganizationTable(
        organization_id="test_org_id",
        object_permission_id="existing_perm_id_123",
        organization_alias="test_org",
        budget_id="test_budget_id",
        models=["test_model_1", "test_model_2"],
        created_by="test_created_by",
        updated_by="test_updated_by",
    )

    # Mock existing object permission record
    existing_object_permission = MagicMock()
    existing_object_permission.model_dump.return_value = {
        "object_permission_id": "existing_perm_id_123",
        "vector_stores": ["old_store_1", "old_store_2"],
    }

    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=existing_object_permission
    )

    # Mock upsert operation
    updated_permission = MagicMock()
    updated_permission.object_permission_id = "existing_perm_id_123"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(
        return_value=updated_permission
    )

    # Test data with new object permission
    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(
            vector_stores=["new_store_1", "new_store_2", "new_store_3"]
        ).model_dump(exclude_unset=True, exclude_none=True),
        "organization_alias": "updated_org",
    }

    # Call the function
    result = await handle_update_object_permission(
        data_json=data_json,
        existing_organization_row=existing_organization_row,
    )

    # Verify the object_permission was removed from data_json and object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "existing_perm_id_123"

    # Verify database operations were called correctly
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": "existing_perm_id_123"}
    )
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_organization_update_object_permissions_no_existing_permission(
    monkeypatch,
):
    """
    Test creating object permissions when an organization has no existing object_permission_id.

    This test verifies that when updating object permissions for an organization that has
    object_permission_id set to None, a new entry is created in the
    LiteLLM_ObjectPermissionTable and the organization is updated with the new object_permission_id.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_OrganizationTable,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    existing_organization_row_no_perm = LiteLLM_OrganizationTable(
        organization_id="test_org_id_2",
        object_permission_id=None,
        organization_alias="test_org_2",
        budget_id="test_budget_id_2",
        models=["test_model_1", "test_model_2"],
        created_by="test_created_by_2",
        updated_by="test_updated_by_2",
    )

    # Mock find_unique to return None (no existing permission)
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=None
    )

    # Mock upsert to create new record
    new_permission = MagicMock()
    new_permission.object_permission_id = "new_perm_id_456"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(
        return_value=new_permission
    )

    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(
            vector_stores=["brand_new_store"]
        ).model_dump(exclude_unset=True, exclude_none=True),
        "organization_alias": "updated_org_2",
    }

    result = await handle_update_object_permission(
        data_json=data_json,
        existing_organization_row=existing_organization_row_no_perm,
    )

    # Verify new object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "new_perm_id_456"

    # Verify upsert was called to create new record
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_organization_update_object_permissions_missing_permission_record(
    monkeypatch,
):
    """
    Test creating object permissions when existing object_permission_id record is not found.

    This test verifies that when updating object permissions for an organization that has an
    object_permission_id but the corresponding record cannot be found in the database,
    a new entry is created in the LiteLLM_ObjectPermissionTable with the new permissions.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_OrganizationTable,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    existing_organization_row_missing_perm = LiteLLM_OrganizationTable(
        organization_id="test_org_id_3",
        object_permission_id="missing_perm_id_789",
        organization_alias="test_org_3",
        budget_id="test_budget_id_3",
        models=["test_model_1", "test_model_2"],
        created_by="test_created_by_3",
        updated_by="test_updated_by_3",
    )

    # Mock find_unique to return None (permission record not found)
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=None
    )

    # Mock upsert to create new record
    new_permission = MagicMock()
    new_permission.object_permission_id = "recreated_perm_id_789"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(
        return_value=new_permission
    )

    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(
            vector_stores=["recreated_store"]
        ).model_dump(exclude_unset=True, exclude_none=True),
        "organization_alias": "updated_org_3",
    }

    result = await handle_update_object_permission(
        data_json=data_json,
        existing_organization_row=existing_organization_row_missing_perm,
    )

    # Verify new object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "recreated_perm_id_789"

    # Verify find_unique was called with the missing permission ID
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": "missing_perm_id_789"}
    )

    # Verify upsert was called to create new record
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()

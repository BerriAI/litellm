import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)

from unittest.mock import AsyncMock, MagicMock

from litellm.proxy.management_helpers.object_permission_utils import (
    _set_object_permission,
)


@pytest.mark.asyncio
async def test_set_object_permission():
    """
    Test that _set_object_permission correctly:
    1. Creates an object permission record in the database
    2. Excludes None values from the data
    3. Excludes object_permission_id from the data sent to create
    4. Serializes mcp_tool_permissions to JSON string
    5. Returns data_json with object_permission_id set and object_permission removed
    """
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_created_permission = MagicMock()
    mock_created_permission.object_permission_id = "test_perm_id_123"
    
    mock_prisma_client.db.litellm_objectpermissiontable.create = AsyncMock(
        return_value=mock_created_permission
    )

    # Test data with object_permission
    data_json = {
        "user_id": "test_user",
        "models": ["gpt-4"],
        "object_permission": {
            "vector_stores": ["store_1", "store_2"],
            "mcp_servers": ["server_a"],
            "mcp_tool_permissions": {
                "server_a": ["tool1", "tool2"]
            },
            "object_permission_id": "should_be_excluded",
            "mcp_access_groups": None,  # This should be excluded
        }
    }

    # Call the function
    result = await _set_object_permission(
        data_json=data_json,
        prisma_client=mock_prisma_client
    )

    # Verify object_permission_id was added to result
    assert result["object_permission_id"] == "test_perm_id_123"
    
    # Verify object_permission was removed from result
    assert "object_permission" not in result
    
    # Verify create was called
    mock_prisma_client.db.litellm_objectpermissiontable.create.assert_called_once()
    
    # Verify the data passed to create excludes None values and object_permission_id
    call_args = mock_prisma_client.db.litellm_objectpermissiontable.create.call_args
    created_data = call_args.kwargs["data"]
    
    assert "object_permission_id" not in created_data
    assert "mcp_access_groups" not in created_data  # None value should be excluded
    assert created_data["vector_stores"] == ["store_1", "store_2"]
    assert created_data["mcp_servers"] == ["server_a"]
    
    # Verify mcp_tool_permissions was serialized to JSON string
    assert isinstance(created_data["mcp_tool_permissions"], str)
    mcp_tools_parsed = json.loads(created_data["mcp_tool_permissions"])
    assert mcp_tools_parsed == {"server_a": ["tool1", "tool2"]}
    
    # Verify other fields remain in result
    assert result["user_id"] == "test_user"
    assert result["models"] == ["gpt-4"]


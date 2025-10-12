import json
import os
import sys
from litellm._uuid import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_UserTable,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.management_helpers.utils import add_new_member


@pytest.mark.asyncio
async def test_add_new_member_uses_default_team_budget_id():
    """
    Test that add_new_member uses the default_team_budget_id when max_budget_in_team is None.

    This test verifies that:
    1. When max_budget_in_team is None
    2. And default_team_budget_id is provided
    3. The team membership is created with the default_team_budget_id
    """
    from litellm.proxy._types import LitellmUserRoles

    # Setup test data
    test_user_id = "test_user_123"
    test_team_id = "test_team_456"
    test_default_budget_id = "default_budget_789"
    test_admin_name = "test_admin"

    # Create a Member object with user_id
    new_member = Member(user_id=test_user_id, role="user")

    # Create UserAPIKeyAuth object
    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin_user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Mock the user table upsert operation
    mock_user_response = MagicMock()
    mock_user_response.model_dump.return_value = {
        "user_id": test_user_id,
        "user_email": None,
        "teams": [test_team_id],
        "user_role": "internal_user",
    }
    mock_prisma_client.db.litellm_usertable.upsert = AsyncMock(
        return_value=mock_user_response
    )

    # Mock the team membership creation
    mock_team_membership_response = MagicMock()
    mock_team_membership_response.model_dump.return_value = {
        "team_id": test_team_id,
        "user_id": test_user_id,
        "budget_id": test_default_budget_id,
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_teammembership.create = AsyncMock(
        return_value=mock_team_membership_response
    )

    # Call the function with max_budget_in_team=None and a default_team_budget_id
    result_user, result_team_membership = await add_new_member(
        new_member=new_member,
        max_budget_in_team=None,  # This is the key - no max budget specified
        prisma_client=mock_prisma_client,
        team_id=test_team_id,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=test_admin_name,
        default_team_budget_id=test_default_budget_id,  # This should be used
    )

    # Verify that the user was created/updated correctly
    assert result_user is not None
    assert result_user.user_id == test_user_id

    # Verify that the team membership was created correctly
    assert result_team_membership is not None
    assert result_team_membership.team_id == test_team_id
    assert result_team_membership.user_id == test_user_id
    assert result_team_membership.budget_id == test_default_budget_id

    # Verify that the prisma client methods were called correctly
    mock_prisma_client.db.litellm_usertable.upsert.assert_called_once()
    mock_prisma_client.db.litellm_teammembership.create.assert_called_once()

    # Verify that no budget table creation was called (since max_budget_in_team is None)
    assert (
        not hasattr(mock_prisma_client.db, "litellm_budgettable")
        or not mock_prisma_client.db.litellm_budgettable.create.called
    )

    # Verify the team membership was created with the correct budget_id
    team_membership_call_args = (
        mock_prisma_client.db.litellm_teammembership.create.call_args
    )
    assert team_membership_call_args is not None
    create_data = team_membership_call_args.kwargs["data"]
    assert create_data["budget_id"] == test_default_budget_id


@pytest.mark.asyncio
async def test_add_new_member_creates_new_budget_when_max_budget_provided():
    """
    Test that add_new_member creates a new budget when max_budget_in_team is provided.

    This test verifies that:
    1. When max_budget_in_team is provided
    2. A new budget is created in the litellm_budgettable
    3. The new budget_id is used for the team membership
    """
    from litellm.proxy._types import LitellmUserRoles

    # Setup test data
    test_user_id = "test_user_123"
    test_team_id = "test_team_456"
    test_max_budget = 100.0
    test_new_budget_id = "new_budget_789"
    test_admin_name = "test_admin"

    # Create a Member object with user_id
    new_member = Member(user_id=test_user_id, role="user")

    # Create UserAPIKeyAuth object
    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin_user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Mock the user table upsert operation
    mock_user_response = MagicMock()
    mock_user_response.model_dump.return_value = {
        "user_id": test_user_id,
        "user_email": None,
        "teams": [test_team_id],
        "user_role": "internal_user",
    }
    mock_prisma_client.db.litellm_usertable.upsert = AsyncMock(
        return_value=mock_user_response
    )

    # Mock the budget table creation
    mock_budget_response = MagicMock()
    mock_budget_response.budget_id = test_new_budget_id
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_budget_response
    )

    # Mock the team membership creation
    mock_team_membership_response = MagicMock()
    mock_team_membership_response.model_dump.return_value = {
        "team_id": test_team_id,
        "user_id": test_user_id,
        "budget_id": test_new_budget_id,
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_teammembership.create = AsyncMock(
        return_value=mock_team_membership_response
    )

    # Call the function with max_budget_in_team provided
    result_user, result_team_membership = await add_new_member(
        new_member=new_member,
        max_budget_in_team=test_max_budget,  # This should trigger budget creation
        prisma_client=mock_prisma_client,
        team_id=test_team_id,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=test_admin_name,
        default_team_budget_id=None,  # Should be ignored since max_budget_in_team is provided
    )

    # Verify that the budget was created
    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()
    budget_call_args = mock_prisma_client.db.litellm_budgettable.create.call_args
    budget_data = budget_call_args.kwargs["data"]
    assert budget_data["max_budget"] == test_max_budget
    assert budget_data["created_by"] == user_api_key_dict.user_id
    assert budget_data["updated_by"] == user_api_key_dict.user_id

    # Verify that the team membership was created with the new budget_id
    assert result_team_membership is not None
    assert result_team_membership.budget_id == test_new_budget_id

    # Verify the team membership was created with the correct budget_id
    team_membership_call_args = (
        mock_prisma_client.db.litellm_teammembership.create.call_args
    )
    assert team_membership_call_args is not None
    create_data = team_membership_call_args.kwargs["data"]
    assert create_data["budget_id"] == test_new_budget_id


@pytest.mark.asyncio
async def test_add_new_member_with_user_email():
    """
    Test add_new_member with user_email instead of user_id and default budget.

    This test verifies that:
    1. When new_member has user_email instead of user_id
    2. And max_budget_in_team is None
    3. The default_team_budget_id is used correctly
    """
    from litellm.proxy._types import LitellmUserRoles

    # Setup test data
    test_user_email = "test@example.com"
    test_team_id = "test_team_456"
    test_default_budget_id = "default_budget_789"
    test_admin_name = "test_admin"

    # Create a Member object with user_email
    new_member = Member(user_email=test_user_email, role="user")

    # Create UserAPIKeyAuth object
    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin_user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Mock get_data to return empty list (no existing user)
    mock_prisma_client.get_data = AsyncMock(return_value=[])

    # Mock insert_data for new user creation
    mock_user_response = MagicMock()
    mock_user_response.model_dump.return_value = {
        "user_id": "generated_user_id",
        "user_email": test_user_email,
        "teams": [test_team_id],
        "user_role": "internal_user",
    }
    mock_prisma_client.insert_data = AsyncMock(return_value=mock_user_response)

    # Mock the team membership creation
    mock_team_membership_response = MagicMock()
    mock_team_membership_response.model_dump.return_value = {
        "team_id": test_team_id,
        "user_id": "generated_user_id",
        "budget_id": test_default_budget_id,
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_teammembership.create = AsyncMock(
        return_value=mock_team_membership_response
    )

    # Call the function
    result_user, result_team_membership = await add_new_member(
        new_member=new_member,
        max_budget_in_team=None,
        prisma_client=mock_prisma_client,
        team_id=test_team_id,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=test_admin_name,
        default_team_budget_id=test_default_budget_id,
    )

    # Verify that the user was created correctly
    assert result_user is not None
    assert result_user.user_email == test_user_email

    # Verify that the team membership was created with the default budget_id
    assert result_team_membership is not None
    assert result_team_membership.budget_id == test_default_budget_id

    # Verify that get_data was called to check for existing user
    mock_prisma_client.get_data.assert_called_once_with(
        key_val={"user_email": test_user_email},
        table_name="user",
        query_type="find_all",
    )

    # Verify that insert_data was called to create new user
    mock_prisma_client.insert_data.assert_called_once()
    insert_call_args = mock_prisma_client.insert_data.call_args
    insert_data = insert_call_args.kwargs["data"]
    assert insert_data["user_email"] == test_user_email
    assert insert_data["teams"] == [test_team_id]


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_with_object_permission_id():
    """
    Test that attach_object_permission_to_dict correctly attaches object_permission
    when object_permission_id is present and found in database.
    """
    from litellm.proxy.management_helpers.object_permission_utils import attach_object_permission_to_dict

    # Setup test data
    test_object_permission_id = "test_perm_123"
    test_data_dict = {
        "user_id": "test_user_456",
        "object_permission_id": test_object_permission_id,
        "other_field": "other_value"
    }
    
    expected_object_permission = {
        "object_permission_id": test_object_permission_id,
        "vector_stores": ["store1", "store2"],
        "assistants": ["assistant1"],
        "models": ["gpt-4", "claude-3"]
    }

    # Mock the prisma client
    mock_prisma_client = AsyncMock()
    
    # Mock the object permission response
    mock_object_permission = MagicMock()
    mock_object_permission.model_dump.return_value = expected_object_permission
    
    # Mock the database query
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=mock_object_permission
    )

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict,
        prisma_client=mock_prisma_client
    )

    # Verify the result
    assert result is not None
    assert result["user_id"] == "test_user_456"
    assert result["object_permission_id"] == test_object_permission_id
    assert result["other_field"] == "other_value"
    assert result["object_permission"] == expected_object_permission

    # Verify the database query was called correctly
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": test_object_permission_id}
    )


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_without_object_permission_id():
    """
    Test that attach_object_permission_to_dict returns the original dict unchanged
    when object_permission_id is not present.
    """
    from litellm.proxy.management_helpers.object_permission_utils import attach_object_permission_to_dict

    # Setup test data without object_permission_id
    test_data_dict = {
        "user_id": "test_user_456",
        "other_field": "other_value"
    }

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict,
        prisma_client=mock_prisma_client
    )

    # Verify the result is unchanged
    assert result == test_data_dict
    assert "object_permission" not in result

    # Verify no database query was made
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_object_permission_not_found():
    """
    Test that attach_object_permission_to_dict returns the original dict unchanged
    when object_permission_id is present but not found in database.
    """
    from litellm.proxy.management_helpers.object_permission_utils import attach_object_permission_to_dict

    # Setup test data
    test_object_permission_id = "test_perm_123"
    test_data_dict = {
        "user_id": "test_user_456",
        "object_permission_id": test_object_permission_id,
        "other_field": "other_value"
    }

    # Mock the prisma client
    mock_prisma_client = AsyncMock()
    
    # Mock the database query to return None (not found)
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=None
    )

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict,
        prisma_client=mock_prisma_client
    )

    # Verify the result is unchanged
    assert result == test_data_dict
    assert "object_permission" not in result

    # Verify the database query was called
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": test_object_permission_id}
    )


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_with_dict_method():
    """
    Test that attach_object_permission_to_dict handles object permissions that use .dict() method
    instead of .model_dump() method.
    """
    from litellm.proxy.management_helpers.object_permission_utils import attach_object_permission_to_dict

    # Setup test data
    test_object_permission_id = "test_perm_123"
    test_data_dict = {
        "user_id": "test_user_456",
        "object_permission_id": test_object_permission_id,
        "other_field": "other_value"
    }
    
    expected_object_permission = {
        "object_permission_id": test_object_permission_id,
        "vector_stores": ["store1"],
        "assistants": []
    }

    # Mock the prisma client
    mock_prisma_client = AsyncMock()
    
    # Mock the object permission response that uses .dict() method
    mock_object_permission = MagicMock()
    mock_object_permission.model_dump.side_effect = AttributeError("No model_dump method")
    mock_object_permission.dict.return_value = expected_object_permission
    
    # Mock the database query
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=mock_object_permission
    )

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict,
        prisma_client=mock_prisma_client
    )

    # Verify the result
    assert result is not None
    assert result["object_permission"] == expected_object_permission

    # Verify both methods were attempted
    mock_object_permission.model_dump.assert_called_once()
    mock_object_permission.dict.assert_called_once()


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_with_none_prisma_client():
    """
    Test that attach_object_permission_to_dict raises ValueError when prisma_client is None.
    """
    from litellm.proxy.management_helpers.object_permission_utils import attach_object_permission_to_dict

    # Setup test data
    test_data_dict = {
        "user_id": "test_user_456",
        "object_permission_id": "test_perm_123"
    }

    # Call the function with None prisma_client
    with pytest.raises(ValueError, match="Prisma client not found"):
        await attach_object_permission_to_dict(
            data_dict=test_data_dict,
            prisma_client=None  # type: ignore
        )


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_with_empty_dict():
    """
    Test that attach_object_permission_to_dict handles empty dictionaries correctly.
    """
    from litellm.proxy.management_helpers.object_permission_utils import attach_object_permission_to_dict

    # Setup empty test data
    test_data_dict = {}

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict,
        prisma_client=mock_prisma_client
    )

    # Verify the result is unchanged
    assert result == {}
    assert "object_permission" not in result

    # Verify no database query was made
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_with_none_object_permission_id():
    """
    Test that attach_object_permission_to_dict handles None object_permission_id correctly.
    """
    from litellm.proxy.management_helpers.object_permission_utils import attach_object_permission_to_dict

    # Setup test data with None object_permission_id
    test_data_dict = {
        "user_id": "test_user_456",
        "object_permission_id": None,
        "other_field": "other_value"
    }

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict,
        prisma_client=mock_prisma_client
    )

    # Verify the result is unchanged
    assert result == test_data_dict
    assert "object_permission" not in result

    # Verify no database query was made
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_not_called()

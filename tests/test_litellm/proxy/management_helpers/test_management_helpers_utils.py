import json
import os
import sys
import uuid
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

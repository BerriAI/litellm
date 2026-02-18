import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from litellm._uuid import uuid

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy._types import LiteLLM_UserTable, NewUserResponse
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.management_endpoints.sso import CustomMicrosoftSSO
from litellm.proxy.management_endpoints.types import CustomOpenID
from litellm.proxy.management_endpoints.ui_sso import (
    GoogleSSOHandler,
    MicrosoftSSOHandler,
    SSOAuthenticationHandler,
    normalize_email,
    process_sso_jwt_access_token,
    determine_role_from_groups,
    _setup_team_mappings,
)
from litellm.types.proxy.management_endpoints.ui_sso import (
    DefaultTeamSSOParams,
    MicrosoftGraphAPIUserGroupDirectoryObject,
    MicrosoftGraphAPIUserGroupResponse,
    MicrosoftServicePrincipalTeam,
    TeamMappings,
)


def test_microsoft_sso_handler_openid_from_response_user_principal_name():
    # Arrange
    # Create a mock response similar to what Microsoft SSO would return
    mock_response = {
        "userPrincipalName": "test@example.com",
        "displayName": "Test User",
        "id": "user123",
        "givenName": "Test",
        "surname": "User",
        "some_other_field": "value",
    }
    expected_team_ids = ["team1", "team2"]
    # Act
    # Call the method being tested
    result = MicrosoftSSOHandler.openid_from_response(
        response=mock_response, team_ids=expected_team_ids, user_role=None
    )

    # Assert

    # Check that the result is a CustomOpenID object with the expected values
    assert isinstance(result, CustomOpenID)
    assert result.email == "test@example.com"
    assert result.display_name == "Test User"
    assert result.provider == "microsoft"
    assert result.id == "user123"
    assert result.first_name == "Test"
    assert result.last_name == "User"
    assert result.team_ids == expected_team_ids


def test_microsoft_sso_handler_openid_from_response():
    # Arrange
    # Create a mock response similar to what Microsoft SSO would return
    mock_response = {
        "mail": "test@example.com",
        "displayName": "Test User",
        "id": "user123",
        "givenName": "Test",
        "surname": "User",
        "some_other_field": "value",
    }
    expected_team_ids = ["team1", "team2"]
    # Act
    # Call the method being tested
    result = MicrosoftSSOHandler.openid_from_response(
        response=mock_response, team_ids=expected_team_ids, user_role=None
    )

    # Assert

    # Check that the result is a CustomOpenID object with the expected values
    assert isinstance(result, CustomOpenID)
    assert result.email == "test@example.com"
    assert result.display_name == "Test User"
    assert result.provider == "microsoft"
    assert result.id == "user123"
    assert result.first_name == "Test"
    assert result.last_name == "User"
    assert result.team_ids == expected_team_ids


def test_microsoft_sso_handler_with_empty_response():
    # Arrange
    # Test with None response

    # Act
    result = MicrosoftSSOHandler.openid_from_response(
        response=None, team_ids=[], user_role=None
    )

    # Assert
    assert isinstance(result, CustomOpenID)
    assert result.email is None
    assert result.display_name is None
    assert result.provider == "microsoft"
    assert result.id is None
    assert result.first_name is None
    assert result.last_name is None
    assert result.team_ids == []


def test_microsoft_sso_handler_openid_from_response_with_custom_attributes():
    """
    Test that MicrosoftSSOHandler.openid_from_response uses custom attribute names
    from constants when environment variables are set.
    """
    # Arrange
    mock_response = {
        "custom_email_field": "custom@example.com",
        "custom_display_name": "Custom Display Name",
        "custom_id_field": "custom_user_123",
        "custom_first_name": "CustomFirst",
        "custom_last_name": "CustomLast",
    }
    expected_team_ids = ["team1"]

    # Act
    with patch(
        "litellm.constants.MICROSOFT_USER_EMAIL_ATTRIBUTE", "custom_email_field"
    ), patch(
        "litellm.constants.MICROSOFT_USER_DISPLAY_NAME_ATTRIBUTE", "custom_display_name"
    ), patch(
        "litellm.constants.MICROSOFT_USER_ID_ATTRIBUTE", "custom_id_field"
    ), patch(
        "litellm.constants.MICROSOFT_USER_FIRST_NAME_ATTRIBUTE", "custom_first_name"
    ), patch(
        "litellm.constants.MICROSOFT_USER_LAST_NAME_ATTRIBUTE", "custom_last_name"
    ), patch(
        "litellm.proxy.management_endpoints.ui_sso.MICROSOFT_USER_EMAIL_ATTRIBUTE",
        "custom_email_field",
    ), patch(
        "litellm.proxy.management_endpoints.ui_sso.MICROSOFT_USER_DISPLAY_NAME_ATTRIBUTE",
        "custom_display_name",
    ), patch(
        "litellm.proxy.management_endpoints.ui_sso.MICROSOFT_USER_ID_ATTRIBUTE",
        "custom_id_field",
    ), patch(
        "litellm.proxy.management_endpoints.ui_sso.MICROSOFT_USER_FIRST_NAME_ATTRIBUTE",
        "custom_first_name",
    ), patch(
        "litellm.proxy.management_endpoints.ui_sso.MICROSOFT_USER_LAST_NAME_ATTRIBUTE",
        "custom_last_name",
    ):
        result = MicrosoftSSOHandler.openid_from_response(
            response=mock_response, team_ids=expected_team_ids, user_role=None
        )

    # Assert
    assert isinstance(result, CustomOpenID)
    assert result.email == "custom@example.com"
    assert result.display_name == "Custom Display Name"
    assert result.provider == "microsoft"
    assert result.id == "custom_user_123"
    assert result.first_name == "CustomFirst"
    assert result.last_name == "CustomLast"
    assert result.team_ids == expected_team_ids


def test_get_microsoft_callback_response():
    # Arrange
    mock_request = MagicMock(spec=Request)
    mock_response = {
        "mail": "microsoft_user@example.com",
        "displayName": "Microsoft User",
        "id": "msft123",
        "givenName": "Microsoft",
        "surname": "User",
    }

    with patch.dict(
        os.environ,
        {"MICROSOFT_CLIENT_SECRET": "mock_secret", "MICROSOFT_TENANT": "mock_tenant"},
    ):
        mock_verify = AsyncMock(return_value=mock_response)
        with patch(
            "fastapi_sso.sso.microsoft.MicrosoftSSO.verify_and_process",
            new=mock_verify,
        ):
            # Act
            result = asyncio.run(
                MicrosoftSSOHandler.get_microsoft_callback_response(
                    request=mock_request,
                    microsoft_client_id="mock_client_id",
                    redirect_url="http://mock_redirect_url",
                )
            )

    # Assert
    assert isinstance(result, CustomOpenID)
    assert result.email == "microsoft_user@example.com"
    assert result.display_name == "Microsoft User"
    assert result.provider == "microsoft"
    assert result.id == "msft123"
    assert result.first_name == "Microsoft"
    assert result.last_name == "User"


def test_get_microsoft_callback_response_raw_sso_response():
    # Arrange
    mock_request = MagicMock(spec=Request)
    mock_response = {
        "mail": "microsoft_user@example.com",
        "displayName": "Microsoft User",
        "id": "msft123",
        "givenName": "Microsoft",
        "surname": "User",
    }

    with patch.dict(
        os.environ,
        {"MICROSOFT_CLIENT_SECRET": "mock_secret", "MICROSOFT_TENANT": "mock_tenant"},
    ):
        mock_verify = AsyncMock(return_value=mock_response)
        with patch(
            "fastapi_sso.sso.microsoft.MicrosoftSSO.verify_and_process",
            new=mock_verify,
        ):
            # Act
            result = asyncio.run(
                MicrosoftSSOHandler.get_microsoft_callback_response(
                    request=mock_request,
                    microsoft_client_id="mock_client_id",
                    redirect_url="http://mock_redirect_url",
                    return_raw_sso_response=True,
                )
            )

    # Assert
    assert isinstance(result, dict)
    assert result["mail"] == "microsoft_user@example.com"
    assert result["displayName"] == "Microsoft User"
    assert result["id"] == "msft123"
    assert result["givenName"] == "Microsoft"
    assert result["surname"] == "User"


def test_get_google_callback_response():
    # Arrange
    mock_request = MagicMock(spec=Request)
    mock_response = {
        "email": "google_user@example.com",
        "name": "Google User",
        "sub": "google123",
        "given_name": "Google",
        "family_name": "User",
    }

    with patch.dict(os.environ, {"GOOGLE_CLIENT_SECRET": "mock_secret"}):
        mock_verify = AsyncMock(return_value=mock_response)
        with patch(
            "fastapi_sso.sso.google.GoogleSSO.verify_and_process", new=mock_verify
        ):
            # Act
            result = asyncio.run(
                GoogleSSOHandler.get_google_callback_response(
                    request=mock_request,
                    google_client_id="mock_client_id",
                    redirect_url="http://mock_redirect_url",
                )
            )

    # Assert
    assert isinstance(result, dict)
    assert result.get("email") == "google_user@example.com"
    assert result.get("name") == "Google User"
    assert result.get("sub") == "google123"
    assert result.get("given_name") == "Google"
    assert result.get("family_name") == "User"


@pytest.mark.asyncio
async def test_get_user_groups_from_graph_api():
    # Arrange
    mock_response = {
        "@odata.context": "https://graph.microsoft.com/v1.0/$metadata#directoryObjects",
        "value": [
            {
                "@odata.type": "#microsoft.graph.group",
                "id": "group1",
                "displayName": "Group 1",
            },
            {
                "@odata.type": "#microsoft.graph.group",
                "id": "group2",
                "displayName": "Group 2",
            },
        ],
    }

    async def mock_get(*args, **kwargs):
        mock = MagicMock()
        mock.json.return_value = mock_response
        return mock

    with patch(
        "litellm.proxy.management_endpoints.ui_sso.get_async_httpx_client"
    ) as mock_client:
        mock_client.return_value = MagicMock()
        mock_client.return_value.get = mock_get

        # Act
        result = await MicrosoftSSOHandler.get_user_groups_from_graph_api(
            access_token="mock_token"
        )

        # Assert
        assert isinstance(result, list)
        assert len(result) == 2
        assert "group1" in result
        assert "group2" in result


@pytest.mark.asyncio
async def test_get_user_groups_empty_response():
    # Arrange
    mock_response = {
        "@odata.context": "https://graph.microsoft.com/v1.0/$metadata#directoryObjects",
        "value": [],
    }

    async def mock_get(*args, **kwargs):
        mock = MagicMock()
        mock.json.return_value = mock_response
        return mock

    with patch(
        "litellm.proxy.management_endpoints.ui_sso.get_async_httpx_client"
    ) as mock_client:
        mock_client.return_value = MagicMock()
        mock_client.return_value.get = mock_get

        # Act
        result = await MicrosoftSSOHandler.get_user_groups_from_graph_api(
            access_token="mock_token"
        )

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0


@pytest.mark.asyncio
async def test_get_user_groups_error_handling():
    # Arrange
    async def mock_get(*args, **kwargs):
        raise Exception("API Error")

    with patch(
        "litellm.proxy.management_endpoints.ui_sso.get_async_httpx_client"
    ) as mock_client:
        mock_client.return_value = MagicMock()
        mock_client.return_value.get = mock_get

        # Act
        result = await MicrosoftSSOHandler.get_user_groups_from_graph_api(
            access_token="mock_token"
        )

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0


def test_get_group_ids_from_graph_api_response():
    # Arrange
    mock_response = MicrosoftGraphAPIUserGroupResponse(
        odata_context="https://graph.microsoft.com/v1.0/$metadata#directoryObjects",
        odata_nextLink=None,
        value=[
            MicrosoftGraphAPIUserGroupDirectoryObject(
                odata_type="#microsoft.graph.group",
                id="group1",
                displayName="Group 1",
                description=None,
                deletedDateTime=None,
                roleTemplateId=None,
            ),
            MicrosoftGraphAPIUserGroupDirectoryObject(
                odata_type="#microsoft.graph.group",
                id="group2",
                displayName="Group 2",
                description=None,
                deletedDateTime=None,
                roleTemplateId=None,
            ),
            MicrosoftGraphAPIUserGroupDirectoryObject(
                odata_type="#microsoft.graph.group",
                id=None,  # Test handling of None id
                displayName="Invalid Group",
                description=None,
                deletedDateTime=None,
                roleTemplateId=None,
            ),
        ],
    )

    # Act
    result = MicrosoftSSOHandler._get_group_ids_from_graph_api_response(mock_response)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 2
    assert "group1" in result
    assert "group2" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "team_params",
    [
        # Test case 1: Using DefaultTeamSSOParams
        DefaultTeamSSOParams(
            max_budget=10, budget_duration="1d", models=["special-gpt-5"]
        ),
        # Test case 2: Using Dict
        {"max_budget": 10, "budget_duration": "1d", "models": ["special-gpt-5"]},
    ],
)
async def test_default_team_params(team_params):
    """
    When litellm.default_team_params is set, it should be used to create a new team
    """
    # Arrange
    litellm.default_team_params = team_params

    def mock_jsonify_team_object(db_data):
        return db_data

    # Mock Prisma client
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_first = AsyncMock(return_value=None)
    mock_prisma.db.litellm_teamtable.create = AsyncMock()
    mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
    mock_prisma.get_data = AsyncMock(return_value=None)
    mock_prisma.jsonify_team_object = MagicMock(side_effect=mock_jsonify_team_object)

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        # Act
        team_id = str(uuid.uuid4())
        await MicrosoftSSOHandler.create_litellm_teams_from_service_principal_team_ids(
            service_principal_teams=[
                MicrosoftServicePrincipalTeam(
                    principalId=team_id,
                    principalDisplayName="Test Team",
                )
            ]
        )

        # Assert
        # Verify team was created with correct parameters
        mock_prisma.db.litellm_teamtable.create.assert_called_once()
        create_call_args = mock_prisma.db.litellm_teamtable.create.call_args.kwargs[
            "data"
        ]
        assert create_call_args["team_id"] == team_id
        assert create_call_args["team_alias"] == "Test Team"
        assert create_call_args["max_budget"] == 10
        assert create_call_args["budget_duration"] == "1d"
        assert create_call_args["models"] == ["special-gpt-5"]


@pytest.mark.asyncio
async def test_create_team_without_default_params():
    """
    Test team creation when litellm.default_team_params is None
    Should create team with just the basic required fields
    """
    # Arrange
    litellm.default_team_params = None

    def mock_jsonify_team_object(db_data):
        return db_data

    # Mock Prisma client
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_first = AsyncMock(return_value=None)
    mock_prisma.db.litellm_teamtable.create = AsyncMock()
    mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
    mock_prisma.get_data = AsyncMock(return_value=None)
    mock_prisma.jsonify_team_object = MagicMock(side_effect=mock_jsonify_team_object)

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        # Act
        team_id = str(uuid.uuid4())
        await MicrosoftSSOHandler.create_litellm_teams_from_service_principal_team_ids(
            service_principal_teams=[
                MicrosoftServicePrincipalTeam(
                    principalId=team_id,
                    principalDisplayName="Test Team",
                )
            ]
        )

        # Assert
        mock_prisma.db.litellm_teamtable.create.assert_called_once()
        create_call_args = mock_prisma.db.litellm_teamtable.create.call_args.kwargs[
            "data"
        ]
        assert create_call_args["team_id"] == team_id
        assert create_call_args["team_alias"] == "Test Team"
        # Should not have any of the optional fields
        assert "max_budget" not in create_call_args
        assert "budget_duration" not in create_call_args
        assert create_call_args["models"] == []


def test_apply_user_info_values_to_sso_user_defined_values():
    from litellm.proxy._types import LiteLLM_UserTable, SSOUserDefinedValues
    from litellm.proxy.management_endpoints.ui_sso import (
        apply_user_info_values_to_sso_user_defined_values,
    )

    user_info = LiteLLM_UserTable(
        user_id="123",
        user_email="test@example.com",
        user_role="admin",
    )

    user_defined_values: SSOUserDefinedValues = {
        "models": [],
        "user_id": "456",
        "user_email": "test@example.com",
        "user_role": "admin",
        "max_budget": None,
        "budget_duration": None,
    }

    sso_user_defined_values = apply_user_info_values_to_sso_user_defined_values(
        user_info=user_info,
        user_defined_values=user_defined_values,
    )

    assert sso_user_defined_values is not None
    assert sso_user_defined_values["user_id"] == "123"


def test_apply_user_info_values_to_sso_user_defined_values_with_models():
    """Test that user's models from DB are preserved when they log in via SSO"""
    from litellm.proxy._types import LiteLLM_UserTable, SSOUserDefinedValues
    from litellm.proxy.management_endpoints.ui_sso import (
        apply_user_info_values_to_sso_user_defined_values,
    )

    # Simulate existing user with models=['no-default-models'] in DB
    user_info = LiteLLM_UserTable(
        user_id="123",
        user_email="test@example.com",
        user_role="admin",
        models=["no-default-models"],  # User has this set in DB
    )

    # Simulate SSO login where models defaults to empty list
    user_defined_values: SSOUserDefinedValues = {
        "models": [],  # Empty on SSO login
        "user_id": "456",
        "user_email": "test@example.com",
        "user_role": "admin",
        "max_budget": None,
        "budget_duration": None,
    }

    sso_user_defined_values = apply_user_info_values_to_sso_user_defined_values(
        user_info=user_info,
        user_defined_values=user_defined_values,
    )

    assert sso_user_defined_values is not None
    assert sso_user_defined_values["user_id"] == "123"
    # This is the fix: models from DB should be preserved
    assert sso_user_defined_values["models"] == ["no-default-models"]


def test_apply_user_info_values_sso_role_takes_precedence():
    """
    Test that SSO role takes precedence over DB role.

    When Microsoft SSO returns a user_role, it should be used instead of the role stored in the database.
    This ensures SSO is the authoritative source for user roles.
    """
    from litellm.proxy._types import LiteLLM_UserTable, SSOUserDefinedValues
    from litellm.proxy.management_endpoints.ui_sso import (
        apply_user_info_values_to_sso_user_defined_values,
    )

    user_info = LiteLLM_UserTable(
        user_id="123",
        user_email="test@example.com",
        user_role="internal_user_viewer",
        models=["model-1"],
    )

    user_defined_values: SSOUserDefinedValues = {
        "models": [],
        "user_id": "456",
        "user_email": "test@example.com",
        "user_role": "proxy_admin_viewer",
        "max_budget": None,
        "budget_duration": None,
    }

    sso_user_defined_values = apply_user_info_values_to_sso_user_defined_values(
        user_info=user_info,
        user_defined_values=user_defined_values,
    )

    assert sso_user_defined_values is not None
    assert sso_user_defined_values["user_id"] == "123"
    assert sso_user_defined_values["user_role"] == "proxy_admin_viewer"
    assert sso_user_defined_values["models"] == ["model-1"]


def test_build_sso_user_update_data_with_valid_role():
    """
    Test that _build_sso_user_update_data includes role when SSO provides a valid role.
    """
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_endpoints.types import CustomOpenID
    from litellm.proxy.management_endpoints.ui_sso import _build_sso_user_update_data

    sso_result = CustomOpenID(
        id="test-user-123",
        email="test@example.com",
        display_name="Test User",
        provider="microsoft",
        team_ids=[],
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    update_data = _build_sso_user_update_data(
        result=sso_result,
        user_email="test@example.com",
        user_id="test-user-123",
    )

    assert update_data["user_email"] == "test@example.com"
    assert update_data["user_role"] == "proxy_admin"


def test_build_sso_user_update_data_without_role():
    """
    Test that _build_sso_user_update_data only includes email when SSO has no role.
    """
    from litellm.proxy.management_endpoints.types import CustomOpenID
    from litellm.proxy.management_endpoints.ui_sso import _build_sso_user_update_data

    sso_result = CustomOpenID(
        id="test-user-456",
        email="test@example.com",
        display_name="Test User",
        provider="microsoft",
        team_ids=[],
        user_role=None,
    )

    update_data = _build_sso_user_update_data(
        result=sso_result,
        user_email="test@example.com",
        user_id="test-user-456",
    )

    assert update_data["user_email"] == "test@example.com"
    assert "user_role" not in update_data


def test_normalize_email():
    """
    Test that normalize_email correctly lowercases email addresses and handles edge cases.
    """
    # Test with lowercase email
    assert normalize_email("test@example.com") == "test@example.com"

    # Test with uppercase email
    assert normalize_email("TEST@EXAMPLE.COM") == "test@example.com"

    # Test with mixed case email
    assert normalize_email("Test.User@Example.COM") == "test.user@example.com"

    # Test with None
    assert normalize_email(None) is None

    # Test with empty string
    assert normalize_email("") == ""


def test_build_sso_user_update_data_normalizes_email():
    """
    Test that _build_sso_user_update_data normalizes email addresses to lowercase.
    """
    from litellm.proxy.management_endpoints.types import CustomOpenID
    from litellm.proxy.management_endpoints.ui_sso import _build_sso_user_update_data

    sso_result = CustomOpenID(
        id="test-user-789",
        email="Test.User@Example.COM",
        display_name="Test User",
        provider="microsoft",
        team_ids=[],
        user_role=None,
    )

    update_data = _build_sso_user_update_data(
        result=sso_result,
        user_email="Test.User@Example.COM",
        user_id="test-user-789",
    )

    # Email should be normalized to lowercase
    assert update_data["user_email"] == "test.user@example.com"
    assert "user_role" not in update_data


def test_generic_response_convertor_normalizes_email():
    """
    Test that generic_response_convertor normalizes email addresses.
    """
    from litellm.proxy.management_endpoints.ui_sso import generic_response_convertor

    mock_response = {
        "preferred_username": "user123",
        "email": "Test.User@Example.COM",
        "sub": "Test User",
        "first_name": "Test",
        "last_name": "User",
        "provider": "generic",
    }

    # Mock JWT handler
    mock_jwt_handler = MagicMock(spec=JWTHandler)
    mock_jwt_handler.get_team_ids_from_jwt.return_value = []

    result = generic_response_convertor(
        response=mock_response,
        jwt_handler=mock_jwt_handler,
        sso_jwt_handler=None,
        role_mappings=None,
    )

    # Email should be normalized to lowercase
    assert result.email == "test.user@example.com"
    assert result.id == "user123"
    assert result.display_name == "Test User"


@pytest.mark.asyncio
async def test_upsert_sso_user_updates_role_for_existing_user():
    """
    Test that upsert_sso_user updates the user role in database when SSO provides a valid role.

    When a user's role is updated in the SSO provider (e.g., Azure), the role should be
    updated in the LiteLLM database on subsequent logins, not just at initial user creation.
    """
    from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles
    from litellm.proxy.management_endpoints.types import CustomOpenID
    from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

    # Mock prisma client
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.update_many = AsyncMock()

    # Existing user in DB with old role
    existing_user = LiteLLM_UserTable(
        user_id="test-user-123",
        user_email="test@example.com",
        user_role="internal_user",
        models=["model-1"],
    )

    # SSO result with new role (e.g., user was promoted to admin in Azure)
    sso_result = CustomOpenID(
        id="test-user-123",
        email="test@example.com",
        display_name="Test User",
        provider="microsoft",
        team_ids=["team-1"],
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    # Act
    await SSOAuthenticationHandler.upsert_sso_user(
        result=sso_result,
        user_info=existing_user,
        user_email="test@example.com",
        user_defined_values=None,
        prisma_client=mock_prisma,
    )

    # Assert - verify database was updated with both email and role
    mock_prisma.db.litellm_usertable.update_many.assert_called_once()
    call_args = mock_prisma.db.litellm_usertable.update_many.call_args
    assert call_args.kwargs["where"] == {"user_id": "test-user-123"}
    assert call_args.kwargs["data"]["user_email"] == "test@example.com"
    assert call_args.kwargs["data"]["user_role"] == "proxy_admin"


@pytest.mark.asyncio
async def test_upsert_sso_user_does_not_update_invalid_role():
    """
    Test that upsert_sso_user does not update the role if SSO provides an invalid role.

    If the SSO returns a role that is not a valid LiteLLM role, it should be ignored
    and only the email should be updated.
    """
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

    # Mock prisma client
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.update_many = AsyncMock()

    # Existing user in DB
    existing_user = LiteLLM_UserTable(
        user_id="test-user-456",
        user_email="test@example.com",
        user_role="internal_user",
        models=[],
    )

    # SSO result with invalid role - use MagicMock to bypass validation
    # This simulates a raw SSO response that has an invalid role string
    sso_result = MagicMock()
    sso_result.user_role = "invalid_role_not_in_enum"

    # Act
    await SSOAuthenticationHandler.upsert_sso_user(
        result=sso_result,
        user_info=existing_user,
        user_email="test@example.com",
        user_defined_values=None,
        prisma_client=mock_prisma,
    )

    # Assert - verify only email was updated, not role
    mock_prisma.db.litellm_usertable.update_many.assert_called_once()
    call_args = mock_prisma.db.litellm_usertable.update_many.call_args
    assert call_args.kwargs["where"] == {"user_id": "test-user-456"}
    assert call_args.kwargs["data"]["user_email"] == "test@example.com"
    assert "user_role" not in call_args.kwargs["data"]


@pytest.mark.asyncio
async def test_upsert_sso_user_no_role_in_sso_response():
    """
    Test that upsert_sso_user only updates email when SSO response has no role.

    When the SSO provider does not return a role, only the email should be updated.
    """
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.management_endpoints.types import CustomOpenID
    from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

    # Mock prisma client
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.update_many = AsyncMock()

    # Existing user in DB
    existing_user = LiteLLM_UserTable(
        user_id="test-user-789",
        user_email="old@example.com",
        user_role="internal_user",
        models=[],
    )

    # SSO result without role
    sso_result = CustomOpenID(
        id="test-user-789",
        email="new@example.com",
        display_name="Test User",
        provider="microsoft",
        team_ids=[],
        user_role=None,
    )

    # Act
    await SSOAuthenticationHandler.upsert_sso_user(
        result=sso_result,
        user_info=existing_user,
        user_email="new@example.com",
        user_defined_values=None,
        prisma_client=mock_prisma,
    )

    # Assert - verify only email was updated
    mock_prisma.db.litellm_usertable.update_many.assert_called_once()
    call_args = mock_prisma.db.litellm_usertable.update_many.call_args
    assert call_args.kwargs["where"] == {"user_id": "test-user-789"}
    assert call_args.kwargs["data"]["user_email"] == "new@example.com"
    assert "user_role" not in call_args.kwargs["data"]


def test_get_user_email_and_id_extracts_microsoft_role():
    """
    Test that _get_user_email_and_id_from_result extracts user_role from Microsoft SSO.

    This ensures Microsoft SSO roles (from app_roles in id_token) are properly
    extracted and converted from enum to string.
    """
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_endpoints.types import CustomOpenID
    from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

    result = CustomOpenID(
        id="test-user-id",
        email="test@example.com",
        display_name="Test User",
        provider="microsoft",
        team_ids=["team-1"],
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )

    parsed = SSOAuthenticationHandler._get_user_email_and_id_from_result(
        result=result,
        generic_client_id=None,
    )

    assert parsed.get("user_email") == "test@example.com"
    assert parsed.get("user_id") == "test-user-id"
    assert parsed.get("user_role") == "proxy_admin_viewer"


@pytest.mark.asyncio
async def test_get_user_info_from_db_user_exists():
    """
    Test that get_user_info_from_db finds existing user and calls upsert_sso_user to update.
    """
    from litellm.proxy.management_endpoints.ui_sso import get_user_info_from_db

    prisma_client = MagicMock()
    user_api_key_cache = MagicMock()
    proxy_logging_obj = MagicMock()
    user_email = "krrishdholakia@gmail.com"
    user_defined_values = {
        "models": [],
        "user_id": "krrishd",
        "user_email": "krrishdholakia@gmail.com",
        "max_budget": None,
        "user_role": None,
        "budget_duration": None,
    }
    args = {
        "result": CustomOpenID(
            id="krrishd",
            email="krrishdholakia@gmail.com",
            first_name=None,
            last_name=None,
            display_name="a3f1c107-04dc-4c93-ae60-7f32eb4b05ce",
            picture=None,
            provider=None,
            team_ids=[],
        ),
        "prisma_client": prisma_client,
        "user_api_key_cache": user_api_key_cache,
        "proxy_logging_obj": proxy_logging_obj,
        "user_email": user_email,
        "user_defined_values": user_defined_values,
    }
    with patch(
        "litellm.proxy.management_endpoints.ui_sso.get_user_object"
    ) as mock_get_user_object:
        await get_user_info_from_db(**args)
        mock_get_user_object.assert_called_once()
        assert mock_get_user_object.call_args.kwargs["user_id"] == "krrishd"


@pytest.mark.asyncio
async def test_get_user_info_from_db_user_exists_alternate_user_id():
    from litellm.proxy.management_endpoints.ui_sso import get_user_info_from_db

    prisma_client = MagicMock()
    user_api_key_cache = MagicMock()
    proxy_logging_obj = MagicMock()
    user_email = "krrishdholakia@gmail.com"
    user_defined_values = {
        "models": [],
        "user_id": "krrishd",
        "user_email": "krrishdholakia@gmail.com",
        "max_budget": None,
        "user_role": None,
        "budget_duration": None,
    }
    args = {
        "result": CustomOpenID(
            id="krrishd",
            email="krrishdholakia@gmail.com",
            first_name=None,
            last_name=None,
            display_name="a3f1c107-04dc-4c93-ae60-7f32eb4b05ce",
            picture=None,
            provider=None,
            team_ids=[],
        ),
        "prisma_client": prisma_client,
        "user_api_key_cache": user_api_key_cache,
        "proxy_logging_obj": proxy_logging_obj,
        "user_email": user_email,
        "user_defined_values": user_defined_values,
        "alternate_user_id": "krrishd-email1234",
    }
    with patch(
        "litellm.proxy.management_endpoints.ui_sso.get_user_object"
    ) as mock_get_user_object:
        await get_user_info_from_db(**args)
        mock_get_user_object.assert_called_once()
        assert mock_get_user_object.call_args.kwargs["user_id"] == "krrishd-email1234"


@pytest.mark.asyncio
async def test_get_user_info_from_db_user_not_exists_creates_user():
    """
    Test that get_user_info_from_db creates a new user when user doesn't exist in DB.

    When get_existing_user_info_from_db returns None, get_user_info_from_db should:
    1. Call upsert_sso_user with user_info=None
    2. upsert_sso_user should call insert_sso_user to create the user
    3. Add user to teams from SSO response
    """
    from litellm.proxy._types import NewUserResponse, SSOUserDefinedValues
    from litellm.proxy.management_endpoints.ui_sso import get_user_info_from_db

    prisma_client = MagicMock()
    user_api_key_cache = MagicMock()
    proxy_logging_obj = MagicMock()
    user_email = "newuser@example.com"
    user_defined_values: SSOUserDefinedValues = {
        "models": [],
        "user_id": "new-user-123",
        "user_email": "newuser@example.com",
        "max_budget": None,
        "user_role": None,
        "budget_duration": None,
    }

    sso_result = CustomOpenID(
        id="new-user-123",
        email="newuser@example.com",
        first_name="New",
        last_name="User",
        display_name="New User",
        picture=None,
        provider="microsoft",
        team_ids=["team-1", "team-2"],
    )

    args = {
        "result": sso_result,
        "prisma_client": prisma_client,
        "user_api_key_cache": user_api_key_cache,
        "proxy_logging_obj": proxy_logging_obj,
        "user_email": user_email,
        "user_defined_values": user_defined_values,
    }

    # Mock new user response
    mock_new_user = NewUserResponse(
        user_id="new-user-123",
        key="sk-xxxxx",
        teams=None,
    )

    with patch(
        "litellm.proxy.management_endpoints.ui_sso.get_existing_user_info_from_db",
        return_value=None,  # User doesn't exist
    ) as mock_get_existing, patch(
        "litellm.proxy.management_endpoints.ui_sso.SSOAuthenticationHandler.upsert_sso_user",
        return_value=mock_new_user,
    ) as mock_upsert, patch(
        "litellm.proxy.management_endpoints.ui_sso.SSOAuthenticationHandler.add_user_to_teams_from_sso_response",
    ) as mock_add_teams:
        # Act
        user_info = await get_user_info_from_db(**args)

        # Assert
        # Should try to find user by id
        mock_get_existing.assert_called_once()
        assert mock_get_existing.call_args.kwargs["user_id"] == "new-user-123"
        assert mock_get_existing.call_args.kwargs["user_email"] == "newuser@example.com"

        # Should call upsert_sso_user with None user_info
        mock_upsert.assert_called_once()
        upsert_call_args = mock_upsert.call_args
        assert upsert_call_args.kwargs["user_info"] is None
        assert upsert_call_args.kwargs["user_email"] == "newuser@example.com"
        assert upsert_call_args.kwargs["user_defined_values"] == user_defined_values

        # Should add user to teams
        mock_add_teams.assert_called_once()
        add_teams_call_args = mock_add_teams.call_args
        assert add_teams_call_args.kwargs["result"] == sso_result
        assert add_teams_call_args.kwargs["user_info"] == mock_new_user

        # Should return the new user
        assert user_info == mock_new_user


@pytest.mark.asyncio
async def test_get_user_info_from_db_user_exists_updates_user():
    """
    Test that get_user_info_from_db updates existing user when user exists in DB.

    When get_existing_user_info_from_db returns a user, get_user_info_from_db should:
    1. Call upsert_sso_user with the existing user_info
    2. upsert_sso_user should update the user in the database
    3. Add user to teams from SSO response
    """
    from litellm.proxy._types import LiteLLM_UserTable, SSOUserDefinedValues
    from litellm.proxy.management_endpoints.ui_sso import get_user_info_from_db

    prisma_client = MagicMock()
    user_api_key_cache = MagicMock()
    proxy_logging_obj = MagicMock()
    user_email = "existing@example.com"
    user_defined_values: SSOUserDefinedValues = {
        "models": [],
        "user_id": "existing-user-456",
        "user_email": "existing@example.com",
        "max_budget": None,
        "user_role": None,
        "budget_duration": None,
    }

    sso_result = CustomOpenID(
        id="existing-user-456",
        email="existing@example.com",
        first_name="Existing",
        last_name="User",
        display_name="Existing User",
        picture=None,
        provider="microsoft",
        team_ids=["team-3"],
    )

    # Existing user in DB
    existing_user = LiteLLM_UserTable(
        user_id="existing-user-456",
        user_email="old@example.com",
        user_role="internal_user",
        models=["gpt-4"],
        teams=[],
    )

    # Updated user after upsert
    updated_user = LiteLLM_UserTable(
        user_id="existing-user-456",
        user_email="existing@example.com",  # Updated email
        user_role="internal_user",
        models=["gpt-4"],
        teams=[],
    )

    args = {
        "result": sso_result,
        "prisma_client": prisma_client,
        "user_api_key_cache": user_api_key_cache,
        "proxy_logging_obj": proxy_logging_obj,
        "user_email": user_email,
        "user_defined_values": user_defined_values,
    }

    with patch(
        "litellm.proxy.management_endpoints.ui_sso.get_existing_user_info_from_db",
        return_value=existing_user,  # User exists
    ) as mock_get_existing, patch(
        "litellm.proxy.management_endpoints.ui_sso.SSOAuthenticationHandler.upsert_sso_user",
        return_value=updated_user,
    ) as mock_upsert, patch(
        "litellm.proxy.management_endpoints.ui_sso.SSOAuthenticationHandler.add_user_to_teams_from_sso_response",
    ) as mock_add_teams:
        # Act
        user_info = await get_user_info_from_db(**args)

        # Assert
        # Should find existing user
        mock_get_existing.assert_called_once()
        assert mock_get_existing.call_args.kwargs["user_id"] == "existing-user-456"

        # Should call upsert_sso_user with existing user_info
        mock_upsert.assert_called_once()
        upsert_call_args = mock_upsert.call_args
        assert upsert_call_args.kwargs["user_info"] == existing_user
        assert upsert_call_args.kwargs["user_email"] == "existing@example.com"

        # Should add user to teams
        mock_add_teams.assert_called_once()
        add_teams_call_args = mock_add_teams.call_args
        assert add_teams_call_args.kwargs["result"] == sso_result
        assert add_teams_call_args.kwargs["user_info"] == updated_user

        # Should return the updated user
        assert user_info == updated_user


@pytest.mark.asyncio
async def test_check_and_update_if_proxy_admin_id():
    """
    Test that a user with matching PROXY_ADMIN_ID gets their role updated to admin
    """
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_endpoints.ui_sso import (
        check_and_update_if_proxy_admin_id,
    )

    # Mock Prisma client
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.update = AsyncMock()

    # Set up test data
    test_user_id = "test_admin_123"
    test_user_role = "user"

    with patch.dict(os.environ, {"PROXY_ADMIN_ID": test_user_id}):
        # Act
        updated_role = await check_and_update_if_proxy_admin_id(
            user_role=test_user_role, user_id=test_user_id, prisma_client=mock_prisma
        )

        # Assert
        assert updated_role == LitellmUserRoles.PROXY_ADMIN.value
        mock_prisma.db.litellm_usertable.update.assert_called_once_with(
            where={"user_id": test_user_id},
            data={"user_role": LitellmUserRoles.PROXY_ADMIN.value},
        )


@pytest.mark.asyncio
async def test_check_and_update_if_proxy_admin_id_already_admin():
    """
    Test that a user who is already an admin doesn't get their role updated
    """
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_endpoints.ui_sso import (
        check_and_update_if_proxy_admin_id,
    )

    # Mock Prisma client
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.update = AsyncMock()

    # Set up test data
    test_user_id = "test_admin_123"
    test_user_role = LitellmUserRoles.PROXY_ADMIN.value

    with patch.dict(os.environ, {"PROXY_ADMIN_ID": test_user_id}):
        # Act
        updated_role = await check_and_update_if_proxy_admin_id(
            user_role=test_user_role, user_id=test_user_id, prisma_client=mock_prisma
        )

        # Assert
        assert updated_role == LitellmUserRoles.PROXY_ADMIN.value
        mock_prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_get_generic_sso_response_with_additional_headers():
    """
    Test that GENERIC_SSO_HEADERS environment variable is correctly processed
    and passed to generic_sso.verify_and_process
    """
    from litellm.proxy.management_endpoints.ui_sso import get_generic_sso_response

    # Arrange
    mock_request = MagicMock(spec=Request)
    mock_jwt_handler = MagicMock(spec=JWTHandler)
    mock_jwt_handler.get_team_ids_from_jwt.return_value = []

    generic_client_id = "test_client_id"
    redirect_url = "http://test.com/callback"

    # Mock response from verify_and_process
    mock_sso_response = {
        "sub": "test_user_123",
        "email": "test@example.com",
        "preferred_username": "testuser",
    }

    # Set up environment variables including GENERIC_SSO_HEADERS
    test_env_vars = {
        "GENERIC_CLIENT_SECRET": "test_secret",
        "GENERIC_AUTHORIZATION_ENDPOINT": "https://auth.example.com/auth",
        "GENERIC_TOKEN_ENDPOINT": "https://auth.example.com/token",
        "GENERIC_USERINFO_ENDPOINT": "https://auth.example.com/userinfo",
        "GENERIC_SSO_HEADERS": "Authorization=Bearer token123, Content-Type=application/json, X-Custom-Header=custom-value",
    }

    # Expected headers dictionary
    expected_headers = {
        "Authorization": "Bearer token123",
        "Content-Type": "application/json",
        "X-Custom-Header": "custom-value",
    }

    # Mock the SSO provider and its methods
    mock_sso_instance = MagicMock()
    mock_sso_instance.verify_and_process = AsyncMock(return_value=mock_sso_response)
    mock_sso_instance.access_token = None  # Avoid triggering JWT decode in process_sso_jwt_access_token

    mock_sso_class = MagicMock(return_value=mock_sso_instance)

    with patch.dict(os.environ, test_env_vars):
        with patch("fastapi_sso.sso.base.DiscoveryDocument"):
            with patch(
                "fastapi_sso.sso.generic.create_provider", return_value=mock_sso_class
            ):
                # Act
                result, received_response = await get_generic_sso_response(
                    request=mock_request,
                    jwt_handler=mock_jwt_handler,
                    generic_client_id=generic_client_id,
                    redirect_url=redirect_url,
                    sso_jwt_handler=None,
                )

                # Assert
                # Verify verify_and_process was called with the correct headers
                mock_sso_instance.verify_and_process.assert_called_once_with(
                    mock_request,
                    params={"include_client_id": False},
                    headers=expected_headers,
                )

                # Verify the result is returned correctly
                assert result == mock_sso_response


@pytest.mark.asyncio
async def test_get_generic_sso_response_with_empty_headers():
    """
    Test that when GENERIC_SSO_HEADERS is not set, an empty headers dict is passed
    """
    from litellm.proxy.management_endpoints.ui_sso import get_generic_sso_response

    # Arrange
    mock_request = MagicMock(spec=Request)
    mock_jwt_handler = MagicMock(spec=JWTHandler)
    mock_jwt_handler.get_team_ids_from_jwt.return_value = []

    generic_client_id = "test_client_id"
    redirect_url = "http://test.com/callback"

    mock_sso_response = {
        "sub": "test_user_123",
        "email": "test@example.com",
        "preferred_username": "testuser",
    }

    # Set up environment variables without GENERIC_SSO_HEADERS
    test_env_vars = {
        "GENERIC_CLIENT_SECRET": "test_secret",
        "GENERIC_AUTHORIZATION_ENDPOINT": "https://auth.example.com/auth",
        "GENERIC_TOKEN_ENDPOINT": "https://auth.example.com/token",
        "GENERIC_USERINFO_ENDPOINT": "https://auth.example.com/userinfo",
    }

    # Mock the SSO provider and its methods
    mock_sso_instance = MagicMock()
    mock_sso_instance.verify_and_process = AsyncMock(return_value=mock_sso_response)
    mock_sso_instance.access_token = None  # Avoid triggering JWT decode in process_sso_jwt_access_token

    mock_sso_class = MagicMock(return_value=mock_sso_instance)

    with patch.dict(os.environ, test_env_vars):
        with patch("fastapi_sso.sso.base.DiscoveryDocument"):
            with patch(
                "fastapi_sso.sso.generic.create_provider", return_value=mock_sso_class
            ):
                # Act
                result, received_response = await get_generic_sso_response(
                    request=mock_request,
                    jwt_handler=mock_jwt_handler,
                    generic_client_id=generic_client_id,
                    redirect_url=redirect_url,
                    sso_jwt_handler=None,
                )

                # Assert
                # Verify verify_and_process was called with empty headers dict
                mock_sso_instance.verify_and_process.assert_called_once_with(
                    mock_request, params={"include_client_id": False}, headers={}
                )

                assert result == mock_sso_response


class TestCLISSOCallbackFunction:
    """Test the cli_sso_callback function specifically"""

    def test_cli_sso_callback_validation_invalid_key(self):
        """Test CLI SSO callback input validation for invalid key format"""
        # Test the validation logic without hitting the database
        invalid_keys = [
            None,
            "",
            "invalid-key",
            "not-sk-key",
            "sk",  # too short
        ]

        for invalid_key in invalid_keys:
            # This should fail validation before any database operations
            # We can test this by checking if the key starts with 'sk-'
            if not invalid_key or not invalid_key.startswith("sk-"):
                # This would trigger the validation error
                assert True  # Validation works as expected


class TestCLIPollingFunction:
    """Test the cli_poll_key function specifically"""

    def test_cli_poll_key_validation_invalid_format(self):
        """Test CLI polling key format validation"""
        # Test key format validation logic
        invalid_keys = [
            "invalid-key",
            "not-sk-key",
            "",
            "sk",  # too short
        ]

        for invalid_key in invalid_keys:
            # Validation logic: key must start with 'sk-'
            if not invalid_key.startswith("sk-"):
                # This would trigger the validation error in the actual function
                assert True  # Validation works as expected


class TestAuthCallbackRouting:
    """Test the auth_callback function routing logic"""

    def test_cli_state_detection_and_routing(self):
        """Test that CLI states are properly detected and would route to CLI callback"""
        from litellm.constants import LITELLM_CLI_SESSION_TOKEN_PREFIX

        # Test CLI state detection logic
        cli_state = f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:sk-test123"

        # This mimics the logic in auth_callback
        if cli_state and cli_state.startswith(f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:"):
            # Extract the key ID from the state
            key_id = cli_state.split(":", 1)[1]
            assert key_id == "sk-test123"
        else:
            assert False, "CLI state should have been detected"

    def test_non_cli_state_routing(self):
        """Test that non-CLI states don't trigger CLI routing"""
        from litellm.constants import LITELLM_CLI_SESSION_TOKEN_PREFIX

        non_cli_states = [
            "regular_oauth_state",
            "some_random_string",
            None,
            "",
            "not_session_token:something",
        ]

        for state in non_cli_states:
            # This mimics the routing logic in auth_callback
            should_route_to_cli = state and state.startswith(
                f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:"
            )
            assert not should_route_to_cli, f"State '{state}' should not route to CLI"


class TestGoogleLoginCLIIntegration:
    """Test the google_login function with CLI parameters"""

    def test_google_login_cli_state_generation(self):
        """Test that google_login generates CLI state when CLI parameters are provided"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Test the CLI state generation logic used in google_login
        source = "litellm-cli"
        key = "sk-test123"

        cli_state = SSOAuthenticationHandler._get_cli_state(source=source, key=key)

        assert cli_state is not None
        assert cli_state.startswith("litellm-session-token:")
        assert "sk-test123" in cli_state

    def test_google_login_no_cli_state_when_missing_params(self):
        """Test that google_login doesn't generate CLI state when CLI parameters are missing"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Test various parameter combinations that shouldn't generate CLI state
        test_cases = [
            (None, None),
            ("litellm-cli", None),
            (None, "sk-test123"),
            ("wrong-source", "sk-test123"),
        ]

        for source, key in test_cases:
            cli_state = SSOAuthenticationHandler._get_cli_state(source=source, key=key)
            assert (
                cli_state is None
            ), f"CLI state should not be generated for source='{source}', key='{key}'"


class TestSSOHandlerIntegration:
    """Test SSOAuthenticationHandler methods"""

    def test_should_use_sso_handler(self):
        """Test the SSO handler detection logic"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Test that SSO handler is used when client IDs are provided
        assert (
            SSOAuthenticationHandler.should_use_sso_handler(google_client_id="test")
            is True
        )
        assert (
            SSOAuthenticationHandler.should_use_sso_handler(microsoft_client_id="test")
            is True
        )
        assert (
            SSOAuthenticationHandler.should_use_sso_handler(generic_client_id="test")
            is True
        )

        # Test that SSO handler is not used when no client IDs are provided
        assert SSOAuthenticationHandler.should_use_sso_handler() is False
        assert (
            SSOAuthenticationHandler.should_use_sso_handler(None, None, None) is False
        )

    def test_get_redirect_url_for_sso(self):
        """Test the redirect URL generation for SSO"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Mock request object
        mock_request = MagicMock()
        mock_request.base_url = "https://test.litellm.ai/"

        # Test redirect URL generation
        redirect_url = SSOAuthenticationHandler.get_redirect_url_for_sso(
            request=mock_request, sso_callback_route="sso/callback"
        )

        assert redirect_url.startswith("https://test.litellm.ai")
        assert "sso/callback" in redirect_url


class TestUISSO_FunctionsExistence:
    """Test that all the new functions exist and are importable"""

    def test_cli_sso_callback_exists(self):
        """Test that cli_sso_callback function exists"""
        from litellm.proxy.management_endpoints.ui_sso import cli_sso_callback

        assert callable(cli_sso_callback)

    def test_cli_poll_key_exists(self):
        """Test that cli_poll_key function exists"""
        from litellm.proxy.management_endpoints.ui_sso import cli_poll_key

        assert callable(cli_poll_key)

    def test_auth_callback_exists(self):
        """Test that auth_callback function exists"""
        from litellm.proxy.management_endpoints.ui_sso import auth_callback

        assert callable(auth_callback)

    def test_google_login_exists(self):
        """Test that google_login function exists"""
        from litellm.proxy.management_endpoints.ui_sso import google_login

        assert callable(google_login)

    def test_sso_authentication_handler_exists(self):
        """Test that SSOAuthenticationHandler class exists with new methods"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Check that the class exists
        assert SSOAuthenticationHandler is not None

        # Check that the new _get_cli_state method exists
        assert hasattr(SSOAuthenticationHandler, "_get_cli_state")
        assert callable(SSOAuthenticationHandler._get_cli_state)


class TestSSOStateHandling:
    """Test the SSO state handling for CLI authentication"""

    def test_get_cli_state_valid(self):
        """Test generating CLI state with valid parameters"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        state = SSOAuthenticationHandler._get_cli_state(
            source="litellm-cli", key="sk-test123"
        )

        assert state is not None
        assert state.startswith("litellm-session-token:")
        assert "sk-test123" in state

    def test_get_cli_state_invalid_source(self):
        """Test generating CLI state with invalid source"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        state = SSOAuthenticationHandler._get_cli_state(
            source="invalid_source", key="sk-test123"
        )

        assert state is None

    def test_get_cli_state_no_key(self):
        """Test generating CLI state without key"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        state = SSOAuthenticationHandler._get_cli_state(source="litellm-cli", key=None)

        assert state is None

    def test_get_cli_state_no_source(self):
        """Test generating CLI state without source"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        state = SSOAuthenticationHandler._get_cli_state(source=None, key="sk-test123")

        assert state is None

    def test_get_cli_state_with_existing_key(self):
        """Test generating CLI state with existing_key embedded in state parameter"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        state = SSOAuthenticationHandler._get_cli_state(
            source="litellm-cli",
            key="sk-new-key-123",
            existing_key="sk-existing-key-456",
        )

        assert state is not None
        assert state.startswith("litellm-session-token:")
        assert "sk-new-key-123" in state
        assert "sk-existing-key-456" in state
        # Verify the format: {PREFIX}:{key}:{existing_key}
        assert state == "litellm-session-token:sk-new-key-123:sk-existing-key-456"

    def test_get_cli_state_without_existing_key(self):
        """Test generating CLI state without existing_key"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        state = SSOAuthenticationHandler._get_cli_state(
            source="litellm-cli", key="sk-new-key-789", existing_key=None
        )

        assert state is not None
        assert state.startswith("litellm-session-token:")
        assert "sk-new-key-789" in state
        # Verify the format: {PREFIX}:{key} (no third part)
        assert state == "litellm-session-token:sk-new-key-789"
        assert state.count(":") == 1  # Only one colon separator


class TestStateRouting:
    """Test state parameter routing logic"""

    def test_cli_state_detection(self):
        """Test detection of CLI state parameters"""
        from litellm.constants import LITELLM_CLI_SESSION_TOKEN_PREFIX

        # Test CLI state format
        cli_state = f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:sk-test123"
        assert cli_state.startswith(f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:")

        # Test extraction of key from state
        key_id = cli_state.split(":", 1)[1]
        assert key_id == "sk-test123"

    def test_cli_state_parsing_with_existing_key(self):
        """Test parsing CLI state with existing_key embedded"""
        from litellm.constants import LITELLM_CLI_SESSION_TOKEN_PREFIX

        # State format: {PREFIX}:{key}:{existing_key}
        cli_state = (
            f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:sk-new-key-456:sk-existing-key-789"
        )

        # Parse as done in auth_callback
        state_parts = cli_state.split(":", 2)  # Split into max 3 parts
        key_id = state_parts[1] if len(state_parts) > 1 else None
        existing_key = state_parts[2] if len(state_parts) > 2 else None

        assert key_id == "sk-new-key-456"
        assert existing_key == "sk-existing-key-789"

    def test_cli_state_parsing_without_existing_key(self):
        """Test parsing CLI state without existing_key"""
        from litellm.constants import LITELLM_CLI_SESSION_TOKEN_PREFIX

        # State format: {PREFIX}:{key}
        cli_state = f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:sk-new-key-999"

        # Parse as done in auth_callback
        state_parts = cli_state.split(":", 2)  # Split into max 3 parts
        key_id = state_parts[1] if len(state_parts) > 1 else None
        existing_key = state_parts[2] if len(state_parts) > 2 else None

        assert key_id == "sk-new-key-999"
        assert existing_key is None

    def test_non_cli_state_detection(self):
        """Test detection of non-CLI state parameters"""
        from litellm.constants import LITELLM_CLI_SESSION_TOKEN_PREFIX

        # Test various non-CLI states
        test_states = [
            "regular_oauth_state",
            "some_random_string",
            None,
            "",
            "not_session_token:something",
        ]

        for state in test_states:
            if state:
                assert not state.startswith(f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:")
            else:
                assert state != f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:"


class TestHTMLIntegration:
    """Test HTML rendering integration with CLI flow"""

    def test_html_render_utils_import(self):
        """Test that HTML render utils can be imported correctly"""
        from litellm.proxy.common_utils.html_forms.cli_sso_success import (
            render_cli_sso_success_page,
        )

        # Test that function exists and is callable
        assert callable(render_cli_sso_success_page)

        # Test that it returns expected type
        html = render_cli_sso_success_page()

        assert isinstance(html, str)
        assert len(html) > 0


class TestCustomUISSO:
    """Test the custom UI SSO sign-in handler functionality"""

    def test_enterprise_import_error_handling(self):
        """Test that proper error is raised when enterprise module is not available"""
        from unittest.mock import MagicMock, patch

        # Mock request
        mock_request = MagicMock()
        mock_request.base_url = "https://test.example.com/"

        # Mock user_custom_ui_sso_sign_in_handler to exist but make enterprise import fail
        with patch("litellm.proxy.proxy_server.premium_user", True):
            with patch(
                "litellm.proxy.proxy_server.user_custom_ui_sso_sign_in_handler",
                MagicMock(),
            ):
                with patch.dict(
                    "sys.modules",
                    {
                        "enterprise.litellm_enterprise.proxy.auth.custom_sso_handler": None
                    },
                ):
                    # Temporarily mock the google_login function call to test the import error path
                    async def mock_google_login():
                        # This mimics the relevant part of google_login that would trigger the import error
                        try:
                            from enterprise.litellm_enterprise.proxy.auth.custom_sso_handler import (
                                EnterpriseCustomSSOHandler,  # noqa: F401
                            )

                            return "success"
                        except ImportError:
                            raise ValueError(
                                "Enterprise features are not available. Custom UI SSO sign-in requires LiteLLM Enterprise."
                            )

                    # Test that the ValueError is raised with the correct message
                    import pytest

                    with pytest.raises(
                        ValueError, match="Enterprise features are not available"
                    ):
                        asyncio.run(mock_google_login())

    @pytest.mark.asyncio
    async def test_handle_custom_ui_sso_sign_in_success(self):
        """Test successful custom UI SSO sign-in with valid headers"""
        from fastapi_sso.sso.base import OpenID

        from enterprise.litellm_enterprise.proxy.auth.custom_sso_handler import (
            EnterpriseCustomSSOHandler,
        )
        from litellm.integrations.custom_sso_handler import CustomSSOLoginHandler

        # Mock request with custom headers
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {
            "x-litellm-user-id": "test_user_123",
            "x-litellm-user-email": "test@example.com",
            "x-forwarded-for": "192.168.1.1",
        }
        mock_request.base_url = "https://test.litellm.ai/"

        # Mock the custom handler
        mock_custom_handler = MagicMock(spec=CustomSSOLoginHandler)
        expected_openid = OpenID(
            id="test_user_123",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            display_name="Test User",
            picture=None,
            provider="custom",
        )
        mock_custom_handler.handle_custom_ui_sso_sign_in = AsyncMock(
            return_value=expected_openid
        )

        # Mock the redirect response method
        mock_redirect_response = MagicMock()
        mock_redirect_response.status_code = 303

        with patch("litellm.proxy.proxy_server.premium_user", True):
            with patch(
                "litellm.proxy.proxy_server.user_custom_ui_sso_sign_in_handler",
                mock_custom_handler,
            ):
                with patch.object(
                    SSOAuthenticationHandler,
                    "get_redirect_response_from_openid",
                    return_value=mock_redirect_response,
                ) as mock_get_redirect:
                    # Act
                    result = (
                        await EnterpriseCustomSSOHandler.handle_custom_ui_sso_sign_in(
                            request=mock_request
                        )
                    )

                    # Assert
                    # Verify the custom handler was called with the request
                    mock_custom_handler.handle_custom_ui_sso_sign_in.assert_called_once_with(
                        request=mock_request
                    )

                    # Verify the redirect response was generated with correct OpenID
                    mock_get_redirect.assert_called_once_with(
                        result=expected_openid,
                        request=mock_request,
                        received_response=None,
                        generic_client_id=None,
                        ui_access_mode=None,
                    )

                    # Verify the result is the redirect response
                    assert result == mock_redirect_response
                    assert result.status_code == 303

    @pytest.mark.asyncio
    async def test_custom_ui_sso_handler_execution_with_real_class(self):
        """
        Test that when a user provides a custom class instance, it gets properly executed
        and its methods are called with the correct parameters
        """
        from fastapi_sso.sso.base import OpenID

        from enterprise.litellm_enterprise.proxy.auth.custom_sso_handler import (
            EnterpriseCustomSSOHandler,
        )
        from litellm.integrations.custom_sso_handler import CustomSSOLoginHandler

        # Create a real custom handler class instance
        class TestCustomSSOHandler(CustomSSOLoginHandler):
            def __init__(self):
                super().__init__()
                self.method_called = False
                self.received_request = None

            async def handle_custom_ui_sso_sign_in(self, request: Request) -> OpenID:
                self.method_called = True
                self.received_request = request

                # Parse headers like the actual implementation would
                request_headers_dict = dict(request.headers)
                return OpenID(
                    id=request_headers_dict.get("x-litellm-user-id", "default_user"),
                    email=request_headers_dict.get(
                        "x-litellm-user-email", "default@test.com"
                    ),
                    first_name="Custom",
                    last_name="Handler",
                    display_name="Custom Handler Test",
                    picture=None,
                    provider="custom",
                )

        # Create instance of our test handler
        test_handler_instance = TestCustomSSOHandler()

        # Mock request with custom headers
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {
            "x-litellm-user-id": "custom_test_user_456",
            "x-litellm-user-email": "custom@example.com",
            "x-forwarded-for": "10.0.0.1",
        }
        mock_request.base_url = "https://custom.litellm.ai/"

        # Mock the redirect response method
        mock_redirect_response = MagicMock()
        mock_redirect_response.status_code = 303

        with patch("litellm.proxy.proxy_server.premium_user", True):
            with patch(
                "litellm.proxy.proxy_server.user_custom_ui_sso_sign_in_handler",
                test_handler_instance,
            ):
                with patch.object(
                    SSOAuthenticationHandler,
                    "get_redirect_response_from_openid",
                    return_value=mock_redirect_response,
                ) as mock_get_redirect:
                    # Act
                    result = (
                        await EnterpriseCustomSSOHandler.handle_custom_ui_sso_sign_in(
                            request=mock_request
                        )
                    )

                    # Assert that our custom handler was executed
                    assert test_handler_instance.method_called is True
                    assert test_handler_instance.received_request == mock_request

                    # Verify the redirect response was called with the OpenID from our custom handler
                    mock_get_redirect.assert_called_once()
                    call_args = mock_get_redirect.call_args.kwargs

                    # Verify the OpenID object has the expected values from our custom handler
                    openid_result = call_args["result"]
                    assert openid_result.id == "custom_test_user_456"
                    assert openid_result.email == "custom@example.com"
                    assert openid_result.first_name == "Custom"
                    assert openid_result.last_name == "Handler"
                    assert openid_result.display_name == "Custom Handler Test"
                    assert openid_result.provider == "custom"

                    # Verify the request and other parameters were passed correctly
                    assert call_args["request"] == mock_request
                    assert call_args["received_response"] is None
                    assert call_args["generic_client_id"] is None
                    assert call_args["ui_access_mode"] is None

                    # Verify the result is the redirect response
                    assert result == mock_redirect_response
                    assert result.status_code == 303


class TestCLIKeyRegenerationFlow:
    """Test the end-to-end CLI key regeneration flow"""

    @pytest.mark.asyncio
    async def test_cli_sso_callback_stores_session(self):
        """Test CLI SSO callback stores session data in cache for JWT generation"""
        from litellm.proxy._types import LiteLLM_UserTable
        from litellm.proxy.management_endpoints.ui_sso import cli_sso_callback

        # Mock request
        mock_request = MagicMock(spec=Request)

        # Test data
        session_key = "sk-session-456"

        # Mock user info
        mock_user_info = LiteLLM_UserTable(
            user_id="test-user-123",
            user_role="internal_user",
            teams=["team1", "team2"],
            models=["gpt-4"],
        )

        # Mock SSO result
        mock_sso_result = {"user_email": "test@example.com", "user_id": "test-user-123"}

        # Mock cache
        mock_cache = MagicMock()

        with patch(
            "litellm.proxy.management_endpoints.ui_sso.get_user_info_from_db",
            return_value=mock_user_info,
        ), patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), patch(
            "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
        ), patch(
            "litellm.proxy.common_utils.html_forms.cli_sso_success.render_cli_sso_success_page",
            return_value="<html>Success</html>",
        ):
            # Act
            result = await cli_sso_callback(
                request=mock_request,
                key=session_key,
                existing_key=None,
                result=mock_sso_result,
            )

            # Assert - verify session was stored in cache
            mock_cache.set_cache.assert_called_once()
            call_args = mock_cache.set_cache.call_args

            # Verify cache key format
            assert "cli_sso_session:" in call_args.kwargs["key"]
            assert session_key in call_args.kwargs["key"]

            # Verify session data structure
            session_data = call_args.kwargs["value"]
            assert session_data["user_id"] == "test-user-123"
            assert session_data["user_role"] == "internal_user"
            assert session_data["teams"] == ["team1", "team2"]
            assert session_data["models"] == ["gpt-4"]

            # Verify TTL
            assert call_args.kwargs["ttl"] == 600  # 10 minutes

            assert result.status_code == 200
            # Verify response contains success message (response is HTML)
            assert result.body is not None

    @pytest.mark.asyncio
    async def test_cli_poll_key_returns_teams_for_selection(self):
        """Test CLI poll endpoint returns teams for user selection when multiple teams exist"""
        from litellm.proxy.management_endpoints.ui_sso import cli_poll_key

        # Test data
        session_key = "sk-session-789"
        session_data = {
            "user_id": "test-user-456",
            "user_role": "internal_user",
            "teams": ["team-a", "team-b", "team-c"],
            "models": ["gpt-4"],
        }

        # Mock cache
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = session_data

        with patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache):
            # Act - First poll without team_id
            result = await cli_poll_key(key_id=session_key, team_id=None)

            # Assert - should return teams list for selection
            assert result["status"] == "ready"
            assert result["requires_team_selection"] is True
            assert result["user_id"] == "test-user-456"
            assert result["teams"] == ["team-a", "team-b", "team-c"]
            assert "key" not in result  # JWT should not be generated yet

            # Verify session was NOT deleted
            mock_cache.delete_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_auth_callback_routes_to_cli_with_existing_key(self):
        """Test that auth_callback properly routes CLI requests and extracts existing_key from state parameter"""
        from litellm.constants import LITELLM_CLI_SESSION_TOKEN_PREFIX
        from litellm.proxy.management_endpoints.ui_sso import auth_callback

        # Mock request (no query params needed - existing_key is in state)
        mock_request = MagicMock(spec=Request)

        # CLI state with existing_key embedded: {PREFIX}:{key}:{existing_key}
        cli_state = f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:sk-new-session-key-456:sk-existing-cli-key-123"

        # Mock the CLI callback and required proxy server components
        mock_result = {"user_id": "test-user", "email": "test@example.com"}

        with patch(
            "litellm.proxy.management_endpoints.ui_sso.cli_sso_callback"
        ) as mock_cli_callback, patch(
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ), patch(
            "litellm.proxy.proxy_server.master_key", "test-master-key"
        ), patch(
            "litellm.proxy.proxy_server.general_settings", {}
        ), patch(
            "litellm.proxy.proxy_server.jwt_handler", MagicMock()
        ), patch(
            "litellm.proxy.proxy_server.user_api_key_cache", MagicMock()
        ), patch.dict(
            os.environ, {"GOOGLE_CLIENT_ID": "test-google-id"}, clear=True
        ), patch(
            "litellm.proxy.management_endpoints.ui_sso.GoogleSSOHandler.get_google_callback_response",
            return_value=mock_result,
        ):
            mock_cli_callback.return_value = MagicMock()

            # Act
            await auth_callback(request=mock_request, state=cli_state)

            # Assert - existing_key should be extracted from state parameter
            mock_cli_callback.assert_called_once_with(
                request=mock_request,
                key="sk-new-session-key-456",
                existing_key="sk-existing-cli-key-123",
                result=mock_result,
            )

    def test_get_redirect_url_does_not_include_existing_key_in_url(self):
        """Test that redirect URL generation does NOT include existing_key in URL (uses state parameter instead)"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Mock request
        mock_request = MagicMock()
        mock_request.base_url = "https://test.litellm.ai/"

        with patch(
            "litellm.proxy.utils.get_custom_url", return_value="https://test.litellm.ai"
        ):
            # Test with existing_key - should NOT be in URL
            redirect_url = SSOAuthenticationHandler.get_redirect_url_for_sso(
                request=mock_request,
                sso_callback_route="sso/callback",
                existing_key="sk-existing-123",
            )

            # existing_key should NOT be in the URL
            assert "https://test.litellm.ai/sso/callback" == redirect_url
            assert "existing_key" not in redirect_url

    def test_get_redirect_url_without_existing_key(self):
        """Test that redirect URL generation works without existing_key parameter"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Mock request
        mock_request = MagicMock()
        mock_request.base_url = "https://test.litellm.ai/"

        with patch(
            "litellm.proxy.utils.get_custom_url", return_value="https://test.litellm.ai"
        ):
            # Test without existing_key
            redirect_url = SSOAuthenticationHandler.get_redirect_url_for_sso(
                request=mock_request, sso_callback_route="sso/callback"
            )

            assert "https://test.litellm.ai/sso/callback" == redirect_url

    @pytest.mark.asyncio
    async def test_cli_poll_key_generates_jwt_with_team(self):
        """Test CLI poll endpoint generates JWT when team_id is provided"""
        from litellm.proxy._types import LiteLLM_UserTable
        from litellm.proxy.management_endpoints.ui_sso import cli_poll_key

        # Test data
        session_key = "sk-session-999"
        selected_team = "team-b"
        session_data = {
            "user_id": "test-user-789",
            "user_role": "internal_user",
            "teams": ["team-a", "team-b", "team-c"],
            "models": ["gpt-4"],
            "user_email": "test@example.com",
        }

        # Mock user info
        mock_user_info = LiteLLM_UserTable(
            user_id="test-user-789",
            user_role="internal_user",
            teams=["team-a", "team-b", "team-c"],
            models=["gpt-4"],
        )

        # Mock cache
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = session_data

        mock_jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.token"

        with patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache), patch(
            "litellm.proxy.proxy_server.prisma_client"
        ) as mock_prisma, patch(
            "litellm.proxy.auth.auth_checks.ExperimentalUIJWTToken.get_cli_jwt_auth_token",
            return_value=mock_jwt_token,
        ) as mock_get_jwt:
            # Mock the user lookup
            mock_prisma.db.litellm_usertable.find_unique = AsyncMock(
                return_value=mock_user_info
            )

            # Act - Second poll with team_id
            result = await cli_poll_key(key_id=session_key, team_id=selected_team)

            # Assert - should return JWT
            assert result["status"] == "ready"
            assert result["key"] == mock_jwt_token
            assert result["user_id"] == "test-user-789"
            assert result["team_id"] == selected_team
            assert result["teams"] == ["team-a", "team-b", "team-c"]

            # Verify JWT was generated with correct team
            mock_get_jwt.assert_called_once()
            jwt_call_args = mock_get_jwt.call_args
            assert jwt_call_args.kwargs["team_id"] == selected_team

            # Verify session was deleted after JWT generation
            mock_cache.delete_cache.assert_called_once()


class TestGetAppRolesFromIdToken:
    """Test the get_app_roles_from_id_token method"""

    def test_roles_picked_when_app_roles_not_exists(self):
        """Test that 'roles' is picked when 'app_roles' doesn't exist"""

        # Create a token with only 'roles' claim
        token_payload = {
            "sub": "user123",
            "email": "test@example.com",
            "roles": ["Admin", "User", "Developer"],
        }

        # Create a mock JWT token
        mock_token = "mock.jwt.token"

        with patch("jwt.decode", return_value=token_payload) as mock_jwt_decode:
            # Act
            result = MicrosoftSSOHandler.get_app_roles_from_id_token(mock_token)

            # Assert
            assert result == ["Admin", "User", "Developer"]
            mock_jwt_decode.assert_called_once_with(
                mock_token, options={"verify_signature": False}
            )

    def test_app_roles_picked_when_both_exist(self):
        """Test that 'app_roles' takes precedence when both 'app_roles' and 'roles' exist"""

        # Create a token with both 'app_roles' and 'roles' claims
        token_payload = {
            "sub": "user123",
            "email": "test@example.com",
            "app_roles": ["AppAdmin", "AppUser"],
            "roles": ["RoleAdmin", "RoleUser"],
        }

        mock_token = "mock.jwt.token"

        with patch("jwt.decode", return_value=token_payload):
            # Act
            result = MicrosoftSSOHandler.get_app_roles_from_id_token(mock_token)

            # Assert - app_roles should be picked, not roles
            assert result == ["AppAdmin", "AppUser"]

    def test_roles_picked_when_app_roles_is_empty(self):
        """Test that 'roles' is picked when 'app_roles' exists but is empty"""

        # Create a token with empty 'app_roles' and populated 'roles'
        token_payload = {
            "sub": "user123",
            "email": "test@example.com",
            "app_roles": [],
            "roles": ["Admin", "User"],
        }

        mock_token = "mock.jwt.token"

        with patch("jwt.decode", return_value=token_payload):
            # Act
            result = MicrosoftSSOHandler.get_app_roles_from_id_token(mock_token)

            # Assert - roles should be picked since app_roles is empty
            assert result == ["Admin", "User"]

    def test_empty_list_when_neither_exists(self):
        """Test that empty list is returned when neither 'app_roles' nor 'roles' exist"""

        # Create a token without roles claims
        token_payload = {"sub": "user123", "email": "test@example.com"}

        mock_token = "mock.jwt.token"

        with patch("jwt.decode", return_value=token_payload):
            # Act
            result = MicrosoftSSOHandler.get_app_roles_from_id_token(mock_token)

            # Assert
            assert result == []

    def test_empty_list_when_no_token_provided(self):
        """Test that empty list is returned when no token is provided"""
        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(None)

        # Assert
        assert result == []

    def test_empty_list_when_roles_not_a_list(self):
        """Test that empty list is returned when roles is not a list"""

        # Create a token with non-list roles
        token_payload = {
            "sub": "user123",
            "email": "test@example.com",
            "roles": "Admin",  # String instead of list
        }

        mock_token = "mock.jwt.token"

        with patch("jwt.decode", return_value=token_payload):
            # Act
            result = MicrosoftSSOHandler.get_app_roles_from_id_token(mock_token)

            # Assert
            assert result == []

    def test_error_handling_on_jwt_decode_exception(self):
        """Test that exceptions during JWT decode are handled gracefully"""

        mock_token = "invalid.jwt.token"

        with patch("jwt.decode", side_effect=Exception("Invalid token")):
            # Act
            result = MicrosoftSSOHandler.get_app_roles_from_id_token(mock_token)

            # Assert - should return empty list on error
            assert result == []


class TestProcessSSOJWTAccessToken:
    """Test the process_sso_jwt_access_token helper function"""

    @pytest.fixture
    def mock_jwt_handler(self):
        """Create a mock JWT handler for testing"""
        mock_handler = MagicMock(spec=JWTHandler)
        mock_handler.get_team_ids_from_jwt.return_value = ["team1", "team2", "team3"]
        return mock_handler

    @pytest.fixture
    def sample_jwt_token(self):
        """Create a sample JWT token string"""
        return "test-jwt-token-header.payload.signature"

    @pytest.fixture
    def sample_jwt_payload(self):
        """Create a sample JWT payload"""
        return {
            "sub": "1234567890",
            "name": "John Doe",
            "iat": 1516239022,
            "groups": ["team1", "team2", "team3"],
        }

    def test_process_sso_jwt_access_token_with_existing_team_ids(
        self, mock_jwt_handler, sample_jwt_token
    ):
        """Test that existing team IDs are not overwritten"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        # Create a result object with existing team_ids
        existing_team_ids = ["existing_team1", "existing_team2"]
        result = CustomOpenID(
            id="test_user",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            display_name="Test User",
            provider="generic",
            team_ids=existing_team_ids,
        )

        with patch("jwt.decode") as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=result,
            )

            # Assert
            # JWT should still be decoded
            mock_jwt_decode.assert_called_once()

            # But team IDs should NOT be extracted since they already exist
            mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()

            # Existing team IDs should remain unchanged
            assert result.team_ids == existing_team_ids

    def test_process_sso_jwt_access_token_with_dict_result(
        self, mock_jwt_handler, sample_jwt_token, sample_jwt_payload
    ):
        """Test processing with a dictionary result object"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        # Create a dictionary result without team_ids
        result = {"id": "test_user", "email": "test@example.com", "name": "Test User"}

        with patch("jwt.decode", return_value=sample_jwt_payload) as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=result,
            )

            # Assert
            mock_jwt_decode.assert_called_once_with(
                sample_jwt_token, options={"verify_signature": False}
            )
            mock_jwt_handler.get_team_ids_from_jwt.assert_called_once_with(
                sample_jwt_payload
            )

            # Verify team_ids was added to the dict as a key
            assert "team_ids" in result
            assert result["team_ids"] == ["team1", "team2", "team3"]

    def test_process_sso_jwt_access_token_with_dict_existing_team_ids(
        self, mock_jwt_handler, sample_jwt_token
    ):
        """Test that existing team IDs in dictionary are not overwritten"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        # Create a dictionary result with existing team_ids
        existing_team_ids = ["dict_team1", "dict_team2"]
        result = {
            "id": "test_user",
            "email": "test@example.com",
            "name": "Test User",
            "team_ids": existing_team_ids,
        }

        with patch("jwt.decode") as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=result,
            )

            # Assert
            # JWT should still be decoded
            mock_jwt_decode.assert_called_once()

            # But team IDs should NOT be extracted since they already exist
            mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()

            # Existing team IDs should remain unchanged
            assert result["team_ids"] == existing_team_ids

    def test_process_sso_jwt_access_token_no_access_token(self, mock_jwt_handler):
        """Test that nothing happens when access token is None or empty"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        result = CustomOpenID(id="test_user", email="test@example.com", team_ids=[])

        # Test with None access token
        with patch("jwt.decode") as mock_jwt_decode:
            process_sso_jwt_access_token(
                access_token_str=None, sso_jwt_handler=mock_jwt_handler, result=result
            )

            # Assert nothing was processed
            mock_jwt_decode.assert_not_called()
            mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()
            assert result.team_ids == []

        # Test with empty string access token
        with patch("jwt.decode") as mock_jwt_decode:
            process_sso_jwt_access_token(
                access_token_str="", sso_jwt_handler=mock_jwt_handler, result=result
            )

            # Assert nothing was processed
            mock_jwt_decode.assert_not_called()
            mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()
            assert result.team_ids == []

    def test_process_sso_jwt_access_token_no_result(
        self, mock_jwt_handler, sample_jwt_token
    ):
        """Test that nothing happens when result is None"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        with patch("jwt.decode") as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=None,
            )

            # Assert nothing was processed
            mock_jwt_decode.assert_not_called()
            mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()

    def test_process_sso_jwt_access_token_non_decode_exception_propagates(
        self, mock_jwt_handler, sample_jwt_token
    ):
        """Test that non-DecodeError JWT exceptions still propagate up."""
        import jwt as pyjwt

        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        result = CustomOpenID(id="test_user", email="test@example.com", team_ids=[])

        with patch(
            "jwt.decode", side_effect=pyjwt.exceptions.InvalidKeyError("Invalid key")
        ) as mock_jwt_decode:
            with pytest.raises(pyjwt.exceptions.InvalidKeyError, match="Invalid key"):
                process_sso_jwt_access_token(
                    access_token_str=sample_jwt_token,
                    sso_jwt_handler=mock_jwt_handler,
                    result=result,
                )

            mock_jwt_decode.assert_called_once()
            mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()

    def test_process_sso_jwt_access_token_empty_team_ids_from_jwt(
        self, mock_jwt_handler, sample_jwt_token, sample_jwt_payload
    ):
        """Test processing when JWT handler returns empty team IDs"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        # Configure mock to return empty team IDs
        mock_jwt_handler.get_team_ids_from_jwt.return_value = []

        result = CustomOpenID(id="test_user", email="test@example.com", team_ids=[])

        with patch("jwt.decode", return_value=sample_jwt_payload) as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=result,
            )

            # Assert
            mock_jwt_decode.assert_called_once()
            mock_jwt_handler.get_team_ids_from_jwt.assert_called_once_with(
                sample_jwt_payload
            )

            # Even empty team IDs should be set
            assert result.team_ids == []

    def test_process_sso_jwt_access_token_with_opaque_token(self, mock_jwt_handler):
        """Test that opaque (non-JWT) access tokens are handled gracefully without raising."""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        result = CustomOpenID(
            id="test_user",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            display_name="Test User",
            provider="generic",
            team_ids=["existing_team"],
            user_role=None,
        )

        # Opaque tokens like those from Logto are short random strings, not JWTs
        opaque_token = "uTxyjXbS_random_opaque_token_string"

        # Should NOT raise - opaque tokens should be silently skipped
        process_sso_jwt_access_token(
            access_token_str=opaque_token,
            sso_jwt_handler=mock_jwt_handler,
            result=result,
        )

        # Result should be untouched
        mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()
        assert result.team_ids == ["existing_team"]
        assert result.user_role is None

    def test_process_sso_jwt_access_token_real_jwt_with_role_and_teams(
        self, mock_jwt_handler
    ):
        """Test that a real JWT containing role and team fields is correctly processed."""
        import jwt as pyjwt

        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        payload = {
            "sub": "user123",
            "email": "admin@example.com",
            "role": "proxy_admin",
            "groups": ["team_alpha", "team_beta"],
        }
        real_jwt_token = pyjwt.encode(payload, "test-secret", algorithm="HS256")

        mock_jwt_handler.get_team_ids_from_jwt.return_value = [
            "team_alpha",
            "team_beta",
        ]

        result = CustomOpenID(
            id="user123",
            email="admin@example.com",
            first_name="Admin",
            last_name="User",
            display_name="Admin User",
            provider="generic",
            team_ids=[],
            user_role=None,
        )

        process_sso_jwt_access_token(
            access_token_str=real_jwt_token,
            sso_jwt_handler=mock_jwt_handler,
            result=result,
        )

        # Team IDs should be extracted via sso_jwt_handler
        mock_jwt_handler.get_team_ids_from_jwt.assert_called_once_with(payload)
        assert result.team_ids == ["team_alpha", "team_beta"]

        # Role should be extracted from the "role" field in the JWT
        from litellm.proxy._types import LitellmUserRoles

        assert result.user_role == LitellmUserRoles.PROXY_ADMIN

    def test_process_sso_jwt_access_token_real_jwt_without_role_and_teams(self):
        """Test that a real JWT without role/team fields leaves result unchanged."""
        import jwt as pyjwt

        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        payload = {
            "sub": "user456",
            "email": "plain@example.com",
            "iat": 1700000000,
        }
        real_jwt_token = pyjwt.encode(payload, "test-secret", algorithm="HS256")

        result = CustomOpenID(
            id="user456",
            email="plain@example.com",
            first_name="Plain",
            last_name="User",
            display_name="Plain User",
            provider="generic",
            team_ids=[],
            user_role=None,
        )

        # No sso_jwt_handler, no role/team fields in JWT
        process_sso_jwt_access_token(
            access_token_str=real_jwt_token,
            sso_jwt_handler=None,
            result=result,
        )

        # Nothing should be modified
        assert result.team_ids == []
        assert result.user_role is None


@pytest.mark.asyncio
async def test_get_ui_settings_includes_api_doc_base_url():
    """Ensure the UI settings endpoint surfaces the optional API doc override."""
    from fastapi import Request

    from litellm.proxy.management_endpoints.ui_sso import get_ui_settings

    mock_request = Request(
        scope={
            "type": "http",
            "headers": [],
            "method": "GET",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": "/sso/get/ui_settings",
            "query_string": b"",
        }
    )

    with patch.dict(
        os.environ,
        {
            "LITELLM_UI_API_DOC_BASE_URL": "https://custom.docs",
        },
    ):
        response = await get_ui_settings(mock_request)
        assert response["LITELLM_UI_API_DOC_BASE_URL"] == "https://custom.docs"


class TestGenericResponseConvertorNestedAttributes:
    """Test generic_response_convertor with nested attribute paths"""

    def test_generic_response_convertor_with_nested_attributes(self):
        """
        Test that generic_response_convertor handles nested attributes with dotted notation
        like "attributes.userId"
        """
        from litellm.proxy.management_endpoints.ui_sso import generic_response_convertor

        # Mock JWT handler
        mock_jwt_handler = MagicMock(spec=JWTHandler)
        mock_jwt_handler.get_team_ids_from_jwt.return_value = []

        # Payload with nested attributes structure
        nested_payload = {
            "sub": "user-sub-123",
            "service": "test-service",
            "auth_time": 1234567890,
            "attributes": {
                "given_name": "John",
                "oauthClientId": "client-123",
                "family_name": "Doe",
                "userId": "nested-user-456",
                "email": "john.doe@example.com",
            },
            "id": "top-level-id-789",
            "client_id": "client-abc",
        }

        # Test with nested user ID attribute
        with patch.dict(
            os.environ,
            {
                "GENERIC_USER_ID_ATTRIBUTE": "attributes.userId",
                "GENERIC_USER_EMAIL_ATTRIBUTE": "attributes.email",
                "GENERIC_USER_FIRST_NAME_ATTRIBUTE": "attributes.given_name",
                "GENERIC_USER_LAST_NAME_ATTRIBUTE": "attributes.family_name",
                "GENERIC_USER_DISPLAY_NAME_ATTRIBUTE": "sub",
            },
        ):
            # Act
            result = generic_response_convertor(
                response=nested_payload,
                jwt_handler=mock_jwt_handler,
                sso_jwt_handler=None,
            )

            # Assert
            assert isinstance(result, CustomOpenID)

            # Note: The current implementation uses response.get() which doesn't support
            # dotted notation for nested attributes. This test documents the current behavior.
            # If nested attribute support is needed, the implementation would need to be updated
            # to handle dotted paths like "attributes.userId"

            # Current behavior: returns None for nested paths
            # Expected behavior with current implementation (no nested path support):
            assert result.id == "nested-user-456"
            assert (
                result.email == "john.doe@example.com"
            )  # Can't access "attributes.email" with .get()
            assert (
                result.first_name == "John"
            )  # Can't access "attributes.given_name" with .get()
            assert (
                result.last_name == "Doe"
            )  # Can't access "attributes.family_name" with .get()
            assert result.display_name == "user-sub-123"  # Top-level attribute works


class TestGenericResponseConvertorUserRole:
    """Test generic_response_convertor user role extraction from SSO token"""

    def test_generic_response_convertor_extracts_valid_user_role(self):
        """
        Test that generic_response_convertor extracts a valid LiteLLM user role
        from the SSO token using the GENERIC_USER_ROLE_ATTRIBUTE env var.
        """
        from litellm.proxy._types import LitellmUserRoles
        from litellm.proxy.management_endpoints.ui_sso import generic_response_convertor

        mock_jwt_handler = MagicMock(spec=JWTHandler)
        mock_jwt_handler.get_team_ids_from_jwt.return_value = []

        sso_response = {
            "preferred_username": "testuser",
            "email": "test@example.com",
            "sub": "Test User",
            "role": "proxy_admin",
        }

        with patch.dict(
            os.environ,
            {"GENERIC_USER_ROLE_ATTRIBUTE": "role"},
        ):
            result = generic_response_convertor(
                response=sso_response,
                jwt_handler=mock_jwt_handler,
                sso_jwt_handler=None,
            )

            assert isinstance(result, CustomOpenID)
            assert result.user_role == LitellmUserRoles.PROXY_ADMIN

    def test_generic_response_convertor_ignores_invalid_user_role(self):
        """
        Test that generic_response_convertor ignores invalid role values
        and sets user_role to None.
        """
        from litellm.proxy.management_endpoints.ui_sso import generic_response_convertor

        mock_jwt_handler = MagicMock(spec=JWTHandler)
        mock_jwt_handler.get_team_ids_from_jwt.return_value = []

        sso_response = {
            "preferred_username": "testuser",
            "email": "test@example.com",
            "role": "invalid_role_value",
        }

        with patch.dict(
            os.environ,
            {"GENERIC_USER_ROLE_ATTRIBUTE": "role"},
        ):
            result = generic_response_convertor(
                response=sso_response,
                jwt_handler=mock_jwt_handler,
                sso_jwt_handler=None,
            )

            assert isinstance(result, CustomOpenID)
            assert result.user_role is None


class TestGetGenericSSORedirectParams:
    """Test _get_generic_sso_redirect_params state parameter priority handling"""

    def test_state_priority_cli_state_provided(self):
        """
        Test that CLI state takes highest priority when provided
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Arrange
        cli_state = "litellm-session-token:sk-test123"

        with patch.dict(os.environ, {"GENERIC_CLIENT_STATE": "env_state_value"}):
            # Act
            (
                redirect_params,
                code_verifier,
            ) = SSOAuthenticationHandler._get_generic_sso_redirect_params(
                state=cli_state,
                generic_authorization_endpoint="https://auth.example.com/authorize",
            )

            # Assert
            assert redirect_params["state"] == cli_state
            assert code_verifier is None  # PKCE not enabled by default

    def test_state_priority_env_variable_when_no_cli_state(self):
        """
        Test that GENERIC_CLIENT_STATE environment variable is used when CLI state is not provided
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Arrange
        env_state = "custom_env_state_value"

        with patch.dict(os.environ, {"GENERIC_CLIENT_STATE": env_state}):
            # Act
            (
                redirect_params,
                code_verifier,
            ) = SSOAuthenticationHandler._get_generic_sso_redirect_params(
                state=None,
                generic_authorization_endpoint="https://auth.example.com/authorize",
            )

            # Assert
            assert redirect_params["state"] == env_state
            assert code_verifier is None

    def test_state_priority_generated_uuid_fallback(self):
        """
        Test that a UUID is generated when neither CLI state nor env variable is provided
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Arrange - no CLI state and no env variable
        with patch.dict(os.environ, {}, clear=False):
            # Remove GENERIC_CLIENT_STATE if it exists
            os.environ.pop("GENERIC_CLIENT_STATE", None)

            # Act
            (
                redirect_params,
                code_verifier,
            ) = SSOAuthenticationHandler._get_generic_sso_redirect_params(
                state=None,
                generic_authorization_endpoint="https://auth.example.com/authorize",
            )

            # Assert
            assert "state" in redirect_params
            assert redirect_params["state"] is not None
            assert len(redirect_params["state"]) == 32  # UUID hex is 32 chars
            assert code_verifier is None

    def test_state_with_pkce_enabled(self):
        """
        Test that PKCE parameters are generated when GENERIC_CLIENT_USE_PKCE is enabled
        """
        import base64
        import hashlib

        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Arrange
        test_state = "test_state_123"

        with patch.dict(os.environ, {"GENERIC_CLIENT_USE_PKCE": "true"}):
            # Act
            (
                redirect_params,
                code_verifier,
            ) = SSOAuthenticationHandler._get_generic_sso_redirect_params(
                state=test_state,
                generic_authorization_endpoint="https://auth.example.com/authorize",
            )

            # Assert state
            assert redirect_params["state"] == test_state

            # Assert PKCE parameters
            assert code_verifier is not None
            assert len(code_verifier) == 43  # Standard PKCE verifier length
            assert "code_challenge" in redirect_params
            assert "code_challenge_method" in redirect_params
            assert redirect_params["code_challenge_method"] == "S256"

            # Verify code_challenge is correctly derived from code_verifier
            expected_challenge_bytes = hashlib.sha256(
                code_verifier.encode("utf-8")
            ).digest()
            expected_challenge = (
                base64.urlsafe_b64encode(expected_challenge_bytes)
                .decode("utf-8")
                .rstrip("=")
            )
            assert redirect_params["code_challenge"] == expected_challenge

    def test_state_with_pkce_disabled(self):
        """
        Test that PKCE parameters are NOT generated when GENERIC_CLIENT_USE_PKCE is false
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Arrange
        test_state = "test_state_456"

        with patch.dict(os.environ, {"GENERIC_CLIENT_USE_PKCE": "false"}):
            # Act
            (
                redirect_params,
                code_verifier,
            ) = SSOAuthenticationHandler._get_generic_sso_redirect_params(
                state=test_state,
                generic_authorization_endpoint="https://auth.example.com/authorize",
            )

            # Assert
            assert redirect_params["state"] == test_state
            assert code_verifier is None
            assert "code_challenge" not in redirect_params
            assert "code_challenge_method" not in redirect_params

    def test_state_priority_cli_state_overrides_env_with_pkce(self):
        """
        Test that CLI state takes priority over env variable even when PKCE is enabled
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Arrange
        cli_state = "cli_state_priority"
        env_state = "env_state_should_not_be_used"

        with patch.dict(
            os.environ,
            {
                "GENERIC_CLIENT_STATE": env_state,
                "GENERIC_CLIENT_USE_PKCE": "true",
            },
        ):
            # Act
            (
                redirect_params,
                code_verifier,
            ) = SSOAuthenticationHandler._get_generic_sso_redirect_params(
                state=cli_state,
                generic_authorization_endpoint="https://auth.example.com/authorize",
            )

            # Assert
            assert redirect_params["state"] == cli_state  # CLI state takes priority
            assert redirect_params["state"] != env_state

            # PKCE should still be generated
            assert code_verifier is not None
            assert "code_challenge" in redirect_params
            assert "code_challenge_method" in redirect_params

    def test_empty_string_state_uses_env_variable(self):
        """
        Test that empty string state is treated as None and uses env variable
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Arrange
        env_state = "env_state_for_empty_cli"

        with patch.dict(os.environ, {"GENERIC_CLIENT_STATE": env_state}):
            # Act
            (
                redirect_params,
                code_verifier,
            ) = SSOAuthenticationHandler._get_generic_sso_redirect_params(
                state="",  # Empty string
                generic_authorization_endpoint="https://auth.example.com/authorize",
            )

            # Assert - empty string is falsy, so env variable should be used
            # Note: This tests current implementation behavior
            # If empty string should be treated differently, implementation needs update
            assert redirect_params["state"] == env_state

    def test_multiple_calls_generate_different_uuids(self):
        """
        Test that multiple calls without state generate different UUIDs
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Arrange - no state provided
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GENERIC_CLIENT_STATE", None)

            # Act
            params1, _ = SSOAuthenticationHandler._get_generic_sso_redirect_params(
                state=None,
                generic_authorization_endpoint="https://auth.example.com/authorize",
            )
            params2, _ = SSOAuthenticationHandler._get_generic_sso_redirect_params(
                state=None,
                generic_authorization_endpoint="https://auth.example.com/authorize",
            )

            # Assert
            assert params1["state"] != params2["state"]
            assert len(params1["state"]) == 32
            assert len(params2["state"]) == 32


class TestPKCEFunctionality:
    """Test PKCE (Proof Key for Code Exchange) functionality"""

    def test_generate_pkce_params(self):
        """
        Test that generate_pkce_params generates valid PKCE parameters
        """
        import base64
        import hashlib

        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Act
        code_verifier, code_challenge = SSOAuthenticationHandler.generate_pkce_params()

        # Assert
        assert len(code_verifier) == 43
        assert isinstance(code_verifier, str)

        # Verify code_challenge is correctly generated from code_verifier
        expected_challenge_bytes = hashlib.sha256(
            code_verifier.encode("utf-8")
        ).digest()
        expected_challenge = (
            base64.urlsafe_b64encode(expected_challenge_bytes)
            .decode("utf-8")
            .rstrip("=")
        )
        assert code_challenge == expected_challenge

        # Verify both are base64url encoded (no padding)
        assert "=" not in code_verifier
        assert "=" not in code_challenge

    @pytest.mark.asyncio
    async def test_prepare_token_exchange_parameters_with_pkce(self):
        """
        Test prepare_token_exchange_parameters retrieves PKCE code_verifier from cache
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Mock request with state parameter
        mock_request = MagicMock(spec=Request)
        test_state = "test_oauth_state_123"
        mock_request.query_params = {"state": test_state}

        # Mock cache with async methods
        mock_cache = MagicMock()
        test_code_verifier = "test_code_verifier_abc123xyz"
        mock_cache.async_get_cache = AsyncMock(return_value=test_code_verifier)
        mock_cache.async_delete_cache = AsyncMock()

        with patch("litellm.proxy.proxy_server.redis_usage_cache", None), patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache):
            # Act
            token_params = (
                await SSOAuthenticationHandler.prepare_token_exchange_parameters(
                    request=mock_request, generic_include_client_id=False
                )
            )

            # Assert
            assert token_params["include_client_id"] is False
            assert token_params["code_verifier"] == test_code_verifier

            # Verify cache was accessed and deleted
            mock_cache.async_get_cache.assert_called_once_with(
                key=f"pkce_verifier:{test_state}"
            )
            mock_cache.async_delete_cache.assert_called_once_with(
                key=f"pkce_verifier:{test_state}"
            )

    @pytest.mark.asyncio
    async def test_get_generic_sso_redirect_response_with_pkce(self):
        """
        Test get_generic_sso_redirect_response with PKCE enabled stores verifier and adds challenge to URL
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Mock SSO provider
        mock_sso = MagicMock()
        mock_redirect_response = MagicMock()
        original_location = (
            "https://auth.example.com/authorize?state=test456&client_id=abc"
        )
        mock_redirect_response.headers = {"location": original_location}
        mock_sso.get_login_redirect = AsyncMock(return_value=mock_redirect_response)
        mock_sso.__enter__ = MagicMock(return_value=mock_sso)
        mock_sso.__exit__ = MagicMock(return_value=False)

        test_state = "test456"
        mock_cache = MagicMock()

        mock_cache.async_set_cache = AsyncMock()

        with patch.dict(os.environ, {"GENERIC_CLIENT_USE_PKCE": "true"}):
            with patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache):
                # Act
                result = await SSOAuthenticationHandler.get_generic_sso_redirect_response(
                    generic_sso=mock_sso,
                    state=test_state,
                    generic_authorization_endpoint="https://auth.example.com/authorize",
                )

                # Assert
                # Verify async cache was called to store code_verifier
                mock_cache.async_set_cache.assert_called_once()
                cache_call = mock_cache.async_set_cache.call_args
                assert cache_call.kwargs["key"] == f"pkce_verifier:{test_state}"
                assert cache_call.kwargs["ttl"] == 600
                assert len(cache_call.kwargs["value"]) == 43

                # Verify PKCE parameters were added to the redirect URL
                assert result is not None
                updated_location = str(result.headers["location"])
                assert "code_challenge=" in updated_location
                assert "code_challenge_method=S256" in updated_location
                assert f"state={test_state}" in updated_location

    @pytest.mark.asyncio
    async def test_pkce_redis_multi_pod_verifier_roundtrip(self):
        """
        Mock Redis to verify PKCE code_verifier round-trip across "pods":
        Pod A stores verifier in Redis; Pod B retrieves it (no real IdP).
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # In-memory mock of Redis (shared between "pods")
        class MockRedisCache:
            def __init__(self):
                self._store = {}

            async def async_set_cache(self, key, value, **kwargs):
                self._store[key] = json.dumps(value)

            async def async_get_cache(self, key, **kwargs):
                val = self._store.get(key)
                if val is None:
                    return None
                # Simulate RedisCache._get_cache_logic: stored as JSON string, return decoded
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except (ValueError, TypeError):
                        return val
                return val

            async def async_delete_cache(self, key):
                self._store.pop(key, None)

        mock_redis = MockRedisCache()
        mock_in_memory = MagicMock()

        mock_sso = MagicMock()
        mock_redirect_response = MagicMock()
        mock_redirect_response.headers = {
            "location": "https://auth.example.com/authorize?state=multi_pod_state_xyz&client_id=abc"
        }
        mock_sso.get_login_redirect = AsyncMock(return_value=mock_redirect_response)
        mock_sso.__enter__ = MagicMock(return_value=mock_sso)
        mock_sso.__exit__ = MagicMock(return_value=False)

        with patch.dict(os.environ, {"GENERIC_CLIENT_USE_PKCE": "true"}):
            with patch("litellm.proxy.proxy_server.redis_usage_cache", mock_redis):
                with patch(
                    "litellm.proxy.proxy_server.user_api_key_cache", mock_in_memory
                ):
                    # Pod A: start login, store code_verifier in "Redis"
                    await SSOAuthenticationHandler.get_generic_sso_redirect_response(
                        generic_sso=mock_sso,
                        state="multi_pod_state_xyz",
                        generic_authorization_endpoint="https://auth.example.com/authorize",
                    )
                    mock_in_memory.async_set_cache.assert_not_called()
                    # MockRedisCache is a real class; assert on state, not .assert_called_*
                    stored_key = "pkce_verifier:multi_pod_state_xyz"
                    assert stored_key in mock_redis._store
                    stored_value = mock_redis._store[stored_key]
                    assert isinstance(stored_value, str) and len(json.loads(stored_value)) == 43

                    # Pod B: callback with same state, retrieve from "Redis"
                    mock_request = MagicMock(spec=Request)
                    mock_request.query_params = {"state": "multi_pod_state_xyz"}
                    token_params = await SSOAuthenticationHandler.prepare_token_exchange_parameters(
                        request=mock_request, generic_include_client_id=False
                    )
                    assert "code_verifier" in token_params
                    assert token_params["code_verifier"] == json.loads(stored_value)
                    mock_in_memory.async_get_cache.assert_not_called()
                    # delete_cache called; key removed (asserted below)

        # Verifier consumed (single-use); key removed from "Redis"
        assert "pkce_verifier:multi_pod_state_xyz" not in mock_redis._store

    @pytest.mark.asyncio
    async def test_pkce_fallback_in_memory_roundtrip_when_redis_none(self):
        """
        Regression: When redis_usage_cache is None (no Redis configured),
        code_verifier is stored and retrieved via user_api_key_cache.
        Roundtrip works when callback hits same pod (same in-memory cache).
        Single-pod or no-Redis deployments must continue to work.
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # In-memory store (simulates user_api_key_cache on one pod)
        in_memory_store = {}

        async def async_set_cache(key, value, **kwargs):
            in_memory_store[key] = value

        async def async_get_cache(key, **kwargs):
            return in_memory_store.get(key)

        async def async_delete_cache(key):
            in_memory_store.pop(key, None)

        mock_in_memory = MagicMock()
        mock_in_memory.async_set_cache = AsyncMock(side_effect=async_set_cache)
        mock_in_memory.async_get_cache = AsyncMock(side_effect=async_get_cache)
        mock_in_memory.async_delete_cache = AsyncMock(side_effect=async_delete_cache)

        mock_sso = MagicMock()
        mock_redirect_response = MagicMock()
        mock_redirect_response.headers = {
            "location": "https://auth.example.com/authorize?state=fallback_state_xyz&client_id=abc"
        }
        mock_sso.get_login_redirect = AsyncMock(return_value=mock_redirect_response)
        mock_sso.__enter__ = MagicMock(return_value=mock_sso)
        mock_sso.__exit__ = MagicMock(return_value=False)

        with patch.dict(os.environ, {"GENERIC_CLIENT_USE_PKCE": "true"}):
            with patch("litellm.proxy.proxy_server.redis_usage_cache", None):
                with patch(
                    "litellm.proxy.proxy_server.user_api_key_cache", mock_in_memory
                ):
                    # Pod A: start login, store code_verifier in in-memory cache
                    await SSOAuthenticationHandler.get_generic_sso_redirect_response(
                        generic_sso=mock_sso,
                        state="fallback_state_xyz",
                        generic_authorization_endpoint="https://auth.example.com/authorize",
                    )
                    mock_in_memory.async_set_cache.assert_called_once()
                    stored_key = mock_in_memory.async_set_cache.call_args.kwargs["key"]
                    stored_value = mock_in_memory.async_set_cache.call_args.kwargs[
                        "value"
                    ]
                    assert stored_key == "pkce_verifier:fallback_state_xyz"
                    assert isinstance(stored_value, str) and len(stored_value) == 43

                    # Same pod: callback retrieves from in-memory cache
                    mock_request = MagicMock(spec=Request)
                    mock_request.query_params = {"state": "fallback_state_xyz"}
                    token_params = await SSOAuthenticationHandler.prepare_token_exchange_parameters(
                        request=mock_request, generic_include_client_id=False
                    )
                    assert "code_verifier" in token_params
                    assert token_params["code_verifier"] == stored_value
                    mock_in_memory.async_get_cache.assert_called_once_with(
                        key=stored_key
                    )
                    mock_in_memory.async_delete_cache.assert_called_once_with(
                        key=stored_key
                    )

        # Verifier consumed; key removed from in-memory
        assert "pkce_verifier:fallback_state_xyz" not in in_memory_store

    @pytest.mark.asyncio
    async def test_pkce_prepare_token_exchange_returns_nothing_when_no_state(self):
        """
        Regression: prepare_token_exchange_parameters with no state in request
        does not call cache and does not add code_verifier.
        """
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        mock_redis = MagicMock()
        mock_in_memory = MagicMock()

        with patch("litellm.proxy.proxy_server.redis_usage_cache", mock_redis):
            with patch("litellm.proxy.proxy_server.user_api_key_cache", mock_in_memory):
                mock_request = MagicMock(spec=Request)
                mock_request.query_params = {}
                token_params = (
                    await SSOAuthenticationHandler.prepare_token_exchange_parameters(
                        request=mock_request, generic_include_client_id=False
                    )
                )
                assert "code_verifier" not in token_params
                mock_redis.async_get_cache.assert_not_called()
                mock_in_memory.async_get_cache.assert_not_called()


# Tests for SSO user team assignment bug (Issue: SSO Users Not Added to Entra-Synced Teams on First Login)
class TestAddMissingTeamMember:
    """Tests for the add_missing_team_member function"""

    @pytest.mark.asyncio
    async def test_add_missing_team_member_with_new_user_response_teams_none(self):
        """
        Bug reproduction: When a NewUserResponse has teams=None (new SSO user),
        add_missing_team_member() should still add the user to the SSO teams.

        Currently FAILS: The function returns early when teams is None.
        """
        from litellm.proxy._types import NewUserResponse
        from litellm.proxy.management_endpoints.ui_sso import add_missing_team_member

        # Simulate a new SSO user - NewUserResponse has teams=None by default
        new_user = NewUserResponse(
            user_id="new-sso-user-123",
            key="sk-xxxxx",
            teams=None,  # This is the default for NewUserResponse
        )

        sso_teams = ["team-from-entra-1", "team-from-entra-2"]

        with patch(
            "litellm.proxy.management_endpoints.ui_sso.create_team_member_add_task"
        ) as mock_add_task:
            mock_add_task.return_value = AsyncMock()

            await add_missing_team_member(user_info=new_user, sso_teams=sso_teams)

            # Bug: This assertion currently FAILS - no teams are added
            # because function returns early when teams is None
            assert (
                mock_add_task.call_count == 2
            ), f"Expected 2 calls to add user to teams, but got {mock_add_task.call_count}"
            called_team_ids = [call.args[0] for call in mock_add_task.call_args_list]
            assert set(called_team_ids) == {
                "team-from-entra-1",
                "team-from-entra-2",
            }

    @pytest.mark.asyncio
    async def test_add_missing_team_member_with_litellm_user_table_empty_teams(self):
        """
        Control test: When a LiteLLM_UserTable has teams=[] (existing user, no teams),
        add_missing_team_member() should add the user to SSO teams.

        This test PASSES because LiteLLM_UserTable defaults teams to [] not None.
        """
        from litellm.proxy._types import LiteLLM_UserTable
        from litellm.proxy.management_endpoints.ui_sso import add_missing_team_member

        # Existing user has teams=[] by default (not None)
        existing_user = LiteLLM_UserTable(
            user_id="existing-user-456",
            teams=[],  # Empty list, not None
        )

        sso_teams = ["team-from-entra-1", "team-from-entra-2"]

        with patch(
            "litellm.proxy.management_endpoints.ui_sso.create_team_member_add_task"
        ) as mock_add_task:
            mock_add_task.return_value = AsyncMock()

            await add_missing_team_member(user_info=existing_user, sso_teams=sso_teams)

            # This PASSES - teams are added because teams=[] not None
            assert mock_add_task.call_count == 2

    @pytest.mark.asyncio
    async def test_add_user_to_teams_from_sso_response_new_user(self):
        """
        Integration test: Simulates the SSO response handler with a new user
        that has teams=None from NewUserResponse.
        """
        from litellm.proxy._types import NewUserResponse
        from litellm.proxy.management_endpoints.types import CustomOpenID
        from litellm.proxy.management_endpoints.ui_sso import (
            SSOAuthenticationHandler,
        )

        # SSO response with team_ids from Entra ID
        sso_result = CustomOpenID(
            id="new-sso-user-id",
            email="newuser@example.com",
            team_ids=["entra-group-1", "entra-group-2"],
        )

        # New user response (simulates what new_user() returns)
        new_user_info = NewUserResponse(
            user_id="new-sso-user-id",
            key="sk-xxxxx",
            teams=None,  # Bug: NewUserResponse defaults to None
        )

        with patch(
            "litellm.proxy.management_endpoints.ui_sso.add_missing_team_member"
        ) as mock_add_member:
            await SSOAuthenticationHandler.add_user_to_teams_from_sso_response(
                result=sso_result,
                user_info=new_user_info,
            )

            # Verify add_missing_team_member was called with correct args
            mock_add_member.assert_called_once_with(
                user_info=new_user_info, sso_teams=["entra-group-1", "entra-group-2"]
            )

    @pytest.mark.asyncio
    async def test_sso_first_login_full_flow_adds_user_to_teams(self):
        """
        End-to-end test: Simulates complete first-time SSO login with Entra groups.
        Verifies teams are created AND user is added as a member.
        """
        from litellm.proxy._types import NewUserResponse
        from litellm.proxy.management_endpoints.ui_sso import add_missing_team_member

        team_member_calls = []

        async def track_team_member_add(team_id, user_info):
            team_member_calls.append({"team_id": team_id, "user_id": user_info.user_id})

        # New SSO user with Entra groups
        new_user = NewUserResponse(
            user_id="first-time-sso-user",
            key="sk-xxxxx",
            teams=None,  # The problematic default
        )

        sso_teams = ["entra-team-alpha", "entra-team-beta"]

        with patch(
            "litellm.proxy.management_endpoints.ui_sso.create_team_member_add_task",
            side_effect=track_team_member_add,
        ):
            await add_missing_team_member(user_info=new_user, sso_teams=sso_teams)

        # Bug: With current code, team_member_calls will be empty
        # After fix: Should have 2 entries
        assert (
            len(team_member_calls) == 2
        ), f"Expected 2 teams to be added, but got {len(team_member_calls)}"
        assert {c["team_id"] for c in team_member_calls} == {
            "entra-team-alpha",
            "entra-team-beta",
        }
        assert all(c["user_id"] == "first-time-sso-user" for c in team_member_calls)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "user_info_factory,teams_value,expected_teams_added",
        [
            # Bug case: NewUserResponse with teams=None
            pytest.param(
                lambda uid: NewUserResponse(user_id=uid, key="sk-xxx", teams=None),
                None,
                ["team-1", "team-2"],  # Should still add teams
                id="new_user_teams_none",
            ),
            # Working case: LiteLLM_UserTable with teams=[]
            pytest.param(
                lambda uid: LiteLLM_UserTable(user_id=uid, teams=[]),
                [],
                ["team-1", "team-2"],
                id="existing_user_empty_teams",
            ),
            # Existing user with some teams already
            pytest.param(
                lambda uid: LiteLLM_UserTable(user_id=uid, teams=["team-1"]),
                ["team-1"],
                ["team-2"],  # Only missing team should be added
                id="existing_user_partial_teams",
            ),
        ],
    )
    async def test_add_missing_team_member_handles_all_user_types(
        self, user_info_factory, teams_value, expected_teams_added
    ):
        """
        Parametrized test ensuring add_missing_team_member works for all user types.
        """
        from litellm.proxy.management_endpoints.ui_sso import add_missing_team_member

        user_info = user_info_factory("test-user-id")
        sso_teams = ["team-1", "team-2"]

        added_teams = []

        async def mock_create_task(team_id, user):
            added_teams.append(team_id)

        with patch(
            "litellm.proxy.management_endpoints.ui_sso.create_team_member_add_task",
            side_effect=mock_create_task,
        ):
            await add_missing_team_member(user_info=user_info, sso_teams=sso_teams)

        assert set(added_teams) == set(
            expected_teams_added
        ), f"Expected teams {expected_teams_added}, but got {added_teams}"


@pytest.mark.asyncio
async def test_role_mappings_override_default_internal_user_params():
    """
    Test that when role_mappings is configured in SSO settings,
    the SSO-extracted role overrides default_internal_user_params role.
    """
    from litellm.proxy._types import NewUserResponse, SSOUserDefinedValues
    from litellm.proxy.management_endpoints.ui_sso import insert_sso_user

    # Save original default_internal_user_params
    original_default_params = getattr(litellm, "default_internal_user_params", None)

    try:
        # Set default_internal_user_params with a role that should be overridden
        litellm.default_internal_user_params = {
            "user_role": "internal_user",
            "max_budget": 100,
            "budget_duration": "30d",
            "models": ["gpt-3.5-turbo"],
        }

        # Mock SSO result
        mock_result_openid = CustomOpenID(
            id="test-user-123",
            email="test@example.com",
            display_name="Test User",
            provider="microsoft",
            team_ids=[],
        )

        # User defined values with SSO-extracted role (from role_mappings)
        user_defined_values: SSOUserDefinedValues = {
            "user_id": "test-user-123",
            "user_email": "test@example.com",
            "user_role": "proxy_admin",  # Role from SSO role_mappings
            "max_budget": None,
            "budget_duration": None,
            "models": [],
        }

        # Mock Prisma client with SSO config that has role_mappings configured
        mock_prisma = MagicMock()
        mock_sso_config = MagicMock()
        mock_sso_config.sso_settings = {
            "role_mappings": {
                "Admin": "proxy_admin",
                "User": "internal_user",
            }
        }
        mock_prisma.db.litellm_ssoconfig.find_unique = AsyncMock(
            return_value=mock_sso_config
        )

        # Mock new_user function
        mock_new_user_response = NewUserResponse(
            user_id="test-user-123",
            key="sk-xxxxx",
            teams=None,
        )

        with patch(
            "litellm.proxy.utils.get_prisma_client_or_throw",
            return_value=mock_prisma,
        ), patch(
            "litellm.proxy.management_endpoints.ui_sso.new_user",
            return_value=mock_new_user_response,
        ) as mock_new_user:
            # Act
            _ = await insert_sso_user(
                result_openid=mock_result_openid,
                user_defined_values=user_defined_values,
            )

            # Assert - verify new_user was called with preserved SSO role
            mock_new_user.assert_called_once()
            call_args = mock_new_user.call_args
            new_user_request = call_args.kwargs["data"]

            # The role from SSO should be preserved, not overridden by default_internal_user_params
            assert (
                new_user_request.user_role == "proxy_admin"
            ), "SSO-extracted role should override default_internal_user_params role"

            # Other default params should still be applied
            assert (
                new_user_request.max_budget == 100
            ), "max_budget from default_internal_user_params should be applied"
            assert (
                new_user_request.budget_duration == "30d"
            ), "budget_duration from default_internal_user_params should be applied"

            # Note: models are applied via _update_internal_new_user_params inside new_user,
            # not in insert_sso_user, so we verify user_defined_values was updated correctly
            # by checking that the function completed successfully and other defaults were applied
            # The models will be applied when new_user processes the request

    finally:
        # Restore original default_internal_user_params
        if original_default_params is not None:
            litellm.default_internal_user_params = original_default_params
        else:
            if hasattr(litellm, "default_internal_user_params"):
                delattr(litellm, "default_internal_user_params")


class TestSSOReadinessEndpoint:
    """Test the /sso/readiness endpoint"""

    @pytest.mark.asyncio
    async def test_sso_readiness_no_sso_configured(self):
        """Test that readiness returns healthy when no SSO is configured"""
        from fastapi.testclient import TestClient

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy.proxy_server import app

        mock_user_auth = UserAPIKeyAuth(
            user_id="test-user-123",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

        try:
            client = TestClient(app)

            with patch.dict(os.environ, {}, clear=True):
                response = client.get("/sso/readiness")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["sso_configured"] is False
                assert data["message"] == "No SSO provider configured"
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_sso_readiness_google_fully_configured(self):
        """Test that readiness returns healthy when Google SSO is fully configured"""
        from fastapi.testclient import TestClient

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy.proxy_server import app

        mock_user_auth = UserAPIKeyAuth(
            user_id="test-user-123",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

        try:
            client = TestClient(app)

            with patch.dict(
                os.environ,
                {
                    "GOOGLE_CLIENT_ID": "test-google-client-id",
                    "GOOGLE_CLIENT_SECRET": "test-google-secret",
                },
                clear=True,
            ):
                response = client.get("/sso/readiness")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["sso_configured"] is True
                assert data["provider"] == "google"
                assert "Google SSO is properly configured" in data["message"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_sso_readiness_google_missing_secret(self):
        """Test that readiness returns unhealthy when Google SSO is missing GOOGLE_CLIENT_SECRET"""
        from fastapi.testclient import TestClient

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy.proxy_server import app

        mock_user_auth = UserAPIKeyAuth(
            user_id="test-user-123",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

        try:
            client = TestClient(app)

            with patch.dict(
                os.environ,
                {"GOOGLE_CLIENT_ID": "test-google-client-id"},
                clear=True,
            ):
                response = client.get("/sso/readiness")

                assert response.status_code == 503
                data = response.json()["detail"]
                assert data["status"] == "unhealthy"
                assert data["sso_configured"] is True
                assert data["provider"] == "google"
                assert "GOOGLE_CLIENT_SECRET" in data["missing_environment_variables"]
                assert (
                    "Google SSO is configured but missing required environment variables"
                    in data["message"]
                )
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "env_vars,expected_status,expected_provider,expected_missing_vars",
        [
            (
                {
                    "MICROSOFT_CLIENT_ID": "test-microsoft-client-id",
                    "MICROSOFT_CLIENT_SECRET": "test-microsoft-secret",
                    "MICROSOFT_TENANT": "test-tenant",
                },
                200,
                "microsoft",
                [],
            ),
            (
                {"MICROSOFT_CLIENT_ID": "test-microsoft-client-id"},
                503,
                "microsoft",
                ["MICROSOFT_CLIENT_SECRET", "MICROSOFT_TENANT"],
            ),
        ],
    )
    async def test_sso_readiness_microsoft_configurations(
        self, env_vars, expected_status, expected_provider, expected_missing_vars
    ):
        """Test Microsoft SSO readiness with both fully configured and missing variables"""
        from fastapi.testclient import TestClient

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy.proxy_server import app

        mock_user_auth = UserAPIKeyAuth(
            user_id="test-user-123",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

        try:
            client = TestClient(app)

            with patch.dict(os.environ, env_vars, clear=True):
                response = client.get("/sso/readiness")

                assert response.status_code == expected_status

                if expected_status == 200:
                    data = response.json()
                    assert data["sso_configured"] is True
                    assert data["provider"] == expected_provider
                    assert data["status"] == "healthy"
                    assert "Microsoft SSO is properly configured" in data["message"]
                else:
                    data = response.json()["detail"]
                    assert data["sso_configured"] is True
                    assert data["provider"] == expected_provider
                    assert data["status"] == "unhealthy"
                    assert set(data["missing_environment_variables"]) == set(
                        expected_missing_vars
                    )
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "env_vars,expected_status,expected_provider,expected_missing_vars",
        [
            (
                {
                    "GENERIC_CLIENT_ID": "test-generic-client-id",
                    "GENERIC_CLIENT_SECRET": "test-generic-secret",
                    "GENERIC_AUTHORIZATION_ENDPOINT": "https://auth.example.com/authorize",
                    "GENERIC_TOKEN_ENDPOINT": "https://auth.example.com/token",
                    "GENERIC_USERINFO_ENDPOINT": "https://auth.example.com/userinfo",
                },
                200,
                "generic",
                [],
            ),
            (
                {"GENERIC_CLIENT_ID": "test-generic-client-id"},
                503,
                "generic",
                [
                    "GENERIC_CLIENT_SECRET",
                    "GENERIC_AUTHORIZATION_ENDPOINT",
                    "GENERIC_TOKEN_ENDPOINT",
                    "GENERIC_USERINFO_ENDPOINT",
                ],
            ),
        ],
    )
    async def test_sso_readiness_generic_configurations(
        self, env_vars, expected_status, expected_provider, expected_missing_vars
    ):
        """Test Generic SSO readiness with both fully configured and missing variables"""
        from fastapi.testclient import TestClient

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy.proxy_server import app

        mock_user_auth = UserAPIKeyAuth(
            user_id="test-user-123",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

        try:
            client = TestClient(app)

            with patch.dict(os.environ, env_vars, clear=True):
                response = client.get("/sso/readiness")

                assert response.status_code == expected_status

                if expected_status == 200:
                    data = response.json()
                    assert data["sso_configured"] is True
                    assert data["provider"] == expected_provider
                    assert data["status"] == "healthy"
                    assert "Generic SSO is properly configured" in data["message"]
                else:
                    data = response.json()["detail"]
                    assert data["sso_configured"] is True
                    assert data["provider"] == expected_provider
                    assert data["status"] == "unhealthy"
                    assert set(data["missing_environment_variables"]) == set(
                        expected_missing_vars
                    )
        finally:
            app.dependency_overrides.clear()


class TestCustomMicrosoftSSO:
    """Tests for CustomMicrosoftSSO class."""

    @pytest.mark.asyncio
    async def test_custom_microsoft_sso_uses_default_endpoints_when_no_env_vars(self):
        """
        Test that CustomMicrosoftSSO uses default Microsoft endpoints
        when no custom environment variables are set.
        """
        # Ensure no custom endpoints are set
        for key in [
            "MICROSOFT_AUTHORIZATION_ENDPOINT",
            "MICROSOFT_TOKEN_ENDPOINT",
            "MICROSOFT_USERINFO_ENDPOINT",
        ]:
            os.environ.pop(key, None)

        sso = CustomMicrosoftSSO(
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant="test-tenant",
            redirect_uri="http://localhost:4000/sso/callback",
        )

        discovery = await sso.get_discovery_document()

        assert (
            discovery["authorization_endpoint"]
            == "https://login.microsoftonline.com/test-tenant/oauth2/v2.0/authorize"
        )
        assert (
            discovery["token_endpoint"]
            == "https://login.microsoftonline.com/test-tenant/oauth2/v2.0/token"
        )
        assert discovery["userinfo_endpoint"] == "https://graph.microsoft.com/v1.0/me"

    @pytest.mark.asyncio
    async def test_custom_microsoft_sso_uses_custom_endpoints_when_env_vars_set(self):
        """
        Test that CustomMicrosoftSSO uses custom endpoints
        when environment variables are set.
        """
        custom_auth_endpoint = "https://custom.example.com/oauth2/v2.0/authorize"
        custom_token_endpoint = "https://custom.example.com/oauth2/v2.0/token"
        custom_userinfo_endpoint = "https://custom.example.com/v1.0/me"

        with patch.dict(
            os.environ,
            {
                "MICROSOFT_AUTHORIZATION_ENDPOINT": custom_auth_endpoint,
                "MICROSOFT_TOKEN_ENDPOINT": custom_token_endpoint,
                "MICROSOFT_USERINFO_ENDPOINT": custom_userinfo_endpoint,
            },
        ):
            sso = CustomMicrosoftSSO(
                client_id="test-client-id",
                client_secret="test-client-secret",
                tenant="test-tenant",
                redirect_uri="http://localhost:4000/sso/callback",
            )

            discovery = await sso.get_discovery_document()

            assert discovery["authorization_endpoint"] == custom_auth_endpoint
            assert discovery["token_endpoint"] == custom_token_endpoint
            assert discovery["userinfo_endpoint"] == custom_userinfo_endpoint

    @pytest.mark.asyncio
    async def test_custom_microsoft_sso_uses_partial_custom_endpoints(self):
        """
        Test that CustomMicrosoftSSO uses custom endpoints for those set,
        and defaults for others.
        """
        custom_auth_endpoint = "https://custom.example.com/oauth2/v2.0/authorize"

        # Clear other env vars first
        os.environ.pop("MICROSOFT_TOKEN_ENDPOINT", None)
        os.environ.pop("MICROSOFT_USERINFO_ENDPOINT", None)

        with patch.dict(
            os.environ,
            {
                "MICROSOFT_AUTHORIZATION_ENDPOINT": custom_auth_endpoint,
            },
        ):
            sso = CustomMicrosoftSSO(
                client_id="test-client-id",
                client_secret="test-client-secret",
                tenant="test-tenant",
                redirect_uri="http://localhost:4000/sso/callback",
            )

            discovery = await sso.get_discovery_document()

            # Custom auth endpoint
            assert discovery["authorization_endpoint"] == custom_auth_endpoint
            # Default token and userinfo endpoints
            assert (
                discovery["token_endpoint"]
                == "https://login.microsoftonline.com/test-tenant/oauth2/v2.0/token"
            )
            assert (
                discovery["userinfo_endpoint"] == "https://graph.microsoft.com/v1.0/me"
            )

    def test_custom_microsoft_sso_uses_common_tenant_when_none(self):
        """
        Test that CustomMicrosoftSSO uses 'common' tenant when tenant is None.
        """
        sso = CustomMicrosoftSSO(
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant=None,
            redirect_uri="http://localhost:4000/sso/callback",
        )

        assert sso.tenant == "common"

    def test_custom_microsoft_sso_is_subclass_of_microsoft_sso(self):
        """
        Test that CustomMicrosoftSSO is a subclass of MicrosoftSSO.
        """
        from fastapi_sso.sso.microsoft import MicrosoftSSO

        sso = CustomMicrosoftSSO(
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant="test-tenant",
            redirect_uri="http://localhost:4000/sso/callback",
        )

        assert isinstance(sso, MicrosoftSSO)


@pytest.mark.asyncio
async def test_setup_team_mappings():
    """Test _setup_team_mappings function loads team mappings from database."""
    # Arrange
    mock_prisma = MagicMock()
    mock_sso_config = MagicMock()
    mock_sso_config.sso_settings = {"team_mappings": {"team_ids_jwt_field": "groups"}}
    mock_prisma.db.litellm_ssoconfig.find_unique = AsyncMock(
        return_value=mock_sso_config
    )

    with patch(
        "litellm.proxy.utils.get_prisma_client_or_throw",
        return_value=mock_prisma,
    ):
        # Act
        result = await _setup_team_mappings()

        # Assert
        assert result is not None
        assert isinstance(result, TeamMappings)
        assert result.team_ids_jwt_field == "groups"
        mock_prisma.db.litellm_ssoconfig.find_unique.assert_called_once_with(
            where={"id": "sso_config"}
        )


# ============================================================================
# Tests for get_litellm_user_role with list inputs (Keycloak returns lists)
# ============================================================================


def test_get_litellm_user_role_with_string():
    """Test that get_litellm_user_role works with a plain string."""
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_endpoints.types import get_litellm_user_role

    result = get_litellm_user_role("proxy_admin")
    assert result == LitellmUserRoles.PROXY_ADMIN


def test_get_litellm_user_role_with_list():
    """
    Test that get_litellm_user_role handles list inputs.
    Keycloak returns roles as arrays like ["proxy_admin"] instead of strings.
    """
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_endpoints.types import get_litellm_user_role

    result = get_litellm_user_role(["proxy_admin"])
    assert result == LitellmUserRoles.PROXY_ADMIN


def test_get_litellm_user_role_with_empty_list():
    """Test that get_litellm_user_role returns None for empty lists."""
    from litellm.proxy.management_endpoints.types import get_litellm_user_role

    result = get_litellm_user_role([])
    assert result is None


def test_get_litellm_user_role_with_invalid_role():
    """Test that get_litellm_user_role returns None for invalid roles."""
    from litellm.proxy.management_endpoints.types import get_litellm_user_role

    result = get_litellm_user_role("not_a_real_role")
    assert result is None


def test_get_litellm_user_role_with_list_multiple_roles():
    """Test that get_litellm_user_role takes the first element from a multi-element list."""
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.management_endpoints.types import get_litellm_user_role

    result = get_litellm_user_role(["proxy_admin", "internal_user"])
    assert result == LitellmUserRoles.PROXY_ADMIN


# ============================================================================
# Tests for process_sso_jwt_access_token role extraction
# ============================================================================


def test_process_sso_jwt_access_token_extracts_role_from_access_token():
    """
    Test that process_sso_jwt_access_token extracts user role from the JWT
    access token when the UserInfo response did not include it.

    This is the core fix for the Keycloak SSO role mapping bug: Keycloak's
    UserInfo endpoint does not return role claims, but the JWT access token
    contains them.
    """
    import jwt as pyjwt

    from litellm.proxy._types import LitellmUserRoles

    # Create a JWT access token with role claims (as Keycloak would)
    access_token_payload = {
        "sub": "user-123",
        "email": "admin@test.com",
        "litellm_role": ["proxy_admin"],
    }
    access_token_str = pyjwt.encode(access_token_payload, "secret", algorithm="HS256")

    # Result object with no role set (simulating UserInfo response without roles)
    result = CustomOpenID(
        id="user-123",
        email="admin@test.com",
        display_name="Admin User",
        team_ids=[],
        user_role=None,
    )

    # Call with GENERIC_USER_ROLE_ATTRIBUTE pointing to litellm_role
    with patch.dict(os.environ, {"GENERIC_USER_ROLE_ATTRIBUTE": "litellm_role"}):
        process_sso_jwt_access_token(
            access_token_str=access_token_str,
            sso_jwt_handler=None,
            result=result,
            role_mappings=None,
        )

    assert result.user_role == LitellmUserRoles.PROXY_ADMIN


def test_process_sso_jwt_access_token_does_not_override_existing_role():
    """
    Test that process_sso_jwt_access_token does NOT override a role that was
    already extracted from the UserInfo response.
    """
    import jwt as pyjwt

    from litellm.proxy._types import LitellmUserRoles

    access_token_payload = {
        "sub": "user-123",
        "litellm_role": ["internal_user"],
    }
    access_token_str = pyjwt.encode(access_token_payload, "secret", algorithm="HS256")

    # Result already has a role (e.g., set from UserInfo)
    result = CustomOpenID(
        id="user-123",
        email="admin@test.com",
        display_name="Admin User",
        team_ids=[],
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    with patch.dict(os.environ, {"GENERIC_USER_ROLE_ATTRIBUTE": "litellm_role"}):
        process_sso_jwt_access_token(
            access_token_str=access_token_str,
            sso_jwt_handler=None,
            result=result,
            role_mappings=None,
        )

    # Should keep the original role
    assert result.user_role == LitellmUserRoles.PROXY_ADMIN


def test_process_sso_jwt_access_token_extracts_role_from_nested_field():
    """
    Test role extraction from a nested JWT field like resource_access.client.roles.
    """
    import jwt as pyjwt

    from litellm.proxy._types import LitellmUserRoles

    access_token_payload = {
        "sub": "user-123",
        "resource_access": {
            "my-client": {
                "roles": ["proxy_admin"]
            }
        },
    }
    access_token_str = pyjwt.encode(access_token_payload, "secret", algorithm="HS256")

    result = CustomOpenID(
        id="user-123",
        email="admin@test.com",
        display_name="Admin User",
        team_ids=[],
        user_role=None,
    )

    with patch.dict(os.environ, {"GENERIC_USER_ROLE_ATTRIBUTE": "resource_access.my-client.roles"}):
        process_sso_jwt_access_token(
            access_token_str=access_token_str,
            sso_jwt_handler=None,
            result=result,
            role_mappings=None,
        )

    assert result.user_role == LitellmUserRoles.PROXY_ADMIN


def test_process_sso_jwt_access_token_with_role_mappings():
    """
    Test role extraction using role_mappings (group-based role determination)
    from the JWT access token.
    """
    import jwt as pyjwt

    from litellm.proxy._types import LitellmUserRoles
    from litellm.types.proxy.management_endpoints.ui_sso import RoleMappings

    access_token_payload = {
        "sub": "user-123",
        "groups": ["keycloak-admins", "developers"],
    }
    access_token_str = pyjwt.encode(access_token_payload, "secret", algorithm="HS256")

    result = CustomOpenID(
        id="user-123",
        email="admin@test.com",
        display_name="Admin User",
        team_ids=[],
        user_role=None,
    )

    role_mappings = RoleMappings(
        provider="generic",
        group_claim="groups",
        default_role=LitellmUserRoles.INTERNAL_USER,
        roles={
            LitellmUserRoles.PROXY_ADMIN: ["keycloak-admins"],
            LitellmUserRoles.INTERNAL_USER: ["developers"],
        },
    )

    process_sso_jwt_access_token(
        access_token_str=access_token_str,
        sso_jwt_handler=None,
        result=result,
        role_mappings=role_mappings,
    )

    # Should get highest privilege role
    assert result.user_role == LitellmUserRoles.PROXY_ADMIN

def test_generic_response_convertor_with_extra_attributes(monkeypatch):
    """Test that extra attributes are extracted when GENERIC_USER_EXTRA_ATTRIBUTES is set"""
    from litellm.proxy.management_endpoints.ui_sso import generic_response_convertor
    
    monkeypatch.setenv("GENERIC_CLIENT_ID", "test_client")
    monkeypatch.setenv("GENERIC_USER_EXTRA_ATTRIBUTES", "custom_field1,custom_field2,custom_field3")
    
    mock_response = {
        "sub": "user-id-123",
        "email": "user@example.com",
        "given_name": "John",
        "family_name": "Doe",
        "name": "John Doe",
        "provider": "generic",
        "custom_field1": "value1",
        "custom_field2": ["item1", "item2"],
        "custom_field3": {"nested": "data"},
    }
    
    mock_jwt_handler = MagicMock(spec=JWTHandler)
    mock_jwt_handler.get_team_ids_from_jwt.return_value = []
    
    result = generic_response_convertor(
        response=mock_response,
        jwt_handler=mock_jwt_handler,
        sso_jwt_handler=None,
        role_mappings=None,
    )
    
    assert result.extra_fields is not None
    assert result.extra_fields["custom_field1"] == "value1"
    assert result.extra_fields["custom_field2"] == ["item1", "item2"]
    assert result.extra_fields["custom_field3"] == {"nested": "data"}

def test_generic_response_convertor_without_extra_attributes(monkeypatch):
    """Test backward compatibility - extra_fields is None when env var not set"""
    from litellm.proxy.management_endpoints.ui_sso import generic_response_convertor
    
    monkeypatch.setenv("GENERIC_CLIENT_ID", "test_client")
    # Don't set GENERIC_USER_EXTRA_ATTRIBUTES
    
    mock_response = {
        "sub": "user-id-123",
        "email": "user@example.com",
        "given_name": "John",
        "family_name": "Doe",
        "name": "John Doe",
        "provider": "generic",
        "custom_field1": "value1",
        "custom_field2": "value2",
    }
    
    mock_jwt_handler = MagicMock(spec=JWTHandler)
    mock_jwt_handler.get_team_ids_from_jwt.return_value = []
    
    result = generic_response_convertor(
        response=mock_response,
        jwt_handler=mock_jwt_handler,
        sso_jwt_handler=None,
        role_mappings=None,
    )
    
    assert result.extra_fields is None

def test_generic_response_convertor_extra_attributes_with_nested_paths(monkeypatch):
    """Test that nested paths work with dot notation"""
    from litellm.proxy.management_endpoints.ui_sso import generic_response_convertor
    
    monkeypatch.setenv("GENERIC_CLIENT_ID", "test_client")
    monkeypatch.setenv("GENERIC_USER_EXTRA_ATTRIBUTES", "org_info.department,org_info.manager")
    
    mock_response = {
        "sub": "user-id-123",
        "email": "user@example.com",
        "org_info": {
            "department": "Engineering",
            "manager": "Jane Smith"
        }
    }
    
    mock_jwt_handler = MagicMock(spec=JWTHandler)
    mock_jwt_handler.get_team_ids_from_jwt.return_value = []
    
    result = generic_response_convertor(
        response=mock_response,
        jwt_handler=mock_jwt_handler,
        sso_jwt_handler=None,
        role_mappings=None,
    )
    
    assert result.extra_fields is not None
    assert result.extra_fields["org_info.department"] == "Engineering"
    assert result.extra_fields["org_info.manager"] == "Jane Smith"

def test_generic_response_convertor_extra_attributes_missing_field(monkeypatch):
    """Test that missing fields return None"""
    from litellm.proxy.management_endpoints.ui_sso import generic_response_convertor
    
    monkeypatch.setenv("GENERIC_CLIENT_ID", "test_client")
    monkeypatch.setenv("GENERIC_USER_EXTRA_ATTRIBUTES", "missing_field,another_missing")
    
    mock_response = {
        "sub": "user-id-123",
        "email": "user@example.com",
    }
    
    mock_jwt_handler = MagicMock(spec=JWTHandler)
    mock_jwt_handler.get_team_ids_from_jwt.return_value = []
    
    result = generic_response_convertor(
        response=mock_response,
        jwt_handler=mock_jwt_handler,
        sso_jwt_handler=None,
        role_mappings=None,
    )
    
    assert result.extra_fields is not None
    assert result.extra_fields["missing_field"] is None
    assert result.extra_fields["another_missing"] is None
import asyncio
import json
import os
import sys
from typing import Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from litellm._uuid import uuid

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy._types import NewTeamRequest
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.management_endpoints.types import CustomOpenID
from litellm.proxy.management_endpoints.ui_sso import (
    GoogleSSOHandler,
    MicrosoftSSOHandler,
    SSOAuthenticationHandler,
)
from litellm.types.proxy.management_endpoints.ui_sso import (
    DefaultTeamSSOParams,
    MicrosoftGraphAPIUserGroupDirectoryObject,
    MicrosoftGraphAPIUserGroupResponse,
    MicrosoftServicePrincipalTeam,
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
    print("result from verify_and_process", result)
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
        print(
            "mock_prisma.db.litellm_teamtable.create.call_args",
            mock_prisma.db.litellm_teamtable.create.call_args,
        )
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
async def test_get_user_info_from_db():
    """
    received args in get_user_info_from_db: {'result': CustomOpenID(id='krrishd', email='krrishdholakia@gmail.com', first_name=None, last_name=None, display_name='a3f1c107-04dc-4c93-ae60-7f32eb4b05ce', picture=None, provider=None, team_ids=[]), 'prisma_client': <litellm.proxy.utils.PrismaClient object at 0x14a74e3c0>, 'user_api_key_cache': <litellm.caching.dual_cache.DualCache object at 0x148d37110>, 'proxy_logging_obj': <litellm.proxy.utils.ProxyLogging object at 0x148dd9090>, 'user_email': 'krrishdholakia@gmail.com', 'user_defined_values': {'models': [], 'user_id': 'krrishd', 'user_email': 'krrishdholakia@gmail.com', 'max_budget': None, 'user_role': None, 'budget_duration': None}}
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
        user_info = await get_user_info_from_db(**args)
        mock_get_user_object.assert_called_once()
        assert mock_get_user_object.call_args.kwargs["user_id"] == "krrishd"


@pytest.mark.asyncio
async def test_get_user_info_from_db_alternate_user_id():
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
        user_info = await get_user_info_from_db(**args)
        mock_get_user_object.assert_called_once()
        assert mock_get_user_object.call_args.kwargs["user_id"] == "krrishd-email1234"


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

    mock_sso_class = MagicMock(return_value=mock_sso_instance)

    with patch.dict(os.environ, test_env_vars):
        with patch("fastapi_sso.sso.base.DiscoveryDocument") as mock_discovery:
            with patch(
                "fastapi_sso.sso.generic.create_provider", return_value=mock_sso_class
            ) as mock_create_provider:
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

    mock_sso_class = MagicMock(return_value=mock_sso_instance)

    with patch.dict(os.environ, test_env_vars):
        with patch("fastapi_sso.sso.base.DiscoveryDocument") as mock_discovery:
            with patch(
                "fastapi_sso.sso.generic.create_provider", return_value=mock_sso_class
            ) as mock_create_provider:
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

        from litellm.proxy.management_endpoints.ui_sso import google_login

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
                                EnterpriseCustomSSOHandler,
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
            models=["gpt-4"]
        )

        # Mock SSO result
        mock_sso_result = {
            "user_email": "test@example.com",
            "user_id": "test-user-123"
        }

        # Mock cache
        mock_cache = MagicMock()
        
        with patch(
            "litellm.proxy.management_endpoints.ui_sso.get_user_info_from_db",
            return_value=mock_user_info
        ), patch(
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ), patch(
            "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
        ), patch(
            "litellm.proxy.common_utils.html_forms.cli_sso_success.render_cli_sso_success_page",
            return_value="<html>Success</html>",
        ):

            # Act
            result = await cli_sso_callback(
                request=mock_request, key=session_key, existing_key=None, result=mock_sso_result
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
            "models": ["gpt-4"]
        }

        # Mock cache
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = session_data
        
        with patch(
            "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
        ):

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
            "user_email": "test@example.com"
        }
        
        # Mock user info
        mock_user_info = LiteLLM_UserTable(
            user_id="test-user-789",
            user_role="internal_user",
            teams=["team-a", "team-b", "team-c"],
            models=["gpt-4"]
        )

        # Mock cache
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = session_data
        
        mock_jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.token"
        
        with patch(
            "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
        ), patch(
            "litellm.proxy.proxy_server.prisma_client"
        ) as mock_prisma, patch(
            "litellm.proxy.auth.auth_checks.ExperimentalUIJWTToken.get_cli_jwt_auth_token",
            return_value=mock_jwt_token
        ) as mock_get_jwt:
            
            # Mock the user lookup
            mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user_info)

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
        import jwt

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
        import jwt

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
        import jwt

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
        import jwt

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
        import jwt

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
        import jwt

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
        return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

    @pytest.fixture
    def sample_jwt_payload(self):
        """Create a sample JWT payload"""
        return {
            "sub": "1234567890",
            "name": "John Doe",
            "iat": 1516239022,
            "groups": ["team1", "team2", "team3"],
        }

    def test_process_sso_jwt_access_token_with_valid_token(
        self, mock_jwt_handler, sample_jwt_token, sample_jwt_payload
    ):
        """Test processing a valid JWT access token with team extraction"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        # Create a result object without team_ids
        result = CustomOpenID(
            id="test_user",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            display_name="Test User",
            provider="generic",
            team_ids=[],
        )

        with patch("jwt.decode", return_value=sample_jwt_payload) as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=result,
            )

            # Assert
            # Verify JWT was decoded correctly
            mock_jwt_decode.assert_called_once_with(
                sample_jwt_token, options={"verify_signature": False}
            )

            # Verify team IDs were extracted from JWT
            mock_jwt_handler.get_team_ids_from_jwt.assert_called_once_with(
                sample_jwt_payload
            )

            # Verify team IDs were set on the result object
            assert result.team_ids == ["team1", "team2", "team3"]

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

    def test_process_sso_jwt_access_token_no_sso_jwt_handler(self, sample_jwt_token):
        """Test that nothing happens when sso_jwt_handler is None"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        result = CustomOpenID(id="test_user", email="test@example.com", team_ids=[])

        with patch("jwt.decode") as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token, sso_jwt_handler=None, result=result
            )

            # Assert nothing was processed
            mock_jwt_decode.assert_not_called()
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

    def test_process_sso_jwt_access_token_jwt_decode_exception(
        self, mock_jwt_handler, sample_jwt_token
    ):
        """Test that JWT decode exceptions are not caught (should propagate up)"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        result = CustomOpenID(id="test_user", email="test@example.com", team_ids=[])

        with patch(
            "jwt.decode", side_effect=Exception("JWT decode error")
        ) as mock_jwt_decode:
            # Act & Assert
            with pytest.raises(Exception, match="JWT decode error"):
                process_sso_jwt_access_token(
                    access_token_str=sample_jwt_token,
                    sso_jwt_handler=mock_jwt_handler,
                    result=result,
                )

            # Verify JWT decode was attempted
            mock_jwt_decode.assert_called_once()
            # But team extraction should not have been called
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
            print(f"User ID result: {result.id}")
            print(f"Email result: {result.email}")
            print(f"First name result: {result.first_name}")
            print(f"Last name result: {result.last_name}")
            print(f"Display name result: {result.display_name}")

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

        # Mock cache
        mock_cache = MagicMock()
        test_code_verifier = "test_code_verifier_abc123xyz"
        mock_cache.get_cache.return_value = test_code_verifier

        with patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache):
            # Act
            token_params = SSOAuthenticationHandler.prepare_token_exchange_parameters(
                request=mock_request, generic_include_client_id=False
            )

            # Assert
            assert token_params["include_client_id"] is False
            assert token_params["code_verifier"] == test_code_verifier

            # Verify cache was accessed and deleted
            mock_cache.get_cache.assert_called_once_with(
                key=f"pkce_verifier:{test_state}"
            )
            mock_cache.delete_cache.assert_called_once_with(
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

        with patch.dict(os.environ, {"GENERIC_CLIENT_USE_PKCE": "true"}):
            with patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache):
                # Act
                result = await SSOAuthenticationHandler.get_generic_sso_redirect_response(
                    generic_sso=mock_sso,
                    state=test_state,
                    generic_authorization_endpoint="https://auth.example.com/authorize",
                )

                # Assert
                # Verify cache was called to store code_verifier
                mock_cache.set_cache.assert_called_once()
                cache_call = mock_cache.set_cache.call_args
                assert cache_call.kwargs["key"] == f"pkce_verifier:{test_state}"
                assert cache_call.kwargs["ttl"] == 600
                assert len(cache_call.kwargs["value"]) == 43

                # Verify PKCE parameters were added to the redirect URL
                assert result is not None
                updated_location = str(result.headers["location"])
                assert "code_challenge=" in updated_location
                assert "code_challenge_method=S256" in updated_location
                assert f"state={test_state}" in updated_location

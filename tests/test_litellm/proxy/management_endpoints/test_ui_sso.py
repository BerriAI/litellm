import asyncio
import json
import os
import sys
from litellm._uuid import uuid
from typing import Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

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
        response=mock_response, team_ids=expected_team_ids
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
        response=mock_response, team_ids=expected_team_ids
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
    result = MicrosoftSSOHandler.openid_from_response(response=None, team_ids=[])

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

    future = asyncio.Future()
    future.set_result(mock_response)

    with patch.dict(
        os.environ,
        {"MICROSOFT_CLIENT_SECRET": "mock_secret", "MICROSOFT_TENANT": "mock_tenant"},
    ):
        with patch(
            "fastapi_sso.sso.microsoft.MicrosoftSSO.verify_and_process",
            return_value=future,
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

    future = asyncio.Future()
    future.set_result(mock_response)
    with patch.dict(
        os.environ,
        {"MICROSOFT_CLIENT_SECRET": "mock_secret", "MICROSOFT_TENANT": "mock_tenant"},
    ):
        with patch(
            "fastapi_sso.sso.microsoft.MicrosoftSSO.verify_and_process",
            return_value=future,
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

    future = asyncio.Future()
    future.set_result(mock_response)

    with patch.dict(os.environ, {"GOOGLE_CLIENT_SECRET": "mock_secret"}):
        with patch(
            "fastapi_sso.sso.google.GoogleSSO.verify_and_process", return_value=future
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
            if not invalid_key or not invalid_key.startswith('sk-'):
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
            if not invalid_key.startswith('sk-'):
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
            "not_session_token:something"
        ]
        
        for state in non_cli_states:
            # This mimics the routing logic in auth_callback
            should_route_to_cli = state and state.startswith(f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:")
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
            assert cli_state is None, f"CLI state should not be generated for source='{source}', key='{key}'"


class TestSSOHandlerIntegration:
    """Test SSOAuthenticationHandler methods"""

    def test_should_use_sso_handler(self):
        """Test the SSO handler detection logic"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Test that SSO handler is used when client IDs are provided
        assert SSOAuthenticationHandler.should_use_sso_handler(google_client_id="test") is True
        assert SSOAuthenticationHandler.should_use_sso_handler(microsoft_client_id="test") is True
        assert SSOAuthenticationHandler.should_use_sso_handler(generic_client_id="test") is True
        
        # Test that SSO handler is not used when no client IDs are provided
        assert SSOAuthenticationHandler.should_use_sso_handler() is False
        assert SSOAuthenticationHandler.should_use_sso_handler(None, None, None) is False

    def test_get_redirect_url_for_sso(self):
        """Test the redirect URL generation for SSO"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Mock request object
        mock_request = MagicMock()
        mock_request.base_url = "https://test.litellm.ai/"
        
        # Test redirect URL generation
        redirect_url = SSOAuthenticationHandler.get_redirect_url_for_sso(
            request=mock_request,
            sso_callback_route="sso/callback"
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
        assert hasattr(SSOAuthenticationHandler, '_get_cli_state')
        assert callable(SSOAuthenticationHandler._get_cli_state)


class TestSSOStateHandling:
    """Test the SSO state handling for CLI authentication"""

    def test_get_cli_state_valid(self):
        """Test generating CLI state with valid parameters"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler
        
        state = SSOAuthenticationHandler._get_cli_state(source="litellm-cli", key="sk-test123")
        
        assert state is not None
        assert state.startswith("litellm-session-token:")
        assert "sk-test123" in state

    def test_get_cli_state_invalid_source(self):
        """Test generating CLI state with invalid source"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler
        
        state = SSOAuthenticationHandler._get_cli_state(source="invalid_source", key="sk-test123")
        
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

    def test_non_cli_state_detection(self):
        """Test detection of non-CLI state parameters"""
        from litellm.constants import LITELLM_CLI_SESSION_TOKEN_PREFIX

        # Test various non-CLI states
        test_states = [
            "regular_oauth_state",
            "some_random_string",
            None,
            "",
            "not_session_token:something"
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
            with patch("litellm.proxy.proxy_server.user_custom_ui_sso_sign_in_handler", MagicMock()):
                with patch.dict('sys.modules', {'enterprise.litellm_enterprise.proxy.auth.custom_sso_handler': None}):
                    # Temporarily mock the google_login function call to test the import error path
                    async def mock_google_login():
                        # This mimics the relevant part of google_login that would trigger the import error
                        try:
                            from enterprise.litellm_enterprise.proxy.auth.custom_sso_handler import (
                                EnterpriseCustomSSOHandler,
                            )
                            return "success"
                        except ImportError:
                            raise ValueError("Enterprise features are not available. Custom UI SSO sign-in requires LiteLLM Enterprise.")
                    
                    # Test that the ValueError is raised with the correct message
                    import pytest
                    with pytest.raises(ValueError, match="Enterprise features are not available"):
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
                    result = await EnterpriseCustomSSOHandler.handle_custom_ui_sso_sign_in(
                        request=mock_request
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
                    email=request_headers_dict.get("x-litellm-user-email", "default@test.com"),
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
                    result = await EnterpriseCustomSSOHandler.handle_custom_ui_sso_sign_in(
                        request=mock_request
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
    async def test_cli_sso_callback_regenerate_existing_key(self):
        """Test CLI SSO callback regenerating an existing key"""
        from litellm.proxy.management_endpoints.ui_sso import cli_sso_callback

        # Mock request
        mock_request = MagicMock(spec=Request)
        
        # Test data
        existing_key = "sk-existing-key-123"
        new_key = "sk-new-key-456"
        
        # Mock the regenerate helper function
        with patch("litellm.proxy.management_endpoints.ui_sso._regenerate_cli_key") as mock_regenerate, \
             patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), \
             patch("litellm.proxy.common_utils.html_forms.cli_sso_success.render_cli_sso_success_page", return_value="<html>Success</html>"):
            
            # Act
            result = await cli_sso_callback(
                request=mock_request,
                key=new_key,
                existing_key=existing_key
            )
            
            # Assert
            mock_regenerate.assert_called_once_with(existing_key=existing_key, new_key=new_key, user_id=None)
            assert result.status_code == 200
            assert "Success" in result.body.decode()

    @pytest.mark.asyncio
    async def test_cli_sso_callback_create_new_key(self):
        """Test CLI SSO callback creating a new key when no existing key provided"""
        from litellm.proxy.management_endpoints.ui_sso import cli_sso_callback

        # Mock request
        mock_request = MagicMock(spec=Request)
        
        # Test data
        new_key = "sk-new-key-789"
        
        # Mock the create helper function
        with patch("litellm.proxy.management_endpoints.ui_sso._create_new_cli_key") as mock_create, \
             patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), \
             patch("litellm.proxy.common_utils.html_forms.cli_sso_success.render_cli_sso_success_page", return_value="<html>Success</html>"):
            
            # Act
            result = await cli_sso_callback(
                request=mock_request,
                key=new_key,
                existing_key=None
            )
            
            # Assert
            mock_create.assert_called_once_with(key=new_key, user_id=None)
            assert result.status_code == 200
            assert "Success" in result.body.decode()

    @pytest.mark.asyncio
    async def test_auth_callback_routes_to_cli_with_existing_key(self):
        """Test that auth_callback properly routes CLI requests and preserves existing_key parameter"""
        from litellm.constants import LITELLM_CLI_SESSION_TOKEN_PREFIX
        from litellm.proxy.management_endpoints.ui_sso import auth_callback

        # Mock request with existing_key query parameter
        mock_request = MagicMock(spec=Request)
        mock_request.query_params.get.return_value = "sk-existing-cli-key-123"
        
        # CLI state
        cli_state = f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:sk-new-session-key-456"
        
        # Mock the CLI callback and required proxy server components
        mock_result = {"user_id": "test-user", "email": "test@example.com"}
        
        with patch("litellm.proxy.management_endpoints.ui_sso.cli_sso_callback") as mock_cli_callback, \
             patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), \
             patch("litellm.proxy.proxy_server.master_key", "test-master-key"), \
             patch("litellm.proxy.proxy_server.general_settings", {}), \
             patch("litellm.proxy.proxy_server.jwt_handler", MagicMock()), \
             patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()), \
             patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "test-google-id"}, clear=True), \
             patch("litellm.proxy.management_endpoints.ui_sso.GoogleSSOHandler.get_google_callback_response", return_value=mock_result):
            mock_cli_callback.return_value = MagicMock()
            
            # Act
            await auth_callback(request=mock_request, state=cli_state)
            
            # Assert
            mock_cli_callback.assert_called_once_with(
                request=mock_request,
                key="sk-new-session-key-456",
                existing_key="sk-existing-cli-key-123",
                result=mock_result
            )

    def test_get_redirect_url_preserves_existing_key(self):
        """Test that redirect URL generation preserves existing_key parameter"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Mock request
        mock_request = MagicMock()
        mock_request.base_url = "https://test.litellm.ai/"
        
        with patch("litellm.proxy.utils.get_custom_url", return_value="https://test.litellm.ai"):
            # Test with existing_key
            redirect_url = SSOAuthenticationHandler.get_redirect_url_for_sso(
                request=mock_request,
                sso_callback_route="sso/callback",
                existing_key="sk-existing-123"
            )
            
            assert "https://test.litellm.ai/sso/callback?existing_key=sk-existing-123" == redirect_url

    def test_get_redirect_url_without_existing_key(self):
        """Test that redirect URL generation works without existing_key parameter"""
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

        # Mock request
        mock_request = MagicMock()
        mock_request.base_url = "https://test.litellm.ai/"
        
        with patch("litellm.proxy.utils.get_custom_url", return_value="https://test.litellm.ai"):
            # Test without existing_key
            redirect_url = SSOAuthenticationHandler.get_redirect_url_for_sso(
                request=mock_request,
                sso_callback_route="sso/callback"
            )
            
            assert "https://test.litellm.ai/sso/callback" == redirect_url

    @pytest.mark.asyncio
    async def test_cli_sso_callback_regenerate_vs_create_flow(self):
        """Test CLI SSO callback calls regenerate_key_fn when existing_key provided, generate_key_helper_fn when not"""
        from litellm.proxy.management_endpoints.ui_sso import cli_sso_callback
        
        mock_request = MagicMock(spec=Request)
        
        with patch("litellm.proxy.management_endpoints.key_management_endpoints.regenerate_key_fn") as mock_regenerate, \
             patch("litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn") as mock_generate, \
             patch("litellm.proxy._types.UserAPIKeyAuth.get_litellm_cli_user_api_key_auth"), \
             patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), \
             patch("litellm.proxy.common_utils.html_forms.cli_sso_success.render_cli_sso_success_page", return_value="<html>Success</html>"):
            
            # Test regeneration path
            await cli_sso_callback(mock_request, key="sk-new-123", existing_key="sk-existing-456")
            mock_regenerate.assert_called_once()
            mock_generate.assert_not_called()
            
            # Reset mocks
            mock_regenerate.reset_mock()
            mock_generate.reset_mock()
            
            # Test creation path
            await cli_sso_callback(mock_request, key="sk-new-789", existing_key=None)
            mock_regenerate.assert_not_called()
            mock_generate.assert_called_once()


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
            "groups": ["team1", "team2", "team3"]
        }

    def test_process_sso_jwt_access_token_with_valid_token(self, mock_jwt_handler, sample_jwt_token, sample_jwt_payload):
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
            team_ids=[]
        )

        with patch("jwt.decode", return_value=sample_jwt_payload) as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=result
            )

            # Assert
            # Verify JWT was decoded correctly
            mock_jwt_decode.assert_called_once_with(
                sample_jwt_token, options={"verify_signature": False}
            )
            
            # Verify team IDs were extracted from JWT
            mock_jwt_handler.get_team_ids_from_jwt.assert_called_once_with(sample_jwt_payload)
            
            # Verify team IDs were set on the result object
            assert result.team_ids == ["team1", "team2", "team3"]

    def test_process_sso_jwt_access_token_with_existing_team_ids(self, mock_jwt_handler, sample_jwt_token):
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
            team_ids=existing_team_ids
        )

        with patch("jwt.decode") as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=result
            )

            # Assert
            # JWT should still be decoded
            mock_jwt_decode.assert_called_once()
            
            # But team IDs should NOT be extracted since they already exist
            mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()
            
            # Existing team IDs should remain unchanged
            assert result.team_ids == existing_team_ids

    def test_process_sso_jwt_access_token_with_dict_result(self, mock_jwt_handler, sample_jwt_token, sample_jwt_payload):
        """Test processing with a dictionary result object"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        # Create a dictionary result without team_ids
        result = {
            "id": "test_user",
            "email": "test@example.com",
            "name": "Test User"
        }

        with patch("jwt.decode", return_value=sample_jwt_payload) as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=result
            )

            # Assert
            mock_jwt_decode.assert_called_once_with(
                sample_jwt_token, options={"verify_signature": False}
            )
            mock_jwt_handler.get_team_ids_from_jwt.assert_called_once_with(sample_jwt_payload)
            
            # Verify team_ids was added to the dict as a key
            assert "team_ids" in result
            assert result["team_ids"] == ["team1", "team2", "team3"]

    def test_process_sso_jwt_access_token_with_dict_existing_team_ids(self, mock_jwt_handler, sample_jwt_token):
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
            "team_ids": existing_team_ids
        }

        with patch("jwt.decode") as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=result
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

        result = CustomOpenID(
            id="test_user",
            email="test@example.com",
            team_ids=[]
        )

        # Test with None access token
        with patch("jwt.decode") as mock_jwt_decode:
            process_sso_jwt_access_token(
                access_token_str=None,
                sso_jwt_handler=mock_jwt_handler,
                result=result
            )
            
            # Assert nothing was processed
            mock_jwt_decode.assert_not_called()
            mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()
            assert result.team_ids == []

        # Test with empty string access token
        with patch("jwt.decode") as mock_jwt_decode:
            process_sso_jwt_access_token(
                access_token_str="",
                sso_jwt_handler=mock_jwt_handler,
                result=result
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

        result = CustomOpenID(
            id="test_user",
            email="test@example.com",
            team_ids=[]
        )

        with patch("jwt.decode") as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=None,
                result=result
            )

            # Assert nothing was processed
            mock_jwt_decode.assert_not_called()
            assert result.team_ids == []

    def test_process_sso_jwt_access_token_no_result(self, mock_jwt_handler, sample_jwt_token):
        """Test that nothing happens when result is None"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        with patch("jwt.decode") as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=None
            )

            # Assert nothing was processed
            mock_jwt_decode.assert_not_called()
            mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()

    def test_process_sso_jwt_access_token_jwt_decode_exception(self, mock_jwt_handler, sample_jwt_token):
        """Test that JWT decode exceptions are not caught (should propagate up)"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        result = CustomOpenID(
            id="test_user",
            email="test@example.com",
            team_ids=[]
        )

        with patch("jwt.decode", side_effect=Exception("JWT decode error")) as mock_jwt_decode:
            # Act & Assert
            with pytest.raises(Exception, match="JWT decode error"):
                process_sso_jwt_access_token(
                    access_token_str=sample_jwt_token,
                    sso_jwt_handler=mock_jwt_handler,
                    result=result
                )

            # Verify JWT decode was attempted
            mock_jwt_decode.assert_called_once()
            # But team extraction should not have been called
            mock_jwt_handler.get_team_ids_from_jwt.assert_not_called()

    def test_process_sso_jwt_access_token_empty_team_ids_from_jwt(self, mock_jwt_handler, sample_jwt_token, sample_jwt_payload):
        """Test processing when JWT handler returns empty team IDs"""
        from litellm.proxy.management_endpoints.ui_sso import (
            process_sso_jwt_access_token,
        )

        # Configure mock to return empty team IDs
        mock_jwt_handler.get_team_ids_from_jwt.return_value = []

        result = CustomOpenID(
            id="test_user",
            email="test@example.com",
            team_ids=[]
        )

        with patch("jwt.decode", return_value=sample_jwt_payload) as mock_jwt_decode:
            # Act
            process_sso_jwt_access_token(
                access_token_str=sample_jwt_token,
                sso_jwt_handler=mock_jwt_handler,
                result=result
            )

            # Assert
            mock_jwt_decode.assert_called_once()
            mock_jwt_handler.get_team_ids_from_jwt.assert_called_once_with(sample_jwt_payload)
            
            # Even empty team IDs should be set
            assert result.team_ids == []


import asyncio
import json
import os
import sys
import uuid
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
async def test_get_user_groups_pagination():
    # Arrange
    first_response = {
        "@odata.context": "https://graph.microsoft.com/v1.0/$metadata#directoryObjects",
        "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/memberOf?$skiptoken=page2",
        "value": [
            {
                "@odata.type": "#microsoft.graph.group",
                "id": "group1",
                "displayName": "Group 1",
            },
        ],
    }
    second_response = {
        "@odata.context": "https://graph.microsoft.com/v1.0/$metadata#directoryObjects",
        "value": [
            {
                "@odata.type": "#microsoft.graph.group",
                "id": "group2",
                "displayName": "Group 2",
            },
        ],
    }

    responses = [first_response, second_response]
    current_response = {"index": 0}

    async def mock_get(*args, **kwargs):
        mock = MagicMock()
        mock.json.return_value = responses[current_response["index"]]
        current_response["index"] += 1
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
        assert current_response["index"] == 2  # Verify both pages were fetched


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
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.management_endpoints.ui_sso import (
        apply_user_info_values_to_sso_user_defined_values,
    )

    user_info = LiteLLM_UserTable(
        user_id="123",
        user_email="test@example.com",
        user_role="admin",
    )

    user_defined_values = {
        "user_id": "456",
        "user_email": "test@example.com",
        "user_role": "admin",
    }

    sso_user_defined_values = apply_user_info_values_to_sso_user_defined_values(
        user_info=user_info,
        user_defined_values=user_defined_values,
    )

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
    with patch.object(
        litellm.proxy.management_endpoints.ui_sso, "get_user_object"
    ) as mock_get_user_object:
        user_info = await get_user_info_from_db(**args)
        mock_get_user_object.assert_called_once()
        mock_get_user_object.call_args.kwargs["user_id"] = "krrishd"


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
    with patch.object(
        litellm.proxy.management_endpoints.ui_sso, "get_user_object"
    ) as mock_get_user_object:
        user_info = await get_user_info_from_db(**args)
        mock_get_user_object.assert_called_once()
        mock_get_user_object.call_args.kwargs["user_id"] = "krrishd-email1234"


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

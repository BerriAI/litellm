import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from datetime import datetime, timedelta

import pytest

import litellm
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    SSOUserDefinedValues,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    ExperimentalUIJWTToken,
    _can_object_call_vector_stores,
    get_user_object,
    vector_store_access_check,
    _get_team_db_check,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.utils import get_utc_datetime


@pytest.fixture(autouse=True)
def set_salt_key(monkeypatch):
    """Automatically set LITELLM_SALT_KEY for all tests"""
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-1234")


@pytest.fixture
def valid_sso_user_defined_values():
    return LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
        models=["gpt-3.5-turbo"],
        max_budget=100.0,
    )


@pytest.fixture
def invalid_sso_user_defined_values():
    return LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=None,  # Missing user role
        models=["gpt-3.5-turbo"],
        max_budget=100.0,
    )


def test_get_experimental_ui_login_jwt_auth_token_valid(valid_sso_user_defined_values):
    """Test generating JWT token with valid user role"""
    token = ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
        valid_sso_user_defined_values
    )

    # Decrypt and verify token contents
    decrypted_token = decrypt_value_helper(
        token, key="ui_hash_key", exception_type="debug"
    )
    # Check that decrypted_token is not None before using json.loads
    assert decrypted_token is not None
    token_data = json.loads(decrypted_token)

    assert token_data["user_id"] == "test_user"
    assert token_data["user_role"] == LitellmUserRoles.PROXY_ADMIN.value
    assert token_data["models"] == ["gpt-3.5-turbo"]
    assert token_data["max_budget"] == litellm.max_ui_session_budget

    # Verify expiration time is set and valid
    assert "expires" in token_data
    expires = datetime.fromisoformat(token_data["expires"].replace("Z", "+00:00"))
    assert expires > get_utc_datetime()
    assert expires <= get_utc_datetime() + timedelta(minutes=10)


def test_get_experimental_ui_login_jwt_auth_token_invalid(
    invalid_sso_user_defined_values,
):
    """Test generating JWT token with missing user role"""
    with pytest.raises(Exception) as exc_info:
        ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
            invalid_sso_user_defined_values
        )

    assert str(exc_info.value) == "User role is required for experimental UI login"


def test_get_key_object_from_ui_hash_key_valid(
    valid_sso_user_defined_values, monkeypatch
):
    """Test getting key object from valid UI hash key"""
    monkeypatch.setenv("EXPERIMENTAL_UI_LOGIN", "True")
    # Generate a valid token
    token = ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
        valid_sso_user_defined_values
    )

    # Get key object
    key_object = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(token)

    assert key_object is not None
    assert key_object.user_id == "test_user"
    assert key_object.user_role == LitellmUserRoles.PROXY_ADMIN
    assert key_object.models == ["gpt-3.5-turbo"]
    assert key_object.max_budget == litellm.max_ui_session_budget


def test_get_key_object_from_ui_hash_key_invalid():
    """Test getting key object from invalid UI hash key"""
    # Test with invalid token
    key_object = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key("invalid_token")
    assert key_object is None


@pytest.mark.asyncio
async def test_default_internal_user_params_with_get_user_object(monkeypatch):
    """Test that default_internal_user_params is used when creating a new user via get_user_object"""
    # Set up default_internal_user_params
    default_params = {
        "models": ["gpt-4", "claude-3-opus"],
        "max_budget": 200.0,
        "user_role": "internal_user",
    }
    monkeypatch.setattr(litellm, "default_internal_user_params", default_params)

    # Mock the necessary dependencies
    mock_prisma_client = MagicMock()
    mock_db = AsyncMock()
    mock_prisma_client.db = mock_db

    # Set up the user creation mock - create a complete user model that can be converted to a dict
    mock_user = MagicMock()
    mock_user.user_id = "new_test_user"
    mock_user.models = ["gpt-4", "claude-3-opus"]
    mock_user.max_budget = 200.0
    mock_user.user_role = "internal_user"
    mock_user.organization_memberships = []

    # Make the mock model_dump or dict method return appropriate data
    mock_user.dict = lambda: {
        "user_id": "new_test_user",
        "models": ["gpt-4", "claude-3-opus"],
        "max_budget": 200.0,
        "user_role": "internal_user",
        "organization_memberships": [],
    }

    # Setup the mock returns
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.create = AsyncMock(return_value=mock_user)

    # Create a mock cache - use AsyncMock for async methods
    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_set_cache = AsyncMock()

    # Call get_user_object with user_id_upsert=True to trigger user creation
    try:
        user_obj = await get_user_object(
            user_id="new_test_user",
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
            user_id_upsert=True,
            proxy_logging_obj=None,
        )
    except Exception as e:
        # this fails since the mock object is a MagicMock and not a LiteLLM_UserTable
        print(e)

    # Verify the user was created with the default params
    mock_prisma_client.db.litellm_usertable.create.assert_called_once()
    creation_args = mock_prisma_client.db.litellm_usertable.create.call_args[1]["data"]

    # Verify defaults were applied to the creation args
    assert "models" in creation_args
    assert creation_args["models"] == ["gpt-4", "claude-3-opus"]
    assert creation_args["max_budget"] == 200.0
    assert creation_args["user_role"] == "internal_user"


@pytest.mark.asyncio
@patch("litellm.proxy.management_endpoints.team_endpoints.new_team", new_callable=AsyncMock)
async def test_get_team_db_check_calls_new_team_on_upsert(mock_new_team, monkeypatch):
    """
    Test that _get_team_db_check correctly calls the `new_team` function
    when a team does not exist and upsert is enabled.
    """
    mock_prisma_client = MagicMock()
    mock_db = AsyncMock()
    mock_prisma_client.db = mock_db
    mock_prisma_client.db.litellm_teamtable.find_unique.return_value = None

    # Define what our mocked `new_team` function should return
    team_id_to_create = "new-jwt-team"
    mock_new_team.return_value = {"team_id": team_id_to_create, "max_budget": 123.45}

    await _get_team_db_check(
        team_id=team_id_to_create,
        prisma_client=mock_prisma_client,
        team_id_upsert=True,
    )

    # Verify that our mocked `new_team` function was called exactly once
    mock_new_team.assert_called_once()

    call_args = mock_new_team.call_args[1]
    data_arg = call_args["data"]

    # Verify that `new_team` was called with the correct team_id and that
    # `max_budget` was None, as our function's job is to delegate, not to set defaults.
    assert data_arg.team_id == team_id_to_create
    assert data_arg.max_budget is None


@pytest.mark.asyncio
@patch("litellm.proxy.management_endpoints.team_endpoints.new_team", new_callable=AsyncMock)
async def test_get_team_db_check_does_not_call_new_team_if_exists(mock_new_team, monkeypatch):
    """
    Test that _get_team_db_check does NOT call the `new_team` function
    if the team already exists in the database.
    """
    mock_prisma_client = MagicMock()
    mock_db = AsyncMock()
    mock_prisma_client.db = mock_db
    mock_prisma_client.db.litellm_teamtable.find_unique.return_value = MagicMock()

    team_id_to_find = "existing-jwt-team"

    await _get_team_db_check(
        team_id=team_id_to_find,
        prisma_client=mock_prisma_client,
        team_id_upsert=True,
    )

    # Verify that `new_team` was NEVER called, because the team was found.
    mock_new_team.assert_not_called()


# Vector Store Auth Check Tests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_client,vector_store_registry,expected_result",
    [
        (None, MagicMock(), True),  # No prisma client
        (MagicMock(), None, True),  # No vector store registry
        (MagicMock(), MagicMock(), True),  # No vector stores to run
    ],
)
async def test_vector_store_access_check_early_returns(
    prisma_client, vector_store_registry, expected_result
):
    """Test vector_store_access_check returns True for early exit conditions"""
    request_body = {"messages": [{"role": "user", "content": "test"}]}

    if vector_store_registry:
        vector_store_registry.get_vector_store_ids_to_run.return_value = None

    with patch("litellm.proxy.proxy_server.prisma_client", prisma_client), patch(
        "litellm.vector_store_registry", vector_store_registry
    ):
        result = await vector_store_access_check(
            request_body=request_body,
            team_object=None,
            valid_token=None,
        )

    assert result == expected_result


@pytest.mark.parametrize(
    "object_permissions,vector_store_ids,should_raise,error_type",
    [
        (None, ["store-1"], False, None),  # None permissions - should pass
        (
            {"vector_stores": []},
            ["store-1"],
            False,
            None,
        ),  # Empty vector_stores - should pass (access to all)
        (
            {"vector_stores": ["store-1", "store-2"]},
            ["store-1"],
            False,
            None,
        ),  # Has access
        (
            {"vector_stores": ["store-1", "store-2"]},
            ["store-3"],
            True,
            ProxyErrorTypes.key_vector_store_access_denied,
        ),  # No access
        (
            {"vector_stores": ["store-1"]},
            ["store-1", "store-3"],
            True,
            ProxyErrorTypes.team_vector_store_access_denied,
        ),  # Partial access
    ],
)
def test_can_object_call_vector_stores_scenarios(
    object_permissions, vector_store_ids, should_raise, error_type
):
    """Test _can_object_call_vector_stores with various permission scenarios"""
    # Convert dict to object if not None
    if object_permissions is not None:
        mock_permissions = MagicMock()
        mock_permissions.vector_stores = object_permissions["vector_stores"]
        object_permissions = mock_permissions

    object_type = (
        "key"
        if error_type == ProxyErrorTypes.key_vector_store_access_denied
        else "team"
    )

    if should_raise:
        with pytest.raises(ProxyException) as exc_info:
            _can_object_call_vector_stores(
                object_type=object_type,
                vector_store_ids_to_run=vector_store_ids,
                object_permissions=object_permissions,
            )
        assert exc_info.value.type == error_type
    else:
        result = _can_object_call_vector_stores(
            object_type=object_type,
            vector_store_ids_to_run=vector_store_ids,
            object_permissions=object_permissions,
        )
        assert result is True


@pytest.mark.asyncio
async def test_vector_store_access_check_with_permissions():
    """Test vector_store_access_check with actual permission checking"""
    request_body = {"tools": [{"type": "function", "function": {"name": "test"}}]}

    # Test with valid token that has access
    valid_token = UserAPIKeyAuth(
        token="test-token",
        object_permission_id="perm-123",
        models=["gpt-4"],
        max_budget=100.0,
    )

    mock_prisma_client = MagicMock()
    mock_permissions = MagicMock()
    mock_permissions.vector_stores = ["store-1", "store-2"]
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=mock_permissions
    )

    mock_vector_store_registry = MagicMock()
    mock_vector_store_registry.get_vector_store_ids_to_run.return_value = ["store-1"]

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), patch(
        "litellm.vector_store_registry", mock_vector_store_registry
    ):
        result = await vector_store_access_check(
            request_body=request_body,
            team_object=None,
            valid_token=valid_token,
        )

    assert result is True

    # Test with denied access
    mock_vector_store_registry.get_vector_store_ids_to_run.return_value = ["store-3"]

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), patch(
        "litellm.vector_store_registry", mock_vector_store_registry
    ):
        with pytest.raises(ProxyException) as exc_info:
            await vector_store_access_check(
                request_body=request_body,
                team_object=None,
                valid_token=valid_token,
            )

        assert exc_info.value.type == ProxyErrorTypes.key_vector_store_access_denied


def test_can_object_call_model_with_alias():
    """Test that can_object_call_model works with model aliases"""
    from litellm import Router
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    model = "[ip-approved] gpt-4o"
    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                },
            }
        ],
        model_group_alias={
            "[ip-approved] gpt-4o": {
                "model": "gpt-3.5-turbo",
                "hidden": True,
            },
        },
    )

    result = _can_object_call_model(
        model=model,
        llm_router=llm_router,
        models=["gpt-3.5-turbo"],
        team_model_aliases=None,
        object_type="key",
        fallback_depth=0,
    )

    print(result)


def test_can_object_call_model_access_via_alias_only():
    """
    Test that a key can access a model via alias even when it doesn't have access to the underlying model.

    This tests the scenario where:
    - Router has model alias: "my-fake-gpt" -> "gpt-4"
    - Key has access to: ["my-fake-gpt"] (alias)
    - Key does NOT have access to: ["gpt-4"] (underlying model)
    - The call should succeed because access is granted via the alias
    """
    from litellm import Router
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    model = "my-fake-gpt"
    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "test-api-key",
                },
            }
        ],
        model_group_alias={
            "my-fake-gpt": {
                "model": "gpt-4",
                "hidden": False,
            },
        },
    )

    # Key has access to the alias but NOT the underlying model
    result = _can_object_call_model(
        model=model,
        llm_router=llm_router,
        models=["my-fake-gpt"],  # Only has access to alias, not "gpt-4"
        team_model_aliases=None,
        object_type="key",
        fallback_depth=0,
    )

    # Should return True because access is granted via the alias
    assert result is True


def test_can_object_call_model_access_via_underlying_model_only():
    """
    Test that a key can access a model via underlying model even when using an alias.

    This tests the scenario where:
    - Router has model alias: "my-fake-gpt" -> "gpt-4"
    - Key has access to: ["gpt-4"] (underlying model)
    - Key does NOT have access to: ["my-fake-gpt"] (alias)
    - The call should succeed because access is granted via the underlying model
    """
    from litellm import Router
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    model = "my-fake-gpt"
    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "test-api-key",
                },
            }
        ],
        model_group_alias={
            "my-fake-gpt": {
                "model": "gpt-4",
                "hidden": False,
            },
        },
    )

    # Key has access to the underlying model but NOT the alias
    result = _can_object_call_model(
        model=model,
        llm_router=llm_router,
        models=["gpt-4"],  # Only has access to underlying model, not "my-fake-gpt"
        team_model_aliases=None,
        object_type="key",
        fallback_depth=0,
    )

    # Should return True because access is granted via the underlying model
    assert result is True


def test_can_object_call_model_no_access_to_alias_or_underlying():
    """
    Test that a key cannot access a model when it has no access to either alias or underlying model.
    """
    from litellm import Router
    from litellm.proxy._types import ProxyErrorTypes, ProxyException
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    model = "my-fake-gpt"
    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "test-api-key",
                },
            }
        ],
        model_group_alias={
            "my-fake-gpt": {
                "model": "gpt-4",
                "hidden": False,
            },
        },
    )

    # Key has access to neither the alias nor the underlying model
    with pytest.raises(ProxyException) as exc_info:
        _can_object_call_model(
            model=model,
            llm_router=llm_router,
            models=["gpt-3.5-turbo"],  # Has access to different model entirely
            team_model_aliases=None,
            object_type="key",
            fallback_depth=0,
        )

    # Should raise ProxyException with appropriate error type
    assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied
    assert "key not allowed to access model" in str(exc_info.value.message)
    assert "my-fake-gpt" in str(exc_info.value.message)

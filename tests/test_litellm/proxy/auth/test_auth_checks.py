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
    CallInfo,
    Litellm_EntityType,
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
    _get_fuzzy_user_object,
    _get_team_db_check,
    _log_budget_lookup_failure,
    _virtual_key_max_budget_alert_check,
    _virtual_key_soft_budget_check,
    get_user_object,
    vector_store_access_check,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.utils import get_utc_datetime


@pytest.fixture(autouse=True)
def set_salt_key(monkeypatch):
    """Automatically set LITELLM_SALT_KEY for all tests"""
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-1234")


@pytest.fixture(autouse=True)
def reset_constants_module():
    """Reset constants module to ensure clean state before each test"""
    import importlib
    from litellm import constants
    from litellm.proxy.auth import auth_checks
    
    # Reload modules before test
    importlib.reload(constants)
    importlib.reload(auth_checks)
    
    yield
    
    # Reload modules after test to clean up
    importlib.reload(constants)
    importlib.reload(auth_checks)


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


def test_get_cli_jwt_auth_token_default_expiration(valid_sso_user_defined_values):
    """Test generating CLI JWT token with default 24-hour expiration"""
    token = ExperimentalUIJWTToken.get_cli_jwt_auth_token(valid_sso_user_defined_values)

    # Decrypt and verify token contents
    decrypted_token = decrypt_value_helper(
        token, key="ui_hash_key", exception_type="debug"
    )
    assert decrypted_token is not None
    token_data = json.loads(decrypted_token)

    assert token_data["user_id"] == "test_user"
    assert token_data["user_role"] == LitellmUserRoles.PROXY_ADMIN.value
    assert token_data["models"] == ["gpt-3.5-turbo"]
    assert token_data["max_budget"] == litellm.max_ui_session_budget

    # Verify expiration time is set to 24 hours (default)
    assert "expires" in token_data
    expires = datetime.fromisoformat(token_data["expires"].replace("Z", "+00:00"))
    assert expires > get_utc_datetime()
    assert expires <= get_utc_datetime() + timedelta(hours=24, minutes=1)
    assert expires >= get_utc_datetime() + timedelta(hours=23, minutes=59)


def test_get_cli_jwt_auth_token_custom_expiration(
    valid_sso_user_defined_values, monkeypatch
):
    """Test generating CLI JWT token with custom expiration via environment variable"""
    import importlib
    from litellm import constants
    from litellm.proxy.auth import auth_checks
    
    # Set custom expiration to 48 hours
    monkeypatch.setenv("LITELLM_CLI_JWT_EXPIRATION_HOURS", "48")
    
    # Reload the constants module to pick up the new env var
    importlib.reload(constants)
    # Also reload auth_checks to pick up the new constant value
    importlib.reload(auth_checks)
    
    token = auth_checks.ExperimentalUIJWTToken.get_cli_jwt_auth_token(valid_sso_user_defined_values)

    # Decrypt and verify token contents
    decrypted_token = decrypt_value_helper(
        token, key="ui_hash_key", exception_type="debug"
    )
    assert decrypted_token is not None
    token_data = json.loads(decrypted_token)

    # Verify expiration time is set to 48 hours
    assert "expires" in token_data
    expires = datetime.fromisoformat(token_data["expires"].replace("Z", "+00:00"))
    assert expires > get_utc_datetime() + timedelta(hours=47, minutes=59)
    assert expires <= get_utc_datetime() + timedelta(hours=48, minutes=1)



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


def test_log_budget_lookup_failure_dry_run():
    """Dry run: verify _log_budget_lookup_failure logs for schema/DB errors."""
    with patch("litellm.proxy.auth.auth_checks.verbose_proxy_logger") as mock_logger:
        err = Exception("column 'policies' does not exist in prisma schema")
        _log_budget_lookup_failure("user", err)
        mock_logger.error.assert_called_once()
        call_msg = mock_logger.error.call_args[0][0]
        assert "user" in call_msg
        assert "cache will not be populated" in call_msg
        assert "policies" in call_msg or "prisma" in call_msg
        assert "prisma db push" in call_msg


def test_log_budget_lookup_failure_skips_user_not_found():
    """Verify _log_budget_lookup_failure does NOT log for expected user-not-found."""
    with patch("litellm.proxy.auth.auth_checks.verbose_proxy_logger") as mock_logger:
        err = Exception()  # bare Exception from get_user_object when user not found
        _log_budget_lookup_failure("user", err)
        mock_logger.error.assert_not_called()


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


@pytest.mark.asyncio
async def test_vector_store_access_check_with_team_permissions():
    """Ensure teams restricted to specific vector stores cannot access others."""
    request_body = {}
    valid_token = UserAPIKeyAuth(token="team-test-token", object_permission_id=None)

    team_object = MagicMock()
    team_object.object_permission_id = "team-permission"

    mock_prisma_client = MagicMock()
    team_permissions = MagicMock()
    team_permissions.vector_stores = ["team-store-allowed"]
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=team_permissions
    )

    mock_vector_store_registry = MagicMock()
    mock_vector_store_registry.get_vector_store_ids_to_run.return_value = [
        "team-store-allowed"
    ]

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), patch(
        "litellm.vector_store_registry", mock_vector_store_registry
    ):
        result = await vector_store_access_check(
            request_body=request_body,
            team_object=team_object,
            valid_token=valid_token,
        )

    assert result is True

    mock_vector_store_registry.get_vector_store_ids_to_run.return_value = [
        "team-store-denied"
    ]

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), patch(
        "litellm.vector_store_registry", mock_vector_store_registry
    ):
        with pytest.raises(ProxyException) as exc_info:
            await vector_store_access_check(
                request_body=request_body,
                team_object=team_object,
                valid_token=valid_token,
            )

    assert exc_info.value.type == ProxyErrorTypes.team_vector_store_access_denied


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


# Tag Budget Enforcement Tests


@pytest.mark.asyncio
async def test_get_tag_objects_batch():
    """
    Test batch fetching of tags validates:
    - Cached tags are fetched from cache (no DB call for them)
    - Uncached tags are fetched in ONE batch DB query
    - After fetching, uncached tags are cached
    """
    from litellm.proxy._types import LiteLLM_TagTable
    from litellm.proxy.auth.auth_checks import get_tag_objects_batch

    mock_prisma = MagicMock()
    mock_cache = MagicMock()
    mock_proxy_logging = MagicMock()

    # Simulate 5 tags: 2 cached, 3 uncached
    tag_names = ["cached-1", "uncached-1", "cached-2", "uncached-2", "uncached-3"]

    # Mock cached tags
    cached_tag_1 = {
        "tag_name": "cached-1",
        "spend": 10.0,
        "models": [],
        "litellm_budget_table": None,
    }
    cached_tag_2 = {
        "tag_name": "cached-2",
        "spend": 20.0,
        "models": [],
        "litellm_budget_table": None,
    }

    # Mock DB response for uncached tags
    uncached_tag_1 = MagicMock()
    uncached_tag_1.tag_name = "uncached-1"
    uncached_tag_1.spend = 30.0
    uncached_tag_1.models = []
    uncached_tag_1.litellm_budget_table = None
    uncached_tag_1.dict = MagicMock(
        return_value={
            "tag_name": "uncached-1",
            "spend": 30.0,
            "models": [],
            "litellm_budget_table": None,
        }
    )

    uncached_tag_2 = MagicMock()
    uncached_tag_2.tag_name = "uncached-2"
    uncached_tag_2.spend = 40.0
    uncached_tag_2.models = []
    uncached_tag_2.litellm_budget_table = None
    uncached_tag_2.dict = MagicMock(
        return_value={
            "tag_name": "uncached-2",
            "spend": 40.0,
            "models": [],
            "litellm_budget_table": None,
        }
    )

    uncached_tag_3 = MagicMock()
    uncached_tag_3.tag_name = "uncached-3"
    uncached_tag_3.spend = 50.0
    uncached_tag_3.models = []
    uncached_tag_3.litellm_budget_table = None
    uncached_tag_3.dict = MagicMock(
        return_value={
            "tag_name": "uncached-3",
            "spend": 50.0,
            "models": [],
            "litellm_budget_table": None,
        }
    )

    # Mock cache behavior - return cached tags, None for uncached
    async def mock_get_cache(key):
        if key == "tag:cached-1":
            return cached_tag_1
        elif key == "tag:cached-2":
            return cached_tag_2
        else:
            return None

    mock_cache.async_get_cache = AsyncMock(side_effect=mock_get_cache)
    mock_cache.async_set_cache = AsyncMock()

    # Mock DB to return all uncached tags in ONE query
    mock_prisma.db.litellm_tagtable.find_many = AsyncMock(
        return_value=[uncached_tag_1, uncached_tag_2, uncached_tag_3]
    )

    # Call batch fetch
    tag_objects = await get_tag_objects_batch(
        tag_names=tag_names,
        prisma_client=mock_prisma,
        user_api_key_cache=mock_cache,
        proxy_logging_obj=mock_proxy_logging,
    )

    # Verify results
    assert len(tag_objects) == 5
    assert "cached-1" in tag_objects
    assert "cached-2" in tag_objects
    assert "uncached-1" in tag_objects
    assert "uncached-2" in tag_objects
    assert "uncached-3" in tag_objects

    # Verify cached tags have correct values
    assert tag_objects["cached-1"].spend == 10.0
    assert tag_objects["cached-2"].spend == 20.0

    # Verify uncached tags have correct values
    assert tag_objects["uncached-1"].spend == 30.0
    assert tag_objects["uncached-2"].spend == 40.0
    assert tag_objects["uncached-3"].spend == 50.0

    # Verify DB was called ONCE with all 3 uncached tags
    mock_prisma.db.litellm_tagtable.find_many.assert_called_once()
    call_args = mock_prisma.db.litellm_tagtable.find_many.call_args
    assert call_args.kwargs["where"]["tag_name"]["in"] == [
        "uncached-1",
        "uncached-2",
        "uncached-3",
    ]

    # Verify uncached tags were cached after fetching
    assert mock_cache.async_set_cache.call_count == 3
    cache_calls = mock_cache.async_set_cache.call_args_list
    cached_keys = [call.kwargs["key"] for call in cache_calls]
    assert "tag:uncached-1" in cached_keys
    assert "tag:uncached-2" in cached_keys
    assert "tag:uncached-3" in cached_keys


@pytest.mark.asyncio
async def test_get_team_object_raises_404_when_not_found():
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import HTTPException

    from litellm.proxy.auth.auth_checks import get_team_object

    mock_prisma_client = MagicMock()
    mock_db = AsyncMock()
    mock_prisma_client.db = mock_db
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        await get_team_object(
            team_id="nonexistent-team",
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
            check_cache_only=False,
            check_db_only=True,
        )

    assert exc_info.value.status_code == 404
    assert "Team doesn't exist in db" in str(exc_info.value.detail)


# Reject Client-Side Metadata Tags Tests


@pytest.mark.asyncio
async def test_reject_clientside_metadata_tags_enabled_with_tags():
    """Test that common_checks rejects request when reject_clientside_metadata_tags is True and metadata.tags is present."""
    from fastapi import Request

    from litellm.proxy.auth.auth_checks import common_checks

    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {"tags": ["custom-tag"]},
    }

    general_settings = {"reject_clientside_metadata_tags": True}

    # Create a mock request object
    mock_request = MagicMock(spec=Request)

    # Create a valid token for the test
    valid_token = UserAPIKeyAuth(token="test-token", models=["gpt-3.5-turbo"])

    with pytest.raises(ProxyException) as exc_info:
        await common_checks(
            request_body=request_body,
            team_object=None,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings=general_settings,
            route="/chat/completions",
            llm_router=None,
            proxy_logging_obj=MagicMock(),
            valid_token=valid_token,
            request=mock_request,
        )

    assert exc_info.value.type == ProxyErrorTypes.bad_request_error
    assert "metadata.tags" in exc_info.value.message
    assert exc_info.value.param == "metadata.tags"
    assert exc_info.value.code == "400"


@pytest.mark.asyncio
async def test_reject_clientside_metadata_tags_enabled_without_tags():
    """Test that common_checks allows request when reject_clientside_metadata_tags is True but no metadata.tags is present."""
    from fastapi import Request

    from litellm.proxy.auth.auth_checks import common_checks

    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {"custom_field": "value"},  # No tags field
    }

    general_settings = {"reject_clientside_metadata_tags": True}

    # Create a mock request object
    mock_request = MagicMock(spec=Request)

    # Create a valid token for the test
    valid_token = UserAPIKeyAuth(token="test-token", models=["gpt-3.5-turbo"])

    # Should not raise an exception
    result = await common_checks(
        request_body=request_body,
        team_object=None,
        user_object=None,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings=general_settings,
        route="/chat/completions",
        llm_router=None,
        proxy_logging_obj=MagicMock(),
        valid_token=valid_token,
        request=mock_request,
    )

    assert result is True


@pytest.mark.asyncio
async def test_reject_clientside_metadata_tags_disabled_with_tags():
    """Test that common_checks allows request with metadata.tags when reject_clientside_metadata_tags is False."""
    from fastapi import Request

    from litellm.proxy.auth.auth_checks import common_checks

    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {"tags": ["custom-tag"]},
    }

    general_settings = {"reject_clientside_metadata_tags": False}

    # Create a mock request object
    mock_request = MagicMock(spec=Request)

    # Create a valid token for the test
    valid_token = UserAPIKeyAuth(token="test-token", models=["gpt-3.5-turbo"])

    # Should not raise an exception
    result = await common_checks(
        request_body=request_body,
        team_object=None,
        user_object=None,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings=general_settings,
        route="/chat/completions",
        llm_router=None,
        proxy_logging_obj=MagicMock(),
        valid_token=valid_token,
        request=mock_request,
    )

    assert result is True


@pytest.mark.asyncio
async def test_reject_clientside_metadata_tags_not_set_with_tags():
    """Test that common_checks allows request with metadata.tags when reject_clientside_metadata_tags is not set."""
    from fastapi import Request

    from litellm.proxy.auth.auth_checks import common_checks

    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {"tags": ["custom-tag"]},
    }

    general_settings = {}  # No reject_clientside_metadata_tags setting

    # Create a mock request object
    mock_request = MagicMock(spec=Request)

    # Create a valid token for the test
    valid_token = UserAPIKeyAuth(token="test-token", models=["gpt-3.5-turbo"])

    # Should not raise an exception
    result = await common_checks(
        request_body=request_body,
        team_object=None,
        user_object=None,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings=general_settings,
        route="/chat/completions",
        llm_router=None,
        proxy_logging_obj=MagicMock(),
        valid_token=valid_token,
        request=mock_request,
    )

    assert result is True


@pytest.mark.asyncio
async def test_reject_clientside_metadata_tags_non_llm_route():
    """Test that reject_clientside_metadata_tags check only applies to LLM API routes."""
    from fastapi import Request

    from litellm.proxy.auth.auth_checks import common_checks

    request_body = {
        "metadata": {"tags": ["custom-tag"]},
    }

    general_settings = {"reject_clientside_metadata_tags": True}

    # Create a mock request object
    mock_request = MagicMock(spec=Request)

    # Create a valid token for the test
    valid_token = UserAPIKeyAuth(token="test-token", models=["gpt-3.5-turbo"])

    # Create an admin user object for the management route
    admin_user = LiteLLM_UserTable(
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    # Should not raise an exception for non-LLM route
    result = await common_checks(
        request_body=request_body,
        team_object=None,
        user_object=admin_user,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings=general_settings,
        route="/key/generate",  # Management route, not LLM route
        llm_router=None,
        proxy_logging_obj=MagicMock(),
        valid_token=valid_token,
        request=mock_request,
    )

    assert result is True


@pytest.mark.asyncio
async def test_virtual_key_soft_budget_check_with_user_obj():
    """Test _virtual_key_soft_budget_check includes user_email when user_obj is provided"""
    alert_triggered = False
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered, captured_call_info
            alert_triggered = True
            captured_call_info = user_info
            assert type == "soft_budget"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=100.0,
        soft_budget=50.0,
        user_id="test-user",
        team_id="test-team",
        team_alias="test-team-alias",
        org_id="test-org",
        key_alias="test-key",
        max_budget=200.0,
    )

    user_obj = LiteLLM_UserTable(
        user_id="test-user",
        user_email="test@example.com",
        max_budget=None,
    )

    proxy_logging_obj = MockProxyLogging()

    await _virtual_key_soft_budget_check(
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
        user_obj=user_obj,
    )

    await asyncio.sleep(0.1)

    assert alert_triggered is True
    assert captured_call_info is not None
    assert captured_call_info.user_email == "test@example.com"
    assert captured_call_info.token == "test-token"
    assert captured_call_info.spend == 100.0
    assert captured_call_info.soft_budget == 50.0
    assert captured_call_info.max_budget == 200.0
    assert captured_call_info.user_id == "test-user"
    assert captured_call_info.team_id == "test-team"
    assert captured_call_info.team_alias == "test-team-alias"
    assert captured_call_info.organization_id == "test-org"
    assert captured_call_info.key_alias == "test-key"
    assert captured_call_info.event_group == Litellm_EntityType.KEY


@pytest.mark.asyncio
async def test_virtual_key_soft_budget_check_without_user_obj():
    """Test _virtual_key_soft_budget_check sets user_email to None when user_obj is not provided"""
    alert_triggered = False
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered, captured_call_info
            alert_triggered = True
            captured_call_info = user_info
            assert type == "soft_budget"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=100.0,
        soft_budget=50.0,
        user_id="test-user",
        team_id="test-team",
        key_alias="test-key",
    )

    proxy_logging_obj = MockProxyLogging()

    await _virtual_key_soft_budget_check(
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
        user_obj=None,
    )

    await asyncio.sleep(0.1)

    assert alert_triggered is True
    assert captured_call_info is not None
    assert captured_call_info.user_email is None


@pytest.mark.parametrize(
    "spend, soft_budget, expect_alert",
    [
        (100.0, 50.0, True),  # Over soft budget
        (50.0, 50.0, True),  # At soft budget
        (25.0, 50.0, False),  # Under soft budget
        (100.0, None, False),  # No soft budget set
    ],
)
@pytest.mark.asyncio
async def test_virtual_key_soft_budget_check_scenarios(
    spend, soft_budget, expect_alert
):
    """Test _virtual_key_soft_budget_check with various spend and soft_budget scenarios"""
    alert_triggered = False

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered
            alert_triggered = True
            assert type == "soft_budget"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=spend,
        soft_budget=soft_budget,
        user_id="test-user",
        key_alias="test-key",
    )

    proxy_logging_obj = MockProxyLogging()

    await _virtual_key_soft_budget_check(
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
        user_obj=None,
    )

    await asyncio.sleep(0.1)

    assert (
        alert_triggered == expect_alert
    ), f"Expected alert_triggered to be {expect_alert} for spend={spend}, soft_budget={soft_budget}"


@pytest.mark.asyncio
async def test_virtual_key_max_budget_alert_check_with_user_obj():
    """Test _virtual_key_max_budget_alert_check includes user_email when user_obj is provided"""
    alert_triggered = False
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered, captured_call_info
            alert_triggered = True
            captured_call_info = user_info
            assert type == "max_budget_alert"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=90.0,
        max_budget=100.0,
        user_id="test-user",
        team_id="test-team",
        team_alias="test-team-alias",
        org_id="test-org",
        key_alias="test-key",
        soft_budget=50.0,
    )

    user_obj = LiteLLM_UserTable(
        user_id="test-user",
        user_email="test@example.com",
        max_budget=None,
    )

    proxy_logging_obj = MockProxyLogging()

    await _virtual_key_max_budget_alert_check(
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
        user_obj=user_obj,
    )

    await asyncio.sleep(0.1)

    assert alert_triggered is True
    assert captured_call_info is not None
    assert captured_call_info.user_email == "test@example.com"
    assert captured_call_info.token == "test-token"
    assert captured_call_info.spend == 90.0
    assert captured_call_info.max_budget == 100.0
    assert captured_call_info.soft_budget == 50.0
    assert captured_call_info.user_id == "test-user"
    assert captured_call_info.team_id == "test-team"
    assert captured_call_info.team_alias == "test-team-alias"
    assert captured_call_info.organization_id == "test-org"
    assert captured_call_info.key_alias == "test-key"
    assert captured_call_info.event_group == Litellm_EntityType.KEY


@pytest.mark.asyncio
async def test_virtual_key_max_budget_alert_check_without_user_obj():
    """Test _virtual_key_max_budget_alert_check sets user_email to None when user_obj is not provided"""
    alert_triggered = False
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered, captured_call_info
            alert_triggered = True
            captured_call_info = user_info
            assert type == "max_budget_alert"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=90.0,
        max_budget=100.0,
        user_id="test-user",
        team_id="test-team",
        key_alias="test-key",
    )

    proxy_logging_obj = MockProxyLogging()

    await _virtual_key_max_budget_alert_check(
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
        user_obj=None,
    )

    await asyncio.sleep(0.1)

    assert alert_triggered is True
    assert captured_call_info is not None
    assert captured_call_info.user_email is None


@pytest.mark.parametrize(
    "spend, max_budget, expect_alert",
    [
        (80.0, 100.0, True),  # At 80% threshold (alert threshold)
        (90.0, 100.0, True),  # Above threshold, below max_budget
        (79.0, 100.0, False),  # Below threshold
        (100.0, 100.0, False),  # At max_budget (not below, so no alert)
        (110.0, 100.0, False),  # Above max_budget (already exceeded)
        (100.0, None, False),  # No max_budget set
        (0.0, 100.0, False),  # Spend is 0
    ],
)
@pytest.mark.asyncio
async def test_virtual_key_max_budget_alert_check_scenarios(
    spend, max_budget, expect_alert
):
    """Test _virtual_key_max_budget_alert_check with various spend and max_budget scenarios"""
    alert_triggered = False

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered
            alert_triggered = True
            assert type == "max_budget_alert"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=spend,
        max_budget=max_budget,
        user_id="test-user",
        key_alias="test-key",
    )

    proxy_logging_obj = MockProxyLogging()

    await _virtual_key_max_budget_alert_check(
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
        user_obj=None,
    )

    await asyncio.sleep(0.1)

    assert (
        alert_triggered == expect_alert
    ), f"Expected alert_triggered to be {expect_alert} for spend={spend}, max_budget={max_budget}"


@pytest.mark.asyncio
async def test_get_fuzzy_user_object_case_insensitive_email():
    """Test that _get_fuzzy_user_object uses case-insensitive email lookup"""
    # Setup mock Prisma client
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_usertable = MagicMock()

    # Mock user data with mixed case email
    test_user = LiteLLM_UserTable(
        user_id="test_123",
        sso_user_id=None,
        user_email="Test@Example.com",  # Mixed case in DB
        organization_memberships=[],
        max_budget=None,
    )

    # Test: SSO ID not found, find by email with different casing
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=test_user)

    # Search with lowercase email (different from DB)
    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma,
        sso_user_id=None,
        user_email="test@example.com",  # Lowercase search
    )

    # Verify user was found despite case difference
    assert result == test_user

    # Verify the query used case-insensitive mode
    mock_prisma.db.litellm_usertable.find_first.assert_called_once()
    call_args = mock_prisma.db.litellm_usertable.find_first.call_args
    assert call_args.kwargs["where"]["user_email"]["equals"] == "test@example.com"
    assert call_args.kwargs["where"]["user_email"]["mode"] == "insensitive"
    assert call_args.kwargs["include"] == {"organization_memberships": True}

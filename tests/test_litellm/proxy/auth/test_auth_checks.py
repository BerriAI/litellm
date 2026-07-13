import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from datetime import datetime, timedelta

import httpx
import pytest
from fastapi import status

import litellm
from litellm.proxy._types import (
    CallInfo,
    Litellm_EntityType,
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TagTable,
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
    _cache_management_object,
    _can_object_call_model,
    _can_object_call_vector_stores,
    _check_end_user_budget,
    _check_team_member_budget,
    _get_fuzzy_user_object,
    _get_team_db_check,
    _log_budget_lookup_failure,
    _tag_max_budget_check,
    _team_max_budget_check,
    _virtual_key_max_budget_alert_check,
    _virtual_key_max_budget_check,
    _virtual_key_soft_budget_check,
    get_key_object,
    get_user_object,
    vector_store_access_check,
)
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
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

    # Verify expiration time is set and valid (Experimental UI uses fixed 10-min expiry)
    assert "expires" in token_data
    expires = datetime.fromisoformat(token_data["expires"].replace("Z", "+00:00"))
    now = get_utc_datetime()
    # Allow 2 second buffer for test execution timing
    assert expires > now
    assert expires <= now + timedelta(minutes=10, seconds=2)


def test_get_cli_jwt_auth_token_includes_team_alias(valid_sso_user_defined_values):
    token = ExperimentalUIJWTToken.get_cli_jwt_auth_token(
        valid_sso_user_defined_values,
        team_id="team-123",
        team_alias="test-team",
    )

    decrypted_token = decrypt_value_helper(
        token, key="ui_hash_key", exception_type="debug"
    )
    assert decrypted_token is not None
    token_data = json.loads(decrypted_token)

    assert token_data["team_id"] == "team-123"
    assert token_data["team_alias"] == "test-team"


def test_get_experimental_ui_login_jwt_auth_token_uses_10_min_expiry(
    valid_sso_user_defined_values,
):
    """Test that Experimental UI token uses fixed 10-minute expiry (does not use LITELLM_UI_SESSION_DURATION)."""
    token = ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
        valid_sso_user_defined_values
    )
    decrypted_token = decrypt_value_helper(
        token, key="ui_hash_key", exception_type="debug"
    )
    assert decrypted_token is not None
    token_data = json.loads(decrypted_token)
    expires = datetime.fromisoformat(token_data["expires"].replace("Z", "+00:00"))
    now = get_utc_datetime()
    # Should expire in ~10 minutes (allow 2 second buffer)
    assert expires > now + timedelta(minutes=9)
    assert expires <= now + timedelta(minutes=10, seconds=2)


def test_experimental_ui_token_ignores_litellm_ui_session_duration(
    valid_sso_user_defined_values,
):
    """Regression test: LITELLM_UI_SESSION_DURATION must NOT affect Experimental UI token expiry.
    Experimental UI intentionally uses fixed 10-min expiry. If this test fails, the constant
    was incorrectly wired to the experimental flow."""
    # Default LITELLM_UI_SESSION_DURATION is "24h" - token must still expire in ~10 min
    token = ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
        valid_sso_user_defined_values
    )
    decrypted_token = decrypt_value_helper(
        token, key="ui_hash_key", exception_type="debug"
    )
    assert decrypted_token is not None
    token_data = json.loads(decrypted_token)
    expires = datetime.fromisoformat(token_data["expires"].replace("Z", "+00:00"))
    now = get_utc_datetime()
    # Must be ~10 min, NOT 24h. If LITELLM_UI_SESSION_DURATION were incorrectly used, this would fail.
    assert expires <= now + timedelta(
        minutes=11
    ), "Experimental UI must use 10-min expiry, not LITELLM_UI_SESSION_DURATION"


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


@pytest.mark.parametrize(
    "object_type,expected_error_type",
    [
        ("key", ProxyErrorTypes.key_model_access_denied),
        ("team", ProxyErrorTypes.team_model_access_denied),
        ("user", ProxyErrorTypes.user_model_access_denied),
        ("org", ProxyErrorTypes.org_model_access_denied),
        ("project", ProxyErrorTypes.project_model_access_denied),
    ],
)
def test_can_object_call_model_denials_return_forbidden(
    object_type, expected_error_type
):
    with pytest.raises(ProxyException) as exc_info:
        _can_object_call_model(
            model="restricted-model",
            llm_router=None,
            models=["allowed-model"],
            object_type=object_type,
        )

    assert exc_info.value.type == expected_error_type
    assert int(exc_info.value.code) == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_can_user_call_model_no_default_models_returns_forbidden():
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.auth.auth_checks import can_user_call_model

    user_object = LiteLLM_UserTable(
        user_id="test-user",
        models=[SpecialModelNames.no_default_models.value],
    )

    with pytest.raises(ProxyException) as exc_info:
        await can_user_call_model(
            model="restricted-model",
            llm_router=None,
            user_object=user_object,
        )

    assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied
    assert int(exc_info.value.code) == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_can_key_call_model_all_team_models_uses_team_allowlist():
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.auth.auth_checks import can_key_call_model

    valid_token = UserAPIKeyAuth(
        api_key="sk-team-key",
        team_id="team-123",
        models=[SpecialModelNames.all_team_models.value],
        team_models=["openai/openai/gpt-5.5-batch"],
    )

    assert (
        await can_key_call_model(
            model="openai/openai/gpt-5.5-batch",
            llm_model_list=None,
            valid_token=valid_token,
            llm_router=None,
        )
        is True
    )

    with pytest.raises(ProxyException) as exc_info:
        await can_key_call_model(
            model="gpt-4o",
            llm_model_list=None,
            valid_token=valid_token,
            llm_router=None,
        )

    assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied


@pytest.mark.asyncio
async def test_can_key_call_model_all_team_models_empty_team_models_is_unrestricted():
    """Team-bound key with empty team_models expands to [] -> unrestricted (same as get_key_models)."""
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.auth.auth_checks import can_key_call_model

    valid_token = UserAPIKeyAuth(
        api_key="sk-team-key",
        team_id="team-123",
        models=[SpecialModelNames.all_team_models.value],
        team_models=[],
    )

    assert (
        await can_key_call_model(
            model="any-model",
            llm_model_list=None,
            valid_token=valid_token,
            llm_router=None,
        )
        is True
    )


@pytest.mark.asyncio
async def test_can_key_call_model_all_team_models_no_team_id_is_unrestricted():
    """A teamless key with all-team-models inherits the full proxy model list
    (empty resolved list = unrestricted access), the same as leaving the models
    field empty. This test will fail if someone re-introduces a teamless denial."""
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.auth.auth_checks import can_key_call_model

    valid_token = UserAPIKeyAuth(
        api_key="sk-orphan-key",
        models=[SpecialModelNames.all_team_models.value],
        team_models=[],
    )

    assert (
        await can_key_call_model(
            model="gpt-4o",
            llm_model_list=None,
            valid_token=valid_token,
            llm_router=None,
        )
        is True
    )


def test_resolve_key_models_teamless_all_team_models_returns_empty():
    """_resolve_key_models_for_auth_check must return [] for a teamless key
    with all-team-models, making it equivalent to an unscoped key (unrestricted
    access). Fails if someone returns the sentinel list for teamless keys."""
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.auth.auth_checks import _resolve_key_models_for_auth_check

    valid_token = UserAPIKeyAuth(
        api_key="sk-orphan",
        models=[SpecialModelNames.all_team_models.value],
        team_models=[],
    )

    result = _resolve_key_models_for_auth_check(valid_token)
    assert result == [], "teamless all-team-models must resolve to [] (unrestricted)"


@pytest.mark.asyncio
async def test_enforce_key_access_teamless_all_team_models_passes():
    """_enforce_key_and_fallback_model_access must not deny a teamless key with
    all-team-models. The inference path skips the key-level model check when
    the sentinel is present, regardless of team_id. Fails if someone adds a
    team_id guard to the pass branch."""
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.auth.user_api_key_auth import _enforce_key_and_fallback_model_access

    valid_token = UserAPIKeyAuth(
        api_key="sk-orphan",
        models=[SpecialModelNames.all_team_models.value],
        team_models=[],
    )

    await _enforce_key_and_fallback_model_access(
        valid_token=valid_token,
        request_data={"model": "gpt-4o"},
        route="/chat/completions",
        request=None,
        llm_model_list=None,
        llm_router=None,
    )


@pytest.mark.asyncio
async def test_can_key_call_resolved_model_teamless_all_team_models_passes():
    """can_key_call_resolved_model must skip the key model check for a teamless
    key with all-team-models. Fails if someone adds a team_id guard to the
    skip_key_model_check condition."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.auth.auth_checks import can_key_call_resolved_model

    valid_token = UserAPIKeyAuth(
        api_key="sk-orphan",
        models=[SpecialModelNames.all_team_models.value],
        team_models=[],
    )

    with patch("litellm.proxy.auth.auth_checks.can_key_call_model", new_callable=AsyncMock) as mock_call:
        with patch("litellm.proxy.proxy_server.prisma_client", None):
            with patch("litellm.proxy.proxy_server.proxy_logging_obj", None):
                with patch("litellm.proxy.proxy_server.user_api_key_cache", None):
                    await can_key_call_resolved_model(
                        model="gpt-4o",
                        llm_model_list=None,
                        valid_token=valid_token,
                        llm_router=None,
                    )
        mock_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_can_team_access_model_all_team_models_expands_router_models():
    from litellm import Router
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.auth.auth_checks import can_team_access_model

    team_object = LiteLLM_TeamTable(
        team_id="team-123",
        models=[SpecialModelNames.all_team_models.value],
    )
    router = Router(
        model_list=[
            {
                "model_name": "allowed-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
            }
        ]
    )

    assert (
        await can_team_access_model(
            model="allowed-model",
            team_object=team_object,
            llm_router=router,
        )
        is True
    )
    with pytest.raises(ProxyException) as exc_info:
        await can_team_access_model(
            model="blocked-model",
            team_object=team_object,
            llm_router=router,
        )

    assert exc_info.value.type == ProxyErrorTypes.team_model_access_denied


@pytest.mark.asyncio
async def test_get_key_object_should_reconnect_once_on_db_connection_error():
    mock_prisma_client = MagicMock()
    mock_prisma_client.get_data = AsyncMock(
        side_effect=[
            httpx.ConnectError("db connection reset"),
            UserAPIKeyAuth(token="hashed-token-1"),
        ]
    )
    mock_prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)

    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_set_cache = AsyncMock()

    key_obj = await get_key_object(
        hashed_token="hashed-token-1",
        prisma_client=mock_prisma_client,
        user_api_key_cache=mock_cache,
    )

    assert key_obj.token == "hashed-token-1"
    assert mock_prisma_client.get_data.await_count == 2
    mock_prisma_client.attempt_db_reconnect.assert_awaited_once_with(
        reason="auth_get_key_object_lookup_failure",
        timeout_seconds=2.0,
        lock_timeout_seconds=0.1,
    )


@pytest.mark.asyncio
async def test_get_key_object_should_raise_if_reconnect_fails_on_db_connection_error():
    mock_prisma_client = MagicMock()
    mock_prisma_client.get_data = AsyncMock(
        side_effect=httpx.ConnectError("db not reachable after outage")
    )
    mock_prisma_client.attempt_db_reconnect = AsyncMock(return_value=False)

    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_set_cache = AsyncMock()

    with pytest.raises(Exception, match="db not reachable after outage"):
        await get_key_object(
            hashed_token="hashed-token-2",
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )

    mock_prisma_client.attempt_db_reconnect.assert_awaited_once_with(
        reason="auth_get_key_object_lookup_failure",
        timeout_seconds=2.0,
        lock_timeout_seconds=0.1,
    )
    assert mock_prisma_client.get_data.await_count == 1


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
    # CLI session tokens carry no per-key budget; spend is enforced via the
    # shared team/user counters. The $0.25 UI session cap must not leak in.
    assert token_data.get("max_budget") is None
    # is_session_token=True causes key_management_endpoints to use the team
    # budget as the delegation ceiling instead of treating None as unlimited.
    assert token_data.get("is_session_token") is True

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

    token = auth_checks.ExperimentalUIJWTToken.get_cli_jwt_auth_token(
        valid_sso_user_defined_values
    )

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


def test_get_cli_jwt_auth_token_unique_per_session(valid_sso_user_defined_values):
    """Each CLI login mints a unique token id (per-session spend isolation) while
    keeping a stable, user-scoped key_alias for log grouping. A regression that
    pins token back to a constant would collapse both ids and fail here."""
    from litellm.constants import CLI_SESSION_KEY_PREFIX

    def _decode(token: str) -> dict:
        decrypted = decrypt_value_helper(
            token, key="ui_hash_key", exception_type="debug"
        )
        assert decrypted is not None
        return json.loads(decrypted)

    first = _decode(
        ExperimentalUIJWTToken.get_cli_jwt_auth_token(valid_sso_user_defined_values)
    )
    second = _decode(
        ExperimentalUIJWTToken.get_cli_jwt_auth_token(valid_sso_user_defined_values)
    )

    assert first["token"].startswith(f"{CLI_SESSION_KEY_PREFIX}-")
    assert second["token"].startswith(f"{CLI_SESSION_KEY_PREFIX}-")
    assert first["token"] != second["token"]

    expected_alias = f"{CLI_SESSION_KEY_PREFIX}-test_user"
    assert first["key_alias"] == second["key_alias"] == expected_alias
    assert first["key_name"] == second["key_name"] == expected_alias


def test_get_cli_jwt_auth_token_applies_fallback_budget(valid_sso_user_defined_values):
    token = ExperimentalUIJWTToken.get_cli_jwt_auth_token(
        valid_sso_user_defined_values, max_budget=litellm.max_ui_session_budget
    )
    decrypted = decrypt_value_helper(token, key="ui_hash_key", exception_type="debug")
    assert decrypted is not None
    assert json.loads(decrypted).get("max_budget") == litellm.max_ui_session_budget


def test_get_cli_jwt_auth_token_no_fallback_when_budget_provided(
    valid_sso_user_defined_values,
):
    token = ExperimentalUIJWTToken.get_cli_jwt_auth_token(
        valid_sso_user_defined_values, max_budget=None
    )
    decrypted = decrypt_value_helper(token, key="ui_hash_key", exception_type="debug")
    assert decrypted is not None
    assert json.loads(decrypted).get("max_budget") is None


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
async def test_get_user_object_upsert_includes_user_email():
    """Test that user_email is included when creating a new user via get_user_object upsert"""
    # Mock the necessary dependencies
    mock_prisma_client = MagicMock()
    mock_db = AsyncMock()
    mock_prisma_client.db = mock_db

    # Set up the user creation mock
    mock_user = MagicMock()
    mock_user.user_id = "new_test_user"
    mock_user.user_email = "test@example.com"
    mock_user.models = []
    mock_user.max_budget = None
    mock_user.user_role = None
    mock_user.organization_memberships = []

    mock_user.dict = lambda: {
        "user_id": "new_test_user",
        "user_email": "test@example.com",
        "models": [],
        "max_budget": None,
        "user_role": None,
        "organization_memberships": [],
    }

    # Setup the mock returns - user does not exist
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.create = AsyncMock(return_value=mock_user)

    # Create a mock cache
    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_set_cache = AsyncMock()

    # Call get_user_object with user_id_upsert=True and user_email
    try:
        await get_user_object(
            user_id="new_test_user",
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
            user_id_upsert=True,
            proxy_logging_obj=None,
            user_email="test@example.com",
        )
    except Exception as e:
        # May fail since mock object is not a real LiteLLM_UserTable
        print(e)

    # Verify the user was created with user_email included
    mock_prisma_client.db.litellm_usertable.create.assert_called_once()
    creation_args = mock_prisma_client.db.litellm_usertable.create.call_args[1]["data"]

    assert (
        "user_email" in creation_args
    ), "user_email should be included when upserting a new user"
    assert creation_args["user_email"] == "test@example.com"
    assert creation_args["user_id"] == "new_test_user"


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
@patch(
    "litellm.proxy.management_endpoints.team_endpoints.new_team", new_callable=AsyncMock
)
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
@patch(
    "litellm.proxy.management_endpoints.team_endpoints.new_team", new_callable=AsyncMock
)
async def test_get_team_db_check_does_not_call_new_team_if_exists(
    mock_new_team, monkeypatch
):
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
        patch("litellm.vector_store_registry", vector_store_registry),
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch("litellm.vector_store_registry", mock_vector_store_registry),
    ):
        result = await vector_store_access_check(
            request_body=request_body,
            team_object=None,
            valid_token=valid_token,
        )

    assert result is True

    # Test with denied access
    mock_vector_store_registry.get_vector_store_ids_to_run.return_value = ["store-3"]

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch("litellm.vector_store_registry", mock_vector_store_registry),
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch("litellm.vector_store_registry", mock_vector_store_registry),
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch("litellm.vector_store_registry", mock_vector_store_registry),
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


# -- Team-member access-group resolution with team-scoped DB models -----------


def _make_team_scoped_router(team_id: str = "team-a"):
    """
    Build a Router whose model_list looks like what the proxy creates for
    team-scoped BYOK DB models: the internal model_name is
    ``<public_name>_<team_id>_<uuid>`` and the public name lives in
    ``model_info.team_public_model_name``.  Two models belong to the
    access group ``fast-models``; one (``mock-power``) does not.
    """
    from litellm import Router

    model_list = [
        {
            "model_name": f"mock-fast-1_{team_id}_aaa",
            "litellm_params": {
                "model": "openai/mock-fast-1",
                "api_key": "fake",
            },
            "model_info": {
                "id": f"demo-mock-fast-1-{team_id}",
                "team_id": team_id,
                "team_public_model_name": "mock-fast-1",
                "access_groups": ["fast-models"],
            },
        },
        {
            "model_name": f"mock-fast-2_{team_id}_bbb",
            "litellm_params": {
                "model": "openai/mock-fast-2",
                "api_key": "fake",
            },
            "model_info": {
                "id": f"demo-mock-fast-2-{team_id}",
                "team_id": team_id,
                "team_public_model_name": "mock-fast-2",
                "access_groups": ["fast-models"],
            },
        },
        {
            "model_name": f"mock-power_{team_id}_ccc",
            "litellm_params": {
                "model": "openai/mock-power",
                "api_key": "fake",
            },
            "model_info": {
                "id": f"demo-mock-power-{team_id}",
                "team_id": team_id,
                "team_public_model_name": "mock-power",
            },
        },
    ]
    return Router(model_list=model_list)


def test_can_object_call_model_access_group_with_team_id():
    """
    When team_id is passed, _can_object_call_model should resolve
    model_info.access_groups for team-scoped DB models and allow
    access via group name.
    """
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    router = _make_team_scoped_router()

    result = _can_object_call_model(
        model="mock-fast-1",
        llm_router=router,
        models=["fast-models", "mock-power"],
        object_type="team",
        team_id="team-a",
    )
    assert result is True


def test_can_object_call_model_access_group_without_team_id_fails():
    """
    Without team_id the router cannot find team-scoped DB models, so
    access group resolution fails and the call is denied.
    This is the pre-fix behavior.
    """
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    router = _make_team_scoped_router()

    with pytest.raises(ProxyException):
        _can_object_call_model(
            model="mock-fast-1",
            llm_router=router,
            models=["fast-models", "mock-power"],
            object_type="team",
            # team_id intentionally omitted
        )


def test_can_object_call_model_literal_name_with_team_id():
    """
    Literal model name matching should still work when team_id is
    passed — no regression from adding team_id.
    """
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    router = _make_team_scoped_router()

    result = _can_object_call_model(
        model="mock-power",
        llm_router=router,
        models=["fast-models", "mock-power"],
        object_type="team",
        team_id="team-a",
    )
    assert result is True


def test_can_object_call_model_denied_model_with_team_id():
    """
    A model not in the allowed list (by name or access group) should
    still be denied even when team_id is passed.
    """
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    router = _make_team_scoped_router()

    with pytest.raises(ProxyException):
        _can_object_call_model(
            model="mock-vision",
            llm_router=router,
            models=["fast-models", "mock-power"],
            object_type="team",
            team_id="team-a",
        )


def test_can_object_call_model_second_group_member_with_team_id():
    """
    Both models in the access group should be reachable, not just
    the first one.
    """
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    router = _make_team_scoped_router()

    result = _can_object_call_model(
        model="mock-fast-2",
        llm_router=router,
        models=["fast-models"],
        object_type="team",
        team_id="team-a",
    )
    assert result is True


@pytest.mark.asyncio
async def test_check_team_member_model_access_with_access_group():
    """
    End-to-end test of _check_team_member_model_access: a member whose
    allowed_models contains an access group name should be allowed to
    call models in that group for team-scoped DB models.
    """
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_TeamMembership,
        LiteLLM_TeamTable,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.auth_checks import _check_team_member_model_access

    router = _make_team_scoped_router()
    team = LiteLLM_TeamTable(team_id="team-a")
    token = UserAPIKeyAuth(token="sk-test", user_id="alice", team_id="team-a")
    membership = LiteLLM_TeamMembership(
        user_id="alice",
        team_id="team-a",
        litellm_budget_table=LiteLLM_BudgetTable(
            allowed_models=["fast-models", "mock-power"],
        ),
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_membership",
        return_value=membership,
    ):
        # Should not raise — mock-fast-1 is in the fast-models group
        await _check_team_member_model_access(
            model="mock-fast-1",
            team_object=team,
            valid_token=token,
            llm_router=router,
            prisma_client=None,
            user_api_key_cache=MagicMock(),
            proxy_logging_obj=MagicMock(),
        )


@pytest.mark.asyncio
async def test_check_team_member_model_access_denied_model():
    """
    A member with per-member allowed_models should be denied access to
    a model that is neither listed by name nor covered by an access group.
    """
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_TeamMembership,
        LiteLLM_TeamTable,
        ProxyException,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.auth_checks import _check_team_member_model_access

    router = _make_team_scoped_router()
    team = LiteLLM_TeamTable(team_id="team-a")
    token = UserAPIKeyAuth(token="sk-test", user_id="alice", team_id="team-a")
    membership = LiteLLM_TeamMembership(
        user_id="alice",
        team_id="team-a",
        litellm_budget_table=LiteLLM_BudgetTable(
            allowed_models=["fast-models", "mock-power"],
        ),
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_membership",
        return_value=membership,
    ):
        with pytest.raises(ProxyException) as exc_info:
            await _check_team_member_model_access(
                model="mock-vision",
                team_object=team,
                valid_token=token,
                llm_router=router,
                prisma_client=None,
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=MagicMock(),
            )
        assert exc_info.value.type == ProxyErrorTypes.team_model_access_denied
        assert int(exc_info.value.code) == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_check_team_member_model_access_no_override_inherits_team():
    """
    When a member has no allowed_models (empty budget table), the function
    should return without raising — the team-level check applies instead.
    """
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_TeamMembership,
        LiteLLM_TeamTable,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.auth_checks import _check_team_member_model_access

    router = _make_team_scoped_router()
    team = LiteLLM_TeamTable(team_id="team-a")
    token = UserAPIKeyAuth(token="sk-test", user_id="bob", team_id="team-a")
    membership = LiteLLM_TeamMembership(
        user_id="bob",
        team_id="team-a",
        litellm_budget_table=LiteLLM_BudgetTable(),
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_membership",
        return_value=membership,
    ):
        # Should return without raising — no per-member restriction
        await _check_team_member_model_access(
            model="mock-vision",
            team_object=team,
            valid_token=token,
            llm_router=router,
            prisma_client=None,
            user_api_key_cache=MagicMock(),
            proxy_logging_obj=MagicMock(),
        )


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

    # Mock cached tags — must be LiteLLM_TagTable instances: the mocked async_get_cache
    # bypasses UserApiKeyCache deserialization, so returning plain dicts would flow through
    # as dict (production returns models after Codec.deserialize inside the cache).
    cached_tag_1 = LiteLLM_TagTable(
        tag_name="cached-1",
        spend=10.0,
        models=[],
        litellm_budget_table=None,
    )
    cached_tag_2 = LiteLLM_TagTable(
        tag_name="cached-2",
        spend=20.0,
        models=[],
        litellm_budget_table=None,
    )

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
    async def mock_get_cache(*args, **kwargs):
        key = kwargs.get("key")
        if key == "tag:cached-1":
            return cached_tag_1
        if key == "tag:cached-2":
            return cached_tag_2
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
async def test_reject_clientside_metadata_tags_allows_key_tags_without_client_tags():
    """Key metadata.tags are injected after the reject check; requests without
    client metadata.tags must not be blocked when reject_clientside_metadata_tags is on.
    """
    from fastapi import Request

    from litellm.proxy.auth.auth_checks import common_checks

    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    general_settings = {"reject_clientside_metadata_tags": True}
    mock_request = MagicMock(spec=Request)
    valid_token = UserAPIKeyAuth(
        token="test-token",
        models=["gpt-3.5-turbo"],
        metadata={"tags": ["engineering"]},
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_tag_objects_batch",
        new_callable=AsyncMock,
        return_value={},
    ):
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
    assert request_body["metadata"]["tags"] == ["engineering"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        "/bedrock/model/us.anthropic.claude-sonnet-4-6/invoke",
        "/v1/messages",
    ],
)
async def test_common_checks_metadata_route_keeps_key_tags_out_of_provider_metadata(
    route,
):
    """GH#30629: on routes that track tags in litellm_metadata (bedrock, /v1/messages,
    responses, ...) key-level tags must land in litellm_metadata, never in the
    provider-facing metadata field (Bedrock rejects non-user_id metadata with HTTP 400).
    The auth-time pre-seed keys off LITELLM_METADATA_ROUTES, so hardcoding a single route
    or dropping the pre-seed makes apply_key_tags_pre_auth fall back to metadata; this
    guards that regression.
    """
    from fastapi import Request

    from litellm.proxy.auth.auth_checks import common_checks

    request_body = {"messages": [{"role": "user", "content": "test"}]}

    mock_request = MagicMock(spec=Request)
    valid_token = UserAPIKeyAuth(
        token="test-token",
        metadata={"tags": ["engineering"]},
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_tag_objects_batch",
        new_callable=AsyncMock,
        return_value={},
    ):
        result = await common_checks(
            request_body=request_body,
            team_object=None,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route=route,
            llm_router=None,
            proxy_logging_obj=MagicMock(),
            valid_token=valid_token,
            request=mock_request,
        )

    assert result is True
    assert request_body["litellm_metadata"]["tags"] == ["engineering"]
    assert "metadata" not in request_body


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
async def test_virtual_key_max_budget_alert_check_with_multi_threshold_map():
    """Test that max_budget_alert_emails map from metadata is attached to CallInfo on the new path"""
    alert_triggered = False
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered, captured_call_info
            alert_triggered = True
            captured_call_info = user_info

    alert_config = {
        "50": ["finance@co.com"],
        "75": ["finance@co.com", "bu_lead@co.com"],
    }
    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=60.0,
        max_budget=100.0,
        user_id="test-user",
        key_alias="test-key",
        metadata={"max_budget_alert_emails": alert_config},
    )
    user_obj = LiteLLM_UserTable(
        user_id="test-user",
        user_email="owner@co.com",
        max_budget=None,
    )

    await _virtual_key_max_budget_alert_check(
        valid_token=valid_token,
        proxy_logging_obj=MockProxyLogging(),
        user_obj=user_obj,
    )
    await asyncio.sleep(0.1)

    assert alert_triggered is True
    assert captured_call_info is not None
    assert captured_call_info.max_budget_alert_emails == alert_config
    assert captured_call_info.user_email == "owner@co.com"
    assert captured_call_info.event_group == Litellm_EntityType.KEY


@pytest.mark.asyncio
async def test_virtual_key_max_budget_alert_check_old_path_no_map():
    """Test that old single-threshold path is used when no max_budget_alert_emails in metadata"""
    alert_triggered = False
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered, captured_call_info
            alert_triggered = True
            captured_call_info = user_info

    # spend=90 is above 80% of 100 → old path should fire
    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=90.0,
        max_budget=100.0,
        user_id="test-user",
        key_alias="test-key",
        metadata={},
    )

    await _virtual_key_max_budget_alert_check(
        valid_token=valid_token,
        proxy_logging_obj=MockProxyLogging(),
        user_obj=None,
    )
    await asyncio.sleep(0.1)

    assert alert_triggered is True
    assert captured_call_info is not None
    assert captured_call_info.max_budget_alert_emails is None


@pytest.mark.asyncio
async def test_virtual_key_max_budget_alert_check_old_path_below_threshold_no_alert():
    """Test that old path does NOT fire when spend is below 80% and no map is set"""
    alert_triggered = False

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered
            alert_triggered = True

    # spend=50 is below 80% of 100 → should NOT fire
    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=50.0,
        max_budget=100.0,
        user_id="test-user",
        key_alias="test-key",
        metadata={},
    )

    await _virtual_key_max_budget_alert_check(
        valid_token=valid_token,
        proxy_logging_obj=MockProxyLogging(),
        user_obj=None,
    )
    await asyncio.sleep(0.1)

    assert alert_triggered is False


@pytest.mark.asyncio
async def test_virtual_key_max_budget_alert_check_global_fallback():
    """Test that litellm.default_key_max_budget_alert_emails is used when key metadata has no map"""
    alert_triggered = False
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered, captured_call_info
            alert_triggered = True
            captured_call_info = user_info

    global_config = {
        "50": ["global-finance@co.com"],
        "75": ["global-finance@co.com", "global-lead@co.com"],
    }
    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=60.0,
        max_budget=100.0,
        user_id="test-user",
        key_alias="test-key",
        metadata={},  # no per-key config
    )

    import litellm

    original = litellm.default_key_max_budget_alert_emails
    try:
        litellm.default_key_max_budget_alert_emails = global_config
        await _virtual_key_max_budget_alert_check(
            valid_token=valid_token,
            proxy_logging_obj=MockProxyLogging(),
            user_obj=None,
        )
        await asyncio.sleep(0.1)

        assert alert_triggered is True
        assert captured_call_info.max_budget_alert_emails == global_config
    finally:
        litellm.default_key_max_budget_alert_emails = original


@pytest.mark.asyncio
async def test_virtual_key_max_budget_alert_check_per_key_merges_with_global():
    """Test that per-key and global configs are additively merged"""
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal captured_call_info
            captured_call_info = user_info

    per_key_config = {"50": ["per-key@co.com"]}
    global_config = {"75": ["global@co.com"]}

    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=60.0,
        max_budget=100.0,
        user_id="test-user",
        key_alias="test-key",
        metadata={"max_budget_alert_emails": per_key_config},
    )

    import litellm

    original = litellm.default_key_max_budget_alert_emails
    try:
        litellm.default_key_max_budget_alert_emails = global_config
        await _virtual_key_max_budget_alert_check(
            valid_token=valid_token,
            proxy_logging_obj=MockProxyLogging(),
            user_obj=None,
        )
        await asyncio.sleep(0.1)

        # Additive merge: both thresholds present, recipients merged per threshold
        assert captured_call_info.max_budget_alert_emails == {
            "50": ["per-key@co.com"],
            "75": ["global@co.com"],
        }
    finally:
        litellm.default_key_max_budget_alert_emails = original


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


@pytest.mark.asyncio
async def test_custom_auth_common_checks_opt_in():
    """
    Test that common_checks only runs for a custom-auth deployment when
    custom_auth_run_common_checks is explicitly set to True in general_settings.

    After the centralization refactor, common_checks runs in the
    ``user_api_key_auth`` wrapper via ``_run_centralized_common_checks``
    (not inside ``_run_post_custom_auth_checks``). The opt-in flag now
    gates the centralized gate for custom-auth deployments, preserving
    the pre-existing RPS guarantee for custom-auth hot paths.
    """
    import litellm.proxy.proxy_server as _proxy_server_mod
    from litellm.proxy.auth.user_api_key_auth import _run_centralized_common_checks

    valid_token = UserAPIKeyAuth(token="test-token", user_id="u1")
    mock_request = MagicMock()

    def _attrs(flag, user_custom_auth):
        return {
            "prisma_client": None,
            "user_api_key_cache": MagicMock(),
            "proxy_logging_obj": MagicMock(),
            "general_settings": (
                {"custom_auth_run_common_checks": True} if flag else {}
            ),
            "llm_router": None,
            "user_custom_auth": user_custom_auth,
            "litellm_proxy_admin_name": "admin",
            "master_key": "sk-test-master",
        }

    # Default (no flag) with custom auth configured — centralized gate
    # SHOULD skip to preserve custom-auth RPS.
    attrs = _attrs(flag=False, user_custom_auth=AsyncMock())
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_common:
            await _run_centralized_common_checks(
                user_api_key_auth_obj=valid_token,
                request=mock_request,
                request_data={},
                route="/chat/completions",
            )
            mock_common.assert_not_called()
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)

    # With flag=True and custom auth configured — common_checks SHOULD run.
    attrs = _attrs(flag=True, user_custom_auth=AsyncMock())
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_common:
            await _run_centralized_common_checks(
                user_api_key_auth_obj=valid_token,
                request=mock_request,
                request_data={},
                route="/chat/completions",
            )
            mock_common.assert_called_once()
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


# =====================================================================
# Spend counter budget check tests (v2 — Redis-backed spend counters)
# =====================================================================


@pytest.mark.asyncio
async def test_virtual_key_budget_check_reads_from_spend_counter():
    """Budget check should use get_current_spend when counter exists,
    even if cached object shows lower spend."""
    from litellm.proxy.utils import ProxyLogging

    valid_token = UserAPIKeyAuth(
        token="test-hashed-token",
        spend=0.0,  # stale — counter has 1.5
        max_budget=1.0,
        user_id="test-user",
    )

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)
    proxy_logging_obj.budget_alerts = AsyncMock()

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:key:test-hashed-token":
            return 1.5
        return fallback_spend

    with patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _virtual_key_max_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=proxy_logging_obj,
            )
        assert exc_info.value.current_cost == 1.5
        assert exc_info.value.max_budget == 1.0


@pytest.mark.asyncio
async def test_virtual_key_budget_check_fallback_no_counter():
    """When counter doesn't exist, budget check should fall back
    to cached object's spend via fallback_spend."""
    from litellm.proxy.utils import ProxyLogging

    valid_token = UserAPIKeyAuth(
        token="test-hashed-token",
        spend=15.0,
        max_budget=10.0,
        user_id="test-user",
    )

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)
    proxy_logging_obj.budget_alerts = AsyncMock()

    # get_current_spend returns fallback_spend when no counter exists
    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        return fallback_spend

    with patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _virtual_key_max_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=proxy_logging_obj,
            )
        assert exc_info.value.current_cost == 15.0


# =====================================================================
# Throttle-on-budget-exceeded tests (LIT-3894): an over-budget key that
# opted in is throttled to a global % of its TPM/RPM instead of blocked.
# =====================================================================


def _over_budget_token(**overrides) -> UserAPIKeyAuth:
    base = dict(
        token="throttle-token",
        spend=20.0,
        max_budget=10.0,
        user_id="test-user",
    )
    base.update(overrides)
    return UserAPIKeyAuth(**base)


def _patched_spend(value: float):
    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        return value

    return patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend)


def _budget_logging_obj():
    from litellm.proxy.utils import ProxyLogging

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)
    proxy_logging_obj.budget_alerts = AsyncMock()
    return proxy_logging_obj


@pytest.mark.parametrize(
    "limit, pct, expected",
    [
        (1000, 0.1, 100),
        (100, 0.1, 10),
        (1, 0.1, 1),  # floor would be 0; trickle of 1 keeps the key alive
        (None, 0.1, None),
        (50, 0.5, 25),
        (1000, None, 1000),  # no percentage -> limit unchanged
    ],
)
def test_throttled_limit(limit, pct, expected):
    from litellm.proxy.auth.budget_throttle import throttled_limit

    assert throttled_limit(limit, pct) == expected


@pytest.mark.asyncio
async def test_budget_exceeded_throttles_instead_of_blocking(monkeypatch):
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", 0.1)
    valid_token = _over_budget_token(
        tpm_limit=1000,
        rpm_limit=100,
        metadata={"throttle_on_budget_exceeded": True},
    )

    with _patched_spend(20.0):
        await _virtual_key_max_budget_check(
            valid_token=valid_token,
            proxy_logging_obj=_budget_logging_obj(),
        )

    # persistent limits are untouched (so the throttle never compounds); the
    # request-scoped percentage is what the rate limiter scales by
    assert valid_token.budget_throttle_pct == 0.1
    assert valid_token.tpm_limit == 1000
    assert valid_token.rpm_limit == 100
    # the request-scoped decision must not leak into serialized responses
    assert "budget_throttle_pct" not in valid_token.model_dump()


@pytest.mark.asyncio
async def test_budget_throttle_decision_cleared_before_caching():
    """The request-scoped throttle decision must not persist into the key cache,
    otherwise it would re-apply (and compound) on every subsequent request."""
    from litellm.proxy.auth.auth_checks import _copy_user_api_key_auth_for_cache

    valid_token = _over_budget_token(
        tpm_limit=1000, rpm_limit=100, metadata={"throttle_on_budget_exceeded": True}
    )
    valid_token.budget_throttle_pct = 0.1

    cached = _copy_user_api_key_auth_for_cache(user_api_key_obj=valid_token)

    assert cached.budget_throttle_pct is None
    assert cached.tpm_limit == 1000
    assert cached.rpm_limit == 100


@pytest.mark.asyncio
async def test_budget_exceeded_throttle_no_configured_limits(monkeypatch):
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", 0.1)
    valid_token = _over_budget_token(metadata={"throttle_on_budget_exceeded": True})
    assert valid_token.tpm_limit is None
    assert valid_token.rpm_limit is None

    with _patched_spend(20.0):
        with pytest.raises(litellm.BudgetExceededError):
            await _virtual_key_max_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=_budget_logging_obj(),
            )

    assert valid_token.budget_throttle_pct is None


@pytest.mark.asyncio
async def test_budget_exceeded_not_opted_in_still_blocks(monkeypatch):
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", 0.1)
    valid_token = _over_budget_token(tpm_limit=1000, rpm_limit=100)

    with _patched_spend(20.0):
        with pytest.raises(litellm.BudgetExceededError):
            await _virtual_key_max_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=_budget_logging_obj(),
            )

    assert valid_token.budget_throttle_pct is None


@pytest.mark.parametrize("pct", [None, 0, 1.5, -0.1, True])
@pytest.mark.asyncio
async def test_budget_exceeded_invalid_percentage_blocks(monkeypatch, pct):
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", pct)
    valid_token = _over_budget_token(
        tpm_limit=1000,
        rpm_limit=100,
        metadata={"throttle_on_budget_exceeded": True},
    )

    with _patched_spend(20.0):
        with pytest.raises(litellm.BudgetExceededError):
            await _virtual_key_max_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=_budget_logging_obj(),
            )

    assert valid_token.budget_throttle_pct is None


@pytest.mark.asyncio
async def test_under_budget_does_not_throttle(monkeypatch):
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", 0.1)
    valid_token = _over_budget_token(
        max_budget=100.0,
        tpm_limit=1000,
        rpm_limit=100,
        metadata={"throttle_on_budget_exceeded": True},
    )

    with _patched_spend(5.0):
        await _virtual_key_max_budget_check(
            valid_token=valid_token,
            proxy_logging_obj=_budget_logging_obj(),
        )

    assert valid_token.budget_throttle_pct is None


@pytest.mark.asyncio
async def test_team_budget_check_reads_from_spend_counter():
    """Team budget check should use get_current_spend when counter exists."""
    from litellm.proxy.utils import ProxyLogging

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        spend=0.0,  # stale
        max_budget=1.0,
    )
    valid_token = UserAPIKeyAuth(token="test-token", team_id="test-team")

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)
    proxy_logging_obj.budget_alerts = AsyncMock()

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:team:test-team":
            return 1.5
        return fallback_spend

    with patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _team_max_budget_check(
                team_object=team_object,
                valid_token=valid_token,
                proxy_logging_obj=proxy_logging_obj,
            )
        assert exc_info.value.current_cost == 1.5


@pytest.mark.asyncio
async def test_end_user_budget_check_reads_from_spend_counter():
    """End-user budget check should use get_current_spend when counter exists."""
    end_user_object = LiteLLM_EndUserTable(
        user_id="customer-1",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
    )

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:end_user:customer-1":
            return 1.5
        return fallback_spend

    with patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _check_end_user_budget(
                end_user_obj=end_user_object,
                route="/chat/completions",
            )
        assert exc_info.value.current_cost == 1.5
        assert exc_info.value.max_budget == 1.0


@pytest.mark.asyncio
async def test_tag_budget_check_reads_from_spend_counter():
    """Tag budget check should use get_current_spend when counter exists."""
    from litellm.proxy.utils import ProxyLogging

    tag_object = LiteLLM_TagTable(
        tag_name="paid-tag",
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
    )

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:tag:paid-tag":
            return 1.5
        return fallback_spend

    with (
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch(
            "litellm.proxy.auth.auth_checks.get_tag_objects_batch",
            new_callable=AsyncMock,
            return_value={"paid-tag": tag_object},
        ),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _tag_max_budget_check(
                request_body={"metadata": {"tags": ["paid-tag"]}},
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=ProxyLogging(user_api_key_cache=None),
                valid_token=UserAPIKeyAuth(token="test-token"),
            )
        assert exc_info.value.current_cost == 1.5
        assert exc_info.value.max_budget == 1.0


@pytest.mark.asyncio
async def test_team_member_budget_check_reads_from_spend_counter():
    """Team member budget check should use get_current_spend when counter exists."""
    from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_TeamMembership
    from litellm.proxy.utils import ProxyLogging

    team_object = LiteLLM_TeamTable(team_id="test-team")
    user_object = LiteLLM_UserTable(user_id="test-user")
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
    )

    team_membership = LiteLLM_TeamMembership(
        user_id="test-user",
        team_id="test-team",
        spend=0.0,  # stale
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
    )

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:team_member:test-user:test-team":
            return 1.5
        return fallback_spend

    with (
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=team_membership,
        ),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _check_team_member_budget(
                team_object=team_object,
                user_object=user_object,
                valid_token=valid_token,
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=proxy_logging_obj,
            )
        assert exc_info.value.current_cost == 1.5


class TestGuardrailModificationCheck:
    """Defense-in-depth: `_guardrail_modification_check` must 403 when the
    caller's metadata attempts to modify any guardrail-related key and the
    team lacks the `modify_guardrails` permission. Checks both the
    historically-covered `guardrails` list and the bypass toggles that
    `_get_admin_metadata` silently ignores at read time.
    """

    def _call(self, request_body):
        from litellm.proxy.auth.auth_checks import _guardrail_modification_check

        team_object = MagicMock()
        team_object.metadata = {}  # no permission
        return _guardrail_modification_check(
            request_body=request_body, team_object=team_object
        )

    def test_noop_when_no_guardrail_keys_present(self):
        # no-op — should return silently
        self._call({"metadata": {"unrelated": "value"}})

    def test_rejects_guardrails_list(self):
        from fastapi import HTTPException

        with patch(
            "litellm.proxy.guardrails.guardrail_helpers.can_modify_guardrails",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc:
                self._call({"metadata": {"guardrails": ["custom"]}})
            assert exc.value.status_code == 403

    def test_rejects_disable_global_guardrails_plural(self):
        from fastapi import HTTPException

        with patch(
            "litellm.proxy.guardrails.guardrail_helpers.can_modify_guardrails",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc:
                self._call({"metadata": {"disable_global_guardrails": True}})
            assert exc.value.status_code == 403

    def test_rejects_disable_global_guardrail_singular(self):
        """VERIA-28's originally-reported singular-key typo variant."""
        from fastapi import HTTPException

        with patch(
            "litellm.proxy.guardrails.guardrail_helpers.can_modify_guardrails",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc:
                self._call({"metadata": {"disable_global_guardrail": True}})
            assert exc.value.status_code == 403

    def test_rejects_opted_out_global_guardrails(self):
        from fastapi import HTTPException

        with patch(
            "litellm.proxy.guardrails.guardrail_helpers.can_modify_guardrails",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc:
                self._call(
                    {"metadata": {"opted_out_global_guardrails": ["some_guardrail"]}}
                )
            assert exc.value.status_code == 403

    @pytest.mark.parametrize(
        "key",
        [
            "guardrails",
            "disable_global_guardrails",
            "disable_global_guardrail",
            "opted_out_global_guardrails",
        ],
    )
    @pytest.mark.parametrize("empty_value", [{}, [], "", 0, False])
    def test_rejects_empty_value_modification(self, key, empty_value):
        """Regression: an explicitly-supplied empty/falsy value still expresses
        intent to modify and must trigger the permission check. Truthiness-based
        gating let callers bypass the check by sending e.g.
        ``metadata={"guardrails": {}}``, which downstream evaluation interpreted
        as "disable all guardrails" while the auth layer treated it as no-op.
        """
        from fastapi import HTTPException

        with patch(
            "litellm.proxy.guardrails.guardrail_helpers.can_modify_guardrails",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc:
                self._call({"metadata": {key: empty_value}})
            assert exc.value.status_code == 403

    def test_rejects_injection_via_litellm_metadata_key(self):
        """Caller can populate the OTHER metadata key; that must also 403."""
        from fastapi import HTTPException

        with patch(
            "litellm.proxy.guardrails.guardrail_helpers.can_modify_guardrails",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc:
                self._call({"litellm_metadata": {"disable_global_guardrails": True}})
            assert exc.value.status_code == 403

    def test_rejects_root_level_injection(self):
        """Top-level injection (`request_body["disable_global_guardrails"]`)
        was VERIA-28's easiest variant to hit — keep it rejected."""
        from fastapi import HTTPException

        with patch(
            "litellm.proxy.guardrails.guardrail_helpers.can_modify_guardrails",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc:
                self._call({"disable_global_guardrails": True})
            assert exc.value.status_code == 403

    def test_allows_when_team_has_permission(self):
        with patch(
            "litellm.proxy.guardrails.guardrail_helpers.can_modify_guardrails",
            return_value=True,
        ):
            # no-op, should not raise
            self._call({"metadata": {"disable_global_guardrails": True}})

    def test_rejects_string_encoded_metadata_bypass(self):
        """Regression: attacker sends metadata as JSON string to bypass the
        isinstance(dict) guard. The check must coerce the string to dict
        and evaluate guardrail modification keys inside it."""
        import json as _json

        from fastapi import HTTPException

        attacker_payload = {"disable_global_guardrails": True}
        with patch(
            "litellm.proxy.guardrails.guardrail_helpers.can_modify_guardrails",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc:
                self._call({"metadata": _json.dumps(attacker_payload)})
            assert exc.value.status_code == 403

    def test_rejects_string_encoded_litellm_metadata_bypass(self):
        """Same bypass via the litellm_metadata key."""
        import json as _json

        from fastapi import HTTPException

        attacker_payload = {"guardrails": ["evaded"]}
        with patch(
            "litellm.proxy.guardrails.guardrail_helpers.can_modify_guardrails",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc:
                self._call({"litellm_metadata": _json.dumps(attacker_payload)})
            assert exc.value.status_code == 403

    def test_noop_when_string_is_not_json_object(self):
        """Unparseable strings should not trigger a 403 — they have no keys."""
        self._call({"metadata": "not-json"})
        self._call({"metadata": '"just a string"'})


@pytest.mark.asyncio
async def test_team_member_budget_check_falls_back_to_team_default_budget_id():
    """When a member's TeamMembership has no linked budget row, the check
    should fall back to team.metadata["team_member_budget_id"] and still
    enforce the cap. Pre-fix, this path silently skipped enforcement."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_TeamMembership
    from litellm.proxy.utils import ProxyLogging

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        metadata={"team_member_budget_id": "budget-default"},
    )
    user_object = LiteLLM_UserTable(user_id="test-user")
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
    )

    # Membership row without an attached budget.
    team_membership = LiteLLM_TeamMembership(
        user_id="test-user",
        team_id="test-team",
        spend=0.0,
        budget_id=None,
        litellm_budget_table=None,
    )

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)

    fake_budget_row = MagicMock()
    fake_budget_row.max_budget = 50.0
    fake_budget_row.dict = MagicMock(
        return_value={"budget_id": "budget-default", "max_budget": 50.0}
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=fake_budget_row
    )

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:team_member:test-user:test-team":
            return 70.0
        return fallback_spend

    user_api_key_cache = DualCache()

    with (
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=team_membership,
        ),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _check_team_member_budget(
                team_object=team_object,
                user_object=user_object,
                valid_token=valid_token,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
        assert exc_info.value.current_cost == 70.0
        assert exc_info.value.max_budget == 50.0

    # First call did perform the fallback DB lookup.
    prisma_client.db.litellm_budgettable.find_unique.assert_awaited_once()

    # Second call hits the cached budget row, no additional prisma read.
    prisma_client.db.litellm_budgettable.find_unique.reset_mock()
    with (
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=team_membership,
        ),
    ):
        with pytest.raises(litellm.BudgetExceededError) as second_exc_info:
            await _check_team_member_budget(
                team_object=team_object,
                user_object=user_object,
                valid_token=valid_token,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
    # The cached $50 cap is still being applied (not a coincidental skip)
    assert second_exc_info.value.current_cost == 70.0
    assert second_exc_info.value.max_budget == 50.0
    prisma_client.db.litellm_budgettable.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_member_budget_check_per_member_override_wins_over_team_default():
    """If a member has a per-member budget AND the team carries a
    team_member_budget_id default, the per-member value wins and the
    fallback prisma lookup is never performed."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_TeamMembership
    from litellm.proxy.utils import ProxyLogging

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        metadata={"team_member_budget_id": "budget-default"},
    )
    user_object = LiteLLM_UserTable(user_id="test-user")
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
    )

    team_membership = LiteLLM_TeamMembership(
        user_id="test-user",
        team_id="test-team",
        spend=0.0,
        budget_id="budget-override",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=200.0),
    )

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)

    # Team-default row resolves to $50. If the fallback fired (it must
    # not here), spend $70 would exceed that $50 cap and raise.
    fake_budget_row = MagicMock()
    fake_budget_row.max_budget = 50.0

    prisma_client = MagicMock()
    prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=fake_budget_row
    )

    mocked_spend = 70.0

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:team_member:test-user:test-team":
            return mocked_spend
        return fallback_spend

    # 1. spend ($70) < per-member cap ($200) → no raise, no fallback lookup.
    with (
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=team_membership,
        ),
    ):
        await _check_team_member_budget(
            team_object=team_object,
            user_object=user_object,
            valid_token=valid_token,
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            proxy_logging_obj=proxy_logging_obj,
        )

    prisma_client.db.litellm_budgettable.find_unique.assert_not_awaited()

    # 2. Now push spend above the per-member cap ($200). Must raise with
    # max_budget=200 to prove the per-member cap is the value being
    # enforced (not just that enforcement silently skipped).
    mocked_spend = 250.0
    with (
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=team_membership,
        ),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _check_team_member_budget(
                team_object=team_object,
                user_object=user_object,
                valid_token=valid_token,
                prisma_client=prisma_client,
                user_api_key_cache=DualCache(),
                proxy_logging_obj=proxy_logging_obj,
            )
    assert exc_info.value.current_cost == 250.0
    assert exc_info.value.max_budget == 200.0


@pytest.mark.asyncio
async def test_team_member_budget_check_null_clone_falls_back_to_team_default():
    """Per-member NULL max_budget falls through to the team default cap."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_TeamMembership
    from litellm.proxy.utils import ProxyLogging

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        metadata={"team_member_budget_id": "budget-default"},
    )
    user_object = LiteLLM_UserTable(user_id="test-user")
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
    )

    # Per-member row exists with NULL max_budget (the cloned-from-incomplete-default case).
    team_membership = LiteLLM_TeamMembership(
        user_id="test-user",
        team_id="test-team",
        spend=0.0,
        budget_id="budget-clone",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=None),
    )

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)

    fake_default_row = MagicMock()
    fake_default_row.max_budget = 65.0
    fake_default_row.dict = MagicMock(
        return_value={"budget_id": "budget-default", "max_budget": 65.0}
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=fake_default_row
    )

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:team_member:test-user:test-team":
            return 500.0
        return fallback_spend

    with (
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=team_membership,
        ),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _check_team_member_budget(
                team_object=team_object,
                user_object=user_object,
                valid_token=valid_token,
                prisma_client=prisma_client,
                user_api_key_cache=DualCache(),
                proxy_logging_obj=proxy_logging_obj,
            )

    assert exc_info.value.current_cost == 500.0
    assert exc_info.value.max_budget == 65.0
    prisma_client.db.litellm_budgettable.find_unique.assert_awaited_once()


@pytest.mark.asyncio
async def test_team_member_budget_check_null_clone_with_null_default_skips_enforcement():
    """When per-member and team default are both NULL, enforcement still skips."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_TeamMembership
    from litellm.proxy.utils import ProxyLogging

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        metadata={"team_member_budget_id": "budget-default"},
    )
    user_object = LiteLLM_UserTable(user_id="test-user")
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
    )

    team_membership = LiteLLM_TeamMembership(
        user_id="test-user",
        team_id="test-team",
        spend=0.0,
        budget_id="budget-clone",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=None),
    )

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)

    fake_default_row = MagicMock()
    fake_default_row.max_budget = None
    fake_default_row.dict = MagicMock(
        return_value={"budget_id": "budget-default", "max_budget": None}
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=fake_default_row
    )

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:team_member:test-user:test-team":
            return 1000.0
        return fallback_spend

    with (
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=team_membership,
        ),
    ):
        # No raise: both rows are NULL, so enforcement is correctly skipped.
        await _check_team_member_budget(
            team_object=team_object,
            user_object=user_object,
            valid_token=valid_token,
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            proxy_logging_obj=proxy_logging_obj,
        )


@pytest.mark.asyncio
async def test_team_member_budget_check_zero_team_default_treated_as_no_cap():
    """A team default budget with max_budget=0.0 (likely a stale/accidental
    write) must not block every member. The fallback path treats 0 as
    "no cap"; per-member rows still respect 0 as an explicit disable."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_TeamMembership
    from litellm.proxy.utils import ProxyLogging

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        metadata={"team_member_budget_id": "budget-default"},
    )
    user_object = LiteLLM_UserTable(user_id="test-user")
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
    )

    # No per-member row -> falls through to team default.
    team_membership = LiteLLM_TeamMembership(
        user_id="test-user",
        team_id="test-team",
        spend=0.0,
        budget_id=None,
        litellm_budget_table=None,
    )

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)

    # Team default budget row with max_budget=0.0 (the regression trigger).
    fake_default_row = MagicMock()
    fake_default_row.max_budget = 0.0
    fake_default_row.dict = MagicMock(
        return_value={"budget_id": "budget-default", "max_budget": 0.0}
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=fake_default_row
    )

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:team_member:test-user:test-team":
            return 0.0
        return fallback_spend

    with (
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=team_membership,
        ),
    ):
        # No raise: 0.0 cap is treated as "no cap configured".
        await _check_team_member_budget(
            team_object=team_object,
            user_object=user_object,
            valid_token=valid_token,
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            proxy_logging_obj=proxy_logging_obj,
        )


@pytest.mark.asyncio
async def test_team_member_budget_check_zero_per_member_row_still_blocks():
    """A per-member row with max_budget=0.0 is treated as an explicit admin
    disable - enforcement still blocks. Only the team-default fallback
    path treats 0 as no cap."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_TeamMembership
    from litellm.proxy.utils import ProxyLogging

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        metadata={"team_member_budget_id": "budget-default"},
    )
    user_object = LiteLLM_UserTable(user_id="test-user")
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
    )

    # Per-member row with max_budget=0.0 - admin intent: disable this user.
    team_membership = LiteLLM_TeamMembership(
        user_id="test-user",
        team_id="test-team",
        spend=0.0,
        budget_id="budget-disable",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=0.0),
    )

    proxy_logging_obj = ProxyLogging(user_api_key_cache=None)

    prisma_client = MagicMock()
    prisma_client.db.litellm_budgettable.find_unique = AsyncMock(return_value=None)

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:team_member:test-user:test-team":
            return 0.0
        return fallback_spend

    with (
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=team_membership,
        ),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _check_team_member_budget(
                team_object=team_object,
                user_object=user_object,
                valid_token=valid_token,
                prisma_client=prisma_client,
                user_api_key_cache=DualCache(),
                proxy_logging_obj=proxy_logging_obj,
            )
    assert exc_info.value.max_budget == 0.0


# --- resolve_and_validate_end_user_id ---------------------------------------


@pytest.fixture
def _validate_flag_on(monkeypatch):
    """Enable opt-in DB validation for the duration of a test."""
    import litellm

    monkeypatch.setattr(litellm, "validate_end_user_id_in_db", True)
    monkeypatch.setattr(litellm, "max_end_user_budget_id", None)


def _validation_cache():
    cache = MagicMock()
    cache.async_get_cache = AsyncMock(return_value=None)
    cache.async_set_cache = AsyncMock()
    return cache


def _patch_validation_helpers(monkeypatch, *, end_user=None, user=None, fuzzy=None):
    """Stub out the DB helpers resolve_and_validate_end_user_id delegates to."""
    from litellm.proxy.auth import auth_checks

    monkeypatch.setattr(
        auth_checks, "get_end_user_object", AsyncMock(return_value=end_user)
    )
    monkeypatch.setattr(auth_checks, "get_user_object", AsyncMock(return_value=user))
    monkeypatch.setattr(
        auth_checks, "_get_fuzzy_user_object", AsyncMock(return_value=fuzzy)
    )


@pytest.mark.asyncio
async def test_resolve_end_user_returns_none_for_none_input(
    _validate_flag_on, monkeypatch
):
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    _patch_validation_helpers(monkeypatch)
    cache = _validation_cache()
    assert (
        await resolve_and_validate_end_user_id(
            raw_end_user_id=None,
            prisma_client=MagicMock(),
            user_api_key_cache=cache,
        )
        is None
    )


@pytest.mark.asyncio
async def test_resolve_end_user_passes_through_when_flag_disabled(monkeypatch):
    """Default behaviour: flag is off, arbitrary ids pass through untouched."""
    import litellm
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    monkeypatch.setattr(litellm, "validate_end_user_id_in_db", False)
    _patch_validation_helpers(monkeypatch)
    cache = _validation_cache()

    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="codex-session-abc",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    assert result == "codex-session-abc"
    cache.async_set_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_end_user_passes_through_when_no_prisma_client(
    _validate_flag_on, monkeypatch
):
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    _patch_validation_helpers(monkeypatch)
    cache = _validation_cache()

    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="alice@example.com",
        prisma_client=None,
        user_api_key_cache=cache,
    )
    assert result == "alice@example.com"


@pytest.mark.asyncio
async def test_resolve_end_user_matches_end_user_table(_validate_flag_on, monkeypatch):
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    _patch_validation_helpers(monkeypatch, end_user=MagicMock())
    cache = _validation_cache()

    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="customer-123",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    assert result == "customer-123"
    cache.async_set_cache.assert_awaited_once()
    kwargs = cache.async_set_cache.await_args.kwargs
    assert kwargs["key"] == "end_user_validation:customer-123"
    assert kwargs["value"] == "valid"


@pytest.mark.asyncio
async def test_resolve_end_user_matches_user_table_by_user_id(
    _validate_flag_on, monkeypatch
):
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    _patch_validation_helpers(monkeypatch, user=MagicMock())
    cache = _validation_cache()

    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="user-xyz",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    assert result == "user-xyz"
    # email fallback should not run for a non-email input
    auth_checks._get_fuzzy_user_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_end_user_matches_user_table_by_email(
    _validate_flag_on, monkeypatch
):
    """Email-shaped ids route through get_user_object with user_email set.

    The fuzzy lookup must happen inside get_user_object so it shares the
    _should_check_db throttle and user_api_key_cache — no direct raw
    Prisma calls on the auth path.
    """
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    _patch_validation_helpers(monkeypatch, user=MagicMock())
    cache = _validation_cache()

    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="Alice@Example.com",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    assert result == "Alice@Example.com"
    auth_checks.get_user_object.assert_awaited_once()
    user_kwargs = auth_checks.get_user_object.await_args.kwargs
    assert user_kwargs["user_id"] == "Alice@Example.com"
    assert user_kwargs["user_email"] == "Alice@Example.com"
    # email branch must not bypass the cached helper with a raw fuzzy call
    auth_checks._get_fuzzy_user_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_end_user_non_email_id_does_not_pass_user_email(
    _validate_flag_on, monkeypatch
):
    """Non-email ids skip the email fuzzy path to avoid a pointless DB hit."""
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    _patch_validation_helpers(monkeypatch, user=MagicMock())
    cache = _validation_cache()

    await resolve_and_validate_end_user_id(
        raw_end_user_id="user-xyz",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    auth_checks.get_user_object.assert_awaited_once()
    user_kwargs = auth_checks.get_user_object.await_args.kwargs
    assert user_kwargs["user_email"] is None


@pytest.mark.asyncio
async def test_resolve_end_user_drops_codex_opaque_identifier(
    _validate_flag_on, monkeypatch
):
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    _patch_validation_helpers(monkeypatch)  # all helpers return None
    cache = _validation_cache()

    codex_id = (
        "user_8a4a360c36621665b341e06fb76041d9b6def732bb183eea148d4abc9d97c1de"
        "_account__session_a2bce4a5-8887-44ef-b491-fbf0a55c6569"
    )
    result = await resolve_and_validate_end_user_id(
        raw_end_user_id=codex_id,
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    assert result is None
    cache.async_set_cache.assert_awaited_once()
    kwargs = cache.async_set_cache.await_args.kwargs
    assert kwargs["value"] == "invalid"


@pytest.mark.asyncio
async def test_resolve_end_user_preserves_id_when_default_budget_configured(
    _validate_flag_on, monkeypatch
):
    """Don't drop unregistered ids when litellm.max_end_user_budget_id is set.

    The default end-user budget is applied downstream when the id is present
    but not found in the db — dropping the id here would bypass those limits.
    """
    import litellm
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    monkeypatch.setattr(litellm, "max_end_user_budget_id", "default-budget")
    _patch_validation_helpers(monkeypatch)
    cache = _validation_cache()

    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="new-customer",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    assert result == "new-customer"


@pytest.mark.asyncio
async def test_resolve_end_user_drops_unknown_email(_validate_flag_on, monkeypatch):
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    _patch_validation_helpers(monkeypatch)
    cache = _validation_cache()

    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="stranger@example.com",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    assert result is None


@pytest.mark.asyncio
async def test_resolve_end_user_uses_cached_valid_result(
    _validate_flag_on, monkeypatch
):
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    _patch_validation_helpers(monkeypatch)
    cache = _validation_cache()
    cache.async_get_cache = AsyncMock(return_value="valid")

    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="alice@example.com",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    assert result == "alice@example.com"
    auth_checks.get_end_user_object.assert_not_awaited()
    auth_checks.get_user_object.assert_not_awaited()
    auth_checks._get_fuzzy_user_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_end_user_uses_cached_invalid_result(
    _validate_flag_on, monkeypatch
):
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    _patch_validation_helpers(monkeypatch, end_user=MagicMock())
    cache = _validation_cache()
    cache.async_get_cache = AsyncMock(return_value="invalid")

    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="bogus",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    assert result is None
    # Despite a matching row configured, helpers aren't called — cache wins.
    auth_checks.get_end_user_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_end_user_swallows_db_errors_and_returns_none(
    _validate_flag_on, monkeypatch
):
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    monkeypatch.setattr(
        auth_checks,
        "get_end_user_object",
        AsyncMock(side_effect=Exception("db down")),
    )
    monkeypatch.setattr(
        auth_checks,
        "get_user_object",
        AsyncMock(side_effect=Exception("db down")),
    )
    cache = _validation_cache()

    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="alice@example.com",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    # DB errors shouldn't raise through the auth path — treat as unknown.
    assert result is None


@pytest.mark.asyncio
async def test_resolve_end_user(_validate_flag_on, monkeypatch):
    """Verify that resolve_and_validate_end_user_id does NOT raise BudgetExceededError.

    Note: As of the refactor that moved _check_end_user_budget out of
    get_end_user_object, budget enforcement now happens in common_checks().

    The end-user validation path should return the user ID regardless of budget status.
    Budget enforcement for end users happens later in common_checks() via
    _check_end_user_budget(), which respects skip_budget_checks for zero-cost models.

    This test verifies that even when get_end_user_object returns a user with a budget,
    resolve_and_validate_end_user_id does not block the request - budget enforcement
    is deferred to common_checks() where skip_budget_checks logic can be applied.
    """
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id

    # Mock get_end_user_object to return a user with budget info
    # (simulating a user who may have exceeded their budget)
    mock_end_user = MagicMock()
    mock_end_user.user_id = "customer-over-budget"
    monkeypatch.setattr(
        auth_checks,
        "get_end_user_object",
        AsyncMock(return_value=mock_end_user),
    )
    cache = _validation_cache()

    # resolve_and_validate_end_user_id should return the user ID without raising
    # BudgetExceededError - budget enforcement happens in common_checks()
    result = await resolve_and_validate_end_user_id(
        raw_end_user_id="customer-over-budget",
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
    )
    assert result == "customer-over-budget"


@pytest.mark.asyncio
async def test_cache_team_object_writes_team_id_and_invalidates_team_alias():
    """
    Regression pin for LIT-3244 patch/1.86.0 follow-up.

    `_cache_team_object` is the canonical "refresh this team" primitive.
    Two cache keys are in play:
      - "team_id:<id>"    — used by `get_team_object(team_id=...)`,
                            i.e. API-key auth and JWT-with-team_id_jwt_field
      - "team_alias:<alias>" — used by `get_team_object_by_alias(team_alias=...)`,
                               i.e. JWT-with-team_alias_jwt_field

    Invariants this test pins:
      1. Writes the team_id-keyed entry with the refreshed object (team_id
         is the table PK — guaranteed unique, safe to write).
      2. DELETES (does NOT write) the team_alias-keyed entry. `team_alias`
         has no UNIQUE constraint in schema.prisma, so writing it from
         this generic refresh path would let a team admin who renames
         their team to collide with another team's alias silently
         overwrite the cached team for JWT-by-alias auth (veria-ai
         review on #28739). Deleting forces the next JWT-by-alias
         reader through `get_team_object_by_alias`, which enforces
         len(teams)==1 before populating the cache.
      3. When team_alias is None, NO alias-key operation happens (no
         delete of an empty-keyed entry, no spurious write).
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
    from litellm.proxy.auth.auth_checks import _cache_team_object

    base_team_row = {
        "team_id": "team-1234",
        "team_alias": "H-Capacity",
        "models": ["openai/*", "bedrock-claude-sonnet-4"],
    }

    # ===== team_alias is set =====
    team_table = LiteLLM_TeamTableCachedObj(**base_team_row)
    cache = MagicMock()
    cache.async_set_cache = AsyncMock()
    cache.delete_cache = MagicMock()
    logging_obj = MagicMock()
    logging_obj.internal_usage_cache.dual_cache.async_delete_cache = AsyncMock()

    await _cache_team_object(
        team_id="team-1234",
        team_table=team_table,
        user_api_key_cache=cache,
        proxy_logging_obj=logging_obj,
    )

    # (1) team_id-keyed write fires with the refreshed object
    written_keys = [
        (c.kwargs.get("key") or c.args[0])
        for c in cache.async_set_cache.await_args_list
    ]
    assert written_keys == ["team_id:team-1234"], (
        "Only the team_id-keyed write should fire; the alias key must be "
        "deleted, NOT written. "
        f"Got writes: {written_keys}"
    )
    written_value = (
        cache.async_set_cache.await_args.kwargs.get("value")
        or cache.async_set_cache.await_args.args[1]
    )
    assert written_value is team_table

    # (2) team_alias-keyed entry is deleted in BOTH the in-memory cache
    # and the Redis dual cache (mirrors _delete_cache_key_object pattern).
    cache.delete_cache.assert_called_once_with(key="team_alias:H-Capacity")
    logging_obj.internal_usage_cache.dual_cache.async_delete_cache.assert_awaited_once_with(
        key="team_alias:H-Capacity"
    )

    # ===== team_alias is None: no alias-key operation =====
    aliasless = LiteLLM_TeamTableCachedObj(**{**base_team_row, "team_alias": None})
    cache2 = MagicMock()
    cache2.async_set_cache = AsyncMock()
    cache2.delete_cache = MagicMock()
    logging_obj2 = MagicMock()
    logging_obj2.internal_usage_cache.dual_cache.async_delete_cache = AsyncMock()

    await _cache_team_object(
        team_id="team-no-alias",
        team_table=aliasless,
        user_api_key_cache=cache2,
        proxy_logging_obj=logging_obj2,
    )

    cache2.delete_cache.assert_not_called()
    logging_obj2.internal_usage_cache.dual_cache.async_delete_cache.assert_not_awaited()
    written_keys_aliasless = [
        (c.kwargs.get("key") or c.args[0])
        for c in cache2.async_set_cache.await_args_list
    ]
    assert written_keys_aliasless == ["team_id:team-no-alias"]


MODEL_DISCOVERY_ROUTES = [
    "/v1/models",
    "/models",
    "/model/info",
    "/v1/model/info",
    "/v2/model/info",
    "/model_group/info",
]


@pytest.mark.parametrize("route", MODEL_DISCOVERY_ROUTES)
@pytest.mark.asyncio
async def test_model_discovery_route_bypasses_team_budget(route):
    """Regression for #27923: an exhausted team budget must not block model-discovery routes,
    otherwise OpenAI-compatible clients calling GET /v1/models at startup break."""
    from litellm.proxy.auth.auth_checks import common_checks

    team_object = LiteLLM_TeamTable(team_id="test-team", spend=150.0, max_budget=100.0)

    result = await common_checks(
        request_body={},
        team_object=team_object,
        user_object=None,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings={},
        route=route,
        llm_router=None,
        proxy_logging_obj=AsyncMock(),
        valid_token=UserAPIKeyAuth(token="test-token", team_id="test-team"),
        request=MagicMock(),
    )

    assert result is True


@pytest.mark.asyncio
async def test_model_discovery_route_bypasses_user_budget():
    """Regression for #27923: an exhausted user budget must not block model discovery."""
    from litellm.proxy.auth.auth_checks import common_checks

    user_object = LiteLLM_UserTable(user_id="test-user", spend=100.0, max_budget=50.0)

    result = await common_checks(
        request_body={},
        team_object=None,
        user_object=user_object,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings={},
        route="/v1/models",
        llm_router=None,
        proxy_logging_obj=AsyncMock(),
        valid_token=UserAPIKeyAuth(token="test-token", user_id="test-user"),
        request=MagicMock(),
    )

    assert result is True


@pytest.mark.asyncio
async def test_side_effectful_info_route_still_enforces_budget():
    """#27923 keeps the bypass narrow: /health/services can fire Slack/email/webhook test
    messages, so an exhausted budget must still block it. Widening the exemption back to
    is_info_route() would regress this."""
    from litellm.proxy.auth.auth_checks import common_checks

    team_object = LiteLLM_TeamTable(team_id="test-team", spend=150.0, max_budget=100.0)

    with pytest.raises(litellm.BudgetExceededError):
        await common_checks(
            request_body={},
            team_object=team_object,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/health/services",
            llm_router=None,
            proxy_logging_obj=AsyncMock(),
            valid_token=UserAPIKeyAuth(token="test-token", team_id="test-team"),
            request=MagicMock(),
        )


@pytest.mark.asyncio
async def test_inference_route_still_enforces_team_budget():
    """Control for #27923: inference routes stay fully budget-enforced."""
    from litellm.proxy.auth.auth_checks import common_checks

    team_object = LiteLLM_TeamTable(team_id="test-team", spend=150.0, max_budget=100.0)

    with pytest.raises(litellm.BudgetExceededError):
        await common_checks(
            request_body={},
            team_object=team_object,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/v1/chat/completions",
            llm_router=None,
            proxy_logging_obj=AsyncMock(),
            valid_token=UserAPIKeyAuth(token="test-token", team_id="test-team"),
            request=MagicMock(),
        )


@pytest.mark.asyncio
async def test_virtual_key_max_budget_error_names_the_key():
    """BudgetExceededError for a virtual key must name the key (alias + masked key)
    so operators don't have to reverse-map a spend figure back to a key."""
    valid_token = UserAPIKeyAuth(
        token="hashed-token",
        key_alias="payments-prod",
        key_name="sk-...um_g",
        max_budget=10.0,
        spend=0.0,
    )
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.budget_alerts = AsyncMock()

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(return_value=25.0),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _virtual_key_max_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=proxy_logging_obj,
            )

    message = str(exc_info.value)
    assert "payments-prod" in message
    assert "sk-...um_g" in message


@pytest.mark.asyncio
async def test_virtual_key_max_budget_not_exceeded_does_not_raise():
    """Spend below the configured budget must not raise."""
    valid_token = UserAPIKeyAuth(
        token="hashed-token",
        key_alias="payments-prod",
        max_budget=10.0,
        spend=0.0,
    )
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.budget_alerts = AsyncMock()

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(return_value=1.0),
    ):
        await _virtual_key_max_budget_check(
            valid_token=valid_token,
            proxy_logging_obj=proxy_logging_obj,
        )


class _TTLCapturingInMemoryCache(InMemoryCache):
    """Records the ``ttl`` DualCache forwards into the in-memory layer."""

    def __init__(self) -> None:
        super().__init__()
        self.last_ttl = None

    def set_cache(self, key, value, **kwargs):  # type: ignore[override]
        self.last_ttl = kwargs.get("ttl")
        super().set_cache(key, value, **kwargs)


class TestManagementObjectTTLHonored:
    """
    Regression for LIT-3338. ``_cache_management_object`` is the central writer on
    the reported ``get_key_object -> _cache_key_object -> _cache_management_object``
    path. It must cache for the configured ``user_api_key_cache_ttl`` (propagated to
    ``default_in_memory_ttl``) rather than the hardcoded 60s management default.
    """

    @pytest.mark.asyncio
    async def test_uses_configured_user_api_key_cache_ttl(self):
        mem = _TTLCapturingInMemoryCache()
        cache = UserApiKeyCache(in_memory_cache=mem, default_in_memory_ttl=300)

        await _cache_management_object(
            key="team_id:lit-3338",
            value=UserAPIKeyAuth(token="hash-lit-3338"),
            user_api_key_cache=cache,
            proxy_logging_obj=None,
            model_type=UserAPIKeyAuth,
        )

        assert mem.last_ttl == 300

    @pytest.mark.asyncio
    async def test_falls_back_to_management_default_when_unconfigured(self):
        mem = _TTLCapturingInMemoryCache()
        cache = UserApiKeyCache(in_memory_cache=mem)
        assert cache.default_in_memory_ttl is None

        await _cache_management_object(
            key="team_id:lit-3338-default",
            value=UserAPIKeyAuth(token="hash-default"),
            user_api_key_cache=cache,
            proxy_logging_obj=None,
            model_type=UserAPIKeyAuth,
        )

        assert mem.last_ttl == DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL


class _BudgetSpendConcurrencyProbe:
    """Stand-in for get_current_spend that pins how many scope checks are in flight.

    Each call registers itself, records the peak simultaneous count, and blocks on
    ``release`` until the test lets it proceed. ``all_arrived`` only fires once
    ``expected`` distinct scope reads are suspended here at the same time, which can
    happen only if common_checks gathers the per-scope reads instead of awaiting
    them one after another.
    """

    def __init__(self, expected: int):
        self.expected = expected
        self.in_flight = 0
        self.max_in_flight = 0
        self.all_arrived = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, *args, **kwargs) -> float:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.in_flight >= self.expected:
            self.all_arrived.set()
        try:
            await self.release.wait()
        finally:
            self.in_flight -= 1
        return 0.0


@pytest.mark.asyncio
async def test_common_checks_budget_reads_run_concurrently():
    """Independent per-scope budget reads in common_checks must run concurrently.

    team max, team window, key window, and end-user each read a distinct spend
    counter with no cross-scope dependency. With the gather they are all suspended
    in get_current_spend simultaneously; reverting to sequential awaits leaves only
    one in flight at a time, so ``all_arrived`` never fires and this test times out.
    """
    from fastapi import Request

    from litellm.proxy.auth.auth_checks import common_checks

    team = LiteLLM_TeamTable(
        team_id="t1",
        spend=0.0,
        max_budget=100.0,
        budget_limits=[{"budget_duration": "1d", "max_budget": 100.0}],
    )
    token = UserAPIKeyAuth(
        token="k1",
        budget_limits=[{"budget_duration": "1d", "max_budget": 100.0}],
    )
    end_user = LiteLLM_EndUserTable(
        user_id="eu1",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=100.0),
    )

    probe = _BudgetSpendConcurrencyProbe(expected=4)

    with patch("litellm.proxy.proxy_server.prisma_client", None), patch(
        "litellm.proxy.proxy_server.get_current_spend", probe
    ):
        task = asyncio.create_task(
            common_checks(
                request_body={"messages": [{"role": "user", "content": "hi"}]},
                team_object=team,
                user_object=None,
                end_user_object=end_user,
                global_proxy_spend=None,
                general_settings={},
                route="/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=MagicMock(spec=Request),
            )
        )
        try:
            await asyncio.wait_for(probe.all_arrived.wait(), timeout=3.0)
            assert probe.max_in_flight == 4
        finally:
            probe.release.set()
        assert await task is True


@pytest.mark.asyncio
async def test_common_checks_budget_gather_raises_highest_priority_scope():
    """A gathered scope that is over budget must still raise BudgetExceededError.

    When more than one scope is over budget the error from the highest-priority
    scope (team, matching the previous sequential order) propagates; when only a
    lower-priority scope (end-user) is over budget its error still surfaces. This
    fails if any scope is dropped from the gather or if errors are swallowed.
    """
    from fastapi import Request

    from litellm.proxy.auth.auth_checks import common_checks

    async def _spend_by_counter(counter_key, fallback_spend, max_budget=None, **kwargs):
        if counter_key == "spend:team:t1":
            return _spend_by_counter.team
        if counter_key == "spend:end_user:eu1":
            return _spend_by_counter.end_user
        return 0.0

    team = LiteLLM_TeamTable(team_id="t1", spend=0.0, max_budget=100.0)
    end_user = LiteLLM_EndUserTable(
        user_id="eu1",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=100.0),
    )

    async def _run():
        return await common_checks(
            request_body={"messages": [{"role": "user", "content": "hi"}]},
            team_object=team,
            user_object=None,
            end_user_object=end_user,
            global_proxy_spend=None,
            general_settings={},
            route="/chat/completions",
            llm_router=None,
            proxy_logging_obj=MagicMock(),
            valid_token=None,
            request=MagicMock(spec=Request),
        )

    with patch("litellm.proxy.proxy_server.prisma_client", None), patch(
        "litellm.proxy.proxy_server.get_current_spend", _spend_by_counter
    ):
        # Both team and end-user over budget: team wins on priority.
        _spend_by_counter.team = 999.0
        _spend_by_counter.end_user = 999.0
        with pytest.raises(litellm.BudgetExceededError) as both_over:
            await _run()
        assert "Team=t1" in str(both_over.value)

        # Only the lower-priority end-user scope over budget: its error still raises.
        _spend_by_counter.team = 0.0
        _spend_by_counter.end_user = 999.0
        with pytest.raises(litellm.BudgetExceededError) as end_user_over:
            await _run()
        assert "End User=eu1" in str(end_user_over.value)


@pytest.mark.asyncio
async def test_common_checks_personal_user_budget_blocks_in_gather():
    """The personal-key user budget scope is enforced inside the gather.

    For a personal key (no team) whose user is over budget, the gathered user
    check must raise BudgetExceededError. This guards the relocated personal
    user-budget read and fails if that scope is dropped from the gather.
    """
    from fastapi import Request

    from litellm.proxy.auth.auth_checks import common_checks

    user = LiteLLM_UserTable(user_id="u1", spend=0.0, max_budget=100.0)
    token = UserAPIKeyAuth(token="k1", user_id="u1")

    async def _spend_by_counter(counter_key, fallback_spend, max_budget=None, **kwargs):
        return 999.0 if counter_key == "spend:user:u1" else 0.0

    with patch("litellm.proxy.proxy_server.prisma_client", None), patch(
        "litellm.proxy.proxy_server.get_current_spend", _spend_by_counter
    ):
        with pytest.raises(litellm.BudgetExceededError) as over:
            await common_checks(
                request_body={"messages": [{"role": "user", "content": "hi"}]},
                team_object=None,
                user_object=user,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route="/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=MagicMock(spec=Request),
            )
    assert "User=u1" in str(over.value)

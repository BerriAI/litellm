import json
import os
import sys
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest
from prisma.errors import ClientNotConnectedError, HTTPClientClosedError, PrismaError

import litellm.proxy.health_endpoints._health_endpoints as _health_endpoints_module

from litellm.proxy.health_endpoints._health_endpoints import (
    _db_health_readiness_check,
    _resolve_os_environ_variables,
    get_callback_identifier,
    health_license_endpoint,
    health_services_endpoint,
)
from litellm.proxy.health_endpoints._health_endpoints import (
    test_model_connection as health_test_model_connection,
)

# Import shared proxy test helpers from conftest
from tests.test_litellm.proxy.conftest import create_proxy_test_client


@pytest.mark.asyncio
async def test_db_health_cache_hit_returns_cached():
    """
    When cache is 'connected' and within the 15s TTL, return the cache
    without calling health_check.
    """
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now(),
    }

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await _db_health_readiness_check()

    assert result["status"] == "connected"
    mock_prisma.health_check.assert_not_called()


@pytest.mark.asyncio
async def test_db_health_cache_expired_calls_health_check():
    """
    When cache is 'connected' but older than 15s, call health_check
    to re-validate the connection.
    """
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await _db_health_readiness_check()

    assert result["status"] == "connected"
    mock_prisma.health_check.assert_called_once()


@pytest.mark.asyncio
async def test_db_health_non_connected_ignores_cache_ttl():
    """
    When cache status is not 'connected' (e.g. 'disconnected', 'unknown'),
    always call health_check regardless of how fresh the cache is.
    """
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "disconnected",
        "last_updated": datetime.now(),
    }

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await _db_health_readiness_check()

    assert result["status"] == "connected"
    mock_prisma.health_check.assert_called_once()


@pytest.mark.asyncio
async def test_db_health_prisma_client_none():
    """
    When prisma_client is None, return 'disconnected' without attempting
    a health_check call.
    """
    _health_endpoints_module.db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(minutes=5),
    }

    with patch("litellm.proxy.proxy_server.prisma_client", None):
        result = await _db_health_readiness_check()

    assert result["status"] == "disconnected"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        PrismaError(),
        ClientNotConnectedError(),
        HTTPClientClosedError(),
    ],
)
async def test_db_health_error_flag_off_raises_no_reconnect(prisma_error):
    """
    When health_check raises and allow_requests_on_db_unavailable is False,
    handle_db_exception re-raises immediately. The reconnect path is never
    reached, so disconnect/connect are never called.
    """
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=prisma_error)
    mock_prisma.disconnect = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(Exception) as exc_info:
            await _db_health_readiness_check()

        assert exc_info.value is prisma_error
        mock_prisma.disconnect.assert_not_called()
        assert _health_endpoints_module.db_health_cache["status"] == "disconnected"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        PrismaError("Can't reach database server"),
        ClientNotConnectedError(),
        HTTPClientClosedError(),
    ],
)
async def test_db_health_error_flag_on_reconnect_succeeds(prisma_error):
    """
    When health_check raises, allow_requests_on_db_unavailable is True,
    and the reconnect cycle (disconnect -> connect -> health_check) succeeds,
    return 'connected' and update the cache.
    """
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=[prisma_error, None])
    mock_prisma.disconnect = AsyncMock()
    mock_prisma.connect = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": True},
        ),
    ):
        result = await _db_health_readiness_check()

    assert result["status"] == "connected"
    mock_prisma.disconnect.assert_called_once()
    mock_prisma.connect.assert_called_once()
    assert mock_prisma.health_check.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        PrismaError("Can't reach database server"),
        ClientNotConnectedError(),
        HTTPClientClosedError(),
    ],
)
async def test_db_health_error_flag_on_reconnect_fails(prisma_error):
    """
    When health_check raises, allow_requests_on_db_unavailable is True,
    but the reconnect also fails, return 'disconnected' instead of raising.
    This respects the flag's intent: keep serving even without a DB.
    """
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=prisma_error)
    mock_prisma.disconnect = AsyncMock()
    mock_prisma.connect = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": True},
        ),
    ):
        result = await _db_health_readiness_check()

    assert result["status"] == "disconnected"
    mock_prisma.disconnect.assert_called_once()
    mock_prisma.connect.assert_called_once()


@pytest.mark.asyncio
async def test_db_health_non_transport_error_flag_off_raises():
    """
    When health_check raises a non-transport error and
    allow_requests_on_db_unavailable is False, handle_db_exception
    re-raises before reaching the is_database_transport_error guard.
    Cache is still invalidated before the re-raise.
    """
    non_transport_error = PrismaError("UniqueViolationError")
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=non_transport_error)
    mock_prisma.disconnect = AsyncMock()
    mock_prisma.connect = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(PrismaError):
            await _db_health_readiness_check()

    assert _health_endpoints_module.db_health_cache["status"] == "disconnected"
    mock_prisma.disconnect.assert_not_called()
    mock_prisma.connect.assert_not_called()


@pytest.mark.asyncio
async def test_db_health_non_transport_error_flag_on_skips_reconnect():
    """
    When health_check raises a non-transport error (e.g. data-layer) and
    allow_requests_on_db_unavailable is True, handle_db_exception swallows
    the exception, then is_database_transport_error returns False so the
    reconnect cycle is skipped. Returns 'disconnected' without calling
    disconnect/connect.
    """
    non_transport_error = PrismaError("UniqueViolationError")
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=non_transport_error)
    mock_prisma.disconnect = AsyncMock()
    mock_prisma.connect = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": True},
        ),
    ):
        result = await _db_health_readiness_check()

    assert result["status"] == "disconnected"
    mock_prisma.disconnect.assert_not_called()
    mock_prisma.connect.assert_not_called()


@pytest.mark.asyncio
async def test_db_health_reconnect_disconnect_fails():
    """
    When disconnect() itself raises during the reconnect cycle,
    the inner except catches it and returns 'disconnected'.
    connect() and the second health_check() are never called.
    """
    transport_error = ClientNotConnectedError()
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=transport_error)
    mock_prisma.disconnect = AsyncMock(side_effect=RuntimeError("already closed"))
    mock_prisma.connect = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": True},
        ),
    ):
        result = await _db_health_readiness_check()

    assert result["status"] == "disconnected"
    mock_prisma.disconnect.assert_called_once()
    mock_prisma.connect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error_message",
    [
        ("healthy", ""),
        ("unhealthy", "queue not reachable"),
    ],
)
async def test_health_services_endpoint_sqs(status, error_message):
    """
    Verify the /health/services SQS branch returns expected status and message
    based on SQSLogger.async_health_check().
    """
    with patch("litellm.integrations.sqs.SQSLogger") as MockSQSLogger:
        mock_instance = MagicMock()
        mock_instance.async_health_check = AsyncMock(
            return_value={"status": status, "error_message": error_message}
        )
        MockSQSLogger.return_value = mock_instance

        result = await health_services_endpoint(service="sqs")

        assert result["status"] == status
        assert result["message"] == error_message
        mock_instance.async_health_check.assert_awaited_once()


def test_resolve_os_environ_variables_should_use_secret_manager_get_secret():
    params = {
        "api_key": "os.environ/TEST_API_KEY",
        "api_base": "https://example.com",
    }

    with patch(
        "litellm.proxy.health_endpoints._health_endpoints.get_secret",
        return_value="resolved-secret-value",
    ) as mock_get_secret:
        result = _resolve_os_environ_variables(params)

    assert result == {
        "api_key": "resolved-secret-value",
        "api_base": "https://example.com",
    }
    mock_get_secret.assert_called_once_with("os.environ/TEST_API_KEY")


def test_resolve_os_environ_variables_should_resolve_nested_dicts_and_lists():
    params = {
        "api_key": "os.environ/ROOT_SECRET",
        "headers": {
            "Authorization": "os.environ/AUTH_SECRET",
            "static": "value",
        },
        "fallbacks": [
            "os.environ/FALLBACK_SECRET",
            {
                "nested_list_key": "os.environ/NESTED_LIST_SECRET",
            },
            ["os.environ/DEEP_LIST_SECRET", "plain-value"],
        ],
    }

    resolved_values = {
        "os.environ/ROOT_SECRET": "root-secret",
        "os.environ/AUTH_SECRET": "auth-secret",
        "os.environ/FALLBACK_SECRET": "fallback-secret",
        "os.environ/NESTED_LIST_SECRET": "nested-list-secret",
        "os.environ/DEEP_LIST_SECRET": "deep-list-secret",
    }

    with patch(
        "litellm.proxy.health_endpoints._health_endpoints.get_secret",
        side_effect=lambda secret_name: resolved_values[secret_name],
    ) as mock_get_secret:
        result = _resolve_os_environ_variables(params)

    assert result == {
        "api_key": "root-secret",
        "headers": {
            "Authorization": "auth-secret",
            "static": "value",
        },
        "fallbacks": [
            "fallback-secret",
            {"nested_list_key": "nested-list-secret"},
            ["deep-list-secret", "plain-value"],
        ],
    }
    assert mock_get_secret.call_count == 5


@pytest.mark.asyncio
async def test_health_services_endpoint_email_should_use_test_email_address_from_db_when_store_model_in_db_enabled():
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_config.find_unique = AsyncMock(
        side_effect=[
            SimpleNamespace(param_value={"store_model_in_db": True}),
            SimpleNamespace(param_value={"TEST_EMAIL_ADDRESS": "encrypted-db-value"}),
        ]
    )
    mock_slack_alerting = SimpleNamespace(
        send_key_created_or_user_invited_email=AsyncMock()
    )
    mock_proxy_logging_obj = SimpleNamespace(
        slack_alerting_instance=mock_slack_alerting
    )
    mock_user_api_key_dict = SimpleNamespace(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
    )

    with patch.dict(os.environ, {"TEST_EMAIL_ADDRESS": "env@example.com"}), patch(
        "litellm.proxy.proxy_server.general_settings",
        {},
    ), patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        mock_proxy_logging_obj,
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints.get_secret_bool",
        return_value=False,
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints.decrypt_value_helper",
        return_value="db@example.com",
    ):
        result = await health_services_endpoint(
            service="email",
            user_api_key_dict=mock_user_api_key_dict,
        )

    assert result["status"] == "success"
    assert (
        mock_prisma.db.litellm_config.find_unique.await_args_list[0].kwargs["where"]
        == {"param_name": "general_settings"}
    )
    assert (
        mock_prisma.db.litellm_config.find_unique.await_args_list[1].kwargs["where"]
        == {"param_name": "environment_variables"}
    )
    webhook_event = (
        mock_slack_alerting.send_key_created_or_user_invited_email.await_args.kwargs[
            "webhook_event"
        ]
    )
    assert webhook_event.user_email == "db@example.com"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_model_in_db_secret,general_settings_row,environment_variables_row,expected_db_calls",
    [
        (False, SimpleNamespace(param_value={"store_model_in_db": False}), None, 1),
        (True, None, None, 1),
    ],
    ids=["db-disabled", "config-row-missing"],
)
async def test_health_services_endpoint_email_should_fall_back_to_env_test_email_address_when_db_disabled_or_missing(
    store_model_in_db_secret,
    general_settings_row,
    environment_variables_row,
    expected_db_calls,
):
    mock_prisma = MagicMock()
    db_rows = []
    if general_settings_row is not None:
        db_rows.append(general_settings_row)
    if store_model_in_db_secret:
        db_rows.append(environment_variables_row)
    mock_prisma.db.litellm_config.find_unique = AsyncMock(side_effect=db_rows)
    mock_slack_alerting = SimpleNamespace(
        send_key_created_or_user_invited_email=AsyncMock()
    )
    mock_proxy_logging_obj = SimpleNamespace(
        slack_alerting_instance=mock_slack_alerting
    )
    mock_user_api_key_dict = SimpleNamespace(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
    )

    with patch.dict(os.environ, {"TEST_EMAIL_ADDRESS": "env@example.com"}), patch(
        "litellm.proxy.proxy_server.general_settings",
        {},
    ), patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        mock_proxy_logging_obj,
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints.get_secret_bool",
        return_value=store_model_in_db_secret,
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints.decrypt_value_helper"
    ) as decrypt_mock:
        result = await health_services_endpoint(
            service="email",
            user_api_key_dict=mock_user_api_key_dict,
        )

    assert result["status"] == "success"
    webhook_event = (
        mock_slack_alerting.send_key_created_or_user_invited_email.await_args.kwargs[
            "webhook_event"
        ]
    )
    assert webhook_event.user_email == "env@example.com"
    assert mock_prisma.db.litellm_config.find_unique.await_count == expected_db_calls
    decrypt_mock.assert_not_called()


@pytest.mark.asyncio
async def test_health_services_endpoint_email_should_accept_json_string_environment_variables():
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_config.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            param_value=json.dumps(
                {"TEST_EMAIL_ADDRESS": "json-string-db-value"}
            )
        )
    )
    mock_slack_alerting = SimpleNamespace(
        send_key_created_or_user_invited_email=AsyncMock()
    )
    mock_proxy_logging_obj = SimpleNamespace(
        slack_alerting_instance=mock_slack_alerting
    )
    mock_user_api_key_dict = SimpleNamespace(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
    )

    with patch.dict(os.environ, {"TEST_EMAIL_ADDRESS": "env@example.com"}), patch(
        "litellm.proxy.proxy_server.general_settings",
        {},
    ), patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        mock_proxy_logging_obj,
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints.get_secret_bool",
        return_value=True,
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints.decrypt_value_helper",
        return_value="json@example.com",
    ) as decrypt_mock:
        result = await health_services_endpoint(
            service="email",
            user_api_key_dict=mock_user_api_key_dict,
        )

    assert result["status"] == "success"
    decrypt_mock.assert_called_once_with(
        value="json-string-db-value",
        key="TEST_EMAIL_ADDRESS",
        exception_type="debug",
        return_original_value=True,
    )
    webhook_event = (
        mock_slack_alerting.send_key_created_or_user_invited_email.await_args.kwargs[
            "webhook_event"
        ]
    )
    assert webhook_event.user_email == "json@example.com"


@pytest.mark.asyncio
async def test_health_license_endpoint_with_active_license():
    license_data = {
        "expiration_date": "2099-01-01",
        "allowed_features": ["feature-a"],
        "max_users": 100,
        "max_teams": 5,
    }
    mock_license_check = SimpleNamespace(
        license_str="test-license",
        public_key=None,
        airgapped_license_data=license_data,
        verify_license_without_api_request=MagicMock(return_value=True),
    )

    with (
        patch(
            "litellm.proxy.proxy_server._license_check",
            mock_license_check,
        ),
        patch(
            "litellm.proxy.proxy_server.premium_user",
            True,
        ),
        patch(
            "litellm.proxy.proxy_server.premium_user_data",
            license_data,
        ),
    ):
        response = await health_license_endpoint(user_api_key_dict=MagicMock())

    assert response["has_license"] is True
    assert response["license_type"] == "enterprise"
    assert response["expiration_date"] == "2099-01-01"
    assert response["allowed_features"] == ["feature-a"]
    assert response["limits"] == {"max_users": 100, "max_teams": 5}


@pytest.mark.asyncio
async def test_health_license_endpoint_without_valid_license():
    mock_license_check = SimpleNamespace(
        license_str="invalid-key",
        public_key=None,
        airgapped_license_data=None,
        verify_license_without_api_request=MagicMock(return_value=False),
    )

    with (
        patch(
            "litellm.proxy.proxy_server._license_check",
            mock_license_check,
        ),
        patch(
            "litellm.proxy.proxy_server.premium_user",
            False,
        ),
        patch(
            "litellm.proxy.proxy_server.premium_user_data",
            None,
        ),
    ):
        response = await health_license_endpoint(user_api_key_dict=MagicMock())

    assert response["has_license"] is True
    assert response["license_type"] == "community"
    assert response["expiration_date"] is None
    assert response["allowed_features"] == []
    assert response["limits"] == {"max_users": None, "max_teams": None}


@pytest.mark.asyncio
async def test_test_model_connection_loads_config_from_router():
    """
    Test that /health/test_connection automatically loads model configuration
    (including resolved environment variables) from the router when model name is provided.
    """
    # Mock request
    mock_request = MagicMock()

    # Mock user_api_key_dict
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.user_id = "test-user"
    mock_user_api_key_dict.token = "test-token"

    # Mock prisma_client
    mock_prisma_client = MagicMock()

    # Mock router with model configuration
    mock_router = MagicMock()
    mock_deployment = {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "azure/gpt-4o",
            "api_key": "resolved-api-key-from-env",
            "api_base": "https://resolved-endpoint.openai.azure.com/",
            "api_version": "2024-10-21",
        },
        "model_info": {},
    }
    mock_router.get_model_list.return_value = [mock_deployment]

    # Mock ModelManagementAuthChecks - patch at the source module since it's imported inside the function
    mock_can_user_make_model_call = AsyncMock()

    # Mock litellm.ahealth_check
    mock_health_check_result = {
        "status": "healthy",
        "response_time_ms": 100,
    }
    mock_ahealth_check = AsyncMock(return_value=mock_health_check_result)

    # Mock run_with_timeout
    mock_run_with_timeout = AsyncMock(return_value=mock_health_check_result)

    # Mock _update_litellm_params_for_health_check
    def mock_update_params(model_info, litellm_params):
        # Just return params with messages added
        params = litellm_params.copy()
        params["messages"] = [{"role": "user", "content": "test"}]
        return params

    # Mock _reject_os_environ_references
    def mock_reject_os_environ(params):
        return None

    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma_client,
        ),
        patch(
            "litellm.proxy.proxy_server.llm_router",
            mock_router,
        ),
        patch(
            "litellm.proxy.proxy_server.premium_user",
            False,
        ),
        patch(
            "litellm.proxy.management_endpoints.model_management_endpoints.ModelManagementAuthChecks.can_user_make_model_call",
            mock_can_user_make_model_call,
        ),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints.litellm.ahealth_check",
            mock_ahealth_check,
        ),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints.run_with_timeout",
            mock_run_with_timeout,
        ),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints._update_litellm_params_for_health_check",
            mock_update_params,
        ),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints._reject_os_environ_references",
            mock_reject_os_environ,
        ),
    ):
        # Call the endpoint with only model name (no credentials)
        result = await health_test_model_connection(
            request=mock_request,
            mode="chat",
            litellm_params={"model": "gpt-4o"},
            model_info={},
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Verify router.get_model_list was called with the model name
        mock_router.get_model_list.assert_called_once_with(model_name="gpt-4o")

        # Verify that run_with_timeout was called (which wraps ahealth_check)
        assert mock_run_with_timeout.called

        # Get the call args to verify merged params
        call_args = mock_run_with_timeout.call_args
        assert call_args is not None

        # The first arg should be the coroutine from ahealth_check
        # We need to check what was passed to ahealth_check
        ahealth_check_call_args = mock_ahealth_check.call_args
        assert ahealth_check_call_args is not None
        model_params = ahealth_check_call_args.kwargs.get("model_params", {})

        # Verify that config params were loaded and merged
        # Note: request params override config params, so model from request is used
        assert model_params.get("api_key") == "resolved-api-key-from-env"
        assert (
            model_params.get("api_base")
            == "https://resolved-endpoint.openai.azure.com/"
        )
        assert model_params.get("api_version") == "2024-10-21"
        assert (
            model_params.get("model") == "gpt-4o"
        )  # Request param overrides config param

        # Verify result
        assert result["status"] == "success"
        assert "result" in result


@pytest.mark.asyncio
async def test_health_services_endpoint_datadog_llm_observability():
    """
    Verify that 'datadog_llm_observability' is accepted as a valid service
    by the /health/services endpoint and does not raise a 400 error.

    Regression test for: https://github.com/BerriAI/litellm/issues/XXXX
    The service was missing from the allowed services validation list.
    """
    from litellm.proxy.health_endpoints._health_endpoints import (
        health_services_endpoint,
    )

    # Mock datadog_llm_observability to be in success_callback so the generic branch handles it
    with patch("litellm.success_callback", ["datadog_llm_observability"]):
        result = await health_services_endpoint(service="datadog_llm_observability")

    # Should not raise HTTPException(400) and should return success
    assert result["status"] == "success"
    assert "datadog_llm_observability" in result["message"]


@pytest.mark.asyncio
async def test_health_services_endpoint_rejects_unknown_service():
    """
    Verify that an unknown service name is rejected with a 400 error.
    """
    from litellm.proxy._types import ProxyException

    with pytest.raises(ProxyException):
        await health_services_endpoint(service="totally_unknown_service_xyz")


@pytest.fixture(scope="function")
def proxy_client(monkeypatch):
    """
    Fixture that starts a proxy server instance for testing.
    Uses the actual FastAPI app from proxy_server which includes all routers.

    Note: TestClient doesn't start a real HTTP server - it runs the FastAPI app
    in-process. However, it DOES trigger FastAPI's lifespan events (startup/shutdown)
    when used as a context manager, which initializes the proxy server components.

    Database access:
    - If DATABASE_URL is set in environment, the proxy will automatically connect
    - Database connection happens during lifespan startup events
    - To enable database access, set DATABASE_URL environment variable before running tests

    Redis cache:
    - If REDIS_HOST is set in environment, Redis cache will be automatically configured
    - Cache configuration is included in /health/readiness endpoint response
    """
    client = create_proxy_test_client(monkeypatch)
    with client:
        yield client


def test_health_liveliness_endpoint(proxy_client):
    """
    Test that /health/liveliness endpoint returns 200 OK with "I'm alive!" message.
    This is a critical orchestration endpoint that must be simple and fast.
    """
    # Measure the time taken for the health check call
    start_time = time.perf_counter()

    # Make GET request to /health/liveliness
    response = proxy_client.get("/health/liveliness")

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    # Assert response status
    assert (
        response.status_code == 200
    ), f"Expected 200 OK, got {response.status_code}: {response.text}"

    # Assert response content (FastAPI JSON-encodes the string)
    assert (
        response.json() == "I'm alive!"
    ), f"Expected 'I'm alive!' message, got: {response.json()}"

    # Verify response is fast (should be < 100ms for a simple endpoint)
    # This is critical for orchestration systems that poll frequently
    assert (
        duration_ms < 100
    ), f"Health check took {duration_ms:.2f}ms, expected < 100ms for a simple endpoint"

    # Log the duration for visibility (useful for CI/CD monitoring)
    print(f"\n/health/liveliness response time: {duration_ms:.2f}ms")


def test_health_liveness_endpoint(proxy_client):
    """
    Test that /health/liveness endpoint (Kubernetes standard name) also works.
    """
    # Measure the time taken for the health check call
    start_time = time.perf_counter()

    # Make GET request to /health/liveness
    response = proxy_client.get("/health/liveness")

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    # Assert response status
    assert (
        response.status_code == 200
    ), f"Expected 200 OK, got {response.status_code}: {response.text}"

    # Assert response content (FastAPI JSON-encodes the string)
    assert (
        response.json() == "I'm alive!"
    ), f"Expected 'I'm alive!' message, got: {response.json()}"

    # Verify response is fast (should be < 100ms for a simple endpoint)
    assert (
        duration_ms < 100
    ), f"Health check took {duration_ms:.2f}ms, expected < 100ms for a simple endpoint"

    # Log the duration for visibility (useful for CI/CD monitoring)
    print(f"\n/health/liveness response time: {duration_ms:.2f}ms")


def test_health_readiness(proxy_client):
    """
    Test /health/readiness endpoint.
    Database and Redis are optional - the endpoint should work whether they're available or not.

    If DATABASE_URL is set, the endpoint will check database connectivity.
    If REDIS_HOST is set, the endpoint will report cache status.
    If neither is set, the endpoint should still return a valid health status.
    """
    # Measure the time taken for the health check call
    start_time = time.perf_counter()

    # Make GET request to /health/readiness
    response = proxy_client.get("/health/readiness")

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    # Assert response status
    assert (
        response.status_code == 200
    ), f"Expected 200 OK, got {response.status_code}: {response.text}"

    # Verify response is fast (readiness may include DB check if available, so < 500ms is reasonable)
    # This is critical for orchestration systems (Kubernetes) that poll frequently
    assert (
        duration_ms < 500
    ), f"Health check took {duration_ms:.2f}ms, expected < 500ms for readiness endpoint"

    # Assert response contains expected fields
    response_data = response.json()
    assert "status" in response_data, "Response should contain 'status' field"
    assert (
        "litellm_version" in response_data
    ), "Response should contain 'litellm_version' field"

    # Display all health endpoint response fields (matches what /health/readiness returns)
    print("\n" + "-" * 60)
    print("HEALTH ENDPOINT RESPONSE")
    print("-" * 60)
    print(f"Status: {response_data.get('status', 'unknown')}")
    print(f"Database: {response_data.get('db', 'not reported')}")
    print(f"LiteLLM Version: {response_data.get('litellm_version', 'unknown')}")
    print(f"Success Callbacks: {response_data.get('success_callbacks', [])}")
    print(f"Cache: {response_data.get('cache', 'none')}")
    print(
        f"Use AioHTTP Transport: {response_data.get('use_aiohttp_transport', 'unknown')}"
    )
    print(f"Response time: {duration_ms:.2f}ms")

    # If database status is reported, verify it's a valid status
    # Database may be "connected", "disconnected", "unknown", or "Not connected" (when prisma_client is None)
    if "db" in response_data:
        db_status = response_data["db"]
        # Database status can be any of these valid states
        assert db_status in [
            "connected",
            "disconnected",
            "unknown",
            "Not connected",
        ], f"Unexpected db status: {db_status}"

    print("=" * 60 + "\n")


def test_get_callback_identifier_string_and_object_with_callback_name():
    """
    Test get_callback_identifier with string callbacks and objects with callback_name attribute.

    Covers:
    - String callback (returned as-is)
    - Object with callback_name attribute
    - Object with empty/None callback_name (should fall through to other checks)
    """
    from litellm.proxy.health_endpoints._health_endpoints import get_callback_identifier

    # Test 1: String callback should be returned as-is
    assert get_callback_identifier("datadog") == "datadog"
    assert get_callback_identifier("langfuse") == "langfuse"

    # Test 2: Object with callback_name attribute
    class MockCallbackWithName:
        def __init__(self, name):
            self.callback_name = name

    callback_obj = MockCallbackWithName("custom_callback")
    assert get_callback_identifier(callback_obj) == "custom_callback"

    # Test 3: Object with empty callback_name should fall through
    callback_obj_empty = MockCallbackWithName("")
    # This should fall through to CustomLoggerRegistry or callback_name() fallback
    # We'll verify it doesn't return empty string
    result = get_callback_identifier(callback_obj_empty)
    assert result != ""  # Should not return empty string
    assert isinstance(result, str)  # Should still return a string


def test_get_callback_identifier_custom_logger_registry_and_fallback():
    """
    Test get_callback_identifier with CustomLoggerRegistry lookup and fallback scenarios.

    Covers:
    - Object registered in CustomLoggerRegistry
    - Object with callback_name that matches registry entry
    - Fallback to callback_name() helper function
    """
    from litellm.proxy.health_endpoints._health_endpoints import get_callback_identifier
    from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry

    # Test 1: Object registered in CustomLoggerRegistry (without callback_name attribute)
    # Mock a class that's registered in the registry
    class MockRegisteredLogger:
        pass

    # Mock the registry to return callback strings for our mock class
    with patch.object(
        CustomLoggerRegistry,
        "get_all_callback_strs_from_class_type",
        return_value=["mock_logger"],
    ):
        mock_instance = MockRegisteredLogger()
        result = get_callback_identifier(mock_instance)
        assert result == "mock_logger"

    # Test 2: Object with callback_name that matches registry entry
    class MockCallbackWithMatchingName:
        def __init__(self):
            self.callback_name = "matched_name"

    callback_with_matching = MockCallbackWithMatchingName()
    # Mock registry to return list containing the matching name
    with patch.object(
        CustomLoggerRegistry,
        "get_all_callback_strs_from_class_type",
        return_value=["matched_name", "other_name"],
    ):
        result = get_callback_identifier(callback_with_matching)
        assert result == "matched_name"

    # Test 3: Object with falsy callback_name (empty string), should use registry
    class MockCallbackWithEmptyName:
        def __init__(self):
            self.callback_name = ""  # Empty string is falsy

    callback_empty = MockCallbackWithEmptyName()
    # Mock registry to return list - should use first registry entry since callback_name is falsy
    with patch.object(
        CustomLoggerRegistry,
        "get_all_callback_strs_from_class_type",
        return_value=["registry_name"],
    ):
        result = get_callback_identifier(callback_empty)
        assert result == "registry_name"

    # Test 3b: Object with truthy callback_name not in registry - returns callback_name immediately
    # (This tests that truthy callback_name takes precedence over registry)
    class MockCallbackWithNonMatchingName:
        def __init__(self):
            self.callback_name = "non_matching"

    callback_non_matching = MockCallbackWithNonMatchingName()
    # Even if registry has different values, truthy callback_name is returned first
    with patch.object(
        CustomLoggerRegistry,
        "get_all_callback_strs_from_class_type",
        return_value=["registry_name"],
    ):
        result = get_callback_identifier(callback_non_matching)
        # Should return callback_name because it's truthy (checked before registry)
        assert result == "non_matching"

    # Test 4: Object not in registry, falls back to callback_name() helper
    class UnregisteredCallback:
        def __init__(self):
            pass

    unregistered = UnregisteredCallback()
    # Mock registry to return empty list (not registered)
    with patch.object(
        CustomLoggerRegistry, "get_all_callback_strs_from_class_type", return_value=[]
    ):
        result = get_callback_identifier(unregistered)
        # Should fall back to callback_name() which returns __class__.__name__
        assert result == "UnregisteredCallback"

    # Test 5: Function callback (not a class instance)
    def my_callback_function():
        pass

    # Function won't have __class__, so it will skip registry check and go to callback_name()
    result = get_callback_identifier(my_callback_function)
    # Should fall back to callback_name() which returns __name__
    assert result == "my_callback_function"


# ──────────────────────────────────────────────────────────
# _str_to_bool / _get_env_secret / get_secret_bool
# ──────────────────────────────────────────────────────────


def test_str_to_bool():
    from litellm.proxy.health_endpoints._health_endpoints import _str_to_bool

    assert _str_to_bool(None) is None
    assert _str_to_bool("true") is True
    assert _str_to_bool("True") is True
    assert _str_to_bool("  TRUE  ") is True
    assert _str_to_bool("false") is False
    assert _str_to_bool("False") is False
    assert _str_to_bool("  FALSE  ") is False
    assert _str_to_bool("yes") is None
    assert _str_to_bool("1") is None
    assert _str_to_bool("") is None


def test_get_env_secret(monkeypatch):
    from litellm.proxy.health_endpoints._health_endpoints import _get_env_secret

    # Not set – returns default
    monkeypatch.delenv("MY_SECRET", raising=False)
    assert _get_env_secret("MY_SECRET") is None
    assert _get_env_secret("MY_SECRET", default_value="default") == "default"

    # Set via plain name
    monkeypatch.setenv("MY_SECRET", "hello")
    assert _get_env_secret("MY_SECRET") == "hello"

    # Set via os.environ/ prefix
    assert _get_env_secret("os.environ/MY_SECRET") == "hello"

    # Not set, os.environ/ prefix – returns default
    monkeypatch.delenv("MISSING_KEY", raising=False)
    assert _get_env_secret("os.environ/MISSING_KEY", default_value=False) is False


def test_get_secret_bool(monkeypatch):
    from litellm.proxy.health_endpoints._health_endpoints import get_secret_bool

    # Not set – returns default
    monkeypatch.delenv("BOOL_FLAG", raising=False)
    assert get_secret_bool("BOOL_FLAG") is None
    assert get_secret_bool("BOOL_FLAG", default_value=True) is True

    # Set to "true"
    monkeypatch.setenv("BOOL_FLAG", "true")
    assert get_secret_bool("BOOL_FLAG") is True

    # Set to "false"
    monkeypatch.setenv("BOOL_FLAG", "false")
    assert get_secret_bool("BOOL_FLAG") is False

    # Set to unrecognised value – returns None (not default, because env var IS set)
    monkeypatch.setenv("BOOL_FLAG", "maybe")
    assert get_secret_bool("BOOL_FLAG", default_value=True) is None


# ──────────────────────────────────────────────────────────
# _parse_config_row_param_value
# ──────────────────────────────────────────────────────────


def test_parse_config_row_param_value():
    from litellm.proxy.health_endpoints._health_endpoints import (
        _parse_config_row_param_value,
    )

    # None → empty dict
    assert _parse_config_row_param_value(None) == {}

    # Dict → copy of dict
    assert _parse_config_row_param_value({"a": 1}) == {"a": 1}

    # Valid JSON string
    assert _parse_config_row_param_value('{"x": 2}') == {"x": 2}

    # Invalid JSON string → empty dict
    assert _parse_config_row_param_value("not-json") == {}

    # JSON string that parses to non-dict (list) → empty dict
    assert _parse_config_row_param_value("[1, 2, 3]") == {}

    # Arbitrary non-convertible type → empty dict
    assert _parse_config_row_param_value(12345) == {}


# ──────────────────────────────────────────────────────────
# _build_model_param_to_info_mapping
# ──────────────────────────────────────────────────────────


def test_build_model_param_to_info_mapping_basic():
    from litellm.proxy.health_endpoints._health_endpoints import (
        _build_model_param_to_info_mapping,
    )

    model_list = [
        {
            "model_name": "gpt-4",
            "model_info": {"id": "model-id-1"},
            "litellm_params": {"model": "openai/gpt-4"},
        },
        {
            "model_name": "gpt-3.5",
            "model_info": {"id": "model-id-2"},
            "litellm_params": {"model": "openai/gpt-3.5-turbo"},
        },
    ]

    result = _build_model_param_to_info_mapping(model_list)

    assert "openai/gpt-4" in result
    assert result["openai/gpt-4"] == [{"model_name": "gpt-4", "model_id": "model-id-1"}]
    assert result["openai/gpt-3.5-turbo"] == [
        {"model_name": "gpt-3.5", "model_id": "model-id-2"}
    ]


def test_build_model_param_to_info_mapping_multiple_models_same_param():
    from litellm.proxy.health_endpoints._health_endpoints import (
        _build_model_param_to_info_mapping,
    )

    model_list = [
        {
            "model_name": "prod-gpt4",
            "model_info": {"id": "id-a"},
            "litellm_params": {"model": "openai/gpt-4"},
        },
        {
            "model_name": "staging-gpt4",
            "model_info": {"id": "id-b"},
            "litellm_params": {"model": "openai/gpt-4"},
        },
    ]

    result = _build_model_param_to_info_mapping(model_list)

    assert len(result["openai/gpt-4"]) == 2
    names = {entry["model_name"] for entry in result["openai/gpt-4"]}
    assert names == {"prod-gpt4", "staging-gpt4"}


def test_build_model_param_to_info_mapping_skips_missing_fields():
    from litellm.proxy.health_endpoints._health_endpoints import (
        _build_model_param_to_info_mapping,
    )

    # Missing model_name or litellm_params.model → should be skipped
    model_list = [
        {
            "model_info": {"id": "id-1"},
            "litellm_params": {"model": "openai/gpt-4"},
            # no model_name
        },
        {
            "model_name": "gpt-4",
            "model_info": {"id": "id-2"},
            "litellm_params": {},  # no model key
        },
    ]

    result = _build_model_param_to_info_mapping(model_list)
    assert result == {}


# ──────────────────────────────────────────────────────────
# _aggregate_health_check_results
# ──────────────────────────────────────────────────────────


def test_aggregate_health_check_results_healthy():
    from litellm.proxy.health_endpoints._health_endpoints import (
        _aggregate_health_check_results,
    )

    model_param_to_info = {
        "openai/gpt-4": [{"model_name": "gpt-4", "model_id": "id-1"}]
    }
    healthy_endpoints = [{"model": "openai/gpt-4", "latency": 0.1}]

    result = _aggregate_health_check_results(model_param_to_info, healthy_endpoints, [])

    key = ("id-1", "gpt-4")
    assert key in result
    assert result[key]["healthy_count"] == 1
    assert result[key]["unhealthy_count"] == 0
    assert result[key]["error_message"] is None


def test_aggregate_health_check_results_unhealthy():
    from litellm.proxy.health_endpoints._health_endpoints import (
        _aggregate_health_check_results,
    )

    model_param_to_info = {
        "openai/gpt-4": [{"model_name": "gpt-4", "model_id": "id-1"}]
    }
    unhealthy_endpoints = [{"model": "openai/gpt-4", "error": "connection refused"}]

    result = _aggregate_health_check_results(model_param_to_info, [], unhealthy_endpoints)

    key = ("id-1", "gpt-4")
    assert key in result
    assert result[key]["unhealthy_count"] == 1
    assert result[key]["error_message"] == "connection refused"


def test_aggregate_health_check_results_unknown_model_skipped():
    from litellm.proxy.health_endpoints._health_endpoints import (
        _aggregate_health_check_results,
    )

    model_param_to_info = {}
    healthy_endpoints = [{"model": "openai/unknown"}]

    result = _aggregate_health_check_results(model_param_to_info, healthy_endpoints, [])
    assert result == {}


# ──────────────────────────────────────────────────────────
# _save_background_health_checks_to_db
# ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_background_health_checks_to_db_no_prisma():
    """When prisma_client is None the function returns early without error."""
    from litellm.proxy.health_endpoints._health_endpoints import (
        _save_background_health_checks_to_db,
    )

    # Should not raise
    await _save_background_health_checks_to_db(
        prisma_client=None,
        model_list=[],
        healthy_endpoints=[],
        unhealthy_endpoints=[],
        start_time=time.time(),
    )


@pytest.mark.asyncio
async def test_save_background_health_checks_to_db_saves_on_status_change():
    """When status changes, a DB write task should be created."""
    from litellm.proxy.health_endpoints._health_endpoints import (
        _save_background_health_checks_to_db,
    )

    model_list = [
        {
            "model_name": "gpt-4",
            "model_info": {"id": "model-id-1"},
            "litellm_params": {"model": "openai/gpt-4"},
        }
    ]
    healthy_endpoints = [{"model": "openai/gpt-4"}]

    # Simulate last check was "unhealthy" → status changes to "healthy"
    last_check = SimpleNamespace(
        model_id="model-id-1",
        model_name="gpt-4",
        status="unhealthy",
        checked_at=None,
    )

    mock_prisma = MagicMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(return_value=[last_check])
    mock_prisma.save_health_check_result = AsyncMock(return_value=None)

    with patch("asyncio.create_task") as mock_create_task:
        await _save_background_health_checks_to_db(
            prisma_client=mock_prisma,
            model_list=model_list,
            healthy_endpoints=healthy_endpoints,
            unhealthy_endpoints=[],
            start_time=time.time(),
        )
        mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_save_background_health_checks_to_db_skips_when_status_unchanged_recently():
    """When status is unchanged and last check was recent (<1 hour), no write."""
    from datetime import timezone

    from litellm.proxy.health_endpoints._health_endpoints import (
        _save_background_health_checks_to_db,
    )

    model_list = [
        {
            "model_name": "gpt-4",
            "model_info": {"id": "model-id-1"},
            "litellm_params": {"model": "openai/gpt-4"},
        }
    ]
    healthy_endpoints = [{"model": "openai/gpt-4"}]

    recent_time = datetime.now(timezone.utc)
    last_check = SimpleNamespace(
        model_id="model-id-1",
        model_name="gpt-4",
        status="healthy",  # same as current
        checked_at=recent_time,
    )

    mock_prisma = MagicMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(return_value=[last_check])
    mock_prisma.save_health_check_result = AsyncMock(return_value=None)

    with patch("asyncio.create_task") as mock_create_task:
        await _save_background_health_checks_to_db(
            prisma_client=mock_prisma,
            model_list=model_list,
            healthy_endpoints=healthy_endpoints,
            unhealthy_endpoints=[],
            start_time=time.time(),
        )
        mock_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_save_background_health_checks_to_db_handles_db_error():
    """DB errors are caught and swallowed (health check should not fail)."""
    from litellm.proxy.health_endpoints._health_endpoints import (
        _save_background_health_checks_to_db,
    )

    mock_prisma = MagicMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(
        side_effect=Exception("DB connection lost")
    )

    # Should not raise
    await _save_background_health_checks_to_db(
        prisma_client=mock_prisma,
        model_list=[
            {
                "model_name": "gpt-4",
                "model_info": {"id": "id-1"},
                "litellm_params": {"model": "openai/gpt-4"},
            }
        ],
        healthy_endpoints=[{"model": "openai/gpt-4"}],
        unhealthy_endpoints=[],
        start_time=time.time(),
    )

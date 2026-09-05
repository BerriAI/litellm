import asyncio
import copy
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prisma.errors import ClientNotConnectedError, HTTPClientClosedError, PrismaError

import litellm
import litellm.proxy.health_endpoints._health_endpoints as _health_endpoints_module
from litellm.litellm_core_utils.health_check_helpers import TEST_IMAGE_BASE64
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.router import Router
from litellm.proxy.health_endpoints._health_endpoints import (
    _db_health_readiness_check,
    _show_no_redis_warning,
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
    "transport_error",
    [
        httpx.ConnectError("All connection attempts failed"),
        ClientNotConnectedError(),
        HTTPClientClosedError(),
        PrismaError("Can't reach database server"),
    ],
)
async def test_db_health_transport_error_never_raises(transport_error):
    """
    Regression test for the /health/readiness 503 loop bug.

    handle_db_exception() used to re-raise inside _db_health_readiness_check,
    turning any DB outage into a 503 "Service Unhealthy" response that never
    recovered. Transport errors (ClientNotConnectedError, httpx.ConnectError,
    etc.) must return {"status": "disconnected"} — never raise.
    """
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=transport_error)
    mock_prisma.attempt_db_reconnect = AsyncMock(return_value=False)

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await _db_health_readiness_check()

    assert result["status"] == "disconnected"
    mock_prisma.attempt_db_reconnect.assert_called_once_with(
        reason="health_readiness_check",
        timeout_seconds=_health_endpoints_module.DB_READINESS_CHECK_TIMEOUT_SECONDS,
        lock_timeout_seconds=_health_endpoints_module.DB_READINESS_CHECK_TIMEOUT_SECONDS,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_error",
    [
        httpx.ConnectError("All connection attempts failed"),
        ClientNotConnectedError(),
        HTTPClientClosedError(),
    ],
)
async def test_db_health_transport_error_reconnect_succeeds(transport_error):
    """
    When health_check raises a transport error and attempt_db_reconnect
    succeeds, the second health_check passes and we return 'connected'.
    """
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=[transport_error, None])
    mock_prisma.attempt_db_reconnect = AsyncMock(return_value=True)

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await _db_health_readiness_check()

    assert result["status"] == "connected"
    mock_prisma.attempt_db_reconnect.assert_called_once_with(
        reason="health_readiness_check",
        timeout_seconds=_health_endpoints_module.DB_READINESS_CHECK_TIMEOUT_SECONDS,
        lock_timeout_seconds=_health_endpoints_module.DB_READINESS_CHECK_TIMEOUT_SECONDS,
    )
    assert mock_prisma.health_check.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_error",
    [
        httpx.ConnectError("All connection attempts failed"),
        ClientNotConnectedError(),
        HTTPClientClosedError(),
    ],
)
async def test_db_health_transport_error_reconnect_fails(transport_error):
    """
    When health_check raises a transport error and attempt_db_reconnect also
    fails, return 'disconnected' without raising.
    """
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=transport_error)
    mock_prisma.attempt_db_reconnect = AsyncMock(side_effect=RuntimeError("reconnect failed"))

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await _db_health_readiness_check()

    assert result["status"] == "disconnected"


@pytest.mark.asyncio
async def test_db_health_non_transport_error_returns_disconnected():
    """
    When health_check raises a non-transport error (e.g. data-layer error),
    is_database_transport_error returns False so reconnect is skipped.
    Returns 'disconnected' without raising and without calling attempt_db_reconnect.
    """
    non_transport_error = PrismaError("UniqueViolationError")
    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=non_transport_error)
    mock_prisma.attempt_db_reconnect = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "connected",
        "last_updated": datetime.now() - timedelta(seconds=20),
    }

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await _db_health_readiness_check()

    assert result["status"] == "disconnected"
    mock_prisma.attempt_db_reconnect.assert_not_called()


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
        mock_instance.async_health_check = AsyncMock(return_value={"status": status, "error_message": error_message})
        MockSQSLogger.return_value = mock_instance

        result = await health_services_endpoint(service="sqs")

        assert result["status"] == status
        assert result["message"] == error_message
        mock_instance.async_health_check.assert_awaited_once()


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
        assert model_params.get("api_base") == "https://resolved-endpoint.openai.azure.com/"
        assert model_params.get("api_version") == "2024-10-21"
        assert model_params.get("model") == "gpt-4o"  # Request param overrides config param

        # Verify result
        assert result["status"] == "success"
        assert "result" in result


@pytest.mark.asyncio
async def test_test_model_connection_uses_model_info_id_to_disambiguate_duplicate_model_names():
    """
    When two deployments share the same `model_name` (e.g. wildcard
    `openai/*`) but have different `api_base` values, clicking "Test
    Connection" on a specific row in the UI must probe THAT row's
    `api_base` — not whichever happens to be `deployments[0]`.

    The UI passes `model_info.id` to identify the deployment the user
    actually clicked on. The backend must use that id to look up the
    specific deployment rather than always grabbing the first match.

    Regression test for: silent fallback to deployments[0] when
    multiple deployments share a wildcard model_name.
    """
    from litellm.types.router import Deployment, LiteLLM_Params

    mock_request = MagicMock()
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.user_id = "test-user"
    mock_user_api_key_dict.token = "test-token"

    mock_prisma_client = MagicMock()

    deployment_a = {
        "model_name": "openai/*",
        "litellm_params": {
            "model": "openai/*",
            "api_base": "https://deployment-A-base.invalid/v1",
            "api_key": "fake-key-A",
        },
        "model_info": {"id": "deployment-A-id"},
    }
    deployment_b = {
        "model_name": "openai/*",
        "litellm_params": {
            "model": "openai/*",
            "api_base": "https://deployment-B-base.invalid/v1",
            "api_key": "fake-key-B",
        },
        "model_info": {"id": "deployment-B-id"},
    }

    mock_router = MagicMock()
    mock_router.get_model_list.return_value = [deployment_a, deployment_b]

    # Backend uses get_deployment(model_id=...) for O(1) lookup by id.
    def _get_deployment_by_id(model_id):
        if model_id == "deployment-A-id":
            return Deployment(
                model_name="openai/*",
                litellm_params=LiteLLM_Params(**deployment_a["litellm_params"]),
                model_info=deployment_a["model_info"],
            )
        if model_id == "deployment-B-id":
            return Deployment(
                model_name="openai/*",
                litellm_params=LiteLLM_Params(**deployment_b["litellm_params"]),
                model_info=deployment_b["model_info"],
            )
        return None

    mock_router.get_deployment.side_effect = _get_deployment_by_id

    mock_can_user_make_model_call = AsyncMock()

    mock_health_check_result = {"status": "healthy", "response_time_ms": 50}
    mock_ahealth_check = AsyncMock(return_value=mock_health_check_result)
    mock_run_with_timeout = AsyncMock(return_value=mock_health_check_result)

    def mock_update_params(model_info, litellm_params):
        params = litellm_params.copy()
        params["messages"] = [{"role": "user", "content": "test"}]
        return params

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
        # Click "Test Connection" on deployment B (NOT the first one).
        # The UI sends only `model` + `model_info.id` — it does NOT
        # send `api_base`/`api_key`, so the backend must resolve them
        # from the right deployment.
        await health_test_model_connection(
            request=mock_request,
            mode="chat",
            litellm_params={"model": "openai/*"},
            model_info={"id": "deployment-B-id"},
            user_api_key_dict=mock_user_api_key_dict,
        )

        # The outbound health check must hit deployment B's api_base.
        ahealth_check_call_args = mock_ahealth_check.call_args
        assert ahealth_check_call_args is not None
        model_params = ahealth_check_call_args.kwargs.get("model_params", {})

        assert model_params.get("api_base") == ("https://deployment-B-base.invalid/v1"), (
            "Expected /health/test_connection to probe deployment B's "
            "api_base when model_info.id='deployment-B-id' was provided. "
            f"Got: {model_params.get('api_base')!r}. This means the "
            "backend silently fell back to deployments[0] (A) instead "
            "of disambiguating by model_info.id."
        )
        assert model_params.get("api_key") == "fake-key-B"


@pytest.mark.asyncio
async def test_test_model_connection_falls_back_to_deployments_zero_without_id():
    """
    Backwards-compat: when the request body does NOT include
    `model_info.id`, the legacy behavior of using `deployments[0]`
    is preserved (single-deployment case, or callers that haven't
    been updated to pass an id).
    """
    mock_request = MagicMock()
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.user_id = "test-user"
    mock_user_api_key_dict.token = "test-token"

    mock_prisma_client = MagicMock()

    deployment_a = {
        "model_name": "openai/*",
        "litellm_params": {
            "model": "openai/*",
            "api_base": "https://deployment-A-base.invalid/v1",
            "api_key": "fake-key-A",
        },
        "model_info": {"id": "deployment-A-id"},
    }
    deployment_b = {
        "model_name": "openai/*",
        "litellm_params": {
            "model": "openai/*",
            "api_base": "https://deployment-B-base.invalid/v1",
            "api_key": "fake-key-B",
        },
        "model_info": {"id": "deployment-B-id"},
    }

    mock_router = MagicMock()
    mock_router.get_model_list.return_value = [deployment_a, deployment_b]

    mock_can_user_make_model_call = AsyncMock()
    mock_health_check_result = {"status": "healthy"}
    mock_ahealth_check = AsyncMock(return_value=mock_health_check_result)
    mock_run_with_timeout = AsyncMock(return_value=mock_health_check_result)

    def mock_update_params(model_info, litellm_params):
        params = litellm_params.copy()
        params["messages"] = [{"role": "user", "content": "test"}]
        return params

    def mock_reject_os_environ(params):
        return None

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.premium_user", False),
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
        await health_test_model_connection(
            request=mock_request,
            mode="chat",
            litellm_params={"model": "openai/*"},
            model_info={},  # no id provided
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Without id, deployments[0] (A) should be used (legacy behavior).
        model_params = mock_ahealth_check.call_args.kwargs.get("model_params", {})
        assert model_params.get("api_base") == "https://deployment-A-base.invalid/v1"
        assert model_params.get("api_key") == "fake-key-A"


@pytest.mark.asyncio
async def test_test_model_connection_uses_loaded_deployment_team_id():
    """
    /health/test_connection must authorize using the team_id of the
    deployment it actually loaded (by model_info.id), not the team_id
    supplied in the request body. Requesting team A's deployment while
    authenticated as an admin of team B must be denied.
    """
    from fastapi import HTTPException

    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelManagementAuthChecks,
    )
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    mock_request = MagicMock()

    requester_team_id = "team-b"
    deployment_owner_team_id = "team-a"
    deployment_id = "team-a-deployment-id"

    requester_user_api_key_dict = UserAPIKeyAuth(
        token="requester-token",
        user_id="team-b-admin-user",
        team_id=requester_team_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    mock_prisma_client = MagicMock()

    other_team_deployment = Deployment(
        model_name="team-a-model",
        litellm_params=LiteLLM_Params(
            model="openai/gpt-4o",
            api_base="https://team-a-api.invalid/v1",
            api_key="TEAM-A-API-KEY",
        ),
        model_info=ModelInfo(id=deployment_id, team_id=deployment_owner_team_id),
    )

    mock_router = MagicMock()
    mock_router.get_deployment.return_value = other_team_deployment

    async def fake_find_unique(*, where):
        team_id = where["team_id"]
        if team_id == requester_team_id:
            return SimpleNamespace(
                model_dump=lambda: LiteLLM_TeamTable(
                    team_id=requester_team_id,
                    members_with_roles=[
                        {
                            "user_id": "team-b-admin-user",
                            "role": "admin",
                        }
                    ],
                ).model_dump()
            )
        if team_id == deployment_owner_team_id:
            return SimpleNamespace(
                model_dump=lambda: LiteLLM_TeamTable(
                    team_id=deployment_owner_team_id,
                    members_with_roles=[],
                ).model_dump()
            )
        return None

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch.object(
            ModelManagementAuthChecks,
            "can_user_make_model_call",
            wraps=ModelManagementAuthChecks.can_user_make_model_call,
        ) as spy_auth_check,
        patch("litellm.proxy.management_endpoints.model_management_endpoints.TeamRepository") as MockTeamRepo,
    ):
        mock_team_repo_instance = MagicMock()
        mock_team_repo_instance.table.find_unique = AsyncMock(side_effect=fake_find_unique)
        MockTeamRepo.return_value = mock_team_repo_instance

        with pytest.raises(HTTPException) as exc_info:
            await health_test_model_connection(
                request=mock_request,
                mode="chat",
                litellm_params={
                    "model": "openai/gpt-4o",
                    "api_base": "https://swapped-base.invalid/v1",
                },
                model_info={
                    "id": deployment_id,
                    "team_id": requester_team_id,
                },
                user_api_key_dict=requester_user_api_key_dict,
            )

        assert exc_info.value.status_code == 403
        assert spy_auth_check.called
        passed_model_params = spy_auth_check.call_args.kwargs["model_params"]
        assert passed_model_params.model_info.team_id == deployment_owner_team_id, (
            "Auth check must run against the loaded deployment's team_id "
            f"({deployment_owner_team_id!r}); got "
            f"{passed_model_params.model_info.team_id!r}."
        )


@pytest.mark.asyncio
async def test_test_model_connection_uses_loaded_deployment_team_id_via_model_name_fallback():
    """
    Companion to the id-lookup case: when the caller provides only a model
    name (no `model_info.id`) and that name resolves via the router's
    `model_name` fallback to a deployment owned by a different team, the
    auth check must still run against the loaded deployment's `team_id`,
    not the caller-supplied one in the request body.
    """
    from fastapi import HTTPException

    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelManagementAuthChecks,
    )

    mock_request = MagicMock()

    requester_team_id = "team-b-2"
    deployment_owner_team_id = "team-a-2"

    requester_user_api_key_dict = UserAPIKeyAuth(
        token="requester-token-2",
        user_id="team-b-admin-user-2",
        team_id=requester_team_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    mock_prisma_client = MagicMock()

    other_team_deployment_dict = {
        "model_name": "shared-model-name",
        "litellm_params": {
            "model": "openai/gpt-4o",
            "api_base": "https://team-a-api-2.invalid/v1",
            "api_key": "TEAM-A-API-KEY-2",
        },
        "model_info": {
            "id": "team-a-deployment-id-2",
            "team_id": deployment_owner_team_id,
        },
    }

    mock_router = MagicMock()
    mock_router.get_model_list.return_value = [other_team_deployment_dict]

    async def fake_find_unique(*, where):
        return SimpleNamespace(
            model_dump=lambda: LiteLLM_TeamTable(
                team_id=where["team_id"],
                members_with_roles=(
                    [{"user_id": "team-b-admin-user-2", "role": "admin"}]
                    if where["team_id"] == requester_team_id
                    else []
                ),
            ).model_dump()
        )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch.object(
            ModelManagementAuthChecks,
            "can_user_make_model_call",
            wraps=ModelManagementAuthChecks.can_user_make_model_call,
        ) as spy_auth_check,
        patch("litellm.proxy.management_endpoints.model_management_endpoints.TeamRepository") as MockTeamRepo,
    ):
        mock_team_repo_instance = MagicMock()
        mock_team_repo_instance.table.find_unique = AsyncMock(side_effect=fake_find_unique)
        MockTeamRepo.return_value = mock_team_repo_instance

        with pytest.raises(HTTPException) as exc_info:
            await health_test_model_connection(
                request=mock_request,
                mode="chat",
                litellm_params={
                    "model": "shared-model-name",
                    "api_base": "https://swapped-base-2.invalid/v1",
                },
                model_info={"team_id": requester_team_id},
                user_api_key_dict=requester_user_api_key_dict,
            )

        assert exc_info.value.status_code == 403

        passed_model_params = spy_auth_check.call_args.kwargs["model_params"]
        assert passed_model_params.model_info.team_id == deployment_owner_team_id


@pytest.mark.asyncio
async def test_test_model_connection_authorizes_on_params_after_health_check_params_merge():
    """
    Regression guard for the ordering fix: health_check_params from the request
    body are merged into the probe params BEFORE the authorization check, so a
    caller cannot smuggle a field past auth via health_check_params. Auth is
    stubbed to reject, which halts the endpoint right after it records the
    params it was handed, so the outbound probe is never reached. If the merge
    is moved back to after can_user_make_model_call, the marker is absent from
    those params and this test fails.
    """
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelManagementAuthChecks,
    )
    from litellm.types.router import Deployment

    marker = "sentinel-from-health-check-params"
    mock_can_user_make_model_call = AsyncMock(side_effect=HTTPException(status_code=403, detail="denied"))

    with (
        patch(  # test-quality-ok: proxy module global, no injection seam
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ),
        patch(  # test-quality-ok: proxy module global, no injection seam
            "litellm.proxy.proxy_server.llm_router", None
        ),
        patch.object(  # test-quality-ok: capturing the params handed to auth is the assertion
            ModelManagementAuthChecks,
            "can_user_make_model_call",
            mock_can_user_make_model_call,
        ),
        pytest.raises(HTTPException),
    ):
        await health_test_model_connection(
            request=MagicMock(),
            mode="chat",
            litellm_params={"model": "openai/gpt-4o"},
            model_info={"health_check_params": {"probe_marker": marker}},
            user_api_key_dict=UserAPIKeyAuth(
                token="requester-token",
                user_id="admin-user",
                user_role=LitellmUserRoles.PROXY_ADMIN,
            ),
        )

    assert mock_can_user_make_model_call.called
    passed_model_params = mock_can_user_make_model_call.call_args.kwargs["model_params"]
    assert isinstance(passed_model_params, Deployment)
    authorized_params = passed_model_params.litellm_params.model_dump()
    assert authorized_params.get("probe_marker") == marker


@pytest.mark.asyncio
async def test_test_model_connection_authorized_team_admin_passes_real_auth():
    """
    Positive-path companion to the deny tests above. When the caller is a
    genuine admin of the team that owns the loaded deployment, the real
    (unmocked) auth check must pass and the endpoint must reach the outbound
    health probe. Guards against a regression that swaps the auth `team_id`
    for something deny-all on the legit path.
    """
    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelManagementAuthChecks,
    )
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    mock_request = MagicMock()

    owner_team_id = "team-owner"
    owner_admin_user_id = "team-owner-admin"
    owned_deployment_id = "owned-deployment-id"

    owner_admin_api_key_dict = UserAPIKeyAuth(
        token="owner-admin-token",
        user_id=owner_admin_user_id,
        team_id=owner_team_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    mock_prisma_client = MagicMock()

    owned_deployment = Deployment(
        model_name="owner-model",
        litellm_params=LiteLLM_Params(
            model="openai/gpt-4o-mini",
            api_base="https://owner-real-api.invalid/v1",
            api_key="owner-team-api-key",
        ),
        model_info=ModelInfo(id=owned_deployment_id, team_id=owner_team_id),
    )

    mock_router = MagicMock()
    mock_router.get_deployment.return_value = owned_deployment

    async def fake_find_unique(*, where):
        if where["team_id"] == owner_team_id:
            return SimpleNamespace(
                model_dump=lambda: LiteLLM_TeamTable(
                    team_id=owner_team_id,
                    members_with_roles=[{"user_id": owner_admin_user_id, "role": "admin"}],
                ).model_dump()
            )
        return None

    health_result = {"status": "healthy", "response_time_ms": 50}

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch.object(
            ModelManagementAuthChecks,
            "can_user_make_model_call",
            wraps=ModelManagementAuthChecks.can_user_make_model_call,
        ) as spy_auth_check,
        patch("litellm.proxy.management_endpoints.model_management_endpoints.TeamRepository") as MockTeamRepo,
        patch(
            "litellm.proxy.health_endpoints._health_endpoints.litellm.ahealth_check",
            AsyncMock(return_value=health_result),
        ),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints.run_with_timeout",
            AsyncMock(return_value=health_result),
        ),
    ):
        mock_team_repo_instance = MagicMock()
        mock_team_repo_instance.table.find_unique = AsyncMock(side_effect=fake_find_unique)
        MockTeamRepo.return_value = mock_team_repo_instance

        result = await health_test_model_connection(
            request=mock_request,
            mode="chat",
            litellm_params={"model": "openai/gpt-4o-mini"},
            model_info={"id": owned_deployment_id, "team_id": owner_team_id},
            user_api_key_dict=owner_admin_api_key_dict,
        )

        assert result["status"] == "success"
        passed_model_params = spy_auth_check.call_args.kwargs["model_params"]
        assert passed_model_params.model_info.team_id == owner_team_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error_message",
    [
        ("healthy", ""),
        ("unhealthy", "Galileo authentication failed"),
    ],
)
async def test_health_services_endpoint_galileo(status, error_message):
    with patch("litellm.integrations.galileo.GalileoObserve") as MockGalileoObserve:
        mock_instance = MagicMock()
        mock_instance.async_health_check = AsyncMock(return_value={"status": status, "error_message": error_message})
        MockGalileoObserve.return_value = mock_instance

        result = await health_services_endpoint(service="galileo")

        if status == "healthy":
            assert result["status"] == "healthy"
            assert result["message"] == "Galileo is healthy"
        else:
            assert result["status"] == "unhealthy"
            assert result["message"] == error_message
        mock_instance.async_health_check.assert_awaited_once()


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [
        None,
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        LitellmUserRoles.TEAM,
        LitellmUserRoles.CUSTOMER,
    ],
)
async def test_health_services_endpoint_newrelic_blocks_non_admin(role):
    """
    /health/services?service=newrelic emits a real LiteLLMConnectionTest event
    to the configured New Relic account. Only proxy admins (full or view-only)
    should be able to trigger it; every other caller must be rejected before
    the external event is recorded.
    """
    from litellm.proxy._types import ProxyException

    user_api_key_dict = UserAPIKeyAuth(
        token="non-admin-token",
        user_id="non-admin-user",
        user_role=role,
    )

    with patch("litellm.integrations.newrelic.newrelic.NewRelicLogger") as MockNewRelicLogger:
        mock_instance = MagicMock()
        mock_instance.async_health_check = AsyncMock(return_value={"status": "healthy", "error_message": ""})
        MockNewRelicLogger.return_value = mock_instance

        with pytest.raises(ProxyException) as exc_info:
            await health_services_endpoint(
                user_api_key_dict=user_api_key_dict,
                service="newrelic",
            )

        assert str(exc_info.value.code) == "403"
        mock_instance.async_health_check.assert_not_awaited()
        MockNewRelicLogger.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "admin_role",
    [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY],
)
async def test_health_services_endpoint_newrelic_allows_proxy_admin(admin_role):
    """
    Proxy admins (full and view-only) can trigger the New Relic test event.
    """
    user_api_key_dict = UserAPIKeyAuth(
        token="admin-token",
        user_id="admin-user",
        user_role=admin_role,
    )

    with patch("litellm.integrations.newrelic.newrelic.NewRelicLogger") as MockNewRelicLogger:
        mock_instance = MagicMock()
        mock_instance.async_health_check = AsyncMock(return_value={"status": "healthy", "error_message": ""})
        MockNewRelicLogger.return_value = mock_instance

        result = await health_services_endpoint(
            user_api_key_dict=user_api_key_dict,
            service="newrelic",
        )

        assert result["status"] == "healthy"
        mock_instance.async_health_check.assert_awaited_once()


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
    - Cache diagnostics are included in the authenticated /health/readiness/details response
    """
    client = create_proxy_test_client(monkeypatch)
    with client:
        yield client


def test_health_liveliness_endpoint(proxy_client):
    """
    Test that /health/liveliness endpoint returns 200 OK with "I'm alive!" message.
    This is a critical orchestration endpoint that must be simple and fast.
    """
    warm_up: Final = proxy_client.get("/health/liveliness")
    assert warm_up.status_code == 200, f"Expected 200 OK, got {warm_up.status_code}: {warm_up.text}"

    def _timed_poll() -> tuple[float, httpx.Response]:
        start_time: Final = time.perf_counter()
        response: Final = proxy_client.get("/health/liveliness")
        return (time.perf_counter() - start_time) * 1000, response

    polls: Final = tuple(_timed_poll() for _ in range(5))

    for _, response in polls:
        assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
        assert response.json() == "I'm alive!", f"Expected 'I'm alive!' message, got: {response.json()}"

    durations_ms: Final = tuple(sorted(duration_ms for duration_ms, _ in polls))
    median_ms: Final = durations_ms[len(durations_ms) // 2]
    assert median_ms < 100, f"Median of {len(polls)} health checks took {median_ms:.2f}ms, expected < 100ms"


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
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"

    # Assert response content (FastAPI JSON-encodes the string)
    assert response.json() == "I'm alive!", f"Expected 'I'm alive!' message, got: {response.json()}"

    # Verify response is fast (should be < 100ms for a simple endpoint)
    assert duration_ms < 100, f"Health check took {duration_ms:.2f}ms, expected < 100ms for a simple endpoint"

    # Log the duration for visibility (useful for CI/CD monitoring)
    print(f"\n/health/liveness response time: {duration_ms:.2f}ms")


def test_health_backlog_includes_admission_control_stats(proxy_client):
    response = proxy_client.get("/health/backlog")

    assert response.status_code == 200, response.text
    assert set(response.json()) == {
        "in_flight_requests",
        "admitted_requests",
        "queued_requests",
        "rejected_requests",
    }


def test_health_readiness(proxy_client):
    """
    Test /health/readiness endpoint.
    Database and Redis are optional - the public endpoint should work whether they're available or not.
    """
    # Measure the time taken for the health check call
    start_time = time.perf_counter()

    # Make GET request to /health/readiness
    response = proxy_client.get("/health/readiness")

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    # Assert response status
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"

    # Verify response is fast (readiness may include DB check if available, so < 500ms is reasonable)
    # This is critical for orchestration systems (Kubernetes) that poll frequently
    assert duration_ms < 500, f"Health check took {duration_ms:.2f}ms, expected < 500ms for readiness endpoint"

    # Assert response contains only low-detail public probe fields. `db` is
    # included so unauthenticated probes can distinguish "DB unreachable"
    # from a fully-healthy worker; its value depends on whether the test env
    # exposes DATABASE_URL.
    response_data = response.json()
    assert set(response_data.keys()) == {"status", "db"}
    assert response_data["status"] == "healthy"
    assert response_data["db"] in {"connected", "disconnected", "Not connected"}
    print(f"Response time: {duration_ms:.2f}ms")


def test_health_readiness_details_returns_diagnostic_fields(monkeypatch):
    """
    Detailed readiness diagnostics stay available behind the auth dependency.
    """
    app = FastAPI()
    app.include_router(_health_endpoints_module.router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    client = TestClient(app)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    response = client.get("/health/readiness/details")

    assert response.status_code == 200, response.text
    response_data = response.json()
    assert response_data["status"] == "healthy"
    assert "litellm_version" in response_data
    assert "success_callbacks" in response_data
    assert "cache" in response_data


def test_health_readiness_allows_explicit_legacy_public_details(monkeypatch):
    """
    Operators can explicitly preserve the legacy public readiness payload.
    """
    app = FastAPI()
    app.include_router(_health_endpoints_module.router)
    client = TestClient(app)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_public_health_readiness_details": True},
    )

    response = client.get("/health/readiness")

    assert response.status_code == 200, response.text
    response_data = response.json()
    assert response_data["status"] == "healthy"
    assert "litellm_version" in response_data
    assert "success_callbacks" in response_data
    assert "cache" in response_data


def test_get_callback_identifier_string_and_object_with_callback_name():
    """
    Test get_callback_identifier with string callbacks and objects with callback_name attribute.

    Covers:
    - String callback (returned as-is)
    - Object with callback_name attribute
    - Object with empty/None callback_name (should fall through to other checks)
    """

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
    with patch.object(CustomLoggerRegistry, "get_all_callback_strs_from_class_type", return_value=[]):
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


# ---------------------------------------------------------------------------
# /health response shape: model-access scoping and display-field allowlist
# ---------------------------------------------------------------------------
# These tests pin the contract that the /health response (a) only includes
# deployments the calling key is allowed to see, and (b) does not return
# provider routing fields like api_base / api_version. They guard against
# regressions that would widen the response shape.


@pytest.mark.asyncio
async def test_health_endpoint_filters_model_list_by_user_access():
    """
    health_endpoint() should restrict _llm_model_list to deployments whose
    model_name appears in user_api_key_dict.models before running the health
    check. A key scoped to ["model-a"] should only see model-a in the result,
    not other deployments configured on the proxy.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_base": "https://example-a.test",
            },
            "model_info": {"id": "id-a"},
        },
        {
            "model_name": "model-b",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_base": "https://example-b.test",
                "api_version": "2024-10-21",
            },
            "model_info": {"id": "id-b"},
        },
    ]

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-test-key",
        models=["model-a"],
    )

    captured: dict = {}

    async def fake_perform(**kwargs):
        captured["model_list"] = kwargs["model_list"]
        return {
            "healthy_endpoints": [],
            "unhealthy_endpoints": [],
            "healthy_count": 0,
            "unhealthy_count": 0,
        }

    with (
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", False),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", {}),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            side_effect=fake_perform,
        ),
    ):
        from fastapi import Response

        await health_endpoint(response=Response(), user_api_key_dict=user_api_key_dict, model=None, model_id=None)

    assert "model_list" in captured, "health_endpoint did not call _perform_health_check_and_save"
    returned_names = {m["model_name"] for m in captured["model_list"]}
    assert returned_names == {"model-a"}, f"health_endpoint did not scope model_list to caller access: {returned_names}"


@pytest.mark.asyncio
async def test_health_endpoint_keeps_full_model_list_for_all_proxy_models():
    """
    A key granted all model permissions carries the literal
    "all-proxy-models" entry in user_api_key_dict.models. It matches no real
    model_name, so the access filter must be skipped entirely; otherwise the
    model list filters down to nothing and /health reports 0/0 counts.
    """
    from litellm.proxy._types import SpecialModelNames, UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"id": "id-a"},
        },
        {
            "model_name": "model-b",
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"id": "id-b"},
        },
    ]

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-test-key",
        models=[SpecialModelNames.all_proxy_models.value],
    )

    captured: dict = {}

    async def fake_perform(**kwargs):
        captured["model_list"] = kwargs["model_list"]
        return {
            "healthy_endpoints": [],
            "unhealthy_endpoints": [],
            "healthy_count": 0,
            "unhealthy_count": 0,
        }

    with (
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", False),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", {}),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            side_effect=fake_perform,
        ),
    ):
        from fastapi import Response

        await health_endpoint(response=Response(), user_api_key_dict=user_api_key_dict, model=None, model_id=None)

    returned_names = {m["model_name"] for m in captured["model_list"]}
    assert returned_names == {
        "model-a",
        "model-b",
    }, f"all-proxy-models key should health-check every model: {returned_names}"


@pytest.mark.asyncio
async def test_health_endpoint_resolves_all_team_models_to_team_allowlist():
    """
    A key granted "all-team-models" carries the literal sentinel in
    user_api_key_dict.models, which matches no real model_name. With a
    team_id the sentinel must resolve to the team's allowlist (same
    semantics as get_key_models); otherwise the filter would zero out the
    model list just like the all-proxy-models case.
    """
    from litellm.proxy._types import SpecialModelNames, UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"id": "id-a"},
        },
        {
            "model_name": "model-b",
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"id": "id-b"},
        },
    ]

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-test-key",
        models=[SpecialModelNames.all_team_models.value],
        team_id="team-1",
        team_models=["model-b"],
    )

    captured: dict = {}

    async def fake_perform(**kwargs):
        captured["model_list"] = kwargs["model_list"]
        return {
            "healthy_endpoints": [],
            "unhealthy_endpoints": [],
            "healthy_count": 0,
            "unhealthy_count": 0,
        }

    with (
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", False),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", {}),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            side_effect=fake_perform,
        ),
    ):
        from fastapi import Response

        await health_endpoint(response=Response(), user_api_key_dict=user_api_key_dict, model=None, model_id=None)

    returned_names = {m["model_name"] for m in captured["model_list"]}
    assert returned_names == {"model-b"}, f"all-team-models key should health-check the team's models: {returned_names}"


def _router_for(model_list: Sequence[Mapping[str, object]]) -> Router:
    return Router(model_list=copy.deepcopy(list(model_list)))


_ACCESS_GROUP_MODEL_LIST = [
    {
        "model_name": "bedrock-nova",
        "litellm_params": {"model": "bedrock/us.amazon.nova-2-lite-v1:0"},
        "model_info": {"id": "id-bedrock", "access_groups": ["bedrock-group"]},
    },
    {
        "model_name": "gpt-5.4-mini",
        "litellm_params": {"model": "openai/gpt-5.4-mini"},
        "model_info": {"id": "id-openai"},
    },
]
_ACCESS_GROUP_ROUTER = _router_for(_ACCESS_GROUP_MODEL_LIST)
_TEAM_MODEL_LIST = [
    _ACCESS_GROUP_MODEL_LIST[0],
    {
        "model_name": "bedrock-nova_team-b_9f2c",
        "litellm_params": {"model": "bedrock/us.amazon.nova-2-lite-v1:0"},
        "model_info": {
            "id": "id-team-b",
            "team_id": "team-b",
            "team_public_model_name": "bedrock-nova",
            "access_groups": ["bedrock-group"],
        },
    },
]
_TEAM_CACHED_RESULTS = {
    "healthy_endpoints": [
        {"model": "bedrock/us.amazon.nova-2-lite-v1:0", "model_id": "id-bedrock"},
        {"model": "bedrock/us.amazon.nova-2-lite-v1:0", "model_id": "id-team-b"},
    ],
    "unhealthy_endpoints": [],
    "healthy_count": 2,
    "unhealthy_count": 0,
}
_ACCESS_GROUP_CACHED_RESULTS = {
    "healthy_endpoints": [
        {"model": "bedrock/us.amazon.nova-2-lite-v1:0", "model_id": "id-bedrock"},
        {"model": "openai/gpt-5.4-mini", "model_id": "id-openai"},
    ],
    "unhealthy_endpoints": [],
    "healthy_count": 2,
    "unhealthy_count": 0,
}


@contextmanager
def _proxy_health_globals(
    llm_model_list: Sequence[Mapping[str, object]],
    llm_router: object,
    use_background_health_checks: bool = False,
    health_check_results: Mapping[str, object] | None = None,
) -> Iterator[None]:
    with (
        patch(  # test-quality-ok: proxy module global, no injection seam
            "litellm.proxy.proxy_server.llm_model_list", list(llm_model_list)
        ),
        patch(  # test-quality-ok: proxy module global, no injection seam
            "litellm.proxy.proxy_server.llm_router", llm_router
        ),
        patch(  # test-quality-ok: proxy module global, no injection seam
            "litellm.proxy.proxy_server.prisma_client", None
        ),
        patch(  # test-quality-ok: proxy module global, no injection seam
            "litellm.proxy.proxy_server.use_background_health_checks", use_background_health_checks
        ),
        patch(  # test-quality-ok: proxy module global, no injection seam
            "litellm.proxy.proxy_server.user_model", None
        ),
        patch(  # test-quality-ok: proxy module global, no injection seam
            "litellm.proxy.proxy_server.health_check_results", dict(health_check_results or {})
        ),
        patch(  # test-quality-ok: proxy module global, no injection seam
            "litellm.proxy.proxy_server.health_check_details", True
        ),
        patch(  # test-quality-ok: proxy module global, no injection seam
            "litellm.proxy.proxy_server.health_check_concurrency", 1
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_health_endpoint_expands_access_group_on_live_path():
    """
    LIT-6907 / gh-28206: a key granted a model access group carries the group
    name in user_api_key_dict.models. Matching it as a literal model_name
    filtered every deployment out and /health answered 0/0 for a model the
    same key could call.
    """
    from fastapi import Response

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    captured: dict = {}

    async def fake_perform(**kwargs):
        captured["model_list"] = kwargs["model_list"]
        return {"healthy_endpoints": [], "unhealthy_endpoints": [], "healthy_count": 0, "unhealthy_count": 0}

    with (
        _proxy_health_globals(_ACCESS_GROUP_MODEL_LIST, _ACCESS_GROUP_ROUTER),
        patch(  # test-quality-ok: the model list handed to the probe is the assertion; no injection seam
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            side_effect=fake_perform,
        ),
    ):
        await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-group"]),
            model=None,
            model_id=None,
        )

    assert [m["model_name"] for m in captured["model_list"]] == ["bedrock-nova"]


@pytest.mark.asyncio
async def test_health_endpoint_expands_access_group_on_background_cache_path():
    """
    LIT-6907: the background-cache path scoped the cached entries through the
    same literal model_name match, so an access-group key got an empty result
    plus a warning blaming missing model_info.id.
    """
    from fastapi import Response

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with _proxy_health_globals(
        _ACCESS_GROUP_MODEL_LIST,
        _ACCESS_GROUP_ROUTER,
        use_background_health_checks=True,
        health_check_results=_ACCESS_GROUP_CACHED_RESULTS,
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-group"]),
            model=None,
            model_id=None,
        )

    assert [e["model_id"] for e in result["healthy_endpoints"]] == ["id-bedrock"]
    assert result["healthy_count"] == 1
    assert "warnings" not in result


@pytest.mark.asyncio
async def test_health_endpoint_treats_no_team_all_team_models_as_unrestricted():
    """
    A key granted "all-team-models" without a team resolves to an empty
    allowlist in the auth layer, which means unrestricted. /health used to
    keep the unresolved sentinel and filter every deployment out instead.
    """
    from fastapi import Response

    from litellm.proxy._types import SpecialModelNames, UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    captured: dict = {}

    async def fake_perform(**kwargs):
        captured["model_list"] = kwargs["model_list"]
        return {"healthy_endpoints": [], "unhealthy_endpoints": [], "healthy_count": 0, "unhealthy_count": 0}

    with (
        _proxy_health_globals(_ACCESS_GROUP_MODEL_LIST, None),
        patch(  # test-quality-ok: the model list handed to the probe is the assertion; no injection seam
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            side_effect=fake_perform,
        ),
    ):
        await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(
                api_key="hashed-test-key", models=[SpecialModelNames.all_team_models.value], team_id=None
            ),
            model=None,
            model_id=None,
        )

    assert {m["model_name"] for m in captured["model_list"]} == {"bedrock-nova", "gpt-5.4-mini"}


@pytest.mark.asyncio
async def test_health_endpoint_omits_model_id_warning_when_no_deployment_matches():
    """
    The missing-model_info.id warning is only true when a matching deployment
    exists without an id. A key whose grants match no deployment at all gets a
    plain empty result, not advice to populate ids that are already there.
    """
    from fastapi import Response

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with _proxy_health_globals(
        _ACCESS_GROUP_MODEL_LIST,
        _ACCESS_GROUP_ROUTER,
        use_background_health_checks=True,
        health_check_results=_ACCESS_GROUP_CACHED_RESULTS,
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["no-such-model"]),
            model=None,
            model_id=None,
        )

    assert result["healthy_count"] == 0
    assert result["unhealthy_count"] == 0
    assert "warnings" not in result


@pytest.mark.asyncio
async def test_health_endpoint_filters_background_cache_by_user_access():
    """
    When background_health_checks is enabled, health_endpoint() should also
    scope the cached result to the caller's allowed models rather than
    returning the cache verbatim.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_base": "https://example-a.test",
            },
            "model_info": {"id": "id-a"},
        },
        {
            "model_name": "model-b",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_base": "https://example-b.test",
            },
            "model_info": {"id": "id-b"},
        },
    ]

    cached_results = {
        "healthy_endpoints": [
            {
                "model": "openai/gpt-4o",
                "model_id": "id-a",
                "api_base": "https://example-a.test",
            },
            {
                "model": "openai/gpt-4o",
                "model_id": "id-b",
                "api_base": "https://example-b.test",
            },
        ],
        "unhealthy_endpoints": [],
        "healthy_count": 2,
        "unhealthy_count": 0,
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-test-key",
        models=["model-a"],
    )

    with (
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", True),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", cached_results),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
    ):
        from fastapi import Response

        # Pass model=None, model_id=None explicitly: direct calls to the
        # handler skip FastAPI's Query() resolution, so unspecified params
        # would otherwise carry the Query() sentinel (which is truthy).
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=user_api_key_dict,
            model=None,
            model_id=None,
        )

    # Sanity: the source cache had two entries before scoping; the scoping
    # step is what reduces it to one. (This guards against the test passing
    # vacuously when the cache filter drops everything because cached
    # entries lack the model_id key — both entries carry model_id above.)
    assert len(cached_results["healthy_endpoints"]) == 2
    assert all(ep.get("model_id") for ep in cached_results["healthy_endpoints"]), (
        "test fixture invariant: every cached entry must carry a model_id"
    )

    # The non-admin caller must not see api_base on the returned cache entries.
    returned = result.get("healthy_endpoints", [])
    assert len(returned) == 1, f"expected exactly one cached entry after scoping, got {len(returned)}"
    assert returned[0]["model_id"] == "id-a"
    assert "api_base" not in returned[0]
    assert result["healthy_count"] == 1
    assert result["unhealthy_count"] == 0


@pytest.mark.asyncio
async def test_health_endpoint_admin_sees_routing_fields_non_admin_does_not():
    """
    A proxy admin should still see ``api_base`` and ``api_version`` in the
    /health response so they can tell which Vertex region / Azure resource
    + API version is healthy. A non-admin caller must not — both fields
    should be stripped, and the response should carry a notice header so
    non-admin clients can detect the change programmatically.
    """
    from fastapi import Response

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_base": "https://example-a.test",
            },
            "model_info": {"id": "id-a"},
        },
    ]
    cached_results = {
        "healthy_endpoints": [
            {
                "model": "openai/gpt-4o",
                "model_id": "id-a",
                "api_base": "https://us-central1-aiplatform.googleapis.com/v1/projects/p",
                "api_version": "2024-10-21",
            },
        ],
        "unhealthy_endpoints": [],
        "healthy_count": 1,
        "unhealthy_count": 0,
    }

    admin_key = UserAPIKeyAuth(
        api_key="hashed-admin-key",
        models=["model-a"],
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    non_admin_key = UserAPIKeyAuth(
        api_key="hashed-user-key",
        models=["model-a"],
    )

    common_patches = [
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", True),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", cached_results),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
    ]

    for p in common_patches:
        p.start()
    try:
        admin_response = Response()
        non_admin_response = Response()
        admin_result = await health_endpoint(
            response=admin_response,
            user_api_key_dict=admin_key,
            model=None,
            model_id=None,
        )
        non_admin_result = await health_endpoint(
            response=non_admin_response,
            user_api_key_dict=non_admin_key,
            model=None,
            model_id=None,
        )
    finally:
        for p in common_patches:
            p.stop()

    admin_eps = admin_result.get("healthy_endpoints", [])
    non_admin_eps = non_admin_result.get("healthy_endpoints", [])

    assert len(admin_eps) == 1
    assert admin_eps[0]["api_base"] == "https://us-central1-aiplatform.googleapis.com/v1/projects/p", (
        "admin must see the full api_base so they can identify the region"
    )
    assert admin_eps[0]["api_version"] == "2024-10-21", (
        "admin must see api_version so they can distinguish provider deployments"
    )

    assert len(non_admin_eps) == 1
    assert "api_base" not in non_admin_eps[0]
    assert "api_version" not in non_admin_eps[0]

    # Non-admin response must advertise that api_base/api_version were
    # withheld so clients that previously parsed them can detect the change.
    assert (
        non_admin_response.headers.get("Litellm-Health-Field-Notice")
        == "api_base, api_version, aws_bedrock_runtime_endpoint are admin-only on this endpoint"
    )
    assert "Litellm-Health-Field-Notice" not in admin_response.headers

    # Stripping must produce a copy — the shared cache must still carry the
    # routing fields so the next admin caller can read them.
    cached_first = cached_results["healthy_endpoints"][0]
    assert cached_first["api_base"] == "https://us-central1-aiplatform.googleapis.com/v1/projects/p"
    assert cached_first["api_version"] == "2024-10-21"


@pytest.mark.asyncio
async def test_health_endpoint_warns_when_scoped_models_lack_model_id():
    """
    When a scoped key's accessible models exist on the proxy but none of the
    matching deployments expose a ``model_info.id``, the cache filter drops
    everything. The response should include a structured ``warnings`` field
    so the caller can distinguish "no deployments configured" from
    "deployments excluded due to missing model_info.id".
    """
    from fastapi import Response

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_base": "https://example-a.test",
            },
            # Intentionally no model_info.id — this is the misconfiguration
            # the warnings field is meant to flag.
            "model_info": {},
        },
    ]
    cached_results = {
        "healthy_endpoints": [
            {
                "model": "openai/gpt-4o",
                "model_id": "id-a",
                "api_base": "https://example-a.test",
            },
        ],
        "unhealthy_endpoints": [],
        "healthy_count": 1,
        "unhealthy_count": 0,
    }
    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-user-key",
        models=["model-a"],
    )

    with (
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", True),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", cached_results),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=user_api_key_dict,
            model=None,
            model_id=None,
        )

    assert result["healthy_count"] == 0
    assert result["unhealthy_count"] == 0
    assert "warnings" in result, (
        "empty cache result must surface a warnings field so the caller "
        "can distinguish 'no deployments' from 'deployments excluded'"
    )
    assert any("model_info.id" in w for w in result["warnings"])


@pytest.mark.asyncio
async def test_health_endpoint_blocks_cross_scope_model_id_under_background_cache():
    """
    A non-admin scoped to model-a must not be able to read model-b's cached
    health entry by guessing its model_id. Before the fix,
    _resolve_targeted_model_ids returned {model_id} unconditionally, so the
    cache filter was driven by an unvalidated ID and the global cache
    leaked id-b's entry to the caller.
    """
    from fastapi import HTTPException, Response

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"id": "id-a"},
        },
        {
            "model_name": "model-b",  # caller has no access
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"id": "id-b"},
        },
    ]

    cached_results = {
        "healthy_endpoints": [
            {
                "model": "openai/gpt-4o",
                "model_id": "id-b",
                "api_base": "https://leaky-internal.test",
            },
        ],
        "unhealthy_endpoints": [],
        "healthy_count": 1,
        "unhealthy_count": 0,
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-scoped",
        models=["model-a"],
    )

    response = Response()
    with (
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        # llm_router None here means the model_id 404 lookup short-circuits;
        # we patch _llm_model_list directly instead to drive the cache path.
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", True),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", cached_results),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
    ):
        # Calling with model="model-b" rather than model_id="id-b" because
        # the model_id branch raises 404 when llm_router is None. The bug
        # being verified is the same: a target outside the caller's scoped
        # model_list is refused before the cache is read.
        with pytest.raises(HTTPException) as refused:
            await health_endpoint(
                response=response,
                user_api_key_dict=user_api_key_dict,
                model="model-b",
                model_id=None,
            )

    assert refused.value.status_code == 403
    assert "leaky-internal.test" not in str(refused.value.detail)


@pytest.mark.asyncio
async def test_health_endpoint_503_for_targeted_unhealthy_model_under_background_cache_admin():
    """
    With background_health_checks enabled, an admin calling /health?model=foo
    must get 503 when foo specifically has zero healthy endpoints — even if
    other unrelated models in the cache are healthy. Without the cache-path
    filter, the global healthy_count would mask the targeted failure.
    """
    from fastapi import Response

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",  # the unhealthy target
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"id": "id-a"},
        },
        {
            "model_name": "model-b",  # an unrelated healthy model
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"id": "id-b"},
        },
    ]

    cached_results = {
        "healthy_endpoints": [
            {"model": "openai/gpt-4o", "model_id": "id-b"},
        ],
        "unhealthy_endpoints": [
            {"model": "openai/gpt-4o", "model_id": "id-a", "error": "boom"},
        ],
        "healthy_count": 1,
        "unhealthy_count": 1,
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    response = Response()
    with (
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", True),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", cached_results),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
    ):
        result = await health_endpoint(
            response=response,
            user_api_key_dict=user_api_key_dict,
            model="model-a",
            model_id=None,
        )

    assert response.status_code == 503
    # Body must be scoped to the targeted model — not the global cache.
    assert result["healthy_count"] == 0
    assert result["unhealthy_count"] == 1
    returned_ids = {ep["model_id"] for ep in result.get("unhealthy_endpoints", [])}
    assert returned_ids == {"id-a"}


@pytest.mark.asyncio
async def test_health_endpoint_returns_503_when_requested_model_has_no_healthy_endpoints():
    """
    /health?model=foo must return 503 when the targeted model resolves but
    has zero healthy endpoints. Body shape stays the same so existing
    parsers still work; only the HTTP status changes.
    """
    from fastapi import Response

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_base": "https://example-a.test",
            },
            "model_info": {"id": "id-a"},
        },
    ]

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    async def fake_perform(**kwargs):
        return {
            "healthy_endpoints": [],
            "unhealthy_endpoints": [
                {
                    "model": "openai/gpt-4o",
                    "model_id": "id-a",
                    "error": "boom",
                }
            ],
            "healthy_count": 0,
            "unhealthy_count": 1,
        }

    response = Response()
    with (
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", False),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", {}),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            side_effect=fake_perform,
        ),
    ):
        result = await health_endpoint(
            response=response,
            user_api_key_dict=user_api_key_dict,
            model="model-a",
        )

    assert response.status_code == 503
    assert result["healthy_count"] == 0
    assert result["unhealthy_count"] == 1


@pytest.mark.asyncio
async def test_health_endpoint_returns_200_when_requested_model_has_healthy_endpoints():
    """
    /health?model=foo with a healthy endpoint must keep returning the
    default 200. Verifies the 503 path doesn't fire when healthy_count > 0.
    """
    from fastapi import Response

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"id": "id-a"},
        },
    ]

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    async def fake_perform(**kwargs):
        return {
            "healthy_endpoints": [{"model": "openai/gpt-4o", "model_id": "id-a"}],
            "unhealthy_endpoints": [],
            "healthy_count": 1,
            "unhealthy_count": 0,
        }

    response = Response()
    # Default Response() exposes status_code as None; the endpoint should
    # leave it alone for the healthy path.
    with (
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", False),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", {}),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            side_effect=fake_perform,
        ),
    ):
        await health_endpoint(
            response=response,
            user_api_key_dict=user_api_key_dict,
            model="model-a",
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint_no_model_param_returns_200_even_when_zero_healthy():
    """
    The non-targeted /health (no model / model_id query) preserves the
    legacy 200 behavior even when healthy_count == 0. Existing K8s probes
    and dashboards depend on this; only the targeted call became 5xx-aware.
    """
    from fastapi import Response

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    full_model_list = [
        {
            "model_name": "model-a",
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"id": "id-a"},
        },
    ]

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    async def fake_perform(**kwargs):
        return {
            "healthy_endpoints": [],
            "unhealthy_endpoints": [{"model": "openai/gpt-4o", "model_id": "id-a", "error": "boom"}],
            "healthy_count": 0,
            "unhealthy_count": 1,
        }

    response = Response()
    with (
        patch("litellm.proxy.proxy_server.llm_model_list", full_model_list),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.proxy_server.use_background_health_checks", False),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.health_check_results", {}),
        patch("litellm.proxy.proxy_server.health_check_details", True),
        patch("litellm.proxy.proxy_server.health_check_concurrency", 1),
        patch(
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            side_effect=fake_perform,
        ),
    ):
        # Pass model=None, model_id=None explicitly: when invoked through
        # FastAPI, the Query(None) defaults resolve to None, but direct
        # function calls in unit tests receive Query() sentinel objects
        # (which are truthy). The explicit None mirrors production routing.
        await health_endpoint(
            response=response,
            user_api_key_dict=user_api_key_dict,
            model=None,
            model_id=None,
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_readiness_returns_503_when_db_disconnected():
    """
    When a Prisma client is configured but its health_check fails, the
    readiness probe should mark the worker as unhealthy via the HTTP
    status — not just a body field — so K8s removes the pod from the
    Service endpoints.
    """
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_readiness

    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=PrismaError("nope"))
    mock_prisma.attempt_db_reconnect = AsyncMock(side_effect=Exception("still nope"))

    _health_endpoints_module.db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(seconds=60),
    }

    response = Response()
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await health_readiness(response=response)

    assert response.status_code == 503
    assert result == {"status": "healthy", "db": "disconnected"}


@pytest.mark.asyncio
async def test_health_readiness_returns_200_when_db_down_and_allow_requests_on_db_unavailable():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/34934.

    allow_requests_on_db_unavailable keeps the proxy serving through a DB
    outage, so the readiness probe must keep the pod in rotation (200) and
    report the DB state through the body, not the status code. Otherwise
    K8s pulls every replica before the request-layer fail-open can run.
    """
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_readiness

    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=PrismaError("nope"))
    mock_prisma.attempt_db_reconnect = AsyncMock(side_effect=Exception("still nope"))

    _health_endpoints_module.db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(seconds=60),
    }

    response = Response()
    with (
        patch(  # test-quality-ok: the readiness path reads the proxy-global DB client; it has no injection seam
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ),
        patch.dict(  # test-quality-ok: the fail-open flag lives in the proxy-global general_settings; no injection seam
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": True},
        ),
    ):
        result = await health_readiness(response=response)

    assert response.status_code == 200
    assert result == {"status": "healthy", "db": "disconnected"}


@pytest.mark.asyncio
async def test_health_readiness_details_returns_200_when_db_down_and_allow_requests_on_db_unavailable():
    """
    The detailed readiness payload (public via
    allow_public_health_readiness_details, or /health/readiness/details)
    must honor the same flag so probes pointed at it also stay 200.
    """
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import (
        _get_health_readiness_details,
    )

    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=PrismaError("nope"))
    mock_prisma.attempt_db_reconnect = AsyncMock(side_effect=Exception("still nope"))

    _health_endpoints_module.db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(seconds=60),
    }

    response = Response()
    with (
        patch(  # test-quality-ok: the readiness path reads the proxy-global DB client; it has no injection seam
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ),
        patch.dict(  # test-quality-ok: the fail-open flag lives in the proxy-global general_settings; no injection seam
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": True},
        ),
    ):
        result = await _get_health_readiness_details(response=response)

    assert response.status_code == 200
    assert result["db"] == "disconnected"


@pytest.mark.asyncio
async def test_db_health_readiness_check_bounds_hung_health_check():
    """
    A connection that hangs mid-failover must not stall the probe past the
    kubelet's timeoutSeconds; the DB round-trip is bounded and reported as
    disconnected instead.
    """
    from litellm.proxy.health_endpoints._health_endpoints import (
        _db_health_readiness_check,
    )

    async def hang():
        await asyncio.sleep(60)

    mock_prisma = MagicMock()
    mock_prisma.health_check = hang
    mock_prisma.attempt_db_reconnect = AsyncMock(side_effect=Exception("still down"))

    _health_endpoints_module.db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(seconds=60),
    }

    with patch(  # test-quality-ok: lowers the module-level probe timeout so the hung-call test finishes fast
        "litellm.proxy.health_endpoints._health_endpoints.DB_READINESS_CHECK_TIMEOUT_SECONDS",
        0.05,
    ):
        start = time.monotonic()
        with patch(  # test-quality-ok: the readiness path reads the proxy-global DB client; it has no injection seam
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ):
            result = await _db_health_readiness_check()
        elapsed = time.monotonic() - start

    assert result["status"] == "disconnected"
    assert elapsed < 5


@pytest.mark.asyncio
async def test_db_health_readiness_check_overall_deadline_bounds_hung_reconnect():
    """
    The whole probe-path DB check (initial check + reconnect + re-check,
    including reconnect lock waits) runs under one deadline, so a reconnect
    that hangs on the lock still returns disconnected within the deadline.
    """
    from litellm.proxy.health_endpoints._health_endpoints import (
        _db_health_readiness_check,
    )

    async def hang(**kwargs):
        await asyncio.sleep(60)

    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=httpx.ConnectError("down"))
    mock_prisma.attempt_db_reconnect = hang

    _health_endpoints_module.db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(seconds=60),
    }

    with patch(  # test-quality-ok: lowers the module-level probe timeout so the hung-call test finishes fast
        "litellm.proxy.health_endpoints._health_endpoints.DB_READINESS_PROBE_DEADLINE_SECONDS",
        0.05,
    ):
        start = time.monotonic()
        with patch(  # test-quality-ok: the readiness path reads the proxy-global DB client; it has no injection seam
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ):
            result = await _db_health_readiness_check()
        elapsed = time.monotonic() - start

    assert result["status"] == "disconnected"
    assert elapsed < 5


@pytest.mark.asyncio
async def test_health_readiness_returns_200_when_db_connected():
    """Happy path: connected DB keeps the legacy 200."""
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_readiness

    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock()

    _health_endpoints_module.db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(seconds=60),
    }

    response = Response()
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await health_readiness(response=response)

    assert response.status_code == 200
    assert result == {"status": "healthy", "db": "connected"}


@pytest.mark.asyncio
async def test_health_readiness_returns_200_when_no_db_configured():
    """
    `prisma_client is None` means the operator chose not to use a DB. That
    is a valid configuration — the worker should still report ready. We
    only flip to 503 when a DB *was* configured but is unreachable.
    """
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_readiness

    response = Response()
    with patch("litellm.proxy.proxy_server.prisma_client", None):
        result = await health_readiness(response=response)

    assert response.status_code == 200
    assert result == {"status": "healthy", "db": "Not connected"}


def test_clean_endpoint_data_strips_credentials_keeps_routing_fields():
    """
    _clean_endpoint_data() drops credentials but leaves api_base /
    api_version intact — the per-caller hide/show happens in the endpoint
    layer based on user role, not in the cleaning helper. This guarantees
    proxy admins continue to see those fields in the /health response.
    """
    from litellm.proxy.health_check import _clean_endpoint_data

    raw = {
        "model": "openai/gpt-4o",
        "api_key": "sk-test",
        "api_base": "https://example.test/v1",
        "api_version": "2024-10-21",
        "aws_access_key_id": "AKIAEXAMPLE",
    }

    cleaned = _clean_endpoint_data(raw, details=True)

    assert "api_key" not in cleaned
    assert "aws_access_key_id" not in cleaned
    assert cleaned.get("api_base") == "https://example.test/v1"
    assert cleaned.get("api_version") == "2024-10-21"


def test_clean_endpoint_data_strips_extra_headers_and_aws_session_token():
    """
    gh-36898: GET /health must not leak provider credentials that live in
    `extra_headers` / `headers` / `aws_session_token`. Before the fix these
    were returned in plaintext (api_key was stripped, but these were not).
    """
    from litellm.proxy.health_check import _clean_endpoint_data

    raw = {
        "model": "openai/gpt-4o",
        "api_base": "https://example.test/v1",
        "extra_headers": {
            "Authorization": "Bearer CANARY_EXTRA_HEADERS_AUTHORIZATION",
            "x-goog-api-key": "CANARY_X_GOOG_API_KEY_VALUE",
            "api-key": "CANARY_AZURE_STYLE_API_KEY",
        },
        "headers": {"X-Custom": "CANARY_HEADER_VALUE"},
        "aws_session_token": "CANARY_AWS_SESSION_TOKEN_VALUE",
    }

    cleaned = _clean_endpoint_data(raw, details=True)

    assert "extra_headers" not in cleaned
    assert "headers" not in cleaned
    assert "aws_session_token" not in cleaned
    assert cleaned.get("api_base") == "https://example.test/v1"


@pytest.mark.parametrize(
    "credential_field",
    [
        "api_key",
        "client_secret",
        "azure_ad_token",
        "azure_username",
        "azure_password",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_web_identity_token",
        "vertex_credentials",
        "vertex_ai_credentials",
        "extra_headers",
        "headers",
    ],
)
@pytest.mark.parametrize("details", [True, False, None])
def test_clean_endpoint_data_never_displays_credential_fields(credential_field, details):
    """
    LIT-6239 / gh-36898: /health entries, healthy and unhealthy alike, must never
    carry credential-bearing litellm_params, with or without details.
    """
    from litellm.proxy.health_check import _clean_endpoint_data

    canary = f"CANARY-{credential_field}-VALUE"
    cleaned = _clean_endpoint_data(
        {
            "model": "azure/gpt-5-mini",
            "api_base": "https://example.test/v1",
            credential_field: canary,
        },
        details=details,
    )

    assert credential_field not in cleaned
    assert canary not in str(cleaned)


async def _live_probed_model_ids(
    model_list: Sequence[Mapping[str, object]],
    user_api_key_dict: UserAPIKeyAuth,
    model: str | None = None,
    model_id: str | None = None,
    router: Router | None = None,
) -> set[str]:
    from fastapi import Response

    from litellm.proxy.health_check import narrow_to_target
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    captured: dict = {}

    async def fake_perform(**kwargs):
        captured["model_list"] = narrow_to_target(kwargs["model_list"], kwargs["target_model"], kwargs["model_id"])
        return {"healthy_endpoints": [], "unhealthy_endpoints": [], "healthy_count": 0, "unhealthy_count": 0}

    with (
        _proxy_health_globals(model_list, router if router is not None else _router_for(model_list)),
        patch(  # test-quality-ok: the model list handed to the probe is the assertion; no injection seam
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            side_effect=fake_perform,
        ),
    ):
        await health_endpoint(response=Response(), user_api_key_dict=user_api_key_dict, model=model, model_id=model_id)

    return {m["model_info"]["id"] for m in captured["model_list"]}


@pytest.mark.asyncio
async def test_health_endpoint_hides_another_teams_deployment_behind_a_shared_access_group():
    """
    Expanding an access group must not reach past the team boundary: a
    team-a key holding the group name may not probe team-b's deployment even
    though that deployment sits in the same group.
    """
    probed = await _live_probed_model_ids(
        _TEAM_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-group"], team_id="team-a"),
    )

    assert probed == {"id-bedrock"}


@pytest.mark.asyncio
async def test_health_endpoint_hides_team_deployments_from_a_key_with_no_team():
    """
    Routing never serves a team-owned deployment to a caller without a team
    (``filter_team_based_models``), so a team-less access-group key must not
    probe team-b's deployment with team-b's credentials either.
    """
    probed = await _live_probed_model_ids(
        _TEAM_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-group"], team_id=None),
    )

    assert probed == {"id-bedrock"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_id", "expected_ids"),
    [(None, {"id-bedrock"}), ("team-a", {"id-bedrock"}), ("team-b", {"id-bedrock", "id-team-b"})],
)
async def test_health_endpoint_keeps_an_unrestricted_non_admin_key_to_its_own_team(team_id, expected_ids):
    """
    A key with no model restriction is still bound by routing's team rule:
    it may probe global deployments and its own team's, never another team's.
    """
    probed = await _live_probed_model_ids(
        _TEAM_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=[], team_id=team_id),
    )

    assert probed == expected_ids


@pytest.mark.asyncio
async def test_health_endpoint_lets_a_proxy_admin_probe_every_teams_deployment():
    probed = await _live_probed_model_ids(
        _TEAM_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=[], user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert probed == {"id-bedrock", "id-team-b"}


@pytest.mark.asyncio
async def test_health_endpoint_keeps_an_unrestricted_non_admin_key_to_its_own_team_on_background_cache_path():
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with _proxy_health_globals(
        _TEAM_MODEL_LIST,
        _router_for(_TEAM_MODEL_LIST),
        use_background_health_checks=True,
        health_check_results=_TEAM_CACHED_RESULTS,
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=[], team_id="team-a"),
            model=None,
            model_id=None,
        )

    assert [ep["model_id"] for ep in result["healthy_endpoints"]] == ["id-bedrock"]
    assert result["healthy_count"] == 1


@pytest.mark.asyncio
async def test_health_endpoint_shows_a_teams_own_deployment_by_its_public_name():
    """
    A team key names its team deployment by ``team_public_model_name``, while
    the proxy model list carries the internal ``<name>_<team_id>_<uuid>``
    name; the deployment must still be probed for its own team.
    """
    probed = await _live_probed_model_ids(
        _TEAM_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-nova"], team_id="team-b"),
    )

    assert probed == {"id-bedrock", "id-team-b"}


@pytest.mark.asyncio
async def test_health_endpoint_hides_another_teams_deployment_on_background_cache_path():
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with _proxy_health_globals(
        _TEAM_MODEL_LIST,
        _router_for(_TEAM_MODEL_LIST),
        use_background_health_checks=True,
        health_check_results=_TEAM_CACHED_RESULTS,
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-group"], team_id="team-a"),
            model=None,
            model_id=None,
        )

    assert [ep["model_id"] for ep in result["healthy_endpoints"]] == ["id-bedrock"]
    assert result["healthy_count"] == 1


@pytest.mark.asyncio
async def test_health_endpoint_refuses_a_targeted_deployment_outside_the_callers_scope_on_live_path():
    """
    A scoped key asking for a deployment it may not see must get a 403 and no
    probe at all: probing the rest of its scope instead would report another
    deployment's health under the requested id and store it as such.
    """
    from fastapi import HTTPException, Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    fake_perform = AsyncMock()

    with (
        _proxy_health_globals(_TEAM_MODEL_LIST, _router_for(_TEAM_MODEL_LIST)),
        patch(  # test-quality-ok: the probe must never run; no injection seam
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            fake_perform,
        ),
        pytest.raises(HTTPException) as excinfo,
    ):
        await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-group"], team_id=None),
            model=None,
            model_id="id-team-b",
        )

    assert excinfo.value.status_code == 403
    fake_perform.assert_not_awaited()


@pytest.mark.asyncio
async def test_health_endpoint_refuses_a_targeted_deployment_outside_the_callers_scope_on_background_cache_path():
    from fastapi import HTTPException, Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with (
        _proxy_health_globals(
            _TEAM_MODEL_LIST,
            _router_for(_TEAM_MODEL_LIST),
            use_background_health_checks=True,
            health_check_results=_TEAM_CACHED_RESULTS,
        ),
        pytest.raises(HTTPException) as excinfo,
    ):
        await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-group"], team_id="team-a"),
            model="bedrock-nova_team-b_9f2c",
            model_id=None,
        )

    assert excinfo.value.status_code == 403


_PROVIDER_PREFIXED_MODEL_LIST = [
    {
        "model_name": "bedrock/us.amazon.nova-2-lite-v1:0",
        "litellm_params": {"model": "bedrock/us.amazon.nova-2-lite-v1:0"},
        "model_info": {"id": "id-bedrock-prefixed"},
    },
    _ACCESS_GROUP_MODEL_LIST[1],
]
_PROVIDER_PREFIXED_CACHED_RESULTS = {
    "healthy_endpoints": [
        {"model": "bedrock/us.amazon.nova-2-lite-v1:0", "model_id": "id-bedrock-prefixed"},
        {"model": "openai/gpt-5.4-mini", "model_id": "id-openai"},
    ],
    "unhealthy_endpoints": [],
    "healthy_count": 2,
    "unhealthy_count": 0,
}


@pytest.mark.asyncio
async def test_health_endpoint_expands_a_provider_wildcard_key_on_live_path():
    """
    LIT-6971: auth lets a ``bedrock/*`` key call ``bedrock/us.amazon.nova-2-lite-v1:0``,
    but /health compared the pattern to the deployment name literally and probed nothing.
    """
    probed = await _live_probed_model_ids(
        _PROVIDER_PREFIXED_MODEL_LIST, UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock/*"])
    )

    assert probed == {"id-bedrock-prefixed"}


@pytest.mark.asyncio
async def test_health_endpoint_shows_a_deployment_a_team_reaches_through_a_model_alias():
    """Auth accepts a request under a team model alias, so /health must show the deployment the alias points at."""
    probed = await _live_probed_model_ids(
        _ACCESS_GROUP_MODEL_LIST,
        UserAPIKeyAuth(
            api_key="hashed-test-key",
            models=[],
            team_id="team-a",
            team_models=["nova-alias"],
            team_model_aliases={"nova-alias": "bedrock-nova"},
        ),
    )

    assert probed == {"id-bedrock"}


@pytest.mark.asyncio
async def test_health_endpoint_targets_a_deployment_by_the_alias_a_request_would_use():
    """``/health?model=<team alias>`` must reach the deployment the alias points at, as a request under that alias does."""
    probed = await _live_probed_model_ids(
        _ACCESS_GROUP_MODEL_LIST,
        UserAPIKeyAuth(
            api_key="hashed-test-key",
            models=[],
            team_id="team-a",
            team_models=["nova-alias"],
            team_model_aliases={"nova-alias": "bedrock-nova"},
        ),
        model="nova-alias",
    )

    assert probed == {"id-bedrock"}


@pytest.mark.asyncio
async def test_health_endpoint_follows_a_team_alias_to_the_teams_public_model_name():
    """A team alias may point at the public name of the team's own deployment; auth accepts it, so /health must show it."""
    probed = await _live_probed_model_ids(
        [_TEAM_MODEL_LIST[1], _ACCESS_GROUP_MODEL_LIST[1]],
        UserAPIKeyAuth(
            api_key="hashed-test-key",
            models=[],
            team_id="team-b",
            team_models=["nova-alias"],
            team_model_aliases={"nova-alias": "bedrock-nova"},
        ),
    )

    assert probed == {"id-team-b"}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, "nova-latest"])
async def test_health_endpoint_shows_a_deployment_reached_through_a_router_model_group_alias(model: str | None):
    """Auth resolves a router ``model_group_alias`` before the allowlist check, so /health must show the aliased deployment."""
    probed = await _live_probed_model_ids(
        _ACCESS_GROUP_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=["nova-latest"]),
        model=model,
        router=Router(
            model_list=copy.deepcopy(_ACCESS_GROUP_MODEL_LIST), model_group_alias={"nova-latest": "bedrock-nova"}
        ),
    )

    assert probed == {"id-bedrock"}


_WILDCARD_MODEL_LIST = [
    {
        "model_name": "bedrock/*",
        "litellm_params": {"model": "bedrock/*"},
        "model_info": {"id": "id-bedrock-wildcard"},
    },
    _ACCESS_GROUP_MODEL_LIST[1],
]
_CATCH_ALL_MODEL_LIST = [
    {"model_name": "*", "litellm_params": {"model": "*"}, "model_info": {"id": "id-catch-all"}},
    _ACCESS_GROUP_MODEL_LIST[0],
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, "bedrock/us.amazon.nova-2-lite-v1:0"])
async def test_health_endpoint_shows_a_wildcard_deployment_that_serves_a_key_model(model: str | None):
    """A key allowed the concrete model calls it through the ``bedrock/*`` deployment, so /health must show that deployment."""
    probed = await _live_probed_model_ids(
        _WILDCARD_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock/us.amazon.nova-2-lite-v1:0"]),
        model=model,
    )

    assert probed == {"id-bedrock-wildcard"}


@pytest.mark.asyncio
async def test_health_endpoint_hides_a_wildcard_deployment_when_no_single_model_passes_both_the_key_and_the_team_allowlist():
    """Auth checks the one requested model against both allowlists, so a key and a team allowing different models call nothing through ``bedrock/*``."""
    probed = await _live_probed_model_ids(
        _WILDCARD_MODEL_LIST,
        UserAPIKeyAuth(
            api_key="hashed-test-key",
            models=["bedrock/us.amazon.nova-2-lite-v1:0"],
            team_id="team-a",
            team_models=["bedrock/us.amazon.nova-2-pro-v1:0"],
        ),
    )

    assert probed == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_models", "model"),
    [
        (["bedrock/us.amazon.nova-2-pro-v1:0"], "bedrock/us.amazon.nova-2-lite-v1:0"),
        ([], "bedrock/us.amazon.nova-2-pro-v1:0"),
    ],
)
async def test_health_endpoint_refuses_to_target_a_model_through_a_wildcard_deployment_when_auth_would_deny_it(
    team_models: list[str], model: str
):
    """``bedrock/*`` serves many models; targeting one the key or the team does not allow must 403 the way the request would."""
    from fastapi import HTTPException, Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    fake_perform = AsyncMock()

    with (
        _proxy_health_globals(_WILDCARD_MODEL_LIST, _router_for(_WILDCARD_MODEL_LIST)),
        patch(  # test-quality-ok: the probe must never run; no injection seam
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            fake_perform,
        ),
        pytest.raises(HTTPException) as excinfo,
    ):
        await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(
                api_key="hashed-test-key",
                models=["bedrock/us.amazon.nova-2-lite-v1:0"],
                team_id="team-a" if team_models else None,
                team_models=team_models,
            ),
            model=model,
            model_id=None,
        )

    assert excinfo.value.status_code == 403
    fake_perform.assert_not_awaited()


@pytest.mark.asyncio
async def test_health_endpoint_does_not_treat_a_wildcard_shaped_alias_as_a_pattern():
    """Auth rewrites aliases by exact name, so an alias spelled ``openai/*`` never carries ``openai/gpt-5.4-nano`` anywhere."""
    probed = await _live_probed_model_ids(
        _WILDCARD_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=["openai/gpt-5.4-nano"]),
        router=Router(
            model_list=copy.deepcopy(_WILDCARD_MODEL_LIST),
            model_group_alias={"openai/*": "bedrock/us.amazon.nova-2-lite-v1:0"},
        ),
    )

    assert probed == set()


def _router_copying_a_deployment_under_a_wildcard_shaped_alias() -> Router:
    return Router(model_list=copy.deepcopy(_WILDCARD_MODEL_LIST), model_group_alias={"openai/*": "gpt-5.4-mini"})


@pytest.mark.asyncio
async def test_health_endpoint_does_not_expand_the_alias_copy_the_router_adds_to_the_model_list():
    """``Router.get_model_list`` copies an alias target in under the alias name; a copy spelled ``openai/*`` is no wildcard route."""
    router = _router_copying_a_deployment_under_a_wildcard_shaped_alias()

    probed = await _live_probed_model_ids(
        router.get_model_list(),
        UserAPIKeyAuth(api_key="hashed-test-key", models=["openai/gpt-5.4-nano"]),
        router=router,
    )

    assert probed == set()


@pytest.mark.asyncio
async def test_health_endpoint_targeting_a_model_only_an_alias_copy_could_match_probes_nothing():
    """No deployment serves ``openai/gpt-5.4-nano`` through an alias copy spelled ``openai/*``, so targeting it probes nothing."""
    router = _router_copying_a_deployment_under_a_wildcard_shaped_alias()

    probed = await _live_probed_model_ids(
        router.get_model_list(),
        UserAPIKeyAuth(api_key="hashed-test-key", user_role=LitellmUserRoles.PROXY_ADMIN),
        model="openai/gpt-5.4-nano",
        router=router,
    )

    assert probed == set()


_REAL_WILDCARD_WITH_COLLIDING_ALIAS_LIST = [
    {
        "model_name": "openai/*",
        "litellm_params": {"model": "openai/*"},
        "model_info": {"id": "id-openai-wildcard"},
    },
    _ACCESS_GROUP_MODEL_LIST[1],
]


@pytest.mark.asyncio
async def test_health_endpoint_keeps_a_real_wildcard_deployment_when_an_alias_key_shares_its_name():
    """A real ``openai/*`` deployment must survive the alias-copy filter when ``model_group_alias`` also spells one of its keys ``openai/*``; only the shallow copy of the alias target is dropped."""
    router = Router(
        model_list=copy.deepcopy(_REAL_WILDCARD_WITH_COLLIDING_ALIAS_LIST),
        model_group_alias={"openai/*": "gpt-5.4-mini"},
    )

    probed = await _live_probed_model_ids(
        router.get_model_list(),
        UserAPIKeyAuth(api_key="hashed-test-key", models=["openai/gpt-5.4-nano"]),
        router=router,
    )

    assert probed == {"id-openai-wildcard"}


@pytest.mark.asyncio
async def test_health_endpoint_keeps_a_real_deployment_named_like_a_hidden_alias_key():
    """A hidden ``model_group_alias`` entry never adds a copy to the router's list, so a real deployment sharing that key's name must not be filtered out."""
    router = Router(
        model_list=copy.deepcopy(_REAL_WILDCARD_WITH_COLLIDING_ALIAS_LIST),
        model_group_alias={"openai/*": {"model": "gpt-5.4-mini", "hidden": True}},
    )

    probed = await _live_probed_model_ids(
        router.get_model_list(),
        UserAPIKeyAuth(api_key="hashed-test-key", models=["openai/gpt-5.4-nano"]),
        router=router,
    )

    assert probed == {"id-openai-wildcard"}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, "nova-alias"])
async def test_health_endpoint_shows_a_wildcard_deployment_a_team_alias_points_into(model: str | None):
    """Auth accepts a request under a team alias and routes it through ``bedrock/*``, so /health must show that deployment."""
    probed = await _live_probed_model_ids(
        _WILDCARD_MODEL_LIST,
        UserAPIKeyAuth(
            api_key="hashed-test-key",
            models=[],
            team_id="team-a",
            team_models=["nova-alias"],
            team_model_aliases={"nova-alias": "bedrock/us.amazon.nova-2-lite-v1:0"},
        ),
        model=model,
    )

    assert probed == {"id-bedrock-wildcard"}


_OVERLAPPING_WILDCARD_MODEL_LIST = [_PROVIDER_PREFIXED_MODEL_LIST[0], _WILDCARD_MODEL_LIST[0]]
_OVERLAPPING_WILDCARD_CACHED_RESULTS = {
    "healthy_endpoints": [
        {"model": "bedrock/us.amazon.nova-2-lite-v1:0", "model_id": "id-bedrock-prefixed"},
        {"model": "bedrock/us.amazon.nova-2-pro-v1:0", "model_id": "id-bedrock-wildcard"},
    ],
    "unhealthy_endpoints": [],
    "healthy_count": 2,
    "unhealthy_count": 0,
}


@pytest.mark.asyncio
async def test_health_endpoint_targets_the_concrete_deployment_on_background_cache_path_when_a_wildcard_also_serves_it():
    """The live path probes only the concrete deployment for a model it names, so the cache path must not add the wildcard's entry."""
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with _proxy_health_globals(
        _OVERLAPPING_WILDCARD_MODEL_LIST,
        _router_for(_OVERLAPPING_WILDCARD_MODEL_LIST),
        use_background_health_checks=True,
        health_check_results=_OVERLAPPING_WILDCARD_CACHED_RESULTS,
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock/us.amazon.nova-2-lite-v1:0"]),
            model="bedrock/us.amazon.nova-2-lite-v1:0",
            model_id=None,
        )

    assert [ep["model_id"] for ep in result["healthy_endpoints"]] == ["id-bedrock-prefixed"]
    assert result["healthy_count"] == 1


@pytest.mark.asyncio
async def test_health_endpoint_shows_a_catch_all_deployment_to_a_named_model_key_but_not_to_an_access_group_key():
    """``*`` serves every model a key names; an access group entry names a group, not a model, so it reaches only its group."""
    named = await _live_probed_model_ids(
        _CATCH_ALL_MODEL_LIST, UserAPIKeyAuth(api_key="hashed-test-key", models=["some-model"])
    )
    grouped = await _live_probed_model_ids(
        _CATCH_ALL_MODEL_LIST, UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-group"])
    )

    assert (named, grouped) == ({"id-catch-all"}, {"id-bedrock"})


class _ModelNameCountingRouter(Router):
    def __init__(self, model_list: Sequence[Mapping[str, object]]) -> None:
        self.model_name_lookups = 0
        super().__init__(model_list=copy.deepcopy(list(model_list)))
        self.model_name_lookups = 0

    def get_model_names(self, team_id: str | None = None) -> list[str]:
        self.model_name_lookups += 1
        return super().get_model_names(team_id=team_id)


@pytest.mark.asyncio
async def test_health_endpoint_expands_all_team_models_once_rather_than_per_deployment():
    """A team allowlist of ``all-team-models`` is expanded once for the whole list, not once per deployment it is checked against."""
    router = _ModelNameCountingRouter(_ACCESS_GROUP_MODEL_LIST)
    probed = await _live_probed_model_ids(
        _ACCESS_GROUP_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=[], team_id="team-a", team_models=["all-team-models"]),
        router=router,
    )

    assert (probed, router.model_name_lookups) == ({"id-bedrock", "id-openai"}, 1)


@pytest.mark.asyncio
async def test_health_endpoint_skips_the_key_allowlist_for_a_key_with_a_config_the_way_auth_does():
    """Auth applies only the team allowlist to a key that carries a ``config``, so /health must too."""
    probed = await _live_probed_model_ids(
        _PROVIDER_PREFIXED_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock/*"], config={"lit6971": True}),
    )

    assert probed == {"id-bedrock-prefixed", "id-openai"}


@pytest.mark.asyncio
async def test_health_endpoint_lets_a_provider_wildcard_key_target_its_deployment_by_model_id():
    probed = await _live_probed_model_ids(
        _PROVIDER_PREFIXED_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock/*"]),
        model_id="id-bedrock-prefixed",
    )

    assert probed == {"id-bedrock-prefixed"}


@pytest.mark.asyncio
async def test_health_endpoint_expands_a_provider_wildcard_key_on_background_cache_path():
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with _proxy_health_globals(
        _PROVIDER_PREFIXED_MODEL_LIST,
        _router_for(_PROVIDER_PREFIXED_MODEL_LIST),
        use_background_health_checks=True,
        health_check_results=_PROVIDER_PREFIXED_CACHED_RESULTS,
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock/*"]),
            model=None,
            model_id=None,
        )

    assert [ep["model_id"] for ep in result["healthy_endpoints"]] == ["id-bedrock-prefixed"]
    assert result["healthy_count"] == 1
    assert "warnings" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_models", "team_models", "expected_ids"),
    [
        ([], ["gpt-5.4-mini"], {"id-openai"}),
        ([], ["bedrock-group"], {"id-bedrock"}),
        (["gpt-5.4-mini", "bedrock-group"], ["gpt-5.4-mini"], {"id-openai"}),
    ],
)
async def test_health_endpoint_applies_the_team_allowlist_the_way_auth_does(key_models, team_models, expected_ids):
    """
    LIT-6971: a request from a key on a team with restricted ``models`` has to
    pass the team's allowlist too, but /health only read the key's own list, so
    a key with ``models: []`` on such a team probed every global deployment.
    """
    probed = await _live_probed_model_ids(
        _ACCESS_GROUP_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=key_models, team_id="team-a", team_models=team_models),
    )

    assert probed == expected_ids


@pytest.mark.asyncio
async def test_health_endpoint_applies_the_team_allowlist_on_background_cache_path():
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with _proxy_health_globals(
        _ACCESS_GROUP_MODEL_LIST,
        _ACCESS_GROUP_ROUTER,
        use_background_health_checks=True,
        health_check_results=_ACCESS_GROUP_CACHED_RESULTS,
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(
                api_key="hashed-test-key", models=[], team_id="team-a", team_models=["gpt-5.4-mini"]
            ),
            model=None,
            model_id=None,
        )

    assert [ep["model_id"] for ep in result["healthy_endpoints"]] == ["id-openai"]
    assert result["healthy_count"] == 1


@pytest.mark.asyncio
async def test_health_endpoint_refuses_a_targeted_deployment_the_team_allowlist_forbids():
    from fastapi import HTTPException, Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    fake_perform = AsyncMock()

    with (
        _proxy_health_globals(_ACCESS_GROUP_MODEL_LIST, _ACCESS_GROUP_ROUTER),
        patch(  # test-quality-ok: the probe must never run; no injection seam
            "litellm.proxy.health_endpoints._health_endpoints._perform_health_check_and_save",
            fake_perform,
        ),
        pytest.raises(HTTPException) as excinfo,
    ):
        await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(
                api_key="hashed-test-key", models=[], team_id="team-a", team_models=["gpt-5.4-mini"]
            ),
            model=None,
            model_id="id-bedrock",
        )

    assert excinfo.value.status_code == 403
    fake_perform.assert_not_awaited()


@pytest.mark.asyncio
async def test_health_endpoint_hides_team_deployments_from_a_key_with_no_team_on_background_cache_path():
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with _proxy_health_globals(
        _TEAM_MODEL_LIST,
        _router_for(_TEAM_MODEL_LIST),
        use_background_health_checks=True,
        health_check_results=_TEAM_CACHED_RESULTS,
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-group"], team_id=None),
            model=None,
            model_id=None,
        )

    assert [ep["model_id"] for ep in result["healthy_endpoints"]] == ["id-bedrock"]
    assert result["healthy_count"] == 1


_TEAM_ONLY_MODEL_LIST = [_TEAM_MODEL_LIST[1]]


@pytest.mark.asyncio
async def test_health_endpoint_probes_a_team_only_deployment_by_its_public_name_on_live_path():
    """
    A team key targets its deployment by ``team_public_model_name``; when that
    name resolves to nothing but the team deployment, the probe must run rather
    than 403 as if the key were out of scope.
    """
    probed = await _live_probed_model_ids(
        _TEAM_ONLY_MODEL_LIST,
        UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-nova"], team_id="team-b"),
        model="bedrock-nova",
    )

    assert probed == {"id-team-b"}


@pytest.mark.asyncio
async def test_health_endpoint_returns_a_team_only_deployment_by_its_public_name_on_background_cache_path():
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with _proxy_health_globals(
        _TEAM_ONLY_MODEL_LIST,
        _router_for(_TEAM_ONLY_MODEL_LIST),
        use_background_health_checks=True,
        health_check_results=_TEAM_CACHED_RESULTS,
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-nova"], team_id="team-b"),
            model="bedrock-nova",
            model_id=None,
        )

    assert [ep["model_id"] for ep in result["healthy_endpoints"]] == ["id-team-b"]


@pytest.mark.asyncio
async def test_health_endpoint_targets_both_deployments_behind_a_shared_public_name_on_background_cache_path():
    from fastapi import Response

    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    with _proxy_health_globals(
        _TEAM_MODEL_LIST,
        _router_for(_TEAM_MODEL_LIST),
        use_background_health_checks=True,
        health_check_results=_TEAM_CACHED_RESULTS,
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key", models=["bedrock-nova"], team_id="team-b"),
            model="bedrock-nova",
            model_id=None,
        )

    assert [ep["model_id"] for ep in result["healthy_endpoints"]] == ["id-bedrock", "id-team-b"]


def test_health_test_connection_keeps_error_and_raw_request_through_the_allowlist(monkeypatch):
    """
    The dashboard's Test Connect button reads ``result.error`` and
    ``result.raw_request_typed_dict`` from /health/test_connection, so the
    allowlist must keep both while dropping the probe's own params.
    """
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()

    app = FastAPI()
    app.include_router(_health_endpoints_module.router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    client = TestClient(app)

    with (
        patch(  # test-quality-ok: the endpoint reads the proxy-global DB client and 500s when it is None; it has no injection seam
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ),
        respx.mock(assert_all_called=True) as respx_mock,
    ):
        respx_mock.post(host="api.openai.com", path="/v1/chat/completions").respond(
            status_code=401, json={"error": {"message": "Incorrect API key provided"}}
        )
        response = client.post(
            "/health/test_connection",
            json={
                "mode": "chat",
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk-test", "timeout": 7},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "error"
    assert "Incorrect API key provided" in body["result"]["error"]
    assert "api.openai.com" in body["result"]["raw_request_typed_dict"]["raw_request_api_base"]
    assert not {"api_key", "timeout", "exception"} & set(body["result"])


def test_clean_endpoint_data_keeps_only_json_safe_diagnostics():
    """
    LIT-6907: _clean_endpoint_data used to copy every litellm_param not on a
    deny list, so a nested mapping keyed by a tuple reached jsonable_encoder
    and 500'd /health. Only the explicit allowlist survives now.
    """
    from fastapi.encoders import jsonable_encoder

    from litellm.proxy.health_check import _clean_endpoint_data

    cleaned = _clean_endpoint_data(
        {
            "model": "bedrock/us.amazon.nova-2-lite-v1:0",
            "custom_llm_provider": "bedrock",
            "aws_region_name": "us-east-1",
            "metadata": {("us-east-1", "primary"): "canary-nested-mapping"},
            "allow_client_keepalive_override": False,
            "api_key": "CANARY-API-KEY",
            "x-ratelimit-remaining-requests": 99,
            "raw_request_typed_dict": {"raw_request_api_base": "https://example.test"},
            "aws_bedrock_runtime_endpoint": "https://vpce-bedrock.example.test",
        },
        details=True,
    )

    assert cleaned == {
        "model": "bedrock/us.amazon.nova-2-lite-v1:0",
        "custom_llm_provider": "bedrock",
        "aws_region_name": "us-east-1",
        "x-ratelimit-remaining-requests": 99,
        "raw_request_typed_dict": {"raw_request_api_base": "https://example.test"},
        "aws_bedrock_runtime_endpoint": "https://vpce-bedrock.example.test",
    }
    assert jsonable_encoder(cleaned) == cleaned


@pytest.mark.asyncio
async def test_health_endpoint_result_survives_non_json_safe_deployment_params():
    """
    LIT-6907: the full /health path with a deployment carrying a tuple-keyed
    nested mapping must produce a response FastAPI can encode, with the
    approved diagnostics intact and the offending param absent.
    """
    from fastapi import Response
    from fastapi.encoders import jsonable_encoder

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.health_endpoints._health_endpoints import health_endpoint

    model_list = [
        {
            "model_name": "bedrock-nova",
            "litellm_params": {
                "model": "bedrock/us.amazon.nova-2-lite-v1:0",
                "aws_region_name": "us-east-1",
                "aws_access_key_id": "CANARY-ACCESS-KEY",
                "metadata": {("us-east-1", "primary"): "canary-nested-mapping"},
            },
            "model_info": {"id": "id-bedrock"},
        }
    ]

    with (
        _proxy_health_globals(model_list, None),
        patch(  # test-quality-ok: the provider probe is faked; the assertion is the response shaping after it
            "litellm.ahealth_check", AsyncMock(return_value={"x-ratelimit-remaining-requests": 99})
        ),
    ):
        result = await health_endpoint(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-admin-key", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    encoded = jsonable_encoder(result)
    assert encoded["healthy_count"] == 1
    entry = encoded["healthy_endpoints"][0]
    assert entry["model_id"] == "id-bedrock"
    assert entry["aws_region_name"] == "us-east-1"
    assert entry["x-ratelimit-remaining-requests"] == 99
    assert "metadata" not in entry
    assert "CANARY" not in str(encoded)


class TestConfigBaseForHealthCheck:
    """A request that sets its own connection fields gets a base without the
    configuration's credentials; anything it leaves unset still comes from
    the configuration."""

    CONFIG = {
        "model": "openai/gpt-4o",
        "api_key": "sk-configured",
        "api_base": "https://configured.example/v1",
        "vertex_credentials": "configured-creds",
        "rpm": 100,
    }

    def _base(self, config, request, allow_client_side_credentials=False):
        from litellm.proxy.health_endpoints._health_endpoints import (
            _config_base_for_health_check,
        )

        return _config_base_for_health_check(
            config, request, allow_client_side_credentials=allow_client_side_credentials
        )

    def test_request_without_connection_fields_inherits_config(self):
        base = self._base(self.CONFIG, {"model": "openai/gpt-4o"})
        assert base["api_key"] == "sk-configured"
        assert base["api_base"] == "https://configured.example/v1"

    def test_request_setting_api_base_does_not_inherit_config_credentials(self):
        base = self._base(self.CONFIG, {"api_base": "https://caller.example/v1"})
        assert "api_key" not in base
        assert "api_base" not in base
        assert "vertex_credentials" not in base
        assert base["rpm"] == 100

    def test_add_model_flow_keeps_its_own_credentials(self):
        """Adding a second deployment for an already-configured name sends a
        complete connection; it is tested as sent, not as configured."""
        request = {
            "model": "openai/gpt-4o",
            "api_base": "https://new-deployment.example/v1",
            "api_key": "sk-new-deployment",
        }
        merged = {**self._base(self.CONFIG, request), **request}
        assert merged["api_base"] == "https://new-deployment.example/v1"
        assert merged["api_key"] == "sk-new-deployment"
        assert "sk-configured" not in str(merged)

    def test_destination_override_without_own_key_inherits_no_credential(self):
        """A request that redirects the destination but supplies no credential
        of its own gets none from the configuration."""
        request = {"api_base": "https://elsewhere.example"}
        merged = {**self._base(self.CONFIG, request), **request}
        assert "api_key" not in merged
        assert "sk-configured" not in str(merged)

    def test_non_api_base_destination_field_also_drops_credentials(self):
        base = self._base(
            {**self.CONFIG, "aws_secret_access_key": "configured-secret"},
            {"aws_bedrock_runtime_endpoint": "https://caller.example"},
        )
        assert "api_key" not in base
        assert "aws_secret_access_key" not in base

    def test_opt_in_restores_configured_credentials_under_a_request_endpoint(self):
        """With general_settings.allow_client_side_credentials enabled, a request
        may pair its own endpoint with the configured credentials, as before."""
        base = self._base(
            self.CONFIG,
            {"api_base": "https://caller.example/v1"},
            allow_client_side_credentials=True,
        )
        assert base["api_key"] == "sk-configured"

    def test_stored_credential_reference_is_dropped_with_the_credentials(self):
        """A stored-credential name resolves to the same secrets downstream, so a
        request that redirects the destination must not keep it either."""
        config = {**self.CONFIG, "litellm_credential_name": "OpenAI-prod"}
        base = self._base(config, {"api_base": "https://caller.example/v1"})
        assert "litellm_credential_name" not in base
        assert "api_key" not in base

    def test_stored_credential_reference_kept_when_request_sets_no_connection(self):
        """The Admin UI tests a configured model by naming it plus its stored
        credential and nothing else; that keeps working."""
        config = {**self.CONFIG, "litellm_credential_name": "OpenAI-prod"}
        base = self._base(
            config,
            {"model": "openai/gpt-4o", "litellm_credential_name": "OpenAI-prod", "custom_llm_provider": "openai"},
        )
        assert base["litellm_credential_name"] == "OpenAI-prod"
        assert base["api_key"] == "sk-configured"

    def test_request_naming_another_credential_does_not_inherit_config_credentials(self):
        base = self._base(self.CONFIG, {"model": "openai/gpt-4o", "litellm_credential_name": "Another-cred"})
        assert "api_key" not in base
        assert "api_base" not in base
        assert "vertex_credentials" not in base
        assert base["rpm"] == 100

    def test_blank_credential_name_names_no_credential(self):
        base = self._base(self.CONFIG, {"model": "openai/gpt-4o", "litellm_credential_name": ""})
        assert base["api_key"] == "sk-configured"

    def test_opt_in_does_not_put_config_credentials_over_a_named_credential(self):
        base = self._base(
            self.CONFIG,
            {"model": "openai/gpt-4o", "litellm_credential_name": "Another-cred"},
            allow_client_side_credentials=True,
        )
        assert "api_key" not in base


class TestTestConnectionUsesTheNamedCredential:
    CREDENTIAL_KEY = "sk-credential-key"
    OTHER_DEPLOYMENT_KEY = "sk-other-deployment-key"
    OTHER_DEPLOYMENT_BASE = "https://other-deployment.example/v1"
    REQUEST = {
        "model": "xai/grok-4",
        "custom_llm_provider": "xai",
        "litellm_credential_name": "my-xai-cred",
    }
    COMPLETION = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "grok-4",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    @staticmethod
    def _credential(**values: str) -> CredentialItem:
        return CredentialItem(credential_name="my-xai-cred", credential_info={}, credential_values=values)

    @staticmethod
    def _wildcard_deployment(**litellm_params: str) -> dict:
        return {
            "model_name": "xai/*",
            "litellm_params": {"model": "xai/*", **litellm_params},
            "model_info": {"id": "unrelated-wildcard-deployment"},
        }

    def _probe(
        self,
        monkeypatch,
        deployment: dict,
        request_litellm_params: dict,
        deployment_by_id: object | None = None,
        request_model_info: dict | None = None,
    ) -> httpx.Request:
        """Run /health/test_connection and hand back the upstream request it made."""
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        litellm.in_memory_llm_clients_cache.flush_cache()

        app = FastAPI()
        app.include_router(_health_endpoints_module.router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

        router = MagicMock()
        router.get_model_list.return_value = [deployment]
        router.get_deployment.return_value = deployment_by_id

        with (
            patch(  # test-quality-ok: the endpoint reads the proxy-global DB client and 500s when it is None; it has no injection seam
                "litellm.proxy.proxy_server.prisma_client", MagicMock()
            ),
            patch(  # test-quality-ok: the deployment the probe is matched against is a proxy global; it has no injection seam
                "litellm.proxy.proxy_server.llm_router", router
            ),
            respx.mock(assert_all_called=True) as respx_mock,
        ):
            respx_mock.post(path__regex=r".*/chat/completions").respond(json=self.COMPLETION)
            response = TestClient(app).post(
                "/health/test_connection",
                json={
                    "mode": "chat",
                    "litellm_params": request_litellm_params,
                    "model_info": request_model_info or {"mode": "chat"},
                },
            )
            probe = respx_mock.calls.last.request

        assert response.status_code == 200, response.text
        assert response.json()["status"] == "success", response.text
        return probe

    def test_named_credentials_key_is_sent_not_the_matched_deployments_key(self, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [self._credential(api_key=self.CREDENTIAL_KEY)])

        probe = self._probe(
            monkeypatch,
            self._wildcard_deployment(api_key=self.OTHER_DEPLOYMENT_KEY),
            self.REQUEST,
        )

        assert probe.headers["authorization"] == f"Bearer {self.CREDENTIAL_KEY}"

    def test_named_credentials_api_base_is_used_not_the_matched_deployments(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [self._credential(api_key=self.CREDENTIAL_KEY, api_base="https://credential.example/v1")],
        )

        probe = self._probe(
            monkeypatch,
            self._wildcard_deployment(api_base=self.OTHER_DEPLOYMENT_BASE),
            self.REQUEST,
        )

        assert probe.url.host == "credential.example"

    def test_named_credential_without_an_api_base_leaves_the_provider_default(self, monkeypatch):
        monkeypatch.setattr(litellm, "credential_list", [self._credential(api_key=self.CREDENTIAL_KEY)])

        probe = self._probe(
            monkeypatch,
            self._wildcard_deployment(api_base=self.OTHER_DEPLOYMENT_BASE),
            self.REQUEST,
        )

        assert probe.url.host == "api.x.ai"

    def test_configured_model_named_without_a_credential_still_inherits_its_config(self, monkeypatch):
        probe = self._probe(
            monkeypatch,
            self._wildcard_deployment(api_key=self.OTHER_DEPLOYMENT_KEY, api_base=self.OTHER_DEPLOYMENT_BASE),
            {"model": "xai/grok-4", "custom_llm_provider": "xai"},
        )

        assert probe.headers["authorization"] == f"Bearer {self.OTHER_DEPLOYMENT_KEY}"
        assert probe.url.host == "other-deployment.example"

    def test_deployment_probed_by_id_keeps_the_endpoint_it_is_configured_with(self, monkeypatch):
        """The model detail page always echoes back the credential the deployment already uses."""
        from litellm.types.router import Deployment, LiteLLM_Params

        monkeypatch.setattr(litellm, "credential_list", [self._credential(api_key=self.CREDENTIAL_KEY)])

        probe = self._probe(
            monkeypatch,
            self._wildcard_deployment(api_key=self.OTHER_DEPLOYMENT_KEY, api_base=self.OTHER_DEPLOYMENT_BASE),
            self.REQUEST,
            deployment_by_id=Deployment(
                model_name="grok-4",
                litellm_params=LiteLLM_Params(
                    model="xai/grok-4",
                    api_base="https://configured.example/v1",
                    litellm_credential_name="my-xai-cred",
                ),
                model_info={"id": "configured-deployment"},
            ),
            request_model_info={"id": "configured-deployment", "mode": "chat"},
        )

        assert probe.url.host == "configured.example"
        assert probe.headers["authorization"] == f"Bearer {self.CREDENTIAL_KEY}"


class TestNoRedisWarning:
    """`show_no_redis_warning` drives the Admin UI's default-on "no Redis" banner."""

    @staticmethod
    def _router(redis_cache):
        return SimpleNamespace(cache=SimpleNamespace(redis_cache=redis_cache))

    @staticmethod
    def _prisma_with_workers(live_workers=None, error=None):
        prisma = MagicMock()
        if error is not None:
            prisma.db.query_raw = AsyncMock(side_effect=error)
        else:
            prisma.db.query_raw = AsyncMock(return_value=[{"live_workers": live_workers}])
        return prisma

    @pytest.mark.asyncio
    async def test_warns_when_no_redis_and_no_db_to_count_workers(self, monkeypatch):
        monkeypatch.delenv("LITELLM_DISABLE_NO_REDIS_WARNING", raising=False)
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", self._router(None)),
            patch("litellm.proxy.proxy_server.prisma_client", None),
        ):
            assert await _show_no_redis_warning() is True

    @pytest.mark.asyncio
    async def test_warns_when_there_is_no_router_at_all(self, monkeypatch):
        monkeypatch.delenv("LITELLM_DISABLE_NO_REDIS_WARNING", raising=False)
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", None),
            patch("litellm.proxy.proxy_server.prisma_client", None),
        ):
            assert await _show_no_redis_warning() is True

    @pytest.mark.asyncio
    async def test_stays_quiet_for_a_confirmed_single_worker(self, monkeypatch):
        """One live worker needs no cross-worker coordination, so no env var is needed."""
        monkeypatch.delenv("LITELLM_DISABLE_NO_REDIS_WARNING", raising=False)
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", self._router(None)),
            patch("litellm.proxy.proxy_server.prisma_client", self._prisma_with_workers(1)),
        ):
            assert await _show_no_redis_warning() is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("live_workers", [2, 5])
    async def test_warns_when_multiple_workers_share_the_db(self, monkeypatch, live_workers):
        monkeypatch.delenv("LITELLM_DISABLE_NO_REDIS_WARNING", raising=False)
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", self._router(None)),
            patch("litellm.proxy.proxy_server.prisma_client", self._prisma_with_workers(live_workers)),
        ):
            assert await _show_no_redis_warning() is True

    @pytest.mark.asyncio
    async def test_warns_when_the_worker_census_is_empty(self, monkeypatch):
        """Zero rows means the census cannot CONFIRM a single worker, so warn."""
        monkeypatch.delenv("LITELLM_DISABLE_NO_REDIS_WARNING", raising=False)
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", self._router(None)),
            patch("litellm.proxy.proxy_server.prisma_client", self._prisma_with_workers(0)),
        ):
            assert await _show_no_redis_warning() is True

    @pytest.mark.asyncio
    async def test_warns_when_the_worker_census_query_fails(self, monkeypatch):
        monkeypatch.delenv("LITELLM_DISABLE_NO_REDIS_WARNING", raising=False)
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", self._router(None)),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                self._prisma_with_workers(error=RuntimeError("db down")),
            ),
        ):
            assert await _show_no_redis_warning() is True

    @pytest.mark.asyncio
    async def test_stays_quiet_when_a_coordination_redis_is_configured(self, monkeypatch):
        monkeypatch.delenv("LITELLM_DISABLE_NO_REDIS_WARNING", raising=False)
        prisma = self._prisma_with_workers(5)
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", MagicMock()),
            patch("litellm.proxy.proxy_server.llm_router", self._router(None)),
            patch("litellm.proxy.proxy_server.prisma_client", prisma),
        ):
            assert await _show_no_redis_warning() is False
        prisma.db.query_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_stays_quiet_when_only_the_router_has_redis(self, monkeypatch):
        """router_settings.redis_host alone backs cooldowns and usage-based routing."""
        monkeypatch.delenv("LITELLM_DISABLE_NO_REDIS_WARNING", raising=False)
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", self._router(MagicMock())),
            patch("litellm.proxy.proxy_server.prisma_client", self._prisma_with_workers(5)),
        ):
            assert await _show_no_redis_warning() is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["true", "True"])
    async def test_env_var_suppresses_the_warning_despite_multiple_workers(self, monkeypatch, value):
        monkeypatch.setenv("LITELLM_DISABLE_NO_REDIS_WARNING", value)
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", self._router(None)),
            patch("litellm.proxy.proxy_server.prisma_client", self._prisma_with_workers(5)),
        ):
            assert await _show_no_redis_warning() is False

    @pytest.mark.asyncio
    async def test_env_var_set_false_keeps_the_warning_for_multiple_workers(self, monkeypatch):
        monkeypatch.setenv("LITELLM_DISABLE_NO_REDIS_WARNING", "false")
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", self._router(None)),
            patch("litellm.proxy.proxy_server.prisma_client", self._prisma_with_workers(2)),
        ):
            assert await _show_no_redis_warning() is True

    @pytest.mark.asyncio
    async def test_env_var_set_false_does_not_force_the_warning_for_a_single_worker(self, monkeypatch):
        monkeypatch.setenv("LITELLM_DISABLE_NO_REDIS_WARNING", "false")
        with (
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", self._router(None)),
            patch("litellm.proxy.proxy_server.prisma_client", self._prisma_with_workers(1)),
        ):
            assert await _show_no_redis_warning() is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("has_prisma_client", [True, False])
    async def test_readiness_details_carries_the_flag(self, monkeypatch, has_prisma_client):
        monkeypatch.delenv("LITELLM_DISABLE_NO_REDIS_WARNING", raising=False)
        prisma_client = self._prisma_with_workers(2) if has_prisma_client else None
        with (
            patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
            patch("litellm.proxy.proxy_server.redis_usage_cache", None),
            patch("litellm.proxy.proxy_server.llm_router", self._router(None)),
            patch.object(
                _health_endpoints_module,
                "_db_health_readiness_check",
                AsyncMock(return_value={"status": "connected"}),
            ),
        ):
            details = await _health_endpoints_module._get_health_readiness_details()
        assert details["show_no_redis_warning"] is True

        with (
            patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
            patch("litellm.proxy.proxy_server.redis_usage_cache", MagicMock()),
            patch.object(
                _health_endpoints_module,
                "_db_health_readiness_check",
                AsyncMock(return_value={"status": "connected"}),
            ),
        ):
            details = await _health_endpoints_module._get_health_readiness_details()
        assert details["show_no_redis_warning"] is False


@pytest.mark.asyncio
async def test_health_services_endpoint_ms_teams_posts_adaptive_card():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post = AsyncMock(return_value=mock_response)
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.slack_alerting_instance.async_http_handler.post = mock_post

    with (
        patch(  # test-quality-ok: endpoint reads proxy_server module globals, same pattern as sibling tests
            "litellm.proxy.proxy_server.general_settings",
            {"alerting": ["ms_teams"]},
        ),
        patch(  # test-quality-ok: endpoint reads proxy_server module globals, same pattern as sibling tests
            "litellm.proxy.proxy_server.proxy_logging_obj",
            mock_proxy_logging,
        ),
        patch.dict("os.environ", {"MS_TEAMS_WEBHOOK_URL": "https://teams.example/webhook"}),
    ):
        result = await health_services_endpoint(service="ms_teams")

    assert result["status"] == "success"
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"] == "https://teams.example/webhook"
    sent_body = json.loads(call_kwargs["data"])
    assert sent_body["type"] == "message"
    assert sent_body["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"


@pytest.mark.asyncio
async def test_health_services_endpoint_ms_teams_surfaces_delivery_failure():
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Invalid webhook"
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.slack_alerting_instance.async_http_handler.post = AsyncMock(return_value=mock_response)

    with (
        patch(  # test-quality-ok: endpoint reads proxy_server module globals, same pattern as sibling tests
            "litellm.proxy.proxy_server.general_settings",
            {"alerting": ["ms_teams"]},
        ),
        patch(  # test-quality-ok: endpoint reads proxy_server module globals, same pattern as sibling tests
            "litellm.proxy.proxy_server.proxy_logging_obj",
            mock_proxy_logging,
        ),
        patch.dict("os.environ", {"MS_TEAMS_WEBHOOK_URL": "https://teams.example/webhook"}),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await health_services_endpoint(service="ms_teams")

    assert "status 400" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_health_services_endpoint_ms_teams_requires_alerting_config():
    with patch(  # test-quality-ok: endpoint reads proxy_server module globals, same pattern as sibling tests
        "litellm.proxy.proxy_server.general_settings",
        {"alerting": ["slack"]},
    ):
        with pytest.raises(ProxyException):
            await health_services_endpoint(service="ms_teams")


def test_test_model_connection_accepts_image_edit_mode(monkeypatch):
    """
    Regression: /health/test_connection rejected mode=image_edit with a 422
    before image_edit was added to its mode Literal, breaking the UI Test
    Connection button for image edit deployments.
    """
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()

    app = FastAPI()
    app.include_router(_health_endpoints_module.router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    client = TestClient(app)

    with (
        patch(  # test-quality-ok: the endpoint reads the proxy-global DB client and 500s when it is None; it has no injection seam
            "litellm.proxy.proxy_server.prisma_client", MagicMock()
        ),
        respx.mock(assert_all_called=True) as respx_mock,
    ):
        respx_mock.post(host="api.openai.com", path="/v1/images/edits").respond(
            json={"created": 1700000000, "data": [{"b64_json": TEST_IMAGE_BASE64}]}
        )
        response = client.post(
            "/health/test_connection",
            json={
                "mode": "image_edit",
                "litellm_params": {"model": "openai/gpt-image-2", "api_key": "sk-test"},
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "success"

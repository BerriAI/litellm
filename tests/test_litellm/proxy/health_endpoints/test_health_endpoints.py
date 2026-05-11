import os
import sys
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prisma.errors import ClientNotConnectedError, HTTPClientClosedError, PrismaError

import litellm.proxy.health_endpoints._health_endpoints as _health_endpoints_module

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.health_endpoints._health_endpoints import (
    _db_health_readiness_check,
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
        reason="health_readiness_check"
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
        reason="health_readiness_check"
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
    mock_prisma.attempt_db_reconnect = AsyncMock(
        side_effect=RuntimeError("reconnect failed")
    )

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
        mock_instance.async_health_check = AsyncMock(
            return_value={"status": status, "error_message": error_message}
        )
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

        assert model_params.get("api_base") == (
            "https://deployment-B-base.invalid/v1"
        ), (
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
    Database and Redis are optional - the public endpoint should work whether they're available or not.
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

    # Assert response contains only low-detail public probe fields
    response_data = response.json()
    assert response_data == {"status": "healthy"}
    print(f"Response time: {duration_ms:.2f}ms")


def test_health_readiness_details_returns_diagnostic_fields(monkeypatch):
    """
    Detailed readiness diagnostics stay available behind the auth dependency.
    """
    app = FastAPI()
    app.include_router(_health_endpoints_module.router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN
    )
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

        await health_endpoint(response=Response(), user_api_key_dict=user_api_key_dict)

    assert (
        "model_list" in captured
    ), "health_endpoint did not call _perform_health_check_and_save"
    returned_names = {m["model_name"] for m in captured["model_list"]}
    assert returned_names == {
        "model-a"
    }, f"health_endpoint did not scope model_list to caller access: {returned_names}"


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
    assert all(
        ep.get("model_id") for ep in cached_results["healthy_endpoints"]
    ), "test fixture invariant: every cached entry must carry a model_id"

    # The non-admin caller must not see api_base on the returned cache entries.
    returned = result.get("healthy_endpoints", [])
    assert (
        len(returned) == 1
    ), f"expected exactly one cached entry after scoping, got {len(returned)}"
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
    assert (
        admin_eps[0]["api_base"]
        == "https://us-central1-aiplatform.googleapis.com/v1/projects/p"
    ), "admin must see the full api_base so they can identify the region"
    assert (
        admin_eps[0]["api_version"] == "2024-10-21"
    ), "admin must see api_version so they can distinguish provider deployments"

    assert len(non_admin_eps) == 1
    assert "api_base" not in non_admin_eps[0]
    assert "api_version" not in non_admin_eps[0]

    # Non-admin response must advertise that api_base/api_version were
    # withheld so clients that previously parsed them can detect the change.
    assert (
        non_admin_response.headers.get("Litellm-Health-Field-Notice")
        == "api_base and api_version are admin-only on this endpoint"
    )
    assert "Litellm-Health-Field-Notice" not in admin_response.headers

    # Stripping must produce a copy — the shared cache must still carry the
    # routing fields so the next admin caller can read them.
    cached_first = cached_results["healthy_endpoints"][0]
    assert (
        cached_first["api_base"]
        == "https://us-central1-aiplatform.googleapis.com/v1/projects/p"
    )
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
    from fastapi import Response

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
        # being verified is the same: targeted resolver must drop entries
        # not in the caller's scoped model_list. With the fix, the result
        # has no leaked endpoints and the targeted-503 path fires.
        result = await health_endpoint(
            response=response,
            user_api_key_dict=user_api_key_dict,
            model="model-b",
            model_id=None,
        )

    leaked_ids = {ep.get("model_id") for ep in result.get("healthy_endpoints", [])}
    leaked_ids |= {ep.get("model_id") for ep in result.get("unhealthy_endpoints", [])}
    assert (
        "id-b" not in leaked_ids
    ), "background cache leaked an out-of-scope deployment to a scoped caller"
    assert result["healthy_count"] == 0
    assert response.status_code == 503


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
            "unhealthy_endpoints": [
                {"model": "openai/gpt-4o", "model_id": "id-a", "error": "boom"}
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
    assert result == {"status": "healthy"}


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
    assert result == {"status": "healthy"}


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
    assert result == {"status": "healthy"}


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

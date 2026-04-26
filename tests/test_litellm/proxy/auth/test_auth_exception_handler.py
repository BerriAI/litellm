import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, status
from prisma import errors as prisma_errors
from prisma.errors import (
    ClientNotConnectedError,
    DataError,
    ForeignKeyViolationError,
    HTTPClientClosedError,
    MissingRequiredValueError,
    PrismaError,
    RawQueryError,
    RecordNotFoundError,
    TableNotFoundError,
    UniqueViolationError,
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        # Specific connectivity subclasses.
        HTTPClientClosedError(),
        ClientNotConnectedError(),
        # Bare / generic PrismaError defaults to connectivity — we can't
        # tell what it is, so err on the safe side for genuine outages.
        PrismaError(),
    ],
)
async def test_handle_authentication_error_db_unavailable_connectivity(prisma_error):
    """Transport-level / connectivity failures (and generic PrismaError)
    trigger the HA fallback."""
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        result = await handler._handle_authentication_error(
            prisma_error,
            mock_request,
            {},
            "/test",
            None,
            "test-key",
        )
        assert result.key_name == "failed-to-connect-to-db"
        assert result.token == "failed-to-connect-to-db"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        DataError(data={"user_facing_error": {"meta": {"table": "test_table"}}}),
        UniqueViolationError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        ForeignKeyViolationError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        MissingRequiredValueError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        RawQueryError(data={"user_facing_error": {"meta": {"table": "test_table"}}}),
        TableNotFoundError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        RecordNotFoundError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
    ],
)
async def test_handle_authentication_error_data_layer_errors_do_not_fall_back(
    prisma_error,
):
    """Known data-layer PrismaError subclasses (UniqueViolation,
    RecordNotFound, etc.) mean the DB IS reachable — they must propagate
    instead of triggering the HA fallback, which would grant the
    restricted INTERNAL_USER token to a request that should have
    returned 401."""
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        with pytest.raises(ProxyException):
            await handler._handle_authentication_error(
                prisma_error,
                mock_request,
                {},
                "/test",
                None,
                "test-key",
            )


@pytest.mark.asyncio
async def test_handle_authentication_error_budget_exceeded():
    handler = UserAPIKeyAuthExceptionHandler()

    # Mock request and other dependencies
    mock_request = MagicMock()
    mock_request_data = {}
    mock_route = "/test"
    mock_span = None
    mock_api_key = "test-key"

    # Test with budget exceeded error
    with pytest.raises(ProxyException) as exc_info:
        from litellm.exceptions import BudgetExceededError

        budget_error = BudgetExceededError(
            message="Budget exceeded", current_cost=100, max_budget=100
        )
        await handler._handle_authentication_error(
            budget_error,
            mock_request,
            mock_request_data,
            mock_route,
            mock_span,
            mock_api_key,
        )

    assert exc_info.value.type == ProxyErrorTypes.budget_exceeded


@pytest.mark.asyncio
async def test_route_passed_to_post_call_failure_hook():
    """
    This route is used by proxy track_cost_callback's async_post_call_failure_hook to check if the route is an LLM route
    """
    handler = UserAPIKeyAuthExceptionHandler()

    # Mock request and other dependencies
    mock_request = MagicMock()
    mock_request_data = {}
    test_route = "/custom/route"
    mock_span = None
    mock_api_key = "test-key"

    # Mock proxy_logging_obj.post_call_failure_hook
    with patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
    ) as mock_post_call_failure_hook:
        # Test with DB connection error
        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ):
            try:
                await handler._handle_authentication_error(
                    PrismaError(),
                    mock_request,
                    mock_request_data,
                    test_route,
                    mock_span,
                    mock_api_key,
                )
            except Exception as e:
                pass
            asyncio.sleep(1)
            # Verify post_call_failure_hook was called with the correct route
            mock_post_call_failure_hook.assert_called_once()
            call_args = mock_post_call_failure_hook.call_args[1]
            assert call_args["user_api_key_dict"].request_route == test_route

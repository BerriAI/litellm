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
        PrismaError(),
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
        HTTPClientClosedError(),
        ClientNotConnectedError(),
    ],
)
async def test_handle_authentication_error_db_unavailable(prisma_error):
    handler = UserAPIKeyAuthExceptionHandler()

    # Mock request and other dependencies
    mock_request = MagicMock()
    mock_request_data = {}
    mock_route = "/test"
    mock_span = None
    mock_api_key = "test-key"

    # Test with DB connection error when requests are allowed
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        result = await handler._handle_authentication_error(
            prisma_error,
            mock_request,
            mock_request_data,
            mock_route,
            mock_span,
            mock_api_key,
        )
        assert result.key_name == "failed-to-connect-to-db"
        assert result.token == "failed-to-connect-to-db"


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


@pytest.mark.asyncio
async def test_expected_auth_errors_log_at_warning_level():
    """
    Expected auth failures (ProxyException, HTTPException < 500, BudgetExceededError)
    should log at WARNING level, not ERROR, to reduce log noise.
    """
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request_data = {}
    mock_route = "/schedule/model_cost_map_reload/status"
    mock_span = None
    mock_api_key = "sk-test1234"

    expected_auth_errors = [
        ProxyException(
            message="Token not found",
            type=ProxyErrorTypes.token_not_found_in_db,
            param="key",
            code=401,
        ),
        HTTPException(status_code=401, detail="Invalid API key"),
        HTTPException(status_code=403, detail="Forbidden"),
    ]

    for error in expected_auth_errors:
        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ), patch.object(
            verbose_proxy_logger, "warning"
        ) as mock_warning, patch.object(
            verbose_proxy_logger, "exception"
        ) as mock_exception:
            try:
                await handler._handle_authentication_error(
                    error,
                    mock_request,
                    mock_request_data,
                    mock_route,
                    mock_span,
                    mock_api_key,
                )
            except Exception:
                pass

            assert mock_warning.call_count == 1, (
                f"Expected warning log for {type(error).__name__}, got none"
            )
            assert mock_exception.call_count == 0, (
                f"Did not expect exception log for {type(error).__name__}"
            )


@pytest.mark.asyncio
async def test_unexpected_errors_log_at_error_level():
    """
    Unexpected exceptions (bare Exception, HTTPException with 500) should
    still log at ERROR level with full traceback.
    """
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    mock_request.method = "POST"
    mock_request_data = {}
    mock_route = "/chat/completions"
    mock_span = None
    mock_api_key = "sk-test1234"

    unexpected_errors = [
        Exception("Something unexpected broke"),
        HTTPException(status_code=500, detail="Master key type error"),
    ]

    for error in unexpected_errors:
        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ), patch.object(
            verbose_proxy_logger, "warning"
        ) as mock_warning, patch.object(
            verbose_proxy_logger, "exception"
        ) as mock_exception:
            try:
                await handler._handle_authentication_error(
                    error,
                    mock_request,
                    mock_request_data,
                    mock_route,
                    mock_span,
                    mock_api_key,
                )
            except Exception:
                pass

            mock_exception.assert_called_once(), (
                f"Expected exception log for {type(error).__name__}, got none"
            )
            mock_warning.assert_not_called(), (
                f"Did not expect warning log for {type(error).__name__}"
            )


@pytest.mark.asyncio
async def test_auth_error_log_contains_structured_context():
    """
    Log messages should include route, HTTP method, masked API key, error type,
    and error code for easier debugging.
    """
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request_data = {}
    mock_route = "/schedule/model_cost_map_reload/status"
    mock_span = None
    mock_api_key = "sk-test1234"

    error = ProxyException(
        message="Token not found",
        type=ProxyErrorTypes.token_not_found_in_db,
        param="key",
        code=401,
    )

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": False},
    ), patch.object(verbose_proxy_logger, "warning") as mock_warning:
        try:
            await handler._handle_authentication_error(
                error,
                mock_request,
                mock_request_data,
                mock_route,
                mock_span,
                mock_api_key,
            )
        except Exception:
            pass

        mock_warning.assert_called_once()
        log_message = mock_warning.call_args[0][0]
        log_extra = mock_warning.call_args[1].get("extra", {})

        # Verify structured fields are in the log message
        assert mock_route in log_message
        assert "GET" in log_message
        assert "sk-...1234" in log_message

        # Verify structured extra dict for log aggregation tools
        assert log_extra["route"] == mock_route
        assert log_extra["http_method"] == "GET"
        assert log_extra["api_key"] == "sk-...1234"
        assert log_extra["error_type"] == ProxyErrorTypes.token_not_found_in_db


@pytest.mark.asyncio
async def test_auth_error_log_handles_none_api_key():
    """
    When no API key is provided (None or empty), the log should show 'None'
    instead of crashing.
    """
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request_data = {}
    mock_route = "/test"
    mock_span = None

    error = Exception("No api key passed in.")

    for empty_key in [None, ""]:
        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ), patch.object(verbose_proxy_logger, "exception") as mock_exception:
            try:
                await handler._handle_authentication_error(
                    error,
                    mock_request,
                    mock_request_data,
                    mock_route,
                    mock_span,
                    empty_key,
                )
            except Exception:
                pass

            mock_exception.assert_called_once()
            log_message = mock_exception.call_args[0][0]
            assert "None" in log_message

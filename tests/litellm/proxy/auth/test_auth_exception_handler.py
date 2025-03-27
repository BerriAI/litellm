import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, patch

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


# Test is_database_connection_error method
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
def test_is_database_connection_error_prisma_errors(prisma_error):
    """
    Test that all Prisma errors are considered database connection errors
    """
    handler = UserAPIKeyAuthExceptionHandler()
    assert handler.is_database_connection_error(prisma_error) == True


def test_is_database_connection_generic_errors():
    """
    Test non-Prisma error cases for database connection checking
    """
    handler = UserAPIKeyAuthExceptionHandler()

    # Test with ProxyException (DB connection)
    db_proxy_exception = ProxyException(
        message="DB Connection Error",
        type=ProxyErrorTypes.no_db_connection,
        param="test-param",
    )
    assert handler.is_database_connection_error(db_proxy_exception) == True

    # Test with non-DB error
    regular_exception = Exception("Regular error")
    assert handler.is_database_connection_error(regular_exception) == False


# Test should_allow_request_on_db_unavailable method
@patch(
    "litellm.proxy.proxy_server.general_settings",
    {"allow_requests_on_db_unavailable": True},
)
def test_should_allow_request_on_db_unavailable_true():
    handler = UserAPIKeyAuthExceptionHandler()
    assert handler.should_allow_request_on_db_unavailable() == True


@patch(
    "litellm.proxy.proxy_server.general_settings",
    {"allow_requests_on_db_unavailable": False},
)
def test_should_allow_request_on_db_unavailable_false():
    handler = UserAPIKeyAuthExceptionHandler()
    assert handler.should_allow_request_on_db_unavailable() == False


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

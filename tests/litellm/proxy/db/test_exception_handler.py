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

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler


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
    assert PrismaDBExceptionHandler.is_database_connection_error(prisma_error) == True


def test_is_database_connection_generic_errors():
    """
    Test non-Prisma error cases for database connection checking
    """
    assert (
        PrismaDBExceptionHandler.is_database_connection_error(
            Exception("Regular error")
        )
        == False
    )

    # Test with ProxyException (DB connection)
    db_proxy_exception = ProxyException(
        message="DB Connection Error",
        type=ProxyErrorTypes.no_db_connection,
        param="test-param",
    )
    assert (
        PrismaDBExceptionHandler.is_database_connection_error(db_proxy_exception)
        == True
    )

    # Test with non-DB error
    regular_exception = Exception("Regular error")
    assert (
        PrismaDBExceptionHandler.is_database_connection_error(regular_exception)
        == False
    )


# Test should_allow_request_on_db_unavailable method
@patch(
    "litellm.proxy.proxy_server.general_settings",
    {"allow_requests_on_db_unavailable": True},
)
def test_should_allow_request_on_db_unavailable_true():
    assert PrismaDBExceptionHandler.should_allow_request_on_db_unavailable() == True


@patch(
    "litellm.proxy.proxy_server.general_settings",
    {"allow_requests_on_db_unavailable": False},
)
def test_should_allow_request_on_db_unavailable_false():
    assert PrismaDBExceptionHandler.should_allow_request_on_db_unavailable() == False


@patch(
    "litellm.proxy.proxy_server.general_settings",
    {"allow_requests_on_db_unavailable": True},
)
def test_handle_db_exception_with_connection_error():
    """
    Test that DB connection errors are handled gracefully when allow_requests_on_db_unavailable is True
    """
    db_error = ClientNotConnectedError()
    result = PrismaDBExceptionHandler.handle_db_exception(db_error)
    assert result is None


@patch(
    "litellm.proxy.proxy_server.general_settings",
    {"allow_requests_on_db_unavailable": False},
)
def test_handle_db_exception_raises_error():
    """
    Test that DB connection errors are raised when allow_requests_on_db_unavailable is False
    """
    db_error = ClientNotConnectedError()
    with pytest.raises(ClientNotConnectedError):
        PrismaDBExceptionHandler.handle_db_exception(db_error)


def test_handle_db_exception_with_non_db_error():
    """
    Test that non-DB errors are always raised regardless of allow_requests_on_db_unavailable setting
    """
    regular_error = litellm.BudgetExceededError(
        current_cost=10,
        max_budget=10,
    )
    with pytest.raises(litellm.BudgetExceededError):
        PrismaDBExceptionHandler.handle_db_exception(regular_error)

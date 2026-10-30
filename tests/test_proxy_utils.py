import pytest
from fastapi import HTTPException

from litellm.exceptions import BudgetExceededError
from litellm.proxy._types import ProxyException
from litellm.proxy.utils import _is_client_error_exception


def test_is_client_error_exception_budget_exceeded():
    # BudgetExceededError with default 422 status
    budget_error = BudgetExceededError(current_cost=10.0, max_budget=5.0)
    assert _is_client_error_exception(budget_error) is True

    # BudgetExceededError with explicit 4xx / custom status
    budget_error.status_code = 429
    assert _is_client_error_exception(budget_error) is True


def test_is_client_error_exception_http_exception():
    # HTTPException with 4xx codes
    assert _is_client_error_exception(HTTPException(status_code=400, detail="bad request")) is True
    assert _is_client_error_exception(HTTPException(status_code=401, detail="unauthorized")) is True
    assert _is_client_error_exception(HTTPException(status_code=422, detail="unprocessable")) is True
    assert _is_client_error_exception(HTTPException(status_code=429, detail="rate limit")) is True

    # HTTPException with 5xx codes
    assert _is_client_error_exception(HTTPException(status_code=500, detail="internal server error")) is False
    assert _is_client_error_exception(HTTPException(status_code=502, detail="bad gateway")) is False
    assert _is_client_error_exception(HTTPException(status_code=503, detail="service unavailable")) is False


def test_is_client_error_exception_proxy_exception():
    # ProxyException with 4xx codes
    pe_400 = ProxyException(message="bad request", type="invalid_request_error", param=None, code=400)
    assert _is_client_error_exception(pe_400) is True

    pe_422 = ProxyException(message="budget exceeded", type="budget_exceeded", param=None, code=422)
    assert _is_client_error_exception(pe_422) is True

    # ProxyException with 5xx codes
    pe_500 = ProxyException(message="internal server error", type="internal_server_error", param=None, code=500)
    assert _is_client_error_exception(pe_500) is False

    pe_503 = ProxyException(message="service unavailable", type="service_unavailable", param=None, code=503)
    assert _is_client_error_exception(pe_503) is False


def test_is_client_error_exception_generic_and_server_errors():
    # Generic Exception with status_code attribute
    class CustomClientError(Exception):
        status_code = 403

    class CustomServerError(Exception):
        status_code = 500

    assert _is_client_error_exception(CustomClientError("forbidden")) is True
    assert _is_client_error_exception(CustomServerError("server error")) is False

    # Standard 500 / unexpected Exception without 4xx status_code
    assert _is_client_error_exception(Exception("unexpected error")) is False
    assert _is_client_error_exception(ValueError("invalid value")) is False
    assert _is_client_error_exception(RuntimeError("runtime issue")) is False

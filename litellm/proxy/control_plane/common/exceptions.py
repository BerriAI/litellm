"""RFC 9457 problem documents for the control plane, same contract as /management/v1."""

from typing import Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from litellm.proxy.common_utils.validation_error_body import public_validation_errors
from litellm.proxy.list_api.common import (
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    problem_response,
    request_validation_problem,
)
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail

WWW_AUTHENTICATE: Final = "WWW-Authenticate"
BEARER_CHALLENGE: Final = 'Bearer realm="litellm-control-plane"'


def problem(slug: str, title: str, status: int, detail: str) -> ProblemDetail:
    return ProblemDetail(type=f"{PROBLEM_TYPE_BASE}{slug}", title=title, status=status, detail=detail)


def unauthorized(detail: str) -> ProblemDetail:
    return problem("unauthorized", "Unauthorized", 401, detail)


def forbidden(detail: str) -> ProblemDetail:
    return problem("forbidden", "Forbidden", 403, detail)


def not_found(detail: str) -> ProblemDetail:
    return problem("not-found", "Not Found", 404, detail)


def too_many_requests(detail: str) -> ProblemDetail:
    return problem("too-many-requests", "Too Many Requests", 429, detail)


class ControlPlaneException(ManagementProblem):
    """A problem raised from a dependency or handler, rendered by the app-level handler below."""


def _render(problem_detail: ProblemDetail) -> JSONResponse:
    response: Final = problem_response(problem_detail)
    if problem_detail.status == 401:
        response.headers[WWW_AUTHENTICATE] = BEARER_CHALLENGE
    return response


async def _handle_exception(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ManagementProblem):
        raise exc
    return _render(exc.problem)


async def _handle_validation(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        raise exc
    return _render(request_validation_problem(public_validation_errors(exc.errors())))


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ManagementProblem, _handle_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation)

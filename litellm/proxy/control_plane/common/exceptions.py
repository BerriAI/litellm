"""RFC 9457 error bodies for the control plane, same contract as /management/v1."""

from collections.abc import Mapping
from http import HTTPStatus
from types import MappingProxyType
from typing import Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from litellm.proxy.common_utils.validation_error_body import public_validation_errors
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE as ERROR_TYPE_BASE
from litellm.proxy.list_api.common import problem_response as error_response
from litellm.proxy.list_api.common import request_validation_problem as request_validation_error
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail as ErrorBody

WWW_AUTHENTICATE: Final = "WWW-Authenticate"
BEARER_CHALLENGE: Final = 'Bearer realm="litellm-control-plane"'
EMPTY_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})


def error_body(slug: str, title: str, status: int, detail: str) -> ErrorBody:
    return ErrorBody(type=f"{ERROR_TYPE_BASE}{slug}", title=title, status=status, detail=detail)


def unauthorized(detail: str) -> ErrorBody:
    return error_body("unauthorized", "Unauthorized", 401, detail)


def forbidden(detail: str) -> ErrorBody:
    return error_body("forbidden", "Forbidden", 403, detail)


def not_found(detail: str) -> ErrorBody:
    return error_body("not-found", "Not Found", 404, detail)


def too_many_requests(detail: str) -> ErrorBody:
    return error_body("too-many-requests", "Too Many Requests", 429, detail)


class ControlPlaneException(Exception):
    """An error raised from a dependency or handler, rendered by the app-level handler below."""

    def __init__(self, body: ErrorBody, *, headers: Mapping[str, str] = EMPTY_HEADERS) -> None:
        self.body = body
        self.headers = MappingProxyType(dict(headers))
        super().__init__(body.detail)


def _render(body: ErrorBody) -> JSONResponse:
    response: Final = error_response(body)
    if body.status == 401:
        response.headers[WWW_AUTHENTICATE] = BEARER_CHALLENGE
    return response


async def _handle_exception(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ControlPlaneException):
        raise exc
    response: Final = _render(exc.body)
    response.headers.update(exc.headers)
    return response


async def _handle_validation(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        raise exc
    return _render(request_validation_error(public_validation_errors(exc.errors())))


async def _handle_http_exception(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):
        raise exc
    phrase: Final = HTTPStatus(exc.status_code).phrase
    slug: Final = phrase.lower().replace("'", "").replace(" ", "-")
    body: Final = error_body(slug, phrase, exc.status_code, str(exc.detail))
    response: Final = error_response(body)
    response.headers.update(exc.headers or {})
    return response


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ControlPlaneException, _handle_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)

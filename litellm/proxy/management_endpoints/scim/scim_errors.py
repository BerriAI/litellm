from collections.abc import Callable, Coroutine, Mapping
from http import HTTPStatus
from types import MappingProxyType
from typing import Final, TypeAlias

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import TypeAdapter, ValidationError
from starlette.exceptions import HTTPException
from typing_extensions import assert_never

from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.validation_error_body import ValidationErrorDetail, public_validation_errors
from litellm.proxy.utils import handle_exception_on_proxy
from litellm.types.proxy.management_endpoints.scim_v2 import SCIMError, ScimErrorDetail, ScimType

SCIM_CONTENT_TYPE: Final = "application/scim+json"

_DETAIL_ADAPTER: Final[TypeAdapter[str | ScimErrorDetail]] = TypeAdapter(str | ScimErrorDetail)
_STATUS_PHRASES: Final = MappingProxyType({status.value: status.phrase for status in HTTPStatus})

_ScimRouteFailure: TypeAlias = HTTPException | ProxyException | RequestValidationError


def _scim_type(status_code: int) -> ScimType | None:
    return "uniqueness" if status_code == HTTPStatus.CONFLICT else None


class ScimError(Exception):
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Mapping[str, str] | None = None,
        traced_as: Exception | None = None,
    ) -> None:
        self.status_code: Final = status_code
        self.headers: Final = MappingProxyType(dict(headers or {}))
        self.traced_as: Final = traced_as
        self.error: Final = SCIMError(status=str(status_code), detail=detail, scimType=_scim_type(status_code))
        super().__init__(detail)


def _readable_detail(detail: object) -> str:
    try:
        parsed: Final = _DETAIL_ADAPTER.validate_python(detail)
    except ValidationError:
        return str(detail)
    return parsed if isinstance(parsed, str) else parsed["error"]


def _status_phrase(status_code: int) -> str:
    return _STATUS_PHRASES.get(status_code, str(status_code))


def _proxy_exception_detail(exc: ProxyException, status_code: int) -> object:
    origin: Final = exc.__context__
    if isinstance(origin, HTTPException) and str(origin.detail) == exc.message:
        return origin.detail
    if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return _status_phrase(status_code)
    return exc.message


def _proxy_exception_status_code(exc: ProxyException) -> int:
    return int(exc.code) if exc.code.isdecimal() else HTTPStatus.INTERNAL_SERVER_ERROR.value


def _validation_detail(errors: tuple[ValidationErrorDetail, ...]) -> str:
    return "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in errors)


def _scim_error(status_code: int, detail: object, headers: Mapping[str, str] | None, traced_as: Exception) -> ScimError:
    return ScimError(status_code, _readable_detail(detail) or _status_phrase(status_code), headers, traced_as)


def _scim_error_from_proxy_exception(exc: ProxyException) -> ScimError:
    status_code: Final = _proxy_exception_status_code(exc)
    return _scim_error(status_code, _proxy_exception_detail(exc, status_code), exc.headers, exc)


def _scim_validation_error(exc: RequestValidationError) -> ScimError:
    public_errors: Final = public_validation_errors(exc.errors())
    return ScimError(
        HTTPStatus.UNPROCESSABLE_ENTITY.value,
        _validation_detail(public_errors),
        traced_as=RequestValidationError(public_errors),
    )


def scim_error_from(exc: _ScimRouteFailure) -> ScimError:
    match exc:
        case HTTPException():
            return _scim_error(exc.status_code, exc.detail, exc.headers, exc)
        case ProxyException():
            return _scim_error_from_proxy_exception(exc)
        case RequestValidationError():
            return _scim_validation_error(exc)
        case _:
            assert_never(exc)


def _unexpected_scim_error(exc: Exception) -> ScimError:
    status_code: Final = _proxy_exception_status_code(handle_exception_on_proxy(exc))
    return ScimError(status_code, _status_phrase(status_code), traced_as=exc)


def scim_error_response(exc: ScimError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.error.model_dump(mode="json", exclude_none=True),
        headers=dict(exc.headers),
        media_type=SCIM_CONTENT_TYPE,
    )


class ScimRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[object, object, Response]]:
        handler: Final = super().get_route_handler()

        async def answer_errors_as_scim(request: Request) -> Response:
            try:
                return await handler(request)
            except (HTTPException, ProxyException, RequestValidationError) as exc:
                raise scim_error_from(exc) from exc
            except Exception as exc:
                raise _unexpected_scim_error(exc) from exc

        return answer_errors_as_scim

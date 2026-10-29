"""RFC 7644 section 3.12 SCIM Error responses for every error the /scim/v2 routes answer."""

from collections.abc import Callable, Coroutine, Mapping
from http import HTTPStatus
from types import MappingProxyType
from typing import Final, NamedTuple, TypeAlias

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import TypeAdapter, ValidationError
from starlette.exceptions import HTTPException
from typing_extensions import assert_never

from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.validation_error_body import public_validation_errors
from litellm.types.proxy.management_endpoints.scim_v2 import SCIMError, ScimErrorDetail, ScimType

SCIM_CONTENT_TYPE: Final = "application/scim+json"

_DETAIL_ADAPTER: Final[TypeAdapter[str | ScimErrorDetail]] = TypeAdapter(str | ScimErrorDetail)
_STATUS_PHRASES: Final = MappingProxyType({status.value: status.phrase for status in HTTPStatus})

_ScimRouteFailure: TypeAlias = HTTPException | ProxyException | RequestValidationError


class ScimError(Exception):
    """Raised to answer an RFC 7644 SCIM Error instead of the proxy's OpenAI error shape."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        scim_type: ScimType | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code: Final = status_code
        self.headers: Final = MappingProxyType(dict(headers or {}))
        self.error: Final = SCIMError(status=str(status_code), detail=detail, scimType=scim_type)
        super().__init__(detail)


class _Reason(NamedTuple):
    detail: str
    scim_type: ScimType | None


def _reason(detail: object) -> _Reason:
    try:
        parsed: Final = _DETAIL_ADAPTER.validate_python(detail)
    except ValidationError:
        return _Reason("", None)
    if isinstance(parsed, str):
        return _Reason(parsed, None)
    return _Reason(parsed["error"], parsed.get("scimType"))


def _detail_before_proxy_exception_stringified_it(exc: ProxyException) -> object:
    origin: Final = exc.__context__
    if isinstance(origin, HTTPException) and str(origin.detail) == exc.message:
        return origin.detail
    return exc.message


def _proxy_exception_status_code(exc: ProxyException) -> int:
    return int(exc.code) if exc.code.isdecimal() else HTTPStatus.INTERNAL_SERVER_ERROR.value


def _validation_detail(exc: RequestValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in public_validation_errors(exc.errors())
    )


def _scim_error(status_code: int, detail: object, headers: Mapping[str, str] | None) -> ScimError:
    reason: Final = _reason(detail)
    return ScimError(
        status_code,
        reason.detail or _STATUS_PHRASES.get(status_code, str(status_code)),
        reason.scim_type,
        headers,
    )


def scim_error_from(exc: _ScimRouteFailure) -> ScimError:
    match exc:
        case HTTPException():
            return _scim_error(exc.status_code, exc.detail, exc.headers)
        case ProxyException():
            return _scim_error(
                _proxy_exception_status_code(exc), _detail_before_proxy_exception_stringified_it(exc), exc.headers
            )
        case RequestValidationError():
            return ScimError(HTTPStatus.UNPROCESSABLE_ENTITY.value, _validation_detail(exc))
        case _:
            assert_never(exc)


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

        return answer_errors_as_scim

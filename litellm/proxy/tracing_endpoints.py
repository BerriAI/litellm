"""Authenticated OTLP ingestion and trace query routes."""

from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from functools import partial
from http.client import responses
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import JsonValue, TypeAdapter, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from typing_extensions import ReadOnly, TypedDict, assert_never

from litellm._logging import verbose_proxy_logger
from litellm.constants import OTLP_RETRY_AFTER_SECONDS, TRACE_READ_RETRY_AFTER_SECONDS
from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.authorization import AllRows, ReadScope, resolve_trace_read_scope
from litellm.proxy.auth.authorization_dependencies import LogTeamLookupDependency
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import is_otlp_trace_request
from litellm.proxy.tracing_runtime import provide_receiver, require_receiver
from litellm.rust_bridge.trace.errors import TraceChanged, TraceQueryError
from litellm.rust_bridge.trace.generated.models import TraceQueryHelp
from litellm.rust_bridge.trace.generated.requests import (
    TraceErrorPageRequest,
    TraceHistogramRequest,
    TraceInvalidParam,
    TraceListRequest,
    TraceMetadata,
    TraceNoQueryRequest,
    TraceProblem,
    TraceProblemCode,
    TraceQueryRequest,
    TraceSpanPageRequest,
    TraceSpansPage,
    TraceValuesRequest,
)
from litellm.rust_bridge.trace.generated.routes import OPERATIONS
from litellm.rust_bridge.trace.generated.types import (
    AllQueryScope,
    OwnedQueryScope,
    QueryScope,
    RunField,
    RunOrder,
    RunValues,
    SpanDetail,
    SpanErrorPage,
    TraceHistogram,
    TracePage,
)
from litellm.rust_bridge.trace.queries import TraceSQLResponse
from litellm.rust_bridge.trace.storage import ClickHouseStorage, Tenant
from litellm.tracing import TraceReceiver, TracingPayloadTooLargeError
from litellm.tracing.otlp_http import InvalidOTLPPayloadError, encode_otlp_response

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class ValidationIssue(TypedDict):
    loc: ReadOnly[tuple[str | int, ...]]
    msg: ReadOnly[str]


_VALIDATION_ISSUES: Final = TypeAdapter(tuple[ValidationIssue, ...])


def problem(
    status: int,
    code: TraceProblemCode,
    detail: str,
    database_code: int | None = None,
    errors: tuple[TraceInvalidParam, ...] = (),
) -> TraceProblem:
    return TraceProblem(
        type="about:blank",
        title=responses.get(status, "Trace request failed"),
        status=status,
        code=code,
        detail=detail,
        database_code=database_code,
        errors=errors,
    )


def problem_exception(
    status: int, code: TraceProblemCode, detail: str, database_code: int | None = None
) -> HTTPException:
    return HTTPException(status_code=status, detail=problem(status, code, detail, database_code).model_dump())


def status_code(status: int) -> TraceProblemCode:
    match status:
        case 401:
            return "unauthorized"
        case 403:
            return "forbidden"
        case 404:
            return "not_found"
        case 413:
            return "too_large"
        case 500:
            return "internal_error"
        case 501 | 503:
            return "unavailable"
        case _:
            return "invalid_request"


def problem_response(body: TraceProblem, headers: Mapping[str, str] | None = None) -> Response:
    return JSONResponse(
        body.model_dump(mode="json", exclude_none=True),
        status_code=body.status,
        media_type="application/problem+json",
        headers=headers,
    )


def http_problem(error: StarletteHTTPException) -> Response:
    try:
        body: Final = TraceProblem.model_validate(error.detail)
        return problem_response(body, error.headers)
    except ValidationError:
        return problem_response(
            problem(error.status_code, status_code(error.status_code), str(error.detail)), error.headers
        )


class TraceRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler: Final = super().get_route_handler()

        async def handle(request: Request) -> Response:
            if is_otlp_trace_request(request):
                return await handler(request)
            try:
                return await handler(request)
            except StarletteHTTPException as error:
                return http_problem(error)
            except ProxyException as error:
                status: Final = int(error.code) if error.code.isdecimal() else 500
                return problem_response(problem(status, status_code(status), error.message), error.headers)
            except RequestValidationError as error:
                issues: Final = _VALIDATION_ISSUES.validate_python(error.errors())
                details: Final = tuple(
                    TraceInvalidParam(
                        location="/".join(str(part) for part in issue["loc"]),
                        reason=str(issue["msg"]),
                    )
                    for issue in issues
                )
                return problem_response(problem(422, "invalid_request", "Request validation failed", errors=details))
            except Exception as error:
                verbose_proxy_logger.exception("Trace request failed: %s", error)
                return problem_response(problem(500, "internal_error", "Trace request failed"))

        return handle


router = APIRouter(tags=["agent tracing"], route_class=TraceRoute)


def merge_trace_openapi(schema: object) -> Mapping[str, JsonValue]:
    document: Final = _JSON_OBJECT.validate_python(schema)
    generated: Final = _JSON_OBJECT.validate_json(
        (Path(__file__).parents[1] / "rust_bridge" / "trace" / "generated" / "openapi.json").read_text()
    )
    components: Final = _JSON_OBJECT.validate_python(document.get("components", {}))
    generated_components: Final = _JSON_OBJECT.validate_python(generated.get("components", {}))
    return {
        **document,
        "openapi": generated["openapi"],
        "paths": {
            **_JSON_OBJECT.validate_python(document.get("paths", {})),
            **_JSON_OBJECT.validate_python(generated["paths"]),
        },
        "components": {
            **components,
            **generated_components,
            "schemas": {
                **_JSON_OBJECT.validate_python(components.get("schemas", {})),
                **_JSON_OBJECT.validate_python(generated_components.get("schemas", {})),
            },
            "securitySchemes": {
                **_JSON_OBJECT.validate_python(components.get("securitySchemes", {})),
                **_JSON_OBJECT.validate_python(generated_components.get("securitySchemes", {})),
            },
        },
    }


@dataclass(frozen=True, slots=True)
class TraceAccessContext:
    receiver: TraceReceiver | None
    read_scope: ReadScope | None
    write_tenant: Tenant | None

    def reader(self) -> tuple[TraceReceiver, QueryScope]:
        tracing: Final = require_receiver(self.receiver)
        if self.read_scope is None:
            raise problem_exception(403, "forbidden", "Not allowed to view agent traces")
        return tracing, read_access(self.read_scope)

    def writer(self) -> tuple[TraceReceiver, Tenant]:
        if self.write_tenant is None:
            raise HTTPException(status_code=403, detail="Not allowed to ingest agent traces")
        return require_receiver(self.receiver), self.write_tenant


async def provide_trace_access(
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    tracing: Annotated[TraceReceiver | None, Depends(provide_receiver)],
    log_team_lookup: LogTeamLookupDependency,
) -> TraceAccessContext:
    tenant: Final = Tenant(
        team_id=auth.team_id or "", api_key_hash=auth.token or "", org_id=auth.org_id or "", user_id=auth.user_id or ""
    )
    write_tenant: Final = None if auth.user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY else tenant
    read_scope: Final = await resolve_trace_read_scope(auth, partial(log_team_lookup, auth))
    return TraceAccessContext(tracing, read_scope, write_tenant)


def otlp_error_response(
    request: Request, status_code: int, headers: Mapping[str, str] | None = None
) -> Response | None:
    if not is_otlp_trace_request(request):
        return None
    body, media_type = encode_otlp_response(
        request.headers.get("content-type"), responses.get(status_code, "Trace request failed")
    )
    return Response(content=body, status_code=status_code, media_type=media_type, headers=headers)


def _otlp_error(content_type: str | None, status: int, message: str, retry: bool = False) -> Response:
    body, media_type = encode_otlp_response(content_type, message)
    return Response(
        content=body,
        status_code=status,
        media_type=media_type,
        headers=MappingProxyType({"Retry-After": str(OTLP_RETRY_AFTER_SECONDS)}) if retry else None,
    )


@router.api_route(
    OPERATIONS["trace_ingest"].path,
    methods=[OPERATIONS["trace_ingest"].method],
    operation_id=OPERATIONS["trace_ingest"].operation_id,
    include_in_schema=False,
)
async def ingest_otlp_traces(
    request: Request,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
) -> Response:
    content_type: Final = request.headers.get("content-type")
    try:
        tracing, tenant = context.writer()
        await tracing.ingest(
            body=request.stream(),
            content_type=content_type,
            content_encoding=request.headers.get("content-encoding"),
            tenant=tenant,
        )
    except TracingPayloadTooLargeError as error:
        return _otlp_error(content_type, 413, str(error))
    except InvalidOTLPPayloadError as error:
        return _otlp_error(content_type, 400, str(error))
    except RuntimeError:
        return _otlp_error(content_type, 503, "Trace ingestion is temporarily unavailable", retry=True)
    except HTTPException as error:
        return _otlp_error(content_type, error.status_code, str(error.detail))
    body, media_type = encode_otlp_response(content_type)
    return Response(content=body, media_type=media_type)


def read_failure(error: TraceChanged | ValueError | OverflowError | RuntimeError) -> HTTPException:
    match error:
        case TraceChanged():
            return problem_exception(409, "trace_changed", str(error))
        case ValueError():
            return problem_exception(400, "invalid_request", str(error))
        case OverflowError():
            return problem_exception(413, "too_large", "Trace is too large for this view. Use a filtered trace query.")
        case RuntimeError():
            verbose_proxy_logger.warning("Trace read unavailable: %s", error)
            return HTTPException(
                status_code=503,
                detail=problem(
                    503, "unavailable", "Traces are temporarily unavailable. Please try again."
                ).model_dump(),
                headers={"Retry-After": str(TRACE_READ_RETRY_AFTER_SECONDS)},
            )
        case _:
            return assert_never(error)


@router.api_route(
    OPERATIONS["trace_list"].path,
    methods=[OPERATIONS["trace_list"].method],
    operation_id=OPERATIONS["trace_list"].operation_id,
    response_model=TracePage,
)
async def list_agent_traces(
    params: Annotated[TraceListRequest, Query()],
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
) -> TracePage:
    order: Final = RunOrder(key=params.sort_by, descending=params.sort_dir == "desc")
    try:
        tracing, scope = context.reader()
        return await tracing.list_traces(
            scope=scope,
            start_ms=params.start_ms,
            end_ms=params.end_ms,
            q=params.q,
            cursor=params.cursor,
            order=order,
            page_size=params.page_size,
            as_of_ms=params.as_of_ms,
        )
    except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error


@router.api_route(
    OPERATIONS["trace_histogram"].path,
    methods=[OPERATIONS["trace_histogram"].method],
    operation_id=OPERATIONS["trace_histogram"].operation_id,
    response_model=TraceHistogram,
)
async def agent_trace_histogram(
    params: Annotated[TraceHistogramRequest, Query()],
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
) -> TraceHistogram:
    try:
        tracing, scope = context.reader()
        return await tracing.trace_histogram(
            scope, params.start_ms, params.end_ms, params.q, params.buckets, params.as_of_ms
        )
    except (ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error


@router.api_route(
    OPERATIONS["trace_values"].path,
    methods=[OPERATIONS["trace_values"].method],
    operation_id=OPERATIONS["trace_values"].operation_id,
    response_model=RunValues,
)
async def agent_trace_values(
    field: RunField,
    params: Annotated[TraceValuesRequest, Query()],
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
) -> RunValues:
    try:
        tracing, scope = context.reader()
        return await tracing.run_values(
            scope, params.start_ms, params.end_ms, params.q, field, params.contains, params.limit, params.as_of_ms
        )
    except (ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error


@dataclass(frozen=True, slots=True)
class TraceQueryAccess:
    storage: ClickHouseStorage
    scope: ReadScope
    secret: str


def provide_trace_query_secret() -> str:
    from litellm.proxy.proxy_server import master_key

    if not master_key:
        raise problem_exception(503, "query_unavailable", "Trace SQL queries require a configured proxy master key")
    return master_key


def read_access(scope: ReadScope) -> QueryScope:
    if isinstance(scope, AllRows):
        return AllQueryScope(kind="all")
    return OwnedQueryScope(kind="owned", user_id=scope.user_id or "", team_ids=scope.team_ids)


async def provide_trace_query_access(
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    tracing: Annotated[TraceReceiver | None, Depends(provide_receiver)],
    secret: Annotated[str, Depends(provide_trace_query_secret)],
    log_team_lookup: LogTeamLookupDependency,
) -> TraceQueryAccess:
    storage: Final = require_receiver(tracing).storage
    scope: Final = await resolve_trace_read_scope(auth, partial(log_team_lookup, auth))
    if scope is None:
        raise problem_exception(403, "forbidden", "Not allowed to view logs")
    return TraceQueryAccess(storage, scope, secret)


def sql_failure_status(error: TraceQueryError) -> tuple[int, TraceProblemCode]:
    match error.kind:
        case "rejected":
            return 400, "query_rejected"
        case "limited":
            return 422, "query_limit_exceeded"
        case "unavailable":
            return 503, "query_unavailable"


@router.api_route(
    OPERATIONS["trace_query"].path,
    methods=[OPERATIONS["trace_query"].method],
    operation_id=OPERATIONS["trace_query"].operation_id,
    response_model=TraceSQLResponse,
    response_model_exclude_unset=True,
)
async def query_agent_traces(
    _params: Annotated[TraceNoQueryRequest, Query()],
    body: TraceQueryRequest,
    access: Annotated[TraceQueryAccess, Depends(provide_trace_query_access)],
) -> TraceSQLResponse:
    try:
        return await access.storage.query_sql(body.sql, read_access(access.scope), access.secret, body.params)
    except TraceQueryError as error:
        status, code = sql_failure_status(error)
        raise problem_exception(status, code, error.message, error.database_code) from error
    except ValueError as error:
        raise problem_exception(400, "query_rejected", str(error)) from error
    except RuntimeError as error:
        verbose_proxy_logger.warning("Trace SQL query unavailable: %s", error)
        raise problem_exception(503, "query_unavailable", "Trace SQL is temporarily unavailable") from error


@router.api_route(
    OPERATIONS["trace_query_help"].path,
    methods=[OPERATIONS["trace_query_help"].method],
    operation_id=OPERATIONS["trace_query_help"].operation_id,
    response_model=TraceQueryHelp,
    response_model_exclude_unset=True,
)
async def help_agent_trace_queries(
    _params: Annotated[TraceNoQueryRequest, Query()],
    access: Annotated[TraceQueryAccess, Depends(provide_trace_query_access)],
) -> TraceQueryHelp:
    try:
        return await access.storage.query_help(read_access(access.scope), access.secret)
    except RuntimeError as error:
        verbose_proxy_logger.warning("Trace query help unavailable: %s", error)
        raise problem_exception(503, "query_unavailable", "Trace query help is temporarily unavailable") from error


@router.api_route(
    OPERATIONS["trace_get"].path,
    methods=[OPERATIONS["trace_get"].method],
    operation_id=OPERATIONS["trace_get"].operation_id,
    response_model=TraceMetadata,
)
async def get_agent_trace(
    _params: Annotated[TraceNoQueryRequest, Query()],
    id: str,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
) -> TraceMetadata:
    try:
        tracing, scope = context.reader()
        trace: Final = await tracing.get_trace_metadata(scope, id)
    except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error
    if trace is None:
        raise problem_exception(404, "not_found", "Trace not found")
    return trace


@router.api_route(
    OPERATIONS["trace_spans"].path,
    methods=[OPERATIONS["trace_spans"].method],
    operation_id=OPERATIONS["trace_spans"].operation_id,
    response_model=TraceSpansPage,
)
async def get_agent_trace_spans(
    id: str,
    params: Annotated[TraceSpanPageRequest, Query()],
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
) -> TraceSpansPage:
    try:
        tracing, scope = context.reader()
        page: Final = await tracing.get_trace_spans(scope, id, params.cursor, params.page_size)
    except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error
    if page is None:
        raise problem_exception(404, "not_found", "Trace not found")
    return page


@router.api_route(
    OPERATIONS["trace_span"].path,
    methods=[OPERATIONS["trace_span"].method],
    operation_id=OPERATIONS["trace_span"].operation_id,
    response_model=SpanDetail,
)
async def get_agent_trace_span(
    _params: Annotated[TraceNoQueryRequest, Query()],
    id: str,
    span_id: str,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
) -> SpanDetail:
    try:
        tracing, scope = context.reader()
        span: Final = await tracing.get_span_by_id(scope, id, span_id)
    except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error
    if span is None:
        raise problem_exception(404, "not_found", "Span not found")
    return span


@router.api_route(
    OPERATIONS["trace_error"].path,
    methods=[OPERATIONS["trace_error"].method],
    operation_id=OPERATIONS["trace_error"].operation_id,
    response_model=SpanErrorPage,
)
async def get_agent_trace_span_error(
    id: str,
    span_id: str,
    params: Annotated[TraceErrorPageRequest, Query()],
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
) -> SpanErrorPage:
    try:
        tracing, scope = context.reader()
        page: Final = await tracing.get_span_error_by_id(scope, id, span_id, params.cursor)
    except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error
    if page is None:
        raise problem_exception(404, "not_found", "Span diagnostic not found or no longer available")
    return page

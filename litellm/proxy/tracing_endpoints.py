"""
Agent tracing endpoints. Thin wrappers over `TraceReceiver`: auth -> tenant/scope -> one call.

POST /v1/traces                              OTLP/HTTP trace export (protobuf or JSON)
GET  /v1/traces                              TracePage
GET  /v1/traces/{trace_id}                   Trace
GET  /v1/traces/{trace_id}/spans/{span_id}   SpanDetail
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from http.client import responses
from types import MappingProxyType
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.constants import OTLP_RETRY_AFTER_SECONDS, TRACE_READ_RETRY_AFTER_SECONDS
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.authorization import AllRows, ReadScope, resolve_trace_read_scope
from litellm.proxy.auth.authorization_dependencies import LogTeamLookupDependency
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import is_otlp_trace_request
from litellm.proxy.tracing_runtime import provide_receiver, require_receiver
from litellm.rust_bridge.trace.errors import TraceChanged
from litellm.rust_bridge.trace.generated.models import TraceQueryHelp
from litellm.rust_bridge.trace.generated.types import (
    AllQueryScope,
    OwnedQueryScope,
    QueryScope,
    SpanDetail,
    SpanErrorPage,
    Trace,
    TracePage,
    TraceScope,
)
from litellm.rust_bridge.trace.queries import TraceSQLResponse
from litellm.rust_bridge.trace.storage import ClickHouseStorage, Tenant
from litellm.tracing import TraceReceiver, TracingPayloadTooLargeError
from litellm.tracing.otlp_http import InvalidOTLPPayloadError, encode_otlp_response

router = APIRouter(tags=["agent tracing"])

MS_PER_DAY: Final = 24 * 60 * 60 * 1000


@dataclass(frozen=True, slots=True)
class TraceAccessContext:
    receiver: TraceReceiver | None
    read_scope: ReadScope | None
    write_tenant: Tenant | None

    def reader(self) -> tuple[TraceReceiver, TraceScope]:
        tracing: Final = require_receiver(self.receiver)
        if self.read_scope is None:
            raise HTTPException(status_code=403, detail="Not allowed to view agent traces")
        return tracing, _trace_scope(self.read_scope)

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


def _trace_scope(scope: ReadScope) -> TraceScope:
    if isinstance(scope, AllRows):
        return TraceScope(all_teams=1, user_id="", team_ids=())
    return TraceScope(
        all_teams=0,
        user_id=scope.user_id or "",
        team_ids=scope.team_ids,
    )


def otlp_error_response(
    request: Request, status_code: int, headers: Mapping[str, str] | None = None
) -> Response | None:
    if not is_otlp_trace_request(request):
        return None
    body, media_type = encode_otlp_response(
        request.headers.get("content-type"), responses.get(status_code, "Trace request failed")
    )
    return Response(content=body, status_code=status_code, media_type=media_type, headers=headers)


def _otlp_error(content_type: str | None, status_code: int, message: str, retry: bool = False) -> Response:
    body, media_type = encode_otlp_response(content_type, message)
    return Response(
        content=body,
        status_code=status_code,
        media_type=media_type,
        headers=MappingProxyType({"Retry-After": str(OTLP_RETRY_AFTER_SECONDS)}) if retry else None,
    )


@router.post("/v1/traces", include_in_schema=False)
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
    except TracingPayloadTooLargeError as e:
        return _otlp_error(content_type, 413, str(e))
    except InvalidOTLPPayloadError as error:
        return _otlp_error(content_type, 400, str(error))
    except RuntimeError:
        return _otlp_error(content_type, 503, "Trace ingestion is temporarily unavailable", retry=True)
    except HTTPException as error:
        return _otlp_error(content_type, error.status_code, str(error.detail))
    body, media_type = encode_otlp_response(content_type)
    return Response(content=body, media_type=media_type)


class TraceReadFailure(BaseModel):
    """The body of every failed trace read. Clients branch on `code`, never on `message`."""

    model_config = ConfigDict(frozen=True)

    code: Literal["invalid_request", "trace_changed", "too_large", "unavailable"]
    message: str


def read_failure(error: TraceChanged | ValueError | OverflowError | RuntimeError) -> HTTPException:
    """One status per failure kind, so a client can tell a bad cursor (400, fix the request) from a
    traversal it must restart (409), a result it cannot page through (413), and an outage it should
    retry after `Retry-After` (503)."""
    match error:
        case TraceChanged():
            return HTTPException(
                status_code=409,
                detail=TraceReadFailure(code="trace_changed", message=str(error)).model_dump(),
            )
        case ValueError():
            return HTTPException(
                status_code=400, detail=TraceReadFailure(code="invalid_request", message=str(error)).model_dump()
            )
        case OverflowError():
            return HTTPException(
                status_code=413,
                detail=TraceReadFailure(
                    code="too_large", message="Trace is too large for this view. Use a filtered trace query."
                ).model_dump(),
            )
        case RuntimeError():
            verbose_proxy_logger.warning("Trace read unavailable: %s", error)
            return HTTPException(
                status_code=503,
                detail=TraceReadFailure(
                    code="unavailable", message="Traces are temporarily unavailable. Please try again."
                ).model_dump(),
                headers={"Retry-After": str(TRACE_READ_RETRY_AFTER_SECONDS)},
            )
        case _:
            return assert_never(error)


@router.get("/v1/traces", response_model=TracePage)
async def list_agent_traces(
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    start_ms: Annotated[int | None, Query(description="Window start, unix ms. Default: 24h ago")] = None,
    end_ms: Annotated[int | None, Query(description="Window end, unix ms. Default: now")] = None,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
) -> TracePage:
    now_ms: Final = int(time.time() * 1000)
    try:
        tracing, scope = context.reader()
        return await tracing.list_traces(
            scope=scope,
            start_ms=start_ms if start_ms is not None else now_ms - MS_PER_DAY,
            end_ms=end_ms if end_ms is not None else now_ms,
            cursor=cursor,
        )
    except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error


class TraceQueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sql: str


@dataclass(frozen=True, slots=True)
class TraceQueryAccess:
    storage: ClickHouseStorage
    scope: ReadScope
    secret: str


def provide_trace_query_secret() -> str:
    from litellm.proxy.proxy_server import master_key

    if not master_key:
        raise HTTPException(status_code=503, detail="Trace SQL queries require a configured proxy master key")
    return master_key


def trace_query_scope(scope: ReadScope) -> QueryScope:
    if isinstance(scope, AllRows):
        return AllQueryScope(kind="all")
    return OwnedQueryScope(
        kind="owned",
        user_id=scope.user_id or "",
        team_ids=scope.team_ids,
    )


async def provide_trace_query_access(
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    tracing: Annotated[TraceReceiver | None, Depends(provide_receiver)],
    secret: Annotated[str, Depends(provide_trace_query_secret)],
    log_team_lookup: LogTeamLookupDependency,
) -> TraceQueryAccess:
    storage: Final = require_receiver(tracing).storage
    scope: Final = await resolve_trace_read_scope(auth, partial(log_team_lookup, auth))
    if scope is None:
        raise HTTPException(status_code=403, detail="Not allowed to view logs")
    return TraceQueryAccess(storage, scope, secret)


@router.post("/v1/traces/query", response_model=TraceSQLResponse, response_model_exclude_unset=True)
async def query_agent_traces(
    body: TraceQueryRequest,
    access: Annotated[TraceQueryAccess, Depends(provide_trace_query_access)],
) -> TraceSQLResponse:
    try:
        return await access.storage.query_sql(body.sql, trace_query_scope(access.scope), access.secret)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        verbose_proxy_logger.warning("Trace SQL query unavailable: %s", error)
        raise HTTPException(status_code=503, detail="Trace SQL query failed or exceeded reader limits") from error


@router.get("/v1/traces/query/help", response_model=TraceQueryHelp, response_model_exclude_unset=True)
async def help_agent_trace_queries(
    access: Annotated[TraceQueryAccess, Depends(provide_trace_query_access)],
) -> TraceQueryHelp:
    try:
        return await access.storage.query_help(trace_query_scope(access.scope), access.secret)
    except RuntimeError as error:
        verbose_proxy_logger.warning("Trace query help unavailable: %s", error)
        raise HTTPException(status_code=503, detail="Trace query help is temporarily unavailable") from error


@router.get("/v1/traces/{trace_id}", response_model=Trace)
async def get_agent_trace(
    trace_id: str,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    trace_ref: Annotated[str, Query()] = "",
    cursor: Annotated[str | None, Query(max_length=512)] = None,
    page_size: Annotated[int | None, Query(ge=1, le=500)] = None,
) -> Trace:
    tracing, scope = context.reader()
    try:
        trace: Final = await tracing.get_trace(trace_id, scope, trace_ref, cursor, page_size)
    except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace


@router.get("/v1/traces/{trace_id}/spans/{span_id}", response_model=SpanDetail)
async def get_agent_trace_span(
    trace_id: str,
    span_id: str,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    trace_ref: Annotated[str, Query()] = "",
) -> SpanDetail:
    tracing, scope = context.reader()
    try:
        span: Final = await tracing.get_span(trace_id, span_id, scope, trace_ref)
    except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error
    if span is None:
        raise HTTPException(status_code=404, detail=f"Span {span_id} not found")
    return span


@router.get("/v1/traces/{trace_id}/spans/{span_id}/error", response_model=SpanErrorPage)
async def get_agent_trace_span_error(
    trace_id: str,
    span_id: str,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    trace_ref: Annotated[str, Query()] = "",
    cursor: Annotated[str | None, Query(max_length=512)] = None,
) -> SpanErrorPage:
    try:
        tracing, scope = context.reader()
        page: Final = await tracing.get_span_error(trace_id, span_id, scope, trace_ref, cursor)
    except (TraceChanged, ValueError, OverflowError, RuntimeError) as error:
        raise read_failure(error) from error
    if page is None:
        raise HTTPException(status_code=404, detail="Span diagnostic not found or no longer available")
    return page

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
from http.client import responses
from types import MappingProxyType
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from litellm.constants import OTLP_RETRY_AFTER_SECONDS
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import is_otlp_trace_request
from litellm.proxy.spend_tracking.log_visibility import log_visibility
from litellm.proxy.tracing_runtime import provide_receiver, require_receiver
from litellm.tracing import (
    Tenant,
    TraceReceiver,
    TracingPayloadTooLargeError,
)
from litellm.tracing.decode import InvalidOTLPPayloadError, encode_otlp_response
from litellm.tracing.store import AmbiguousTraceError
from litellm.tracing.types import SpanDetail, SpanErrorPage, Trace, TracePage, TraceScope

router = APIRouter(tags=["agent tracing"])

MS_PER_DAY: Final = 24 * 60 * 60 * 1000


@dataclass(frozen=True, slots=True)
class TraceAccessContext:
    receiver: TraceReceiver | None
    auth: UserAPIKeyAuth
    write_tenant: Tenant | None

    async def reader(self) -> tuple[TraceReceiver, TraceScope]:
        tracing: Final = require_receiver(self.receiver)
        return tracing, await read_scope_for(self.auth)

    def writer(self) -> tuple[TraceReceiver, Tenant]:
        if self.write_tenant is None:
            raise HTTPException(status_code=403, detail="Not allowed to ingest agent traces")
        return require_receiver(self.receiver), self.write_tenant


def tenant_for(auth: UserAPIKeyAuth) -> Tenant:
    return Tenant(
        team_id=auth.team_id or "",
        api_key_hash=auth.token or "",
        org_id=auth.org_id or "",
        user_id=auth.user_id or "",
    )


async def read_scope_for(auth: UserAPIKeyAuth) -> TraceScope:
    visibility: Final = await log_visibility(auth)
    return TraceScope(
        all_teams=int(visibility.all_teams),
        user_id=visibility.user_id,
        team_ids=visibility.team_ids,
        api_key_hash=visibility.api_key_hash,
    )


async def provide_trace_access(
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    tracing: Annotated[TraceReceiver | None, Depends(provide_receiver)],
) -> TraceAccessContext:
    tenant: Final = tenant_for(auth)
    write_tenant: Final = None if auth.user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY else tenant
    return TraceAccessContext(receiver=tracing, auth=auth, write_tenant=write_tenant)


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


@router.get("/v1/traces", response_model=TracePage)
async def list_agent_traces(
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    start_ms: Annotated[int | None, Query(description="Window start, unix ms. Default: 24h ago")] = None,
    end_ms: Annotated[int | None, Query(description="Window end, unix ms. Default: now")] = None,
    cursor: Annotated[str | None, Query()] = None,
) -> TracePage:
    now_ms: Final = int(time.time() * 1000)
    try:
        tracing, scope = await context.reader()
        return await tracing.list_traces(
            scope=scope,
            start_ms=start_ms if start_ms is not None else now_ms - MS_PER_DAY,
            end_ms=end_ms if end_ms is not None else now_ms,
            cursor=cursor,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/v1/traces/{trace_id}", response_model=Trace)
async def get_agent_trace(
    trace_id: str,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    trace_ref: Annotated[str, Query()] = "",
) -> Trace:
    try:
        tracing, scope = await context.reader()
        trace: Final = await tracing.get_trace(trace_id, scope, trace_ref)
    except AmbiguousTraceError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
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
    try:
        tracing, scope = await context.reader()
        span: Final = await tracing.get_span(trace_id, span_id, scope, trace_ref)
    except AmbiguousTraceError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
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
        tracing, scope = await context.reader()
        page: Final = await tracing.get_span_error(trace_id, span_id, scope, trace_ref, cursor)
    except AmbiguousTraceError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if page is None:
        raise HTTPException(status_code=404, detail="Span diagnostic not found or no longer available")
    return page

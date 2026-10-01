"""
Agent tracing endpoints. Thin wrappers over `TraceReceiver`: auth -> tenant/scope -> one call.

POST /v1/traces                              OTLP/HTTP trace export (protobuf or JSON)
GET  /v1/traces                              TracePage
GET  /v1/traces/{trace_id}                   Trace
GET  /v1/traces/{trace_id}/spans/{span_id}   SpanDetail
"""

import time
from dataclasses import dataclass
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from litellm.constants import OTLP_MAX_BODY_BYTES, OTLP_RETRY_AFTER_SECONDS
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.tracing_runtime import provide_receiver, require_receiver
from litellm.tracing import (
    Tenant,
    TraceReceiver,
    TracingPayloadTooLargeError,
)
from litellm.tracing.decode import InvalidOTLPPayloadError, encode_otlp_response
from litellm.tracing.types import SpanDetail, Trace, TracePage, TraceScope

router = APIRouter(tags=["agent tracing"])

MS_PER_DAY: Final = 24 * 60 * 60 * 1000


@dataclass(frozen=True, slots=True)
class TraceAccessContext:
    receiver: TraceReceiver | None
    read_scope: TraceScope | None
    write_tenant: Tenant | None

    def reader(self) -> tuple[TraceReceiver, TraceScope]:
        tracing: Final = require_receiver(self.receiver)
        if self.read_scope is None:
            raise HTTPException(status_code=403, detail="Not allowed to view agent traces")
        return tracing, self.read_scope

    def writer(self) -> tuple[TraceReceiver, Tenant]:
        if self.write_tenant is None:
            raise HTTPException(status_code=403, detail="Not allowed to ingest agent traces")
        return require_receiver(self.receiver), self.write_tenant


async def provide_trace_access(
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    tracing: Annotated[TraceReceiver | None, Depends(provide_receiver)],
) -> TraceAccessContext:
    tenant: Final = Tenant(team_id=auth.team_id or "", api_key_hash=auth.token or "", org_id=auth.org_id or "")
    match auth.user_role:
        case LitellmUserRoles.PROXY_ADMIN:
            return TraceAccessContext(tracing, TraceScope(team_ids=(), api_key_hash=""), tenant)
        case LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY:
            return TraceAccessContext(tracing, TraceScope(team_ids=(), api_key_hash=""), None)
        case _ if auth.team_id:
            return TraceAccessContext(tracing, TraceScope(team_ids=(auth.team_id,), api_key_hash=""), tenant)
        case _ if auth.token:
            return TraceAccessContext(tracing, TraceScope(team_ids=("",), api_key_hash=auth.token), tenant)
        case _:
            return TraceAccessContext(tracing, None, tenant)


async def _read_otlp_body(request: Request) -> bytes:
    body: Final = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > OTLP_MAX_BODY_BYTES:
            raise TracingPayloadTooLargeError(f"OTLP body exceeds {OTLP_MAX_BODY_BYTES} bytes")
        body.extend(chunk)
    return bytes(body)


@router.post("/v1/traces", include_in_schema=False)
async def ingest_otlp_traces(
    request: Request,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
) -> Response:
    tracing, tenant = context.writer()
    content_type: Final = request.headers.get("content-type")
    try:
        await tracing.ingest(
            body=await _read_otlp_body(request),
            content_type=content_type,
            content_encoding=request.headers.get("content-encoding"),
            tenant=tenant,
        )
    except TracingPayloadTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except InvalidOTLPPayloadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            headers={"Retry-After": str(OTLP_RETRY_AFTER_SECONDS)},
        )
    body, media_type = encode_otlp_response(content_type)
    return Response(content=body, media_type=media_type)


@router.get("/v1/traces", response_model=None)
async def list_agent_traces(
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    start_ms: Annotated[int | None, Query(description="Window start, unix ms. Default: 24h ago")] = None,
    end_ms: Annotated[int | None, Query(description="Window end, unix ms. Default: now")] = None,
    cursor: Annotated[str | None, Query()] = None,
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
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/v1/traces/{trace_id}", response_model=None)
async def get_agent_trace(
    trace_id: str,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    trace_ref: Annotated[str, Query()] = "",
) -> Trace:
    tracing, scope = context.reader()
    trace: Final = await tracing.get_trace(trace_id, scope, trace_ref)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace


@router.get("/v1/traces/{trace_id}/spans/{span_id}", response_model=None)
async def get_agent_trace_span(
    trace_id: str,
    span_id: str,
    context: Annotated[TraceAccessContext, Depends(provide_trace_access)],
    trace_ref: Annotated[str, Query()] = "",
) -> SpanDetail:
    tracing, scope = context.reader()
    span: Final = await tracing.get_span(trace_id, span_id, scope, trace_ref)
    if span is None:
        raise HTTPException(status_code=404, detail=f"Span {span_id} not found")
    return span

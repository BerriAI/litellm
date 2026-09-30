from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import Final, Protocol, cast

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.rust_bridge.loader import get_native_bridge

router: Final = APIRouter(prefix="/v1/traces", tags=["Traces"])


class NativeTraces(Protocol):
    def list_traces(
        self,
        clickhouse_url: str,
        start_ms: int,
        end_ms: int,
        service: str | None,
        status: str | None,
        search: str | None,
        cursor: str | None,
        limit: int,
    ) -> Awaitable[object]: ...

    def get_trace(self, clickhouse_url: str, trace_id: str) -> Awaitable[object]: ...

    def get_span(self, clickhouse_url: str, trace_id: str, span_id: str) -> Awaitable[object]: ...


def _connection(user_api_key_dict: UserAPIKeyAuth) -> tuple[str, NativeTraces]:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Trace queries require proxy admin access")

    from litellm.proxy.proxy_server import general_settings_view

    clickhouse_url: Final = general_settings_view().get("clickhouse_url")
    if not isinstance(clickhouse_url, str) or not clickhouse_url:
        raise HTTPException(status_code=503, detail="ClickHouse trace queries are not configured")

    native: Final = get_native_bridge()
    if native is None or not all(hasattr(native, name) for name in ("list_traces", "get_trace", "get_span")):
        raise HTTPException(status_code=503, detail="Rust trace query bridge is unavailable")
    return clickhouse_url, cast(NativeTraces, native)


async def _result(awaitable: Awaitable[object]) -> object:
    try:
        return await awaitable
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _utc_ms(value: datetime) -> int:
    utc_value: Final = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return int(utc_value.timestamp() * 1000)


@router.get("")
async def list_traces(
    start: datetime,
    end: datetime,
    service: str | None = None,
    status: str | None = None,
    q: str | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> object:
    clickhouse_url, native = _connection(user_api_key_dict)
    return await _result(native.list_traces(clickhouse_url, _utc_ms(start), _utc_ms(end), service, status, q, cursor, limit))


@router.get("/{trace_id}")
async def get_trace(
    trace_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> object:
    clickhouse_url, native = _connection(user_api_key_dict)
    return await _result(native.get_trace(clickhouse_url, trace_id))


@router.get("/{trace_id}/spans/{span_id}")
async def get_span(
    trace_id: str,
    span_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> object:
    clickhouse_url, native = _connection(user_api_key_dict)
    return await _result(native.get_span(clickhouse_url, trace_id, span_id))

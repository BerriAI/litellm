"""
Background reporter that periodically POSTs gateway request counts to an
external billing endpoint for request-based Enterprise billing.

Activated by setting the ``LITELLM_USAGE_ENDPOINT`` environment variable.
When set, a background asyncio loop flushes accumulated counters every
``LITELLM_USAGE_REPORT_INTERVAL_SECONDS`` seconds (default 300 / 5 min).

The counters are fed by ``GatewayUsageMiddleware`` which sits in the ASGI
middleware stack and counts every LLM-route request that flows through the
proxy.

Billable request definition: any request whose gateway response status code
is in the range [1, 500) is counted as "successful" (i.e. the gateway
handled it). 4xx responses are included because those represent client-side
errors, not gateway failures. 5xx responses and requests where no response
was sent (``status_code == 0``, e.g. client disconnect) are counted as
"failed".

In multi-pod deployments each pod reports independently; the billing service
aggregates by ``deployment_id``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

logger = logging.getLogger("litellm.proxy")

_USAGE_ENDPOINT_ENV = "LITELLM_USAGE_ENDPOINT"
_USAGE_INTERVAL_ENV = "LITELLM_USAGE_REPORT_INTERVAL_SECONDS"
_USAGE_AUTH_TOKEN_ENV = "LITELLM_USAGE_AUTH_TOKEN"
_DEPLOYMENT_ID_ENV = "LITELLM_DEPLOYMENT_ID"
_DEFAULT_FLUSH_INTERVAL_SECONDS = 300
_HTTP_TIMEOUT_SECONDS = 30
_SHUTDOWN_FLUSH_TIMEOUT_SECONDS = 5


@dataclass(slots=True)
class _Counters:
    total_requests: int = 0
    successful_requests: int = 0


_counters = _Counters()
_lock = asyncio.Lock()


async def record_request(*, succeeded: bool) -> None:
    async with _lock:
        _counters.total_requests += 1
        if succeeded:
            _counters.successful_requests += 1


async def _snapshot_counters() -> _Counters:
    async with _lock:
        return _Counters(
            total_requests=_counters.total_requests,
            successful_requests=_counters.successful_requests,
        )


async def _subtract_counters(snapshot: _Counters) -> None:
    async with _lock:
        _counters.total_requests -= snapshot.total_requests
        _counters.successful_requests -= snapshot.successful_requests


@dataclass(frozen=True, slots=True)
class UsageReportPayload:
    total_requests: int
    successful_requests: int
    failed_requests: int
    hostname: str
    pod_id: str
    deployment_id: str
    period_start: str
    period_end: str
    report_id: str


def _build_payload(snapshot: _Counters, period_start: datetime, period_end: datetime) -> UsageReportPayload:
    hostname = socket.gethostname()
    pod_id = os.environ.get("POD_NAME", hostname)
    deployment_id = os.environ.get(_DEPLOYMENT_ID_ENV, "")
    return UsageReportPayload(
        total_requests=snapshot.total_requests,
        successful_requests=snapshot.successful_requests,
        failed_requests=snapshot.total_requests - snapshot.successful_requests,
        hostname=hostname,
        pod_id=pod_id,
        deployment_id=deployment_id,
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
        report_id=str(uuid.uuid4()),
    )


async def _post_usage(
    client: AsyncHTTPHandler,
    endpoint: str,
    payload: UsageReportPayload,
    auth_token: str,
) -> None:
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    await client.post(
        url=endpoint,
        json=dataclasses.asdict(payload),
        headers=headers if headers else None,
        timeout=_HTTP_TIMEOUT_SECONDS,
    )


async def _flush_once(client: AsyncHTTPHandler, endpoint: str, auth_token: str) -> None:
    snapshot = await _snapshot_counters()
    if snapshot.total_requests == 0:
        return
    period_end = datetime.now(timezone.utc)
    payload = _build_payload(snapshot, period_start=period_end, period_end=period_end)
    await _post_usage(client, endpoint, payload, auth_token)
    await _subtract_counters(snapshot)
    logger.debug(
        "gateway_usage_report_sent total=%d successful=%d failed=%d",
        payload.total_requests,
        payload.successful_requests,
        payload.failed_requests,
    )


async def gateway_usage_reporter_loop() -> None:
    endpoint = os.environ.get(_USAGE_ENDPOINT_ENV)
    if not endpoint:
        return

    flush_interval = int(os.environ.get(_USAGE_INTERVAL_ENV, _DEFAULT_FLUSH_INTERVAL_SECONDS))
    auth_token = os.environ.get(_USAGE_AUTH_TOKEN_ENV, "")

    logger.info(
        "gateway_usage_reporter_started endpoint=%s interval_seconds=%d",
        endpoint,
        flush_interval,
    )

    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
    while True:
        try:
            await asyncio.sleep(flush_interval)
            await _flush_once(client, endpoint, auth_token)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("gateway_usage_reporter iteration failed")


async def flush_on_shutdown() -> None:
    endpoint = os.environ.get(_USAGE_ENDPOINT_ENV)
    if not endpoint:
        return
    auth_token = os.environ.get(_USAGE_AUTH_TOKEN_ENV, "")
    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
    try:
        await asyncio.wait_for(
            _flush_once(client, endpoint, auth_token),
            timeout=_SHUTDOWN_FLUSH_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("gateway_usage_reporter shutdown flush failed")

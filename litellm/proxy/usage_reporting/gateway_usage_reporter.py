"""
Background reporter that periodically POSTs gateway request counts to an
external billing endpoint.

Activated by setting the ``LITELLM_USAGE_ENDPOINT`` environment variable.
When set, a background asyncio loop flushes accumulated counters every
``LITELLM_USAGE_REPORT_INTERVAL_SECONDS`` seconds (default 300 / 5 min).

The counters are fed by ``GatewayUsageMiddleware`` which sits in the ASGI
middleware stack and counts every LLM-route request that flows through the
proxy.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import socket
from dataclasses import dataclass

import httpx

logger = logging.getLogger("litellm.proxy")

_USAGE_ENDPOINT_ENV = "LITELLM_USAGE_ENDPOINT"
_USAGE_INTERVAL_ENV = "LITELLM_USAGE_REPORT_INTERVAL_SECONDS"
_DEFAULT_FLUSH_INTERVAL_SECONDS = 300
_HTTP_TIMEOUT_SECONDS = 30


@dataclass(slots=True)
class _Counters:
    total_requests: int = 0
    successful_requests: int = 0


_counters = _Counters()
_lock = asyncio.Lock()


def record_request(*, succeeded: bool) -> None:
    _counters.total_requests += 1
    if succeeded:
        _counters.successful_requests += 1


async def _drain_counters() -> _Counters:
    async with _lock:
        snapshot = _Counters(
            total_requests=_counters.total_requests,
            successful_requests=_counters.successful_requests,
        )
        _counters.total_requests = 0
        _counters.successful_requests = 0
    return snapshot


@dataclass(frozen=True, slots=True)
class UsageReportPayload:
    total_requests: int
    successful_requests: int
    failed_requests: int
    hostname: str
    pod_id: str


def _build_payload(snapshot: _Counters) -> UsageReportPayload:
    hostname = socket.gethostname()
    pod_id = os.environ.get("POD_NAME", hostname)
    return UsageReportPayload(
        total_requests=snapshot.total_requests,
        successful_requests=snapshot.successful_requests,
        failed_requests=snapshot.total_requests - snapshot.successful_requests,
        hostname=hostname,
        pod_id=pod_id,
    )


async def _post_usage(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: UsageReportPayload,
) -> None:
    response = await client.post(
        endpoint,
        json=dataclasses.asdict(payload),
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()


async def gateway_usage_reporter_loop() -> None:
    endpoint = os.environ.get(_USAGE_ENDPOINT_ENV)
    if not endpoint:
        return

    flush_interval = int(os.environ.get(_USAGE_INTERVAL_ENV, _DEFAULT_FLUSH_INTERVAL_SECONDS))

    logger.info(
        "gateway_usage_reporter_started endpoint=%s interval_seconds=%d",
        endpoint,
        flush_interval,
    )

    async with httpx.AsyncClient() as client:
        while True:
            try:
                await asyncio.sleep(flush_interval)
                snapshot = await _drain_counters()
                if snapshot.total_requests == 0:
                    continue
                payload = _build_payload(snapshot)
                await _post_usage(client, endpoint, payload)
                logger.debug(
                    "gateway_usage_report_sent total=%d successful=%d failed=%d",
                    payload.total_requests,
                    payload.successful_requests,
                    payload.failed_requests,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("gateway_usage_reporter iteration failed")

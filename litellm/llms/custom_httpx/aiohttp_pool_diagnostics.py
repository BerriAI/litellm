"""
Diagnostics for aiohttp connection-pool saturation.

When the connector's ``limit`` / ``limit_per_host`` is reached, aiohttp queues
the request until a slot frees up. Without this, the only visible symptom is a
request timeout, which looks identical to a slow upstream. This module attaches
an aiohttp ``TraceConfig`` that logs a warning naming the host, the configured
limits, and how long the request waited for a connection slot.
"""

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

from litellm._logging import verbose_logger
from litellm.constants import (
    AIOHTTP_CONNECTOR_LIMIT,
    AIOHTTP_CONNECTOR_LIMIT_PER_HOST,
    AIOHTTP_POOL_WAIT_WARNING_SECONDS,
)

if TYPE_CHECKING:
    from aiohttp import ClientSession, TraceConfig


async def _on_request_start(session: "ClientSession", ctx: SimpleNamespace, params: object) -> None:
    url = getattr(params, "url", None)
    ctx.pool_wait_host = getattr(url, "host", None) or "unknown"


async def _on_connection_queued_start(session: "ClientSession", ctx: SimpleNamespace, params: object) -> None:
    ctx.pool_wait_started_at = time.monotonic()


def _make_on_connection_queued_end(threshold_seconds: float):
    async def _on_connection_queued_end(session: "ClientSession", ctx: SimpleNamespace, params: object) -> None:
        started_at = getattr(ctx, "pool_wait_started_at", None)
        if started_at is None:
            return
        waited = time.monotonic() - started_at
        if waited < threshold_seconds:
            return
        verbose_logger.warning(
            "aiohttp connection pool saturated: request to host=%s waited %.2fs "
            "for a free connection slot (limit=%s, limit_per_host=%s). The "
            "upstream may be healthy; consider raising "
            "AIOHTTP_CONNECTOR_LIMIT_PER_HOST / AIOHTTP_CONNECTOR_LIMIT or "
            "investigating idle connections.",
            getattr(ctx, "pool_wait_host", "unknown"),
            waited,
            AIOHTTP_CONNECTOR_LIMIT or "unlimited",
            AIOHTTP_CONNECTOR_LIMIT_PER_HOST or "unlimited",
        )

    return _on_connection_queued_end


def build_pool_wait_trace_config(
    threshold_seconds: float | None = None,
) -> "TraceConfig | None":
    """
    Return a TraceConfig that warns when a request waits for a pooled
    connection longer than ``threshold_seconds`` (defaults to
    ``AIOHTTP_POOL_WAIT_WARNING_SECONDS``). Returns None when disabled.
    """
    if threshold_seconds is None:
        threshold_seconds = AIOHTTP_POOL_WAIT_WARNING_SECONDS
    if threshold_seconds <= 0:
        return None
    from aiohttp import TraceConfig

    trace_config = TraceConfig()
    trace_config.on_request_start.append(_on_request_start)
    trace_config.on_connection_queued_start.append(_on_connection_queued_start)
    trace_config.on_connection_queued_end.append(_make_on_connection_queued_end(threshold_seconds))
    return trace_config

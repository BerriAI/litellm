"""Publishes the per-pod concurrency ceiling that is actually in force.

``global_max_parallel_requests`` is only read by the v1 parallel-request
limiter, which is off unless ``LEGACY_MULTI_INSTANCE_RATE_LIMITING`` is set. The
default v3 limiter never looks at it, so publishing the configured number
regardless would hand operators a ceiling that nothing applies.

A registered Prometheus gauge always exposes a value, so simply declining to set
it renders as ``0``, which reads as "no requests allowed". The gauge therefore
reports the effective ceiling in every state: the configured number when it is
enforced, and ``+Inf`` when nothing bounds concurrency.

See LIT-5460 for the enforcement gap itself.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Final

from litellm._logging import verbose_proxy_logger

UNBOUNDED: Final = float("inf")

# Set where the proxy itself declines a request. litellm forwards upstream
# rate limits with the same 429 the proxy uses for its own, so response status
# alone cannot tell "this pod shed load" from "the provider throttled us", and
# only the former should count toward a decision to throttle or scale out.
proxy_shed_request: Final[ContextVar[bool]] = ContextVar("litellm_proxy_shed_request", default=False)


def mark_request_shed_by_proxy() -> None:
    """Record that this proxy, not an upstream, declined the current request."""
    proxy_shed_request.set(True)


def clear_request_shed_marker() -> None:
    """Undo the mark because the rejection is being recovered from, not returned.

    The mark is set when the error is constructed, which is the only point every
    raise site passes through. A caught rejection that falls back to another
    model would otherwise leave the request marked, and a provider 429 on that
    fallback would be counted as this pod shedding load.
    """
    proxy_shed_request.set(False)


def was_request_shed_by_proxy() -> bool:
    return proxy_shed_request.get()


def is_global_limit_enforced() -> bool:
    """Whether the registered limiter reads ``global_max_parallel_requests``.

    Only the v1 handler does. Identity against v1 rather than "anything but v3"
    so an unrecognised limiter reports unbounded instead of claiming a ceiling
    it may not apply.
    """
    # The limiter class is private by naming convention only: it is what the hook
    # registry is built from, and proxy/utils.py imports it the same way.
    from litellm.proxy.hooks import PROXY_HOOKS
    from litellm.proxy.hooks.parallel_request_limiter import (
        _PROXY_MaxParallelRequestsHandler,  # pyright: ignore[reportPrivateUsage]  # the registry's own hook class, imported this way across the proxy
    )

    return PROXY_HOOKS.get("parallel_request_limiter") is _PROXY_MaxParallelRequestsHandler


def effective_global_limit(limit: int | None) -> float:
    """The ceiling actually applied to this worker's concurrency."""
    if limit is None:
        return UNBOUNDED
    if not is_global_limit_enforced():
        verbose_proxy_logger.warning(
            "global_max_parallel_requests=%s is set but the active rate limiter does not enforce it, so the "
            "limit metric reports unbounded. Set LEGACY_MULTI_INSTANCE_RATE_LIMITING=true to enforce it",
            limit,
        )
        return UNBOUNDED
    return float(limit)


def publish_global_max_parallel_requests(limit: int | None) -> None:
    """Publish the effective ceiling, whatever it turns out to be."""
    try:
        from litellm.integrations.prometheus import PrometheusLogger

        logger: Final = PrometheusLogger.get_instance()
        if logger is not None:
            logger.set_global_max_parallel_requests_limit(effective_global_limit(limit))
    except Exception as e:  # noqa: BLE001  # telemetry must not block startup
        verbose_proxy_logger.debug("global max parallel requests metric failed: %s", e)

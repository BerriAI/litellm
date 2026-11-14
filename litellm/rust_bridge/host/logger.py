from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

from litellm._logging import (
    CorrelationContextFilter,
    DiagnosticProcessingFilter,
    session_id_var,
    set_session_id,
    set_trace_id,
    trace_id_var,
    verbose_logger,
    verbose_proxy_logger,
)

_REDACTION: Final = DiagnosticProcessingFilter()
_CORRELATION: Final = CorrelationContextFilter()
_PROXY_TARGETS: Final = ("litellm_callbacks_parallel_request_limiter",)


def context() -> tuple[str, str]:
    return session_id_var.get(), trace_id_var.get()


def _logger(target: str) -> logging.Logger:
    return verbose_proxy_logger if target.startswith(_PROXY_TARGETS) else verbose_logger


def enabled(level: int, target: str = "") -> bool:
    return _logger(target).isEnabledFor(level)


def _redis_timeout_line(level: int, message: str) -> tuple[int, str]:
    """``log_redis_failure``'s timeout rule, on the throttle redis-py's timeouts count against too."""
    from litellm.caching.redis_cache import redis_timeout_log_throttle

    suppressed: Final = redis_timeout_log_throttle.admit()
    if suppressed is None:
        return logging.DEBUG, message
    if suppressed == 0:
        return level, message
    return (
        level,
        f"{message} ({suppressed} more Redis timeouts since the previous Redis timeout line were logged at DEBUG)",
    )


def emit(
    level: int,
    message: str,
    pathname: str,
    lineno: int,
    target: str,
    fields: Mapping[str, JsonValue],
    correlation: tuple[str, str],
) -> None:
    if fields.get("redis.failure") == "timeout":
        throttled_level, throttled_message = _redis_timeout_line(level, message)
        _emit(throttled_level, throttled_message, pathname, lineno, target, fields, correlation)
        return
    _emit(level, message, pathname, lineno, target, fields, correlation)


def _emit(
    level: int,
    message: str,
    pathname: str,
    lineno: int,
    target: str,
    fields: Mapping[str, JsonValue],
    correlation: tuple[str, str],
) -> None:
    logger: Final = _logger(target)
    if not logger.isEnabledFor(level):
        return
    session_token: Final = set_session_id(correlation[0])
    trace_token: Final = set_trace_id(correlation[1])
    try:
        record: Final = logger.makeRecord(
            logger.name,
            level,
            pathname,
            lineno,
            message,
            (),
            None,
            func=target,
            extra={
                "rust_target": target,
                "rust_fields": dict(fields),
            },
        )
        _REDACTION.filter(record)
        _CORRELATION.filter(record)
        logger.handle(record)
    finally:
        trace_id_var.reset(trace_token)
        session_id_var.reset(session_token)

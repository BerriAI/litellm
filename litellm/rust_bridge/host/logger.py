from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import JsonValue

from litellm._logging import (
    RUST_FIELDS_RECORD_ATTR,
    CorrelationContextFilter,
    DiagnosticProcessingFilter,
    session_id_var,
    set_session_id,
    set_trace_id,
    trace_id_var,
)

_REDACTION: Final = DiagnosticProcessingFilter()
_CORRELATION: Final = CorrelationContextFilter()


_LEVELS: Final = (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
_NOTHING: Final = logging.CRITICAL + 1


def _floor(logger: logging.Logger) -> int:
    return next((level for level in _LEVELS if logger.isEnabledFor(level)), _NOTHING)


def context(loggers: Sequence[str]) -> tuple[tuple[str, str], tuple[int, ...]]:
    """The request ids of the calling thread, and the lowest level each named logger takes."""
    return (session_id_var.get(), trace_id_var.get()), tuple(_floor(logging.getLogger(name)) for name in loggers)


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
    logger: str,
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
        _emit(logger, throttled_level, throttled_message, pathname, lineno, target, fields, correlation)
        return
    _emit(logger, level, message, pathname, lineno, target, fields, correlation)


def _emit(
    name: str,
    level: int,
    message: str,
    pathname: str,
    lineno: int,
    target: str,
    fields: Mapping[str, JsonValue],
    correlation: tuple[str, str],
) -> None:
    logger: Final = logging.getLogger(name)
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
                RUST_FIELDS_RECORD_ATTR: dict(fields),
            },
        )
        _REDACTION.filter(record)
        _CORRELATION.filter(record)
        logger.handle(record)
    finally:
        trace_id_var.reset(trace_token)
        session_id_var.reset(session_token)

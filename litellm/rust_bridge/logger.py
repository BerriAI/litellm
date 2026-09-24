from __future__ import annotations

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
)

_REDACTION: Final = DiagnosticProcessingFilter()
_CORRELATION: Final = CorrelationContextFilter()


def context() -> tuple[str, str]:
    return session_id_var.get(), trace_id_var.get()


def enabled(level: int) -> bool:
    return verbose_logger.isEnabledFor(level)


def emit(
    level: int,
    message: str,
    pathname: str,
    lineno: int,
    target: str,
    fields: Mapping[str, JsonValue],
    correlation: tuple[str, str],
) -> None:
    if not enabled(level):
        return
    session_token: Final = set_session_id(correlation[0])
    trace_token: Final = set_trace_id(correlation[1])
    try:
        record: Final = verbose_logger.makeRecord(
            verbose_logger.name,
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
        verbose_logger.handle(record)
    finally:
        trace_id_var.reset(trace_token)
        session_id_var.reset(session_token)

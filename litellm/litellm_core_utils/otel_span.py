import os
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from enum import Enum
from types import TracebackType
from typing import Any, Awaitable, Dict, Iterator, Optional, Protocol, Type

from litellm._logging import verbose_logger
from litellm.types.services import ServiceTypes


class _OtelSpan(Protocol):
    def end(self, end_time: Optional[int] = None) -> None: ...

    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_status(self, status: Any) -> None: ...

    def record_exception(self, exception: BaseException) -> None: ...


class _OtelLogger(Protocol):
    tracer: Any


_current_otel_span: ContextVar[Optional[_OtelSpan]] = ContextVar(
    "litellm_current_otel_span", default=None
)
_registered_otel_logger: Optional[_OtelLogger] = None
_ENABLE_DETAILED_OTEL_SPANS_ENV_VAR = "LITELLM_ENABLE_DETAILED_OTEL_SPANS"


def get_current_otel_span() -> Optional[_OtelSpan]:
    return _current_otel_span.get()


def set_litellm_otel_logger(otel_logger: Optional[_OtelLogger]) -> None:
    """
    Register the OpenTelemetry logger used by LiteLLM internal spans.

    Keep this registry in core utils so SDK-only imports never need to import
    the proxy server just to resolve an optional tracing dependency.
    """
    global _registered_otel_logger
    _registered_otel_logger = otel_logger


@contextmanager
def attach_otel_span(span: Optional[_OtelSpan]) -> Iterator[Optional[_OtelSpan]]:
    """
    Attach an existing span to LiteLLM's async-local tracing context.

    This is useful for request-root spans created outside this helper. Child
    spans opened with ``litellm_otel_tracer.trace(...)`` will use the attached
    span as their parent unless an explicit parent is provided.
    """
    if span is None:
        yield None
        return

    token = _current_otel_span.set(span)
    otel_context_token: Optional[Any] = None
    try:
        otel_context_token = _OtelContext.attach_span(span)
        yield span
    finally:
        if otel_context_token is not None:
            _OtelContext.detach(otel_context_token)
        _current_otel_span.reset(token)


class LiteLLMOtelSpan:
    """
    Context manager for LiteLLM internal OpenTelemetry spans.

    The span is started on ``__enter__``, stored in a ``ContextVar`` for
    async-safe child span propagation, and ended on ``__exit__``. When OTEL is
    not configured, or a parent span is required but unavailable, it becomes a
    cheap no-op.
    """

    def __init__(
        self,
        *,
        span_name: str,
        service: ServiceTypes,
        parent_span: Optional[_OtelSpan] = None,
        attributes: Optional[Dict[str, Any]] = None,
        require_parent: bool = True,
        otel_logger: Optional[Any] = None,
        start_time_ns: Optional[int] = None,
        end_time_ns: Optional[int] = None,
        detailed: bool = False,
    ) -> None:
        self.span_name = span_name
        self.service = service
        self.parent_span = parent_span
        self.attributes = attributes or {}
        self.require_parent = require_parent
        self.otel_logger = otel_logger
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns
        self.detailed = detailed

        self.span: Optional[_OtelSpan] = None
        self._resolved_otel_logger: Optional[Any] = None
        self._context_token: Optional[Token[Optional[_OtelSpan]]] = None
        self._otel_context_token: Optional[Any] = None

    def __enter__(self) -> "LiteLLMOtelSpan":
        if self.detailed and not _DetailedOtelFeatureGate.is_enabled():
            return self

        parent_span = self.parent_span or get_current_otel_span()
        if self.require_parent and parent_span is None:
            return self

        otel_logger = self.otel_logger or _OpenTelemetryLoggerResolver.get()
        if otel_logger is None:
            return self

        try:
            self._resolved_otel_logger = otel_logger
            context = None
            if parent_span is not None:
                from opentelemetry import trace

                context = trace.set_span_in_context(parent_span)
            span = otel_logger.tracer.start_span(
                name=self.span_name,
                context=context,
                start_time=self.start_time_ns or time.time_ns(),
            )
            self.span = span
            self._set_default_attributes(otel_logger=otel_logger)
            self._context_token = _current_otel_span.set(span)
            self._otel_context_token = _OtelContext.attach_span(span)
        except Exception as exc:
            verbose_logger.debug(
                "LiteLLMOtelSpan: failed to start span %s: %s",
                self.span_name,
                str(exc),
            )
            self.span = None

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        span = self.span
        try:
            if span is not None:
                self._set_status(exc_value=exc_value)
                end_time_ns = (
                    self.end_time_ns if self.end_time_ns is not None else time.time_ns()
                )
                span.end(end_time=end_time_ns)
        except Exception as exc:
            verbose_logger.debug(
                "LiteLLMOtelSpan: failed to end span %s: %s",
                self.span_name,
                str(exc),
            )
        finally:
            if self._otel_context_token is not None:
                _OtelContext.detach(self._otel_context_token)
            if self._context_token is not None:
                _current_otel_span.reset(self._context_token)
        return False

    async def __aenter__(self) -> "LiteLLMOtelSpan":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        return self.__exit__(
            exc_type=exc_type,
            exc_value=exc_value,
            traceback=traceback,
        )

    def set_attribute(self, key: str, value: Any) -> None:
        if self.span is None or self._resolved_otel_logger is None:
            return
        _OtelAttributeSetter.set(
            otel_logger=self._resolved_otel_logger,
            span=self.span,
            key=key,
            value=value,
        )

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        for key, value in attributes.items():
            self.set_attribute(key=key, value=value)

    def _set_default_attributes(self, *, otel_logger: Any) -> None:
        if self.span is None:
            return

        _OtelAttributeSetter.set(
            otel_logger=otel_logger,
            span=self.span,
            key="call_type",
            value=self.span_name,
        )
        _OtelAttributeSetter.set(
            otel_logger=otel_logger,
            span=self.span,
            key="service",
            value=self.service.value,
        )
        for key, value in self.attributes.items():
            _OtelAttributeSetter.set(
                otel_logger=otel_logger,
                span=self.span,
                key=key,
                value=value,
            )

    def _set_status(self, *, exc_value: Optional[BaseException]) -> None:
        if self.span is None:
            return

        try:
            from opentelemetry.trace import Status, StatusCode

            if exc_value is None:
                self.span.set_status(Status(StatusCode.OK))
                return

            self.span.record_exception(exc_value)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_value)))
        except Exception:
            return


class LiteLLMOtelTracer:
    def is_enabled(self) -> bool:
        return _OpenTelemetryLoggerResolver.get() is not None

    def is_detailed_enabled(self) -> bool:
        return _DetailedOtelFeatureGate.is_enabled()

    def trace(
        self,
        span_name: str,
        *,
        service: ServiceTypes,
        parent_span: Optional[_OtelSpan] = None,
        attributes: Optional[Dict[str, Any]] = None,
        require_parent: bool = True,
        start_time: Optional[float] = None,
        detailed: bool = False,
    ) -> LiteLLMOtelSpan:
        return LiteLLMOtelSpan(
            span_name=span_name,
            service=service,
            parent_span=parent_span,
            attributes=attributes,
            require_parent=require_parent,
            start_time_ns=int(start_time * 1e9) if start_time is not None else None,
            detailed=detailed,
        )

    async def trace_async(
        self,
        awaitable: Awaitable[Any],
        *,
        span_name: str,
        service: ServiceTypes,
        parent_span: Optional[_OtelSpan] = None,
        attributes: Optional[Dict[str, Any]] = None,
        require_parent: bool = True,
        detailed: bool = False,
    ) -> Any:
        with self.trace(
            span_name=span_name,
            service=service,
            parent_span=parent_span,
            attributes=attributes,
            require_parent=require_parent,
            detailed=detailed,
        ):
            return await awaitable

    def record_completed_span(
        self,
        *,
        span_name: str,
        service: ServiceTypes,
        start_time: float,
        end_time: Optional[float] = None,
        parent_span: Optional[_OtelSpan] = None,
        attributes: Optional[Dict[str, Any]] = None,
        require_parent: bool = True,
        detailed: bool = False,
    ) -> None:
        if detailed and not _DetailedOtelFeatureGate.is_enabled():
            return

        start_time_ns = int(start_time * 1e9)
        end_time_ns = int(end_time * 1e9) if end_time is not None else None
        span = LiteLLMOtelSpan(
            span_name=span_name,
            service=service,
            parent_span=parent_span,
            attributes=attributes,
            require_parent=require_parent,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            detailed=detailed,
        )
        with span:
            pass


litellm_otel_tracer = LiteLLMOtelTracer()


class _DetailedOtelFeatureGate:
    @classmethod
    def is_enabled(cls) -> bool:
        return os.getenv(_ENABLE_DETAILED_OTEL_SPANS_ENV_VAR, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }


class _OpenTelemetryLoggerResolver:
    @staticmethod
    def get() -> Optional[_OtelLogger]:
        return _registered_otel_logger


class _OtelContext:
    @staticmethod
    def attach_span(span: _OtelSpan) -> Optional[Any]:
        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace

            return otel_context.attach(trace.set_span_in_context(span))
        except Exception:
            return None

    @staticmethod
    def detach(token: Any) -> None:
        try:
            from opentelemetry import context as otel_context

            otel_context.detach(token)
        except Exception:
            return


class _OtelAttributeSetter:
    @staticmethod
    def set(
        *,
        otel_logger: Any,
        span: _OtelSpan,
        key: str,
        value: Any,
    ) -> None:
        normalized_value = _OtelAttributeValueNormalizer.normalize(value)
        safe_set_attribute = getattr(otel_logger, "safe_set_attribute", None)
        try:
            if callable(safe_set_attribute):
                safe_set_attribute(span=span, key=key, value=normalized_value)
            else:
                span.set_attribute(key, normalized_value)
        except Exception:
            return


class _OtelAttributeValueNormalizer:
    @classmethod
    def normalize(cls, value: Any) -> Any:
        normalized_scalar = cls._normalize_scalar(value)
        if normalized_scalar is not None:
            return normalized_scalar

        if isinstance(value, (list, tuple)):
            return [cls._normalize_list_item(item) for item in value]

        return cls._stringify(value)

    @classmethod
    def _normalize_list_item(cls, value: Any) -> Any:
        normalized_scalar = cls._normalize_scalar(value)
        if normalized_scalar is not None:
            return normalized_scalar
        return cls._stringify(value)

    @staticmethod
    def _normalize_scalar(value: Any) -> Optional[Any]:
        if value is None:
            return "None"
        if isinstance(value, Enum):
            enum_value = value.value
            return (
                enum_value if isinstance(enum_value, (str, bool, int, float)) else None
            )
        if isinstance(value, (str, bool, int, float)):
            return value
        return None

    @staticmethod
    def _stringify(value: Any) -> str:
        try:
            return str(value)
        except Exception:
            return "litellm logging error - could_not_serialize"

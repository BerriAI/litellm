from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Final, Protocol, cast

from litellm.rust_bridge.bindings import NativeBinding

NATIVE_ORIGIN: Final = object()
_ORIGIN_ATTRIBUTE: Final = "_litellm_native_origin"


class _ForwardingState(threading.local):
    def __init__(self) -> None:
        self.active: bool = False


_STATE: Final = _ForwardingState()


class DiagnosticEmitter(Protocol):
    def emit(self, severity: int, message: str, fields: str) -> None: ...


class NativeDiagnosticLogger(DiagnosticEmitter, Protocol):
    def active(self) -> bool: ...
    def payload_shapes_enabled(self) -> bool: ...
    def emit_payload_shape(self, event: str) -> None: ...
    def extract_payload_shape(
        self, value: object, *, nodes: int = 4096, depth: int = 16, paths: int = 256, bytes: int = 16384
    ) -> tuple[Sequence[str], bool]: ...
    def configure(self, configuration: str) -> None: ...
    def force_flush(self) -> None: ...
    def shutdown(self) -> None: ...


class NativeDiagnosticFactory(Protocol):
    def __call__(self) -> NativeDiagnosticLogger: ...


def _as_factory(value: object) -> NativeDiagnosticFactory | None:
    if not isinstance(value, type):
        return None
    return cast(NativeDiagnosticFactory, value)  # cast-ok: PyO3 factory must be a type


LOGGER: Final = NativeBinding("NativeDiagnosticLogger", validate=_as_factory)


@lru_cache(maxsize=4)
def _construct(factory: NativeDiagnosticFactory) -> NativeDiagnosticLogger:
    return factory()


def _load() -> NativeDiagnosticLogger | None:
    factory: Final = LOGGER.load()
    return None if factory is None else _construct(factory)


def payload_logger() -> NativeDiagnosticLogger | None:
    return _load()


def _active() -> bool:
    native: Final = _load()
    return native is not None and native.active()


class ForwardingHandler(logging.Handler):
    def __init__(
        self,
        native: Callable[[], DiagnosticEmitter | None] = _load,
        active: Callable[[], bool] = _active,
    ) -> None:
        super().__init__()
        self._native: Final = native
        self._active: Final = active
        self.failures: int = 0

    def emit(self, record: logging.LogRecord) -> None:
        if _STATE.active or record.__dict__.get(_ORIGIN_ATTRIBUTE) is NATIVE_ORIGIN:
            return
        _STATE.active = True
        try:
            if not self._active():
                return
            native: Final = self._native()
            if native is None:
                return
            from litellm._logging import diagnostic_snapshot

            message, fields = diagnostic_snapshot(record)
            native.emit(record.levelno, message, fields)
        except Exception:  # noqa: BLE001  # an exporter failure must not interrupt Python logging
            self.failures += 1
        finally:
            _STATE.active = False


def _covered_by_ancestor(logger: logging.Logger, selected: tuple[logging.Logger, ...]) -> bool:
    while logger.propagate and logger.parent is not None:
        if logger.parent in selected or any(
            isinstance(handler, ForwardingHandler) for handler in logger.parent.handlers
        ):
            return True
        logger = logger.parent  # rebind-ok: advance through the logger propagation chain
    return False


def install(loggers: tuple[logging.Logger, ...]) -> None:
    for logger in loggers:
        if _covered_by_ancestor(logger, loggers):
            for handler in tuple(logger.handlers):
                if isinstance(handler, ForwardingHandler):
                    logger.removeHandler(handler)
            continue
        if not any(isinstance(handler, ForwardingHandler) for handler in logger.handlers):
            logger.addHandler(ForwardingHandler())


def configure(configuration: str, loggers: tuple[logging.Logger, ...] | None) -> bool:
    native: Final = _load()
    if native is None:
        return False
    native.configure(configuration)
    from litellm._logging import verbose_logger, verbose_proxy_logger, verbose_router_logger

    if native.active():
        install((verbose_logger, verbose_proxy_logger, verbose_router_logger) if loggers is None else loggers)
    return True


def force_flush() -> bool:
    native: Final = _load()
    if native is None:
        return False
    native.force_flush()
    return True


def shutdown() -> bool:
    native: Final = _load()
    if native is None:
        return False
    native.shutdown()
    return True

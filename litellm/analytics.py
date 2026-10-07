from __future__ import annotations

import asyncio
import atexit
import json
import os
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Final, Literal, ParamSpec, Protocol, TypeVar

from litellm.rust_bridge.forwarding import native_logger

P: Final = ParamSpec("P")
T: Final = TypeVar("T")
_DEPTH: Final = ContextVar("litellm_analytics_depth", default=0)


class AnalyticsEmitter(Protocol):
    def initialize_analytics(self, configuration: str) -> tuple[bool, bool]: ...
    def emit_analytics(self, route: str) -> None: ...
    def shutdown_analytics(self) -> None: ...


class AnalyticsHost:
    def __init__(
        self,
        native: Callable[[], AnalyticsEmitter | None] = native_logger,
        environment: Mapping[str, str] = os.environ,
        warning: Callable[[str], object] | None = None,
    ) -> None:
        self._native: Final = native
        self._environment: Final = environment
        self._warning: Final = warning
        self._lock: threading.RLock = threading.RLock()
        self._pid: int = os.getpid()
        self._gateway: bool = False
        self._license_declared: bool = False
        self._decided: bool = False
        self._active: AnalyticsEmitter | None = None
        self._session: AnalyticsEmitter | None = None

    def _warn(self, message: str) -> None:
        if self._warning is not None:
            self._warning(message)
            return
        from litellm._logging import verbose_logger

        verbose_logger.warning(message)

    def _reset_worker(self) -> None:
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._decided = False
        self._active = None
        self._session = None

    def prepare_gateway(self) -> None:
        self.shutdown()
        with self._lock:
            self._gateway = True
            self._decided = False
            self._license_declared = False

    def declare_license(self) -> None:
        self._license_declared = True

    def initialize(self, surface: Literal["python_sdk", "python_gateway"], license_configured: bool = False) -> None:
        if os.getpid() != self._pid:
            self._reset_worker()
        with self._lock:
            if self._decided or (self._gateway and surface == "python_sdk"):
                return
            self._decided = True
            try:
                native: Final = self._native()
                if native is None:
                    return
                from litellm import _version

                configuration: Final = json.dumps(
                    {
                        "inputs": {
                            "do_not_track": self._environment.get("DO_NOT_TRACK"),
                            "explicit": self._environment.get("LITELLM_TELEMETRY"),
                            "license_configured": license_configured
                            or self._license_declared
                            or "LITELLM_LICENSE" in self._environment,
                        },
                        "surface": surface,
                        "version": _version.version,
                    }
                )
                active, invalid = native.initialize_analytics(configuration)
                self._session = native
                if invalid:
                    self._warn("Invalid analytics control setting: built-in analytics disabled")
                if active:
                    self._active = native
            except Exception:  # noqa: BLE001  # analytics must not change inference or startup failures
                self._warn("Built-in analytics unavailable")

    def api_used(self, route: str) -> None:
        self.initialize("python_sdk")
        with self._lock:
            if self._active is None:
                return
            try:
                self._active.emit_analytics(route)
            except Exception:  # noqa: BLE001  # analytics must not replace an API result
                self._active = None
                self._warn("Built-in analytics unavailable")

    def shutdown(self) -> None:
        if os.getpid() != self._pid:
            self._reset_worker()
            return
        with self._lock:
            self._decided = True
            if self._session is None:
                return
            try:
                self._session.shutdown_analytics()
            except Exception:  # noqa: BLE001  # draining analytics must not replace a host failure
                self._warn("Built-in analytics shutdown failed")
            finally:
                self._active = None
                self._session = None


_HOST: Final = AnalyticsHost()


def prepare_gateway() -> None:
    _HOST.prepare_gateway()


def license_declared(configuration: Mapping[str, object]) -> bool:
    environment: Final = configuration.get("environment_variables")
    general: Final = configuration.get("general_settings")
    return (isinstance(environment, Mapping) and "LITELLM_LICENSE" in environment) or (
        isinstance(general, Mapping) and "litellm_license" in general
    )


def declare_license() -> None:
    _HOST.declare_license()


@contextmanager
def api_use(route: str, host: AnalyticsHost = _HOST) -> Iterator[None]:
    depth: Final = _DEPTH.get()
    token: Final = _DEPTH.set(depth + 1)
    try:
        if depth == 0:
            host.api_used(route)
        yield
    finally:
        _DEPTH.reset(token)


def track_sync(function: Callable[P, T], route: str, *, host: AnalyticsHost = _HOST) -> Callable[P, T]:
    @wraps(function)
    def tracked(*args: P.args, **kwargs: P.kwargs) -> T:
        with api_use(route, host):
            return function(*args, **kwargs)

    return tracked


def track_async(
    function: Callable[P, Awaitable[T]], route: str, *, host: AnalyticsHost = _HOST
) -> Callable[P, Coroutine[object, object, T]]:
    @wraps(function)
    async def tracked(*args: P.args, **kwargs: P.kwargs) -> T:
        with api_use(route, host):
            return await function(*args, **kwargs)

    return tracked


@asynccontextmanager
async def gateway_lifecycle(license_configured: bool) -> AsyncGenerator[None, None]:
    await asyncio.to_thread(_HOST.initialize, "python_gateway", license_configured)
    try:
        yield
    finally:
        await asyncio.to_thread(_HOST.shutdown)


atexit.register(_HOST.shutdown)
os.register_at_fork(after_in_child=_HOST._reset_worker)

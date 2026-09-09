"""Whole-call argument boundary for native OCR execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import Final, Protocol, cast  # noqa: TID251  # native extension exposes dynamically typed callables

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge._lifecycle import (
    LOGGING_OBJECT_KEY,
    NativeLifecycle,
    NativeLifecycleBindings,
    NativeOutcome,
    TerminalAction,
    advance_host,
    deployment_failure,
    deployment_pre,
    deployment_success,
    drive_async,
    drive_sync,
    host_result,
    initialize_logging,
    invoke_terminal,
    restore_correlation_context,
)
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.provenance import mark_native_response


class RustOcr(Protocol):
    def __call__(
        self, arguments: dict[str, object]
    ) -> OCRResponse: ...  # mutable-ok: native bridge retains and updates Python argument objects


class RustAocr(Protocol):
    def __call__(
        self, arguments: dict[str, object]
    ) -> Awaitable[OCRResponse]: ...  # mutable-ok: native bridge retains and updates Python argument objects


def _as_ocr(value: object) -> RustOcr | None:
    return cast(RustOcr, value) if callable(value) else None


def _as_aocr(value: object) -> RustAocr | None:
    return cast(RustAocr, value) if callable(value) else None


_OCR: Final = NativeBinding("ocr", validate=_as_ocr)
_AOCR: Final = NativeBinding("aocr", validate=_as_aocr)


def load_rust_ocr() -> RustOcr | None:
    return _OCR.load()


def load_rust_aocr() -> RustAocr | None:
    return _AOCR.load()


def ocr(
    arguments: dict[str, object],
) -> OCRResponse:  # mutable-ok: native bridge retains and updates Python argument objects
    implementation: Final = load_rust_ocr()
    if implementation is None:
        raise RuntimeError("Rust OCR is enabled but the native OCR extension is unavailable")
    return mark_native_response(implementation(arguments))


async def aocr(
    arguments: dict[str, object],
) -> OCRResponse:  # mutable-ok: native bridge retains and updates Python argument objects
    implementation: Final = load_rust_aocr()
    if implementation is None:
        raise RuntimeError("Rust OCR is enabled but the native OCR extension is unavailable")
    return mark_native_response(await implementation(arguments))


class _OcrLifecycle(NativeLifecycle, Protocol):
    def identity(self) -> tuple[str, str | None]: ...


class _OcrBindings(NativeLifecycleBindings, Protocol):
    Lifecycle: Callable[
        [dict[str, object], object | None, bool, bool], _OcrLifecycle
    ]  # mutable-ok: native bridge retains and updates Python argument objects
    build_request: Callable[
        [dict[str, object], object, bool], object
    ]  # mutable-ok: native bridge retains and updates Python argument objects
    pre_call: Callable[[object], None]
    send: Callable[
        [object], Awaitable[dict[str, object]]
    ]  # mutable-ok: native bridge retains and updates Python argument objects
    send_sync: Callable[[object], OCRResponse]
    finish: Callable[
        [dict[str, object]], OCRResponse
    ]  # mutable-ok: native bridge retains and updates Python argument objects
    terminal_record: Callable[[object], Mapping[str, object]]


class _OcrHost:
    def __init__(
        self, arguments: dict[str, object], asynchronous: bool, bindings: _OcrBindings
    ) -> None:  # mutable-ok: native bridge retains and updates Python argument objects
        from litellm import utils

        self.bindings: _OcrBindings = bindings
        self.arguments: dict[str, object] = (
            arguments  # mutable-ok: native bridge retains and updates Python argument objects
        )
        self.current: dict[str, object] = (
            arguments  # mutable-ok: native bridge retains and updates Python argument objects
        )
        self.asynchronous: bool = asynchronous
        self.logger: object | None = arguments.get(LOGGING_OBJECT_KEY)
        self.machine: _OcrLifecycle = bindings.Lifecycle(
            arguments, self.logger, asynchronous, utils.is_internal_call.get()
        )
        self.state: object | None = None
        self.response: object = None
        self.error: BaseException | None = None
        self.start: datetime = datetime.now()  # noqa: DTZ005  # Logging preserves the legacy naive timestamp contract
        self.end: datetime | None = None

    def invoke(self) -> tuple[bool, object]:
        return self.bindings.invoke(self.machine, self)

    def setup(self) -> None:
        call_id, trace_id = self.machine.identity()
        self.arguments["litellm_call_id"] = call_id
        self.arguments["litellm_trace_id"] = trace_id
        self.logger = initialize_logging(self.arguments, self.asynchronous)
        self.arguments[LOGGING_OBJECT_KEY] = self.logger

    async def deployment_pre(self) -> None:
        self.current = await deployment_pre(self.current, "aocr")
        self.current[LOGGING_OBJECT_KEY] = self.logger
        call_id, trace_id = self.machine.identity()
        self.current["litellm_call_id"] = call_id
        self.current["litellm_trace_id"] = trace_id

    def build_request(self) -> None:
        if self.logger is None:
            raise RuntimeError("OCR logging was not initialized")
        self.state = self.bindings.build_request(self.current, self.logger, self.asynchronous)

    def pre_call(self) -> None:
        self.bindings.pre_call(self.state)

    def send_sync(self) -> None:
        self.response = self.bindings.send_sync(self.state)
        self.end = datetime.now()  # noqa: DTZ005  # Logging preserves the legacy naive timestamp contract

    async def send(self) -> None:
        self.response = self.bindings.finish(await self.bindings.send(self.state))
        self.end = datetime.now()  # noqa: DTZ005  # Logging preserves the legacy naive timestamp contract

    async def deployment_success(self) -> None:
        from litellm.types.utils import CallTypes

        self.response = await deployment_success(self.current, self.response, CallTypes.aocr)

    async def deployment_failure(self) -> None:
        await deployment_failure(self.current, self.error, "aocr")

    def terminal(self, action: TerminalAction, value: object) -> object:
        if self.logger is None or self.end is None:
            raise RuntimeError("OCR terminal state was not initialized")
        record: Final = self.bindings.terminal_record(self.state) if self.state is not None else None
        return invoke_terminal(
            action,
            (self.arguments, self.current, self.state),
            self.logger,
            record,
            value,
            self.start,
            self.end,
        )

    def sync_success(self) -> object:
        return self.terminal("sync_success", self.response)

    def async_success(self) -> object:
        return self.terminal("async_success", self.response)

    def sync_success_if_needed(self) -> object:
        return self.terminal("sync_success_if_needed", self.response)

    def sync_failure(self) -> object:
        return self.terminal("sync_failure", self.error)

    async def async_failure(self) -> object:
        result: Final = self.terminal("async_failure", self.error)
        return await result if isinstance(result, Awaitable) else result

    def restore(self) -> None:
        restore_correlation_context(self.logger)

    def advance(self, outcome: NativeOutcome, error: BaseException | None = None) -> None:
        advance_host(self, outcome, error)

    def result(self) -> object:
        return host_result(self)


def _drive_sync(  # pyright: ignore[reportUnusedFunction]  # called by the native extension
    arguments: dict[str, object],
    bindings: _OcrBindings,  # mutable-ok: native bridge retains and updates Python argument objects
) -> OCRResponse:
    return cast(OCRResponse, drive_sync(_OcrHost(arguments, False, bindings)))


async def _drive_async(  # pyright: ignore[reportUnusedFunction]  # called by the native extension
    arguments: dict[str, object],
    bindings: _OcrBindings,  # mutable-ok: native bridge retains and updates Python argument objects
) -> OCRResponse:
    return cast(OCRResponse, await drive_async(_OcrHost(arguments, True, bindings)))

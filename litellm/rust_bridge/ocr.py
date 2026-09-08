"""Whole-call argument boundary for native OCR execution."""

from __future__ import annotations

import traceback
from collections.abc import Awaitable, Callable, Mapping
from contextvars import copy_context
from datetime import datetime
from typing import Final, Protocol, cast  # noqa: TID251  # native extension exposes dynamically typed callables
from uuid import uuid4

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
    restore_correlation_context,
)
from litellm.rust_bridge.bindings import NativeBinding


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
    return implementation(arguments)


async def aocr(
    arguments: dict[str, object],
) -> OCRResponse:  # mutable-ok: native bridge retains and updates Python argument objects
    implementation: Final = load_rust_aocr()
    if implementation is None:
        raise RuntimeError("Rust OCR is enabled but the native OCR extension is unavailable")
    return await implementation(arguments)


def initialize_logging(
    arguments: dict[str, object], asynchronous: bool, route: str = "ocr"
) -> object:  # mutable-ok: native bridge retains and updates Python argument objects
    import litellm
    from litellm import utils
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils import litellm_logging
    from litellm.litellm_core_utils.coroutine_checker import coroutine_checker
    from litellm.litellm_core_utils.litellm_logging import Logging, set_callbacks

    supplied: Final = arguments.get(LOGGING_OBJECT_KEY)
    if supplied is not None:
        return supplied
    callbacks: Final = tuple(  # cast-ok: callback registry accepts heterogeneous legacy callback objects
        dict.fromkeys(
            utils.get_dynamic_callbacks(
                cast(  # cast-ok: callback registry accepts heterogeneous legacy callback objects
                    list, arguments.get("callbacks")
                )  # cast-ok: callback registry accepts heterogeneous legacy callback objects
            )  # cast-ok: callback registry accepts heterogeneous legacy callback objects
        )  # cast-ok: callback registry accepts heterogeneous legacy callback objects  # mutable-ok: deduplication uses dict keys
    )
    success: Final = tuple(  # cast-ok: per-call callback list is a legacy untyped boundary
        dict.fromkeys(
            (
                *callbacks,
                *cast(  # cast-ok: per-call callback list is a legacy untyped boundary
                    list, arguments.get("success_callback") or ()
                ),  # cast-ok: per-call callback list is a legacy untyped boundary
            )  # cast-ok: per-call callback list is a legacy untyped boundary
        )  # cast-ok: per-call callback list is a legacy untyped boundary  # mutable-ok: deduplication uses dict keys
    )
    failure: Final = tuple(  # cast-ok: per-call callback list is a legacy untyped boundary
        dict.fromkeys(
            (
                *callbacks,
                *cast(  # cast-ok: per-call callback list is a legacy untyped boundary
                    list, arguments.get("failure_callback") or ()
                ),  # cast-ok: per-call callback list is a legacy untyped boundary
            )  # cast-ok: per-call callback list is a legacy untyped boundary
        )  # cast-ok: per-call callback list is a legacy untyped boundary  # mutable-ok: deduplication uses dict keys
    )
    configured: Final = tuple(
        dict.fromkeys(
            (
                *litellm.input_callback,
                *litellm.success_callback,
                *litellm.failure_callback,
                *litellm._async_success_callback,  # pyright: ignore[reportPrivateUsage]  # callback registry has no public accessor
                *litellm._async_failure_callback,  # pyright: ignore[reportPrivateUsage]  # callback registry has no public accessor
                *success,
                *failure,
            )
        )
    )
    uninitialized: Final = [  # mutable-ok: set_callbacks requires a mutable callback list
        cb
        for cb in configured
        if isinstance(cb, str)
        and (
            cb not in litellm._known_custom_logger_compatible_callbacks  # pyright: ignore[reportPrivateUsage]  # callback compatibility registry has no public accessor
            or cb in litellm.input_callback + litellm.success_callback + litellm.failure_callback
        )
        and cb not in (utils.callback_list or ())
    ]
    if uninitialized:
        set_callbacks(uninitialized, function_id=arguments.get("id"))
        utils.callback_list = list(  # mutable-ok: global callback registry is mutable
            dict.fromkeys((*(utils.callback_list or ()), *uninitialized))
        )  # mutable-ok: global callback registry is mutable
    if litellm_logging.customLogger is None:  # pyright: ignore[reportUnnecessaryComparison]  # runtime plugin registry can be reset to None
        set_callbacks(
            [cb for cb in configured if callable(cb)],  # mutable-ok: set_callbacks requires a mutable callback list
            function_id=arguments.get("id"),  # mutable-ok: set_callbacks requires a mutable callback list
        )  # mutable-ok: set_callbacks requires a mutable callback list
    for event, registered, add_async in (
        ("input", litellm.input_callback, litellm.logging_callback_manager.add_litellm_input_callback),
        ("success", litellm.success_callback, litellm.logging_callback_manager.add_litellm_async_success_callback),
        ("failure", litellm.failure_callback, litellm.logging_callback_manager.add_litellm_async_failure_callback),
    ):
        for cb in tuple(registered):
            if coroutine_checker.is_async_callable(cb) or (event == "success" and cb in ("dynamodb", "openmeter")):
                if cb not in getattr(litellm, f"_async_{event}_callback"):
                    add_async(cb)
                registered.remove(cb)
            elif event != "input" and isinstance(cb, str) and cb in litellm._known_custom_logger_compatible_callbacks:  # pyright: ignore[reportPrivateUsage]  # callback compatibility registry has no public accessor
                utils._add_custom_logger_callback_to_specific_event(cb, event)  # pyright: ignore[reportPrivateUsage]  # callback manager only exposes this internal registration path
    for event, registered, add_sync in (
        ("success", litellm._async_success_callback, litellm.logging_callback_manager.add_litellm_success_callback),  # pyright: ignore[reportPrivateUsage]  # callback registry has no public accessor
        ("failure", litellm._async_failure_callback, litellm.logging_callback_manager.add_litellm_failure_callback),  # pyright: ignore[reportPrivateUsage]  # callback registry has no public accessor
    ):
        for cb in tuple(registered):
            if callable(cb) and not isinstance(cb, CustomLogger) and not coroutine_checker.is_async_callable(cb):
                if cb not in getattr(litellm, f"{event}_callback"):
                    add_sync(cb)
                registered.remove(cb)
    call_id: Final = str(arguments.get("litellm_call_id") or uuid4())
    logger: Final = Logging(
        model=str(arguments["model"]),
        messages="default-message-value",
        stream=False,
        call_type=f"a{route}" if asynchronous else route,
        start_time=datetime.now(),  # noqa: DTZ005  # Logging preserves the legacy naive timestamp contract
        litellm_call_id=call_id,
        function_id=str(arguments.get("id") or ""),
        litellm_trace_id=cast(  # cast-ok: public call argument is validated by Logging
            str | None, arguments.get("litellm_trace_id")
        ),  # cast-ok: public call argument is validated by Logging
        dynamic_input_callbacks=[  # mutable-ok: Logging callback configuration is mutable
            cb for cb in callbacks if cb not in litellm.input_callback and not coroutine_checker.is_async_callable(cb)
        ],
        dynamic_success_callbacks=[  # mutable-ok: Logging callback configuration is mutable
            cb for cb in success if not coroutine_checker.is_async_callable(cb) and cb not in ("dynamodb", "s3")
        ],
        dynamic_async_success_callbacks=[  # mutable-ok: Logging callback configuration is mutable
            cb
            for cb in success
            if coroutine_checker.is_async_callable(cb) or isinstance(cb, CustomLogger) or cb in ("dynamodb", "s3")
        ],
        dynamic_failure_callbacks=[  # mutable-ok: Logging callback configuration is mutable
            cb for cb in failure if not coroutine_checker.is_async_callable(cb)
        ],  # mutable-ok: Logging callback configuration is mutable
        dynamic_async_failure_callbacks=[  # mutable-ok: Logging callback configuration is mutable
            cb for cb in failure if coroutine_checker.is_async_callable(cb) or isinstance(cb, CustomLogger)
        ],
        kwargs=arguments,
        supports_correlation_logging=asynchronous,
    )
    logger.dynamic_input_callbacks = [  # mutable-ok: remove callbacks promoted to the global registry
        cb for cb in dict.fromkeys(logger.dynamic_input_callbacks or ()) if cb not in litellm.input_callback
    ]
    arguments["litellm_call_id"] = call_id
    arguments[LOGGING_OBJECT_KEY] = logger
    return logger


def invoke_terminal(
    action: TerminalAction,
    roots: object,
    logger: object,
    record: Mapping[str, object] | None,
    value: object,
    fallback_start_time: datetime,
    fallback_end_time: datetime,
) -> object:
    from litellm import utils
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    if not isinstance(logger, Logging):
        raise TypeError(f"expected Logging, got {type(logger).__name__}")
    logging: Final = logger
    timing_value: Final = record.get("timing") if record is not None else None
    timing: Final = timing_value if isinstance(timing_value, Mapping) else None
    start_value: Final = timing.get("start_time") if timing is not None else None
    end_value: Final = timing.get("end_time") if timing is not None else None
    start_time: Final = (
        datetime.fromtimestamp(start_value, tz=fallback_start_time.tzinfo)
        if isinstance(start_value, (int, float))
        else fallback_start_time
    )
    end_time: Final = (
        datetime.fromtimestamp(end_value, tz=fallback_end_time.tzinfo)
        if isinstance(end_value, (int, float))
        else fallback_end_time
    )
    if action == "sync_success":

        def run() -> None:
            _retained: Final = roots
            logging.success_handler(value, start_time, end_time)

        return utils.executor.submit(copy_context().run, run)
    if action == "async_success":

        async def run_async() -> None:
            _retained: Final = roots
            await logging.async_success_handler(value, start_time, end_time)

        def enqueue() -> None:
            GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(async_coroutine=run_async())

        if getattr(logging, "_defer_async_logging", False) is True:
            logging._enqueue_deferred_logging = enqueue  # pyright: ignore[reportPrivateUsage]  # preserves Logging's deferred callback contract
        else:
            enqueue()
        return None
    if action == "sync_success_if_needed":
        if logging._should_run_sync_callbacks_for_async_calls():  # pyright: ignore[reportPrivateUsage]  # preserves Logging's async callback policy

            def run() -> None:
                _retained: Final = roots
                logging.success_handler(value, start_time, end_time)

            return utils.executor.submit(copy_context().run, run)
        return None
    exception: Final = cast(Exception, value)  # cast-ok: Rust routes terminal failure values as Python exceptions
    trace: Final = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    if action == "sync_failure":
        logging.failure_handler(exception, trace, start_time, end_time)
        return None
    return logging.async_failure_handler(exception, trace, start_time, end_time)


class _OcrLifecycle(NativeLifecycle, Protocol):
    def identity(self) -> tuple[str, str | None]: ...


class _OcrBindings(NativeLifecycleBindings, Protocol):
    Lifecycle: Callable[
        [dict[str, object], object | None, bool, bool], _OcrLifecycle
    ]  # mutable-ok: native bridge retains and updates Python argument objects
    prepare: Callable[
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

    def prepare(self) -> None:
        if self.logger is None:
            raise RuntimeError("OCR logging was not initialized")
        self.state = self.bindings.prepare(self.current, self.logger, self.asynchronous)

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

    def async_failure(self) -> object:
        return self.terminal("async_failure", self.error)

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

"""Whole-call argument boundary for native OCR execution."""

from __future__ import annotations

import traceback
from collections.abc import Awaitable
from contextvars import copy_context
from datetime import datetime
from typing import Final, Protocol, cast  # noqa: TID251  # native extension exposes dynamically typed callables
from uuid import uuid4

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.bindings import NativeBinding


class RustOcr(Protocol):
    def __call__(self, arguments: dict[str, object]) -> OCRResponse: ...


class RustAocr(Protocol):
    def __call__(self, arguments: dict[str, object]) -> Awaitable[OCRResponse]: ...


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


def ocr(arguments: dict[str, object]) -> OCRResponse:
    implementation: Final = load_rust_ocr()
    if implementation is None:
        raise RuntimeError("Rust OCR is enabled but the native OCR extension is unavailable")
    return implementation(arguments)


async def aocr(arguments: dict[str, object]) -> OCRResponse:
    implementation: Final = load_rust_aocr()
    if implementation is None:
        raise RuntimeError("Rust OCR is enabled but the native OCR extension is unavailable")
    return await implementation(arguments)


def initialize_logging(arguments: dict[str, object], asynchronous: bool, route: str = "ocr") -> object:
    import litellm
    from litellm import utils
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils import litellm_logging
    from litellm.litellm_core_utils.coroutine_checker import coroutine_checker
    from litellm.litellm_core_utils.litellm_logging import Logging, set_callbacks

    supplied: Final = arguments.get("litellm_logging_obj")
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
                    list, arguments.get("success_callback") or []
                ),  # cast-ok: per-call callback list is a legacy untyped boundary
            )  # cast-ok: per-call callback list is a legacy untyped boundary
        )  # cast-ok: per-call callback list is a legacy untyped boundary  # mutable-ok: deduplication uses dict keys
    )
    failure: Final = tuple(  # cast-ok: per-call callback list is a legacy untyped boundary
        dict.fromkeys(
            (
                *callbacks,
                *cast(  # cast-ok: per-call callback list is a legacy untyped boundary
                    list, arguments.get("failure_callback") or []
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
        and cb
        not in (utils.callback_list or [])  # mutable-ok: empty list normalizes an uninitialized callback registry
    ]
    if uninitialized:
        set_callbacks(uninitialized, function_id=arguments.get("id"))
        utils.callback_list = list(  # mutable-ok: global callback registry is mutable
            dict.fromkeys((*(utils.callback_list or []), *uninitialized))
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
        cb for cb in dict.fromkeys(logger.dynamic_input_callbacks or []) if cb not in litellm.input_callback
    ]
    arguments["litellm_call_id"] = call_id
    arguments["litellm_logging_obj"] = logger
    return logger


def invoke_terminal(
    action: str,
    roots: object,
    logger: object,
    record: dict[str, object] | None,
    value: object,
    fallback_start_time: datetime,
    fallback_end_time: datetime,
) -> object:
    from litellm import utils
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    logging: Final = cast(Logging, logger)  # cast-ok: Rust passes the logger returned by initialize_logging
    timing: Final = cast(dict[str, object], record["timing"]) if record is not None else None
    start_time: Final = (
        datetime.fromtimestamp(cast(float, timing["start_time"])) if timing is not None else fallback_start_time
    )
    end_time: Final = (
        datetime.fromtimestamp(cast(float, timing["end_time"])) if timing is not None else fallback_end_time
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
            return invoke_terminal("sync_success", roots, logger, record, value, fallback_start_time, fallback_end_time)
        return None
    exception: Final = cast(Exception, value)  # cast-ok: Rust routes terminal failure values as Python exceptions
    trace: Final = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    if action == "sync_failure":
        logging.failure_handler(exception, trace, start_time, end_time)
        return None
    return logging.async_failure_handler(exception, trace, start_time, end_time)

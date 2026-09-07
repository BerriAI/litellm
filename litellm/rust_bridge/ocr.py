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


def initialize_logging(arguments: dict[str, object], asynchronous: bool) -> object:
    import litellm
    from litellm import utils
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils import litellm_logging
    from litellm.litellm_core_utils.coroutine_checker import coroutine_checker
    from litellm.litellm_core_utils.litellm_logging import Logging, set_callbacks

    supplied: Final = arguments.get("litellm_logging_obj")
    if supplied is not None:
        return supplied
    callbacks: Final = tuple(dict.fromkeys(utils.get_dynamic_callbacks(cast(list, arguments.get("callbacks")))))
    success: Final = tuple(dict.fromkeys((*callbacks, *cast(list, arguments.get("success_callback") or []))))
    failure: Final = tuple(dict.fromkeys((*callbacks, *cast(list, arguments.get("failure_callback") or []))))
    configured: Final = tuple(
        dict.fromkeys(
            (
                *litellm.input_callback,
                *litellm.success_callback,
                *litellm.failure_callback,
                *litellm._async_success_callback,
                *litellm._async_failure_callback,
                *success,
                *failure,
            )
        )
    )
    uninitialized: Final = [
        cb
        for cb in configured
        if isinstance(cb, str)
        and (
            cb not in litellm._known_custom_logger_compatible_callbacks
            or cb in litellm.input_callback + litellm.success_callback + litellm.failure_callback
        )
        and cb not in (utils.callback_list or [])
    ]
    if uninitialized:
        set_callbacks(uninitialized, function_id=arguments.get("id"))
        utils.callback_list = list(dict.fromkeys((*(utils.callback_list or []), *uninitialized)))
    if litellm_logging.customLogger is None:
        set_callbacks([cb for cb in configured if callable(cb)], function_id=arguments.get("id"))
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
            elif event != "input" and isinstance(cb, str) and cb in litellm._known_custom_logger_compatible_callbacks:
                utils._add_custom_logger_callback_to_specific_event(cb, event)
    for event, registered, add_sync in (
        ("success", litellm._async_success_callback, litellm.logging_callback_manager.add_litellm_success_callback),
        ("failure", litellm._async_failure_callback, litellm.logging_callback_manager.add_litellm_failure_callback),
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
        call_type="aocr" if asynchronous else "ocr",
        start_time=datetime.now(),
        litellm_call_id=call_id,
        function_id=str(arguments.get("id") or ""),
        litellm_trace_id=cast(str | None, arguments.get("litellm_trace_id")),
        dynamic_input_callbacks=[
            cb for cb in callbacks if cb not in litellm.input_callback and not coroutine_checker.is_async_callable(cb)
        ],
        dynamic_success_callbacks=[
            cb for cb in success if not coroutine_checker.is_async_callable(cb) and cb not in ("dynamodb", "s3")
        ],
        dynamic_async_success_callbacks=[
            cb
            for cb in success
            if coroutine_checker.is_async_callable(cb) or isinstance(cb, CustomLogger) or cb in ("dynamodb", "s3")
        ],
        dynamic_failure_callbacks=[cb for cb in failure if not coroutine_checker.is_async_callable(cb)],
        dynamic_async_failure_callbacks=[
            cb for cb in failure if coroutine_checker.is_async_callable(cb) or isinstance(cb, CustomLogger)
        ],
        kwargs=arguments,
        supports_correlation_logging=asynchronous,
    )
    logger.dynamic_input_callbacks = [
        cb for cb in dict.fromkeys(logger.dynamic_input_callbacks or []) if cb not in litellm.input_callback
    ]
    arguments["litellm_call_id"] = call_id
    arguments["litellm_logging_obj"] = logger
    return logger


def invoke_terminal(
    action: str, roots: object, logger: object, value: object, start_time: datetime, end_time: datetime
) -> object:
    from litellm import utils
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    logging: Final = cast(Logging, logger)
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
            logging._enqueue_deferred_logging = enqueue
        else:
            enqueue()
        return None
    if action == "sync_success_if_needed":
        if logging._should_run_sync_callbacks_for_async_calls():
            return invoke_terminal("sync_success", roots, logger, value, start_time, end_time)
        return None
    exception: Final = cast(Exception, value)
    trace: Final = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    if action == "sync_failure":
        logging.failure_handler(exception, trace, start_time, end_time)
        return None
    return logging.async_failure_handler(exception, trace, start_time, end_time)

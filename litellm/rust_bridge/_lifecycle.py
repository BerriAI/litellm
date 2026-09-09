from __future__ import annotations

import traceback
from collections.abc import Awaitable, Callable, Mapping
from contextvars import copy_context
from datetime import datetime
from enum import Enum, IntEnum
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    cast,  # noqa: TID251  # retained callback boundary accepts legacy logger interfaces
    overload,
)
from uuid import uuid4

from pydantic import InstanceOf, TypeAdapter

if TYPE_CHECKING:
    from litellm.types.utils import CallTypes


LOGGING_OBJECT_KEY: Final = "litellm_logging_obj"
FALLBACKS_KEY: Final = "fallbacks"
TerminalAction = Literal[
    "sync_success",
    "async_success",
    "sync_success_if_needed",
    "sync_failure",
    "async_failure",
]
_OPTIONAL_ARGUMENTS_ADAPTER: Final[TypeAdapter[InstanceOf[dict[str, object]] | None]] = TypeAdapter(
    InstanceOf[dict[str, object]] | None
)  # mutable-ok: native bridge retains and updates Python argument objects


LIFECYCLE_OWNER_KEY: Final = "_rust_lifecycle_owner"
LIFECYCLE_STARTED_KEY: Final = "_rust_lifecycle_started"


class LifecycleOwner(str, Enum):
    BRIDGE = "bridge"
    WRAPPER = "wrapper"


def build_call_arguments(
    request_arguments: Mapping[str, object] | None,
    route_arguments: Mapping[str, object],
    *,
    logging_obj: object | None = None,
    litellm_params: Mapping[str, object] | None = None,
    lifecycle_owner: LifecycleOwner = LifecycleOwner.BRIDGE,
) -> dict[str, object]:
    return {
        **(litellm_params if litellm_params is not None else {}),
        **(request_arguments if request_arguments is not None else {}),
        **route_arguments,
        **({LOGGING_OBJECT_KEY: logging_obj} if logging_obj is not None else {}),
        LIFECYCLE_OWNER_KEY: lifecycle_owner.value,
    }


@overload
def map_native_error(error: None, arguments: Mapping[str, object], route: str) -> None: ...


@overload
def map_native_error(error: BaseException, arguments: Mapping[str, object], route: str) -> BaseException: ...


def map_native_error(error: BaseException | None, arguments: Mapping[str, object], route: str) -> BaseException | None:
    from litellm.rust_bridge.bindings import native_exception_types
    from litellm.rust_bridge.runtime import BridgeErrorContext, upstream_error

    exceptions: Final = native_exception_types()
    if error is None or exceptions is None or not isinstance(error, exceptions[1]):
        return error
    return upstream_error(
        error,
        BridgeErrorContext(
            route=route,
            provider=str(arguments.get("custom_llm_provider") or ""),
            model=str(arguments.get("model") or ""),
        ),
    )


def owns_lifecycle(arguments: Mapping[str, object]) -> bool:
    return arguments.get(LIFECYCLE_OWNER_KEY, LifecycleOwner.BRIDGE.value) == LifecycleOwner.BRIDGE.value


class NativeOutcome(IntEnum):
    SUCCESS = 0
    FAILURE = 1
    ABORT = 2


class NativeLifecycle(Protocol):
    def complete(self) -> bool | None: ...

    def advance(self, outcome: int, logger_available: bool, has_fallbacks: bool) -> bool: ...


class NativeLifecycleBindings(Protocol):
    invoke: Callable[[NativeLifecycle, object], tuple[bool, object]]


class LifecycleHost(Protocol):
    @property
    def machine(self) -> NativeLifecycle: ...

    def invoke(self) -> tuple[bool, object]: ...

    def advance(self, outcome: NativeOutcome, error: BaseException | None = None) -> None: ...

    def result(self) -> object: ...


class MutableLifecycleHost(LifecycleHost, Protocol):
    arguments: dict[str, object]  # mutable-ok: native bridge retains and updates Python argument objects
    current: dict[str, object]  # mutable-ok: native bridge retains and updates Python argument objects
    logger: object | None
    response: object
    error: BaseException | None
    end: datetime | None


def advance_host(host: MutableLifecycleHost, outcome: NativeOutcome, error: BaseException | None) -> None:
    if error is not None and host.end is None:
        host.end = datetime.now()  # noqa: DTZ005  # lifecycle timestamps preserve Logging's naive timestamp contract
    if host.logger is None:
        host.logger = host.arguments.get(LOGGING_OBJECT_KEY)
    replace: Final = host.machine.advance(
        outcome,
        host.logger is not None,
        host.current.get(FALLBACKS_KEY) is not None,
    )
    if replace:
        host.error = error


def host_result(host: MutableLifecycleHost) -> object:
    if host.machine.complete():
        return host.response
    if host.error is None:
        raise RuntimeError("native lifecycle failed without an error")
    raise host.error


async def deployment_pre(
    arguments: dict[str, object], call_type: str
) -> dict[str, object]:  # mutable-ok: native bridge retains and updates Python argument objects
    from litellm import utils

    modified: Final = _OPTIONAL_ARGUMENTS_ADAPTER.validate_python(
        await utils.async_pre_call_deployment_hook(arguments, call_type)
    )
    return arguments if modified is None else modified


async def deployment_success(
    arguments: dict[str, object], response: object, call_type: CallTypes
) -> object:  # mutable-ok: native bridge retains and updates Python argument objects
    from litellm import utils

    updated: object = await utils.async_post_call_success_deployment_hook(  # pyright: ignore[reportUnknownMemberType]  # legacy hook annotations expose an unknown return
        arguments, response, call_type
    )
    return updated


async def deployment_failure(
    arguments: dict[str, object], error: BaseException | None, call_type: str
) -> None:  # mutable-ok: native bridge retains and updates Python argument objects
    from litellm import utils

    if not isinstance(error, Exception):
        raise TypeError("native lifecycle failure did not retain an exception")
    await utils.async_post_call_failure_deployment_hook(arguments, error, call_type)


def restore_correlation_context(logger: object | None) -> None:
    from litellm import utils

    utils._restore_correlation_context_if_supported(logger)  # pyright: ignore[reportPrivateUsage]  # lifecycle cleanup has no public wrapper


def _invoke_sync(host: LifecycleHost) -> None:
    awaiting, _ = host.invoke()
    if awaiting:
        raise RuntimeError("synchronous lifecycle selected an awaited operation")


async def _invoke_async(host: LifecycleHost) -> None:
    awaiting, value = host.invoke()
    if not awaiting:
        return
    if not isinstance(value, Awaitable):
        raise TypeError("native lifecycle operation did not return an awaitable")
    await value


def drive_sync(host: LifecycleHost) -> object:
    while host.machine.complete() is None:
        try:
            _invoke_sync(host)
        except Exception as error:  # noqa: BLE001  # lifecycle converts every ordinary host failure into a machine outcome
            host.advance(NativeOutcome.FAILURE, error)
        except BaseException as error:  # noqa: BLE001  # cancellation and interrupts must advance the machine as aborts
            host.advance(NativeOutcome.ABORT, error)
        else:
            host.advance(NativeOutcome.SUCCESS)
    return host.result()


async def drive_async(host: LifecycleHost) -> object:
    while host.machine.complete() is None:
        try:
            await _invoke_async(host)
        except Exception as error:  # noqa: BLE001  # lifecycle converts every ordinary host failure into a machine outcome
            host.advance(NativeOutcome.FAILURE, error)
        except BaseException as error:  # noqa: BLE001  # cancellation and interrupts must advance the machine as aborts
            host.advance(NativeOutcome.ABORT, error)
        else:
            host.advance(NativeOutcome.SUCCESS)
    return host.result()


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

    logging: Final = cast(  # cast-ok: callers may supply a Logging-compatible test or plugin implementation
        Logging, logger
    )
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

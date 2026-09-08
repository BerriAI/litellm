from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from enum import IntEnum
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import TypeAdapter

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
_OPTIONAL_ARGUMENTS_ADAPTER: Final[TypeAdapter[dict[str, object] | None]] = TypeAdapter(
    dict[str, object] | None
)  # mutable-ok: native bridge retains and updates Python argument objects


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
    host.invoke()


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
    arguments: dict[str, object], asynchronous: bool, route: str
) -> object:  # mutable-ok: native bridge retains and updates Python argument objects
    from litellm.rust_bridge.ocr import initialize_logging as initialize_ocr_logging

    return initialize_ocr_logging(arguments, asynchronous, route)


def invoke_terminal(
    action: TerminalAction,
    roots: object,
    logger: object,
    record: Mapping[str, object] | None,
    value: object,
    start_time: datetime,
    end_time: datetime,
) -> object:
    from litellm.rust_bridge.ocr import invoke_terminal as invoke_ocr_terminal

    return invoke_ocr_terminal(action, roots, logger, record, value, start_time, end_time)

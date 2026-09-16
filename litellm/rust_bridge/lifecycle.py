from __future__ import annotations

import datetime
import os
import uuid
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # bounded compatibility calls into legacy Python integrations
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging


@dataclass(frozen=True, slots=True)
class Await:
    awaitable: Awaitable[object]


@dataclass(frozen=True, slots=True)
class Complete:
    value: object


class Execution(Protocol):
    def start(self) -> Await | Complete: ...

    def resume_value(self, value: object) -> Await | Complete: ...

    def resume_error(self, error: BaseException) -> Await | Complete: ...

    def close(self) -> None: ...


async def drive(execution: Execution) -> object:
    try:
        step = execution.start()  # rebind-ok: the execution protocol advances after each selected await
        while isinstance(step, Await):
            try:
                value = await step.awaitable  # rebind-ok: each selected await produces the next protocol input
            except GeneratorExit:
                raise
            except BaseException as error:
                step = execution.resume_error(error)  # rebind-ok: advance the execution protocol
            else:
                step = execution.resume_value(value)  # rebind-ok: advance the execution protocol
        return step.value
    finally:
        execution.close()


class MetadataUpdater(Protocol):
    def __call__(
        self,
        result: object,
        logging_obj: Logging,
        model: str | None,
        kwargs: dict[str, object],
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CallSetup:
    logger: Logging
    kwargs: dict[str, object]


def setup(
    call_type: str,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    start_time: datetime.datetime,
    asynchronous: bool,
) -> CallSetup:
    from litellm import utils
    from litellm.litellm_core_utils.litellm_logging import Logging

    arguments: Final = {  # mutable-ok: function_setup consumes an owned kwargs dict
        "litellm_call_id": str(uuid.uuid4()),
        **kwargs,
    }
    supplied: Final = arguments.get("litellm_logging_obj")
    if isinstance(supplied, Logging):
        supplied._native_callback_fast_path = False  # pyright: ignore[reportPrivateUsage]  # supplied loggers retain all dispatch contracts
        return CallSetup(supplied, arguments)
    logger, prepared = utils.function_setup(
        call_type, utils.Rules(), start_time, *args, is_async_call=asynchronous, **arguments
    )
    if type(logger) is Logging and call_type in ("ocr", "aocr"):
        logger._native_callback_fast_path = True  # pyright: ignore[reportPrivateUsage]  # only bridge-created OCR loggers opt into callback elision
    return CallSetup(logger, prepared)


def check_limits(kwargs: Mapping[str, object]) -> None:
    import litellm
    from litellm.litellm_core_utils.core_helpers import max_retries_per_request_hit

    current_cost: Final = litellm._current_cost  # pyright: ignore[reportPrivateUsage]  # shared SDK budget counter has no public accessor
    if litellm.max_budget and current_cost > litellm.max_budget:
        raise litellm.BudgetExceededError(current_cost=current_cost, max_budget=litellm.max_budget)
    if max_retries_per_request_hit(kwargs, litellm.num_retries_per_request):
        raise RuntimeError("Max retries per request hit!")


def finalize(
    response: object,
    logger: Logging,
    kwargs: dict[str, object],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> None:
    from litellm.litellm_core_utils.llm_response_utils import response_metadata

    model: Final = kwargs.get("model")
    update: Final = cast(  # cast-ok: legacy metadata function accepts concrete kwargs
        MetadataUpdater, response_metadata.update_response_metadata
    )
    update(response, logger, model if isinstance(model, str) else None, kwargs, start_time, end_time)


def deployment_callbacks_needed() -> bool:
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    return any(isinstance(callback, CustomLogger) for callback in litellm.callbacks)


def callbacks_needed(logger: Logging, phase: str) -> bool:
    import litellm
    from litellm._logging import (
        _is_debugging_on,  # pyright: ignore[reportPrivateUsage]  # use the same debug gate as Logging
    )

    if (
        _is_debugging_on()
        or getattr(logger, "litellm_request_debug", False)
        or os.getenv("LITELLM_PRINT_STANDARD_LOGGING_PAYLOAD")
    ):
        return True
    input_needed: Final = bool(
        litellm.input_callback
        or litellm._async_input_callback  # pyright: ignore[reportPrivateUsage]  # live async registries have no public accessor
        or logger.dynamic_input_callbacks
        or callable(getattr(logger, "logger_fn", None))
        or logger.log_raw_request_response
        or litellm.log_raw_request_response
    )
    match phase:
        case "input":
            return input_needed
        case "sync_success":
            return bool(litellm.success_callback or logger.dynamic_success_callbacks)
        case "sync_success_async":
            return bool(
                (litellm.success_callback or logger.dynamic_success_callbacks)
                and logger._should_run_sync_callbacks_for_async_calls()  # pyright: ignore[reportPrivateUsage]  # preserve async call filtering of sync callbacks
            )
        case "async_success":
            return bool(litellm._async_success_callback or logger.dynamic_async_success_callbacks)  # pyright: ignore[reportPrivateUsage]  # live async registries have no public accessor
        case "sync_failure":
            return bool(litellm.failure_callback or logger.dynamic_failure_callbacks)
        case "async_failure":
            return bool(litellm._async_failure_callback or logger.dynamic_async_failure_callbacks)  # pyright: ignore[reportPrivateUsage]  # live async registries have no public accessor
        case "payload":
            return bool(
                input_needed
                or litellm.success_callback
                or litellm.failure_callback
                or litellm._async_success_callback  # pyright: ignore[reportPrivateUsage]  # live async registries have no public accessor
                or litellm._async_failure_callback  # pyright: ignore[reportPrivateUsage]  # live async registries have no public accessor
                or logger.dynamic_success_callbacks
                or logger.dynamic_async_success_callbacks
                or logger.dynamic_failure_callbacks
                or logger.dynamic_async_failure_callbacks
            )
        case _:
            return True


def success_bookkeeping(
    logger: Logging, response: object, start: datetime.datetime, end: datetime.datetime, asynchronous: bool
) -> None:
    phase: Final = "async_success" if asynchronous else "sync_success"
    if logger.should_run_logging(phase):
        logger._success_handler_helper_fn(  # pyright: ignore[reportPrivateUsage]  # retain success bookkeeping without constructing a callback payload
            result=response, start_time=start, end_time=end, build_logging_payload=False
        )
        logger.has_run_logging(phase)


def failure_bookkeeping(
    logger: Logging, error: BaseException, start: datetime.datetime, end: datetime.datetime, asynchronous: bool
) -> None:
    phase: Final = "async_failure" if asynchronous else "sync_failure"
    if logger.should_run_logging(phase):
        logger._failure_handler_helper_fn(  # pyright: ignore[reportPrivateUsage]  # retain failure accounting without formatting an unused traceback or payload
            error, "", start, end, build_logging_payload=False
        )
        logger.has_run_logging(phase)

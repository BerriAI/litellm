from __future__ import annotations

import datetime
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
        return CallSetup(supplied, arguments)
    logger, prepared = utils.function_setup(
        call_type, utils.Rules(), start_time, *args, is_async_call=asynchronous, **arguments
    )
    return CallSetup(logger, prepared)


def check_limits(kwargs: Mapping[str, object]) -> None:
    import litellm

    current_cost: Final = litellm._current_cost  # pyright: ignore[reportPrivateUsage]  # shared SDK budget counter has no public accessor
    if litellm.max_budget and current_cost > litellm.max_budget:
        raise litellm.BudgetExceededError(current_cost=current_cost, max_budget=litellm.max_budget)
    metadata: Final = kwargs.get("metadata")
    if isinstance(metadata, Mapping):
        typed_metadata: Final = cast(  # cast-ok: runtime Mapping check establishes read-only metadata
            Mapping[str, object], metadata
        )
        previous: Final = typed_metadata.get("previous_models")
        if (
            isinstance(previous, list)
            and litellm.num_retries_per_request is not None
            and len(cast(list[object], previous))  # cast-ok: runtime list check establishes the retry history
            >= litellm.num_retries_per_request
        ):
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

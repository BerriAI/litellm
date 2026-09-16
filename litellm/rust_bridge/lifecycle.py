from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol


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


def check_limits(kwargs: Mapping[str, object]) -> None:
    import litellm
    from litellm.litellm_core_utils.core_helpers import max_retries_per_request_hit

    current_cost: Final = litellm._current_cost  # pyright: ignore[reportPrivateUsage]  # shared SDK budget counter has no public accessor
    if litellm.max_budget and current_cost > litellm.max_budget:
        raise litellm.BudgetExceededError(current_cost=current_cost, max_budget=litellm.max_budget)
    if max_retries_per_request_hit(kwargs, litellm.num_retries_per_request):
        raise RuntimeError("Max retries per request hit!")

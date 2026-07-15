"""
Cost tracking for background interactions.

A create request with ``background=true`` returns ``in_progress`` with no
usage block, and GET polls are deliberately never billed (billing them would
double-charge every poll; the GET response also does not echo ``background``,
so a poll cannot be told apart from a re-fetch of an already-billed
interaction). The create call is therefore the only place that can own
billing: it schedules a poll task that fetches the interaction until it
reaches a terminal status and logs the final usage as a single success event
attributed to the original request.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterator, Optional

from litellm._logging import verbose_logger
from litellm.constants import (
    BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS,
    BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS,
    BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS,
    BACKGROUND_INTERACTION_COST_POLLING_ENABLED,
)
from litellm.types.interactions import InteractionsAPIResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "incomplete", "budget_exceeded"})


@dataclass(frozen=True, slots=True)
class BackgroundInteractionPollContext:
    interaction_id: str
    custom_llm_provider: str
    logging_obj: "LiteLLMLoggingObj"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    initial_interval_seconds: float = BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS
    max_interval_seconds: float = BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS
    timeout_seconds: float = BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS


FetchInteraction = Callable[[BackgroundInteractionPollContext], Awaitable[InteractionsAPIResponse]]


async def _fetch_interaction(context: BackgroundInteractionPollContext) -> InteractionsAPIResponse:
    from litellm.interactions import aget

    return await aget(
        interaction_id=context.interaction_id,
        custom_llm_provider=context.custom_llm_provider,
        **{"api_key": context.api_key, "api_base": context.api_base, "no-log": True},
    )


def _poll_intervals(initial: float, maximum: float, timeout: float) -> Iterator[float]:
    elapsed = 0.0
    interval = initial
    while elapsed + interval <= timeout:
        yield interval
        elapsed += interval
        interval = min(interval * 2, maximum)


async def poll_and_log_background_interaction_cost(
    context: BackgroundInteractionPollContext,
    fetch_interaction: FetchInteraction = _fetch_interaction,
) -> None:
    for interval in _poll_intervals(
        initial=context.initial_interval_seconds,
        maximum=context.max_interval_seconds,
        timeout=context.timeout_seconds,
    ):
        await asyncio.sleep(interval)
        try:
            response = await fetch_interaction(context)
        except Exception as e:  # noqa: BLE001  # any fetch error must not kill the billing poll loop
            verbose_logger.debug(
                "Background interaction cost poll for %s failed, will retry: %s",
                context.interaction_id,
                e,
            )
            continue
        if response.status not in _TERMINAL_STATUSES:
            continue
        if response.usage is not None:
            await context.logging_obj.async_log_background_interaction_completion(result=response)
        return
    verbose_logger.warning(
        "Gave up cost polling for background interaction %s after %ss; its usage will not be tracked",
        context.interaction_id,
        context.timeout_seconds,
    )


_ACTIVE_POLL_TASKS: set["asyncio.Task[None]"] = set()  # mutable-ok: asyncio requires strong refs to running tasks


def maybe_schedule_background_interaction_cost_polling(
    response: Any,
    create_kwargs: dict[str, Any],
    custom_llm_provider: str,
) -> Optional["asyncio.Task[None]"]:
    from litellm.litellm_core_utils.litellm_logging import Logging

    if not BACKGROUND_INTERACTION_COST_POLLING_ENABLED:
        return None
    if not isinstance(response, InteractionsAPIResponse):
        return None
    if response.status != "in_progress" or not response.id:
        return None
    logging_obj = create_kwargs.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return None
    context = BackgroundInteractionPollContext(
        interaction_id=response.id,
        custom_llm_provider=custom_llm_provider,
        logging_obj=logging_obj,
        api_key=create_kwargs.get("api_key"),
        api_base=create_kwargs.get("api_base"),
    )
    task = asyncio.create_task(poll_and_log_background_interaction_cost(context))
    _ACTIVE_POLL_TASKS.add(task)
    task.add_done_callback(_ACTIVE_POLL_TASKS.discard)
    return task

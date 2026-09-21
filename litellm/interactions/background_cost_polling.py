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

``requires_action`` is terminal for the interaction it names. The API has no
operation that resumes one: a caller answers a tool request by creating a new
interaction whose ``previous_interaction_id`` points at it, and that new
interaction bills itself. The paused interaction keeps the tokens it already
spent producing the tool request, so it is billed and settled where it stops
rather than polled until the timeout, which would both lose that usage and
hold its budget reservation open for the whole timeout window.

Deleting an interaction makes every subsequent poll fail, which would let a
caller retrieve the completed output themselves and then delete it before the
poll task settles, leaving the work unbilled and the budget reservation
refunded at the poll timeout. ``adelete`` therefore settles any pending poll
for the interaction before dispatching the delete: it fetches the current
state with the create's credentials, bills it if it is terminal with usage,
and releases the reservation otherwise. A settlement gate on the create's
logging object makes the poll task and the delete path mutually exclusive, so
the interaction is billed exactly once no matter who settles first.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

from litellm._logging import verbose_logger
from litellm.constants import (
    BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS,
    BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS,
    BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS,
    BACKGROUND_INTERACTION_COST_POLLING_ENABLED,
)
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.types.interactions import InteractionsAPIResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_TERMINAL_STATUSES: Final = frozenset(
    {"completed", "failed", "cancelled", "incomplete", "budget_exceeded", "requires_action"}
)

_POLLABLE_STATUSES: Final = frozenset({"in_progress", "queued"})

_STATUSES_THAT_PRODUCED_OUTPUT: Final = frozenset({"completed", "requires_action"})


@dataclass(frozen=True, slots=True)
class BackgroundInteractionPollContext:
    interaction_id: str
    custom_llm_provider: str
    logging_obj: "LiteLLMLoggingObj"
    api_key: str | None = None
    api_base: str | None = None
    initial_interval_seconds: float = BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS
    max_interval_seconds: float = BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS
    timeout_seconds: float = BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS


FetchInteraction: TypeAlias = Callable[[BackgroundInteractionPollContext], Awaitable[InteractionsAPIResponse]]


async def _fetch_interaction(context: BackgroundInteractionPollContext) -> InteractionsAPIResponse:
    from litellm.interactions import aget

    return await aget(
        interaction_id=context.interaction_id,
        custom_llm_provider=context.custom_llm_provider,
        api_key=context.api_key,
        api_base=context.api_base,
        **{
            "no-log": True
        },  # mutable-ok: "no-log" is not a valid identifier, so it can only be passed through a mapping
    )


def _poll_intervals(initial: float, maximum: float, timeout: float) -> Iterator[float]:
    elapsed = 0.0
    interval = initial
    while interval > 0 and elapsed + interval <= timeout:
        yield interval
        elapsed += interval
        interval = min(interval * 2, maximum)


_SETTLED_KEY = "background_interaction_settled"


def _is_settled(logging_obj: "LiteLLMLoggingObj") -> bool:
    return logging_obj.model_call_details.get(_SETTLED_KEY) is True


def _claim_settlement(logging_obj: "LiteLLMLoggingObj") -> bool:
    """
    Exactly-once gate between the poll task and the delete-time settlement:
    both run on the same event loop and neither awaits between reading and
    setting the flag, so whichever claims first owns billing or release.
    """
    if _is_settled(logging_obj):
        return False
    logging_obj.model_call_details[_SETTLED_KEY] = True  # rebind-ok: both settlers must see the same settlement flag
    return True


async def poll_and_log_background_interaction_cost(
    context: BackgroundInteractionPollContext,
    fetch_interaction: FetchInteraction = _fetch_interaction,
) -> None:
    last_seen_status: str | None = None
    for interval in _poll_intervals(
        initial=context.initial_interval_seconds,
        maximum=context.max_interval_seconds,
        timeout=context.timeout_seconds,
    ):
        await asyncio.sleep(interval)
        if _is_settled(context.logging_obj):
            return
        try:
            response = await fetch_interaction(context)
        except Exception as e:  # noqa: BLE001  # any fetch error must not kill the billing poll loop
            verbose_logger.debug(
                "Background interaction cost poll for %s failed, will retry: %s",
                context.interaction_id,
                e,
            )
            continue
        last_seen_status = response.status
        if response.status not in _TERMINAL_STATUSES:
            continue
        if not _claim_settlement(context.logging_obj):
            return
        if response.usage is not None:
            await _bill_settled_interaction(logging_obj=context.logging_obj, response=response)
        else:
            await _release_open_budget_reservation(logging_obj=context.logging_obj)
        return
    if not _claim_settlement(context.logging_obj):
        return
    if last_seen_status is not None and last_seen_status not in _POLLABLE_STATUSES:
        verbose_logger.error(
            "Gave up cost polling for background interaction %s after %ss: its last status %r is in neither "
            "the pollable nor the terminal set, so this proxy never learned how to settle it and its usage "
            "will not be tracked",
            context.interaction_id,
            context.timeout_seconds,
            last_seen_status,
        )
    else:
        verbose_logger.warning(
            "Gave up cost polling for background interaction %s after %ss; its usage will not be tracked",
            context.interaction_id,
            context.timeout_seconds,
        )
    await _release_open_budget_reservation(logging_obj=context.logging_obj)


async def _release_open_budget_reservation(logging_obj: "LiteLLMLoggingObj") -> None:
    """
    The proxy keeps the pre-call budget reservation open for an in-progress
    background interaction so concurrent creates cannot stack past the budget.
    The completion success event reconciles it to the actual cost; when the
    interaction terminates without billable usage (or polling gives up, or it
    is deleted before settling), no such event fires, so whoever claims the
    settlement must release the reservation here or the spend counters stay
    pinned at the estimated cost.
    """
    metadata = get_litellm_metadata_from_kwargs(kwargs=logging_obj.model_call_details)
    budget_reservation = metadata.get("user_api_key_budget_reservation")
    if not isinstance(budget_reservation, dict):
        return

    from litellm.proxy.spend_tracking.budget_reservation import release_budget_reservation

    try:
        await release_budget_reservation(budget_reservation=budget_reservation)
    except Exception:  # noqa: BLE001  # a failed release must not crash the poll task; counters expire via TTL
        verbose_logger.exception("Failed to release budget reservation for an unbilled background interaction")


async def _bill_settled_interaction(logging_obj: "LiteLLMLoggingObj", response: InteractionsAPIResponse) -> None:
    """
    Claiming the settlement makes the claimer solely responsible for the
    reservation, and no one retries a claim that is already set. A billing
    failure here must therefore release the reservation on its way out, or it
    stays pinned at the estimated cost until the whole poll times out.
    """
    try:
        await logging_obj.async_log_background_interaction_completion(result=response)
    except Exception:
        await _release_open_budget_reservation(logging_obj=logging_obj)
        raise


def is_pollable_background_interaction(response: InteractionsAPIResponse) -> bool:
    """
    The single gate deciding whether a create's response gets a poll task.
    The proxy's success callback defers releasing the budget reservation for
    exactly these responses, on the promise that a poll task will settle them,
    so a response one site accepts and the other refuses strands its
    reservation on the spend counters with nothing left to reconcile it.

    ``queued`` belongs here alongside ``in_progress``. It is the API's
    not-started-yet state, so it reaches a terminal status the same way and
    needs polling for the same reason: nothing else in the proxy ever bills a
    create that came back without usage, so a status missing from both this
    set and ``_TERMINAL_STATUSES`` is billed nowhere and alerts nobody.
    """
    return response.status in _POLLABLE_STATUSES and bool(response.id)


def missing_usage_is_expected(response: InteractionsAPIResponse) -> bool:
    """
    Whether a response arriving with no usage block is a normal outcome rather
    than lost billing data. An interaction that is still running, or that
    stopped at ``failed``, ``cancelled``, ``incomplete`` or ``budget_exceeded``,
    has nothing to charge for and should not raise a cost-tracking alarm.

    ``completed`` and ``requires_action`` both mean the model produced output,
    so a usage block is always expected with them. If one arrives without it
    the charge for real work has been lost, which is precisely what the
    proxy's cost-tracking alert exists to surface.
    """
    return response.status not in _STATUSES_THAT_PRODUCED_OUTPUT


@dataclass(frozen=True, slots=True)
class _ActiveBackgroundPoll:
    task: "asyncio.Task[None]"
    context: BackgroundInteractionPollContext


_ACTIVE_POLLS: dict[str, _ActiveBackgroundPoll] = {}  # mutable-ok: asyncio needs strong refs to running poll tasks


def _discard_poll(interaction_id: str, task: "asyncio.Task[None]") -> None:
    entry = _ACTIVE_POLLS.get(interaction_id)
    if entry is not None and entry.task is task:
        del _ACTIVE_POLLS[interaction_id]


def maybe_schedule_background_interaction_cost_polling(
    response: object,
    create_kwargs: Mapping[str, object],
    custom_llm_provider: str,
) -> "asyncio.Task[None] | None":
    from litellm.litellm_core_utils.litellm_logging import Logging

    if not BACKGROUND_INTERACTION_COST_POLLING_ENABLED:
        return None
    if not isinstance(response, InteractionsAPIResponse):
        return None
    if not is_pollable_background_interaction(response):
        return None
    logging_obj = create_kwargs.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return None
    api_key = create_kwargs.get("api_key")
    api_base = create_kwargs.get("api_base")
    context = BackgroundInteractionPollContext(
        interaction_id=response.id,
        custom_llm_provider=custom_llm_provider,
        logging_obj=logging_obj,
        api_key=api_key if isinstance(api_key, str) else None,
        api_base=api_base if isinstance(api_base, str) else None,
    )
    task = asyncio.create_task(poll_and_log_background_interaction_cost(context))
    _ACTIVE_POLLS[context.interaction_id] = _ActiveBackgroundPoll(task=task, context=context)
    task.add_done_callback(
        lambda finished, interaction_id=context.interaction_id: _discard_poll(interaction_id, finished)
    )
    return task


async def maybe_settle_background_interaction_before_delete(
    interaction_id: str,
    fetch_interaction: FetchInteraction = _fetch_interaction,
) -> None:
    entry = _ACTIVE_POLLS.get(interaction_id)
    if entry is None:
        return
    context = entry.context
    try:
        response = await fetch_interaction(context)
    except Exception as e:  # noqa: BLE001  # unfetchable pre-delete state settles by releasing the reservation
        verbose_logger.debug(
            "Could not fetch background interaction %s before delete, releasing its reservation: %s",
            interaction_id,
            e,
        )
        if _claim_settlement(context.logging_obj):
            await _release_open_budget_reservation(logging_obj=context.logging_obj)
        return
    if not _claim_settlement(context.logging_obj):
        return
    if response.status in _TERMINAL_STATUSES and response.usage is not None:
        await _bill_settled_interaction(logging_obj=context.logging_obj, response=response)
        return
    await _release_open_budget_reservation(logging_obj=context.logging_obj)

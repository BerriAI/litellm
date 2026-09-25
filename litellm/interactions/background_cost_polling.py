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
state, bills it if it is terminal with usage, and releases the reservation
otherwise.

The poll task lives in the process that served the create, so a delete
served by another replica, or by the same replica after a restart, finds no
task to settle. A ``BackgroundSettlementStore`` makes the pending settlement
durable across processes: the create registers the request context that
billing needs (never provider credentials), the settlement is claimed
exactly once through the store, and a delete on any replica rebuilds the
billing context from the store when the poll task is not local. Rows left
unclaimed by a process that died are resumed at startup. The default store is
in-memory, which keeps the SDK and single-process behavior unchanged; the
proxy installs a database-backed one.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, to_jsonable_python

from litellm._logging import verbose_logger
from litellm.constants import (
    BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS,
    BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS,
    BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS,
    BACKGROUND_INTERACTION_COST_POLLING_ENABLED,
)
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.types.interactions import InteractionsAPIResponse
from litellm.types.utils import CustomPricingLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_TERMINAL_STATUSES: Final = frozenset(
    {"completed", "failed", "cancelled", "incomplete", "budget_exceeded", "requires_action"}
)

_POLLABLE_STATUSES: Final = frozenset({"in_progress", "queued"})

_STATUSES_THAT_PRODUCED_OUTPUT: Final = frozenset({"completed", "requires_action"})

SettlementOutcome: TypeAlias = Literal["billed", "released", "unsettled"]


class BackgroundInteractionCreateContext(BaseModel):
    """
    The part of a create's logging state that billing its settled result needs,
    in a shape any replica can store and rebuild a logging object from. Provider
    credentials are deliberately absent: the replica that settles fetches the
    interaction with its own, exactly as it would serve the delete itself.
    """

    model_config = ConfigDict(frozen=True)

    model: str | None
    call_type: str
    litellm_call_id: str
    function_id: str
    litellm_trace_id: str
    start_time: datetime
    custom_llm_provider: str
    metadata: Mapping[str, JsonValue]
    custom_pricing: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class PendingBackgroundInteraction:
    interaction_id: str
    custom_llm_provider: str
    create_context: BackgroundInteractionCreateContext
    created_at: datetime


class BackgroundSettlementStore(Protocol):
    async def register(self, pending: PendingBackgroundInteraction) -> None: ...

    async def pending(self, interaction_id: str) -> PendingBackgroundInteraction | None: ...

    async def is_claimed(self, interaction_id: str) -> bool: ...

    async def claim(self, interaction_id: str) -> bool: ...

    async def record_outcome(self, interaction_id: str, outcome: SettlementOutcome) -> None: ...

    async def unclaimed(self) -> Sequence[PendingBackgroundInteraction]: ...


@dataclass(frozen=True, slots=True)
class InMemoryBackgroundSettlementStore:
    """
    Per-process store: a registered interaction maps to its pending row until
    it is claimed, after which it maps to ``None``. Claiming an interaction the
    store never saw succeeds once, which is what a poll built without a
    registration relies on.
    """

    _rows: dict[str, PendingBackgroundInteraction | None] = field(  # mutable-ok: the registry every settler shares
        default_factory=dict
    )

    async def register(self, pending: PendingBackgroundInteraction) -> None:
        self._rows[pending.interaction_id] = pending

    async def pending(self, interaction_id: str) -> PendingBackgroundInteraction | None:
        return self._rows.get(interaction_id)

    async def is_claimed(self, interaction_id: str) -> bool:
        return interaction_id in self._rows and self._rows[interaction_id] is None

    async def claim(self, interaction_id: str) -> bool:
        if await self.is_claimed(interaction_id):
            return False
        self._rows[interaction_id] = None
        return True

    async def record_outcome(self, interaction_id: str, outcome: SettlementOutcome) -> None:
        return None

    async def unclaimed(self) -> Sequence[PendingBackgroundInteraction]:
        return tuple(row for row in self._rows.values() if row is not None)


@dataclass(slots=True)
class _StoreSlot:
    store: BackgroundSettlementStore


_STORE: Final = _StoreSlot(store=InMemoryBackgroundSettlementStore())


def configure_background_settlement_store(store: BackgroundSettlementStore) -> None:
    _STORE.store = store


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
    store: BackgroundSettlementStore = field(default_factory=InMemoryBackgroundSettlementStore)
    resumed: bool = False


FetchInteraction: TypeAlias = Callable[[BackgroundInteractionPollContext], Awaitable[InteractionsAPIResponse]]


async def fetch_background_interaction(context: BackgroundInteractionPollContext) -> InteractionsAPIResponse:
    from litellm.interactions import aget

    return await aget(
        interaction_id=context.interaction_id,
        custom_llm_provider=context.custom_llm_provider,
        api_key=context.api_key,
        api_base=context.api_base,
        **{"no-log": True},
    )


def _poll_intervals(initial: float, maximum: float, timeout: float) -> Iterator[float]:
    elapsed = 0.0  # rebind-ok: the schedule accumulates the time it has already yielded
    interval = initial  # rebind-ok: the schedule doubles the interval up to the cap
    while interval > 0 and elapsed + interval <= timeout:
        yield interval
        elapsed += interval
        interval = min(interval * 2, maximum)


_CUSTOM_PRICING_KEYS: Final = frozenset(CustomPricingLiteLLMParams.model_fields.keys())

_CARRIED_METADATA_KEYS: Final = frozenset(
    {
        "model_info",
        "model_group",
        "deployment",
        "tags",
        "spend_logs_metadata",
        "requester_metadata",
        "requester_ip_address",
        "user_agent",
        "agent_id",
        "session_id",
        "endpoint",
        "team_alias",
        "team_id",
        "applied_guardrails",
        "prompt_management_metadata",
    }
)

_CARRIED_METADATA_PREFIX: Final = "user_api_"

_UNCARRIED_METADATA_KEY: Final = "user_api_key_auth"

_JSON_VALUE: Final = TypeAdapter(JsonValue)


def _carries(key: str) -> bool:
    if key == _UNCARRIED_METADATA_KEY:
        return False
    return key in _CARRIED_METADATA_KEYS or key.startswith(_CARRIED_METADATA_PREFIX)


def _json_value(value: object) -> tuple[JsonValue, ...]:
    try:
        return (_JSON_VALUE.validate_python(to_jsonable_python(value)),)
    except (PydanticSerializationError, ValidationError):
        verbose_logger.debug("Dropping a background interaction metadata value that has no JSON form: %r", type(value))
        return ()


def _json_values(items: Iterable[tuple[str, object]]) -> Mapping[str, JsonValue]:
    return MappingProxyType({key: parsed for key, value in items for parsed in _json_value(value)})


def _as_datetime(start_time: datetime | float) -> datetime:
    return start_time if isinstance(start_time, datetime) else datetime.fromtimestamp(start_time, tz=timezone.utc)


def _create_context(logging_obj: "LiteLLMLoggingObj", custom_llm_provider: str) -> BackgroundInteractionCreateContext:
    metadata: Final = get_litellm_metadata_from_kwargs(kwargs=logging_obj.model_call_details)
    model: Final = logging_obj.model_call_details.get("model")
    return BackgroundInteractionCreateContext(
        model=model if isinstance(model, str) else logging_obj.model,
        call_type=logging_obj.call_type,
        litellm_call_id=logging_obj.litellm_call_id,
        function_id=logging_obj.function_id,
        litellm_trace_id=logging_obj.litellm_trace_id,
        start_time=_as_datetime(logging_obj.start_time),
        custom_llm_provider=custom_llm_provider,
        metadata=_json_values((key, value) for key, value in metadata.items() if _carries(key)),
        custom_pricing=_json_values(
            (key, value)
            for key, value in logging_obj.litellm_params.items()
            if key in _CUSTOM_PRICING_KEYS and value is not None
        ),
    )


def _rebuild_logging_obj(create_context: BackgroundInteractionCreateContext) -> "LiteLLMLoggingObj":
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj: Final = Logging(
        model=create_context.model,  # pyright: ignore[reportArgumentType]  # function_setup builds the live object with the same None for an agent-only create
        messages=None,
        stream=False,
        call_type=create_context.call_type,
        start_time=create_context.start_time,
        litellm_call_id=create_context.litellm_call_id,
        function_id=create_context.function_id,
        litellm_trace_id=create_context.litellm_trace_id,
    )
    litellm_params: Final = {  # mutable-ok: Logging scrubs and merges the params it is handed in place
        "metadata": dict(create_context.metadata),  # mutable-ok: Logging pops keys from the metadata it is handed
        **create_context.custom_pricing,
    }
    logging_obj.update_environment_variables(
        litellm_params=litellm_params,
        optional_params={},  # mutable-ok: Logging stores the optional params it is handed and updates them in place
        model=create_context.model,
        custom_llm_provider=create_context.custom_llm_provider,
    )
    return logging_obj


async def _settled_elsewhere(context: BackgroundInteractionPollContext) -> bool:
    try:
        return await context.store.is_claimed(context.interaction_id)
    except Exception as e:  # noqa: BLE001  # an unreadable store must not stop the poll; the claim below decides
        verbose_logger.debug(
            "Could not read the settlement state of background interaction %s: %s", context.interaction_id, e
        )
        return False


async def _claim(context: BackgroundInteractionPollContext) -> bool | None:
    """
    Exactly-once gate between every settler of one interaction, on every
    replica: whoever claims first owns billing or release. ``None`` means the
    store could not answer, so nothing is owned and the caller retries later.
    """
    try:
        return await context.store.claim(context.interaction_id)
    except Exception:  # noqa: BLE001  # an unanswerable claim is retried on the next poll rather than billed twice
        verbose_logger.exception("Could not claim the settlement of background interaction %s", context.interaction_id)
        return None


async def _record(context: BackgroundInteractionPollContext, outcome: SettlementOutcome) -> SettlementOutcome:
    try:
        await context.store.record_outcome(context.interaction_id, outcome)
    except Exception:  # noqa: BLE001  # the outcome is an audit trail; the claim already made the settlement exclusive
        verbose_logger.exception("Could not record the settlement of background interaction %s", context.interaction_id)
    return outcome


async def poll_and_log_background_interaction_cost(
    context: BackgroundInteractionPollContext,
    fetch_interaction: FetchInteraction = fetch_background_interaction,
) -> SettlementOutcome | None:
    last_response: InteractionsAPIResponse | None = None  # rebind-ok: the give-up path settles from the last poll
    for interval in _poll_intervals(
        initial=context.initial_interval_seconds,
        maximum=context.max_interval_seconds,
        timeout=context.timeout_seconds,
    ):
        await asyncio.sleep(interval)
        if await _settled_elsewhere(context):
            return None
        try:
            response = await fetch_interaction(context)
        except Exception as e:  # noqa: BLE001  # any fetch error must not kill the billing poll loop
            verbose_logger.debug(
                "Background interaction cost poll for %s failed, will retry: %s",
                context.interaction_id,
                e,
            )
            continue
        last_response = response
        if response.status not in _TERMINAL_STATUSES:
            continue
        if (claimed := await _claim(context)) is None:
            continue
        if not claimed:
            return None
        return await _record(context, await _settle_terminal(logging_obj=context.logging_obj, response=response))
    if not await _claim(context):
        return None
    if last_response is not None and last_response.status in _TERMINAL_STATUSES:
        return await _record(context, await _settle_terminal(logging_obj=context.logging_obj, response=last_response))
    if last_response is not None and last_response.status not in _POLLABLE_STATUSES:
        verbose_logger.error(
            "Gave up cost polling for background interaction %s after %ss: its last status %r is in neither "
            "the pollable nor the terminal set, so this proxy never learned how to settle it and its usage "
            "will not be tracked",
            context.interaction_id,
            context.timeout_seconds,
            last_response.status,
        )
    else:
        verbose_logger.warning(
            "Gave up cost polling for background interaction %s after %ss; its usage will not be tracked",
            context.interaction_id,
            context.timeout_seconds,
        )
    await _release_open_budget_reservation(logging_obj=context.logging_obj)
    return await _record(context, "unsettled")


async def _settle_terminal(logging_obj: "LiteLLMLoggingObj", response: InteractionsAPIResponse) -> SettlementOutcome:
    if response.status in _TERMINAL_STATUSES and response.usage is not None:
        await _bill_settled_interaction(logging_obj=logging_obj, response=response)
        return "billed"
    await _release_open_budget_reservation(logging_obj=logging_obj)
    return "released"


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
    metadata: Final = get_litellm_metadata_from_kwargs(kwargs=logging_obj.model_call_details)
    budget_reservation: Final = metadata.get("user_api_key_budget_reservation")
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
    task: "asyncio.Task[SettlementOutcome | None]"
    context: BackgroundInteractionPollContext


_ACTIVE_POLLS: Final[dict[str, _ActiveBackgroundPoll]] = {}  # mutable-ok: asyncio needs strong refs to poll tasks


def _discard_poll(interaction_id: str, task: "asyncio.Task[SettlementOutcome | None]") -> None:
    entry: Final = _ACTIVE_POLLS.get(interaction_id)
    if entry is not None and entry.task is task:
        del _ACTIVE_POLLS[interaction_id]


def _track_poll(
    context: BackgroundInteractionPollContext, fetch_interaction: FetchInteraction
) -> "asyncio.Task[SettlementOutcome | None]":
    task: Final = asyncio.create_task(poll_and_log_background_interaction_cost(context, fetch_interaction))
    _ACTIVE_POLLS[context.interaction_id] = _ActiveBackgroundPoll(task=task, context=context)
    task.add_done_callback(
        lambda finished, interaction_id=context.interaction_id: _discard_poll(interaction_id, finished)
    )
    return task


@dataclass(frozen=True, slots=True)
class _UnverifiedRegistrationStore:
    """
    Store of a create whose registration raised, so whether its row landed is
    unknown until the durable store answers. The settlement claim asks it
    first, and only an interaction it reports as never stored settles through
    the local gate, which no other process can reach.
    """

    durable: BackgroundSettlementStore
    local: InMemoryBackgroundSettlementStore = field(default_factory=InMemoryBackgroundSettlementStore)

    async def register(self, pending: PendingBackgroundInteraction) -> None:
        await self.durable.register(pending)

    async def pending(self, interaction_id: str) -> PendingBackgroundInteraction | None:
        return await self.durable.pending(interaction_id)

    async def is_claimed(self, interaction_id: str) -> bool:
        return await self.local.is_claimed(interaction_id) or await self.durable.is_claimed(interaction_id)

    async def claim(self, interaction_id: str) -> bool:
        if await self.durable.claim(interaction_id):
            return True
        if await self.durable.is_claimed(interaction_id):
            return False
        return await self.local.claim(interaction_id)

    async def record_outcome(self, interaction_id: str, outcome: SettlementOutcome) -> None:
        if await self.local.is_claimed(interaction_id):
            return
        await self.durable.record_outcome(interaction_id, outcome)

    async def unclaimed(self) -> Sequence[PendingBackgroundInteraction]:
        return await self.durable.unclaimed()


async def _registered_store(
    store: BackgroundSettlementStore, pending: PendingBackgroundInteraction
) -> BackgroundSettlementStore:
    try:
        await store.register(pending)
    except Exception:  # noqa: BLE001  # a store outage must not fail the create; the claim learns if the row landed
        verbose_logger.exception(
            "Could not durably register background interaction %s; its settlement claim decides whether the row landed",
            pending.interaction_id,
        )
        return _UnverifiedRegistrationStore(durable=store)
    return store


async def maybe_schedule_background_interaction_cost_polling(
    response: object,
    create_kwargs: Mapping[str, object],
    custom_llm_provider: str,
    store: BackgroundSettlementStore | None = None,
    fetch_interaction: FetchInteraction = fetch_background_interaction,
) -> "asyncio.Task[SettlementOutcome | None] | None":
    from litellm.litellm_core_utils.litellm_logging import Logging

    if not BACKGROUND_INTERACTION_COST_POLLING_ENABLED:
        return None
    if not isinstance(response, InteractionsAPIResponse):
        return None
    if not is_pollable_background_interaction(response):
        return None
    logging_obj: Final = create_kwargs.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return None
    api_key: Final = create_kwargs.get("api_key")
    api_base: Final = create_kwargs.get("api_base")
    pending: Final = PendingBackgroundInteraction(
        interaction_id=response.id,
        custom_llm_provider=custom_llm_provider,
        create_context=_create_context(logging_obj, custom_llm_provider),
        created_at=datetime.now(timezone.utc),
    )
    context: Final = BackgroundInteractionPollContext(
        interaction_id=response.id,
        custom_llm_provider=custom_llm_provider,
        logging_obj=logging_obj,
        api_key=api_key if isinstance(api_key, str) else None,
        api_base=api_base if isinstance(api_base, str) else None,
        store=await _registered_store(store or _STORE.store, pending),
    )
    return _track_poll(context, fetch_interaction)


async def _pending(store: BackgroundSettlementStore, interaction_id: str) -> PendingBackgroundInteraction | None:
    try:
        return await store.pending(interaction_id)
    except Exception:  # noqa: BLE001  # an unreadable store leaves the interaction to its poll or the counter TTL
        verbose_logger.exception("Could not look up background interaction %s before its delete", interaction_id)
        return None


async def _fetch_before_delete(
    context: BackgroundInteractionPollContext, fetch_interaction: FetchInteraction
) -> InteractionsAPIResponse | None:
    try:
        return await fetch_interaction(context)
    except Exception as e:  # noqa: BLE001  # the caller decides what an unfetchable pre-delete state means
        verbose_logger.debug(
            "Could not fetch background interaction %s before its delete: %s", context.interaction_id, e
        )
        return None


async def _settle_before_delete(
    context: BackgroundInteractionPollContext, response: InteractionsAPIResponse | None
) -> SettlementOutcome | None:
    if not await _claim(context):
        return None
    if response is None:
        await _release_open_budget_reservation(logging_obj=context.logging_obj)
        return await _record(context, "released")
    return await _record(context, await _settle_terminal(logging_obj=context.logging_obj, response=response))


async def maybe_settle_background_interaction_before_delete(
    interaction_id: str,
    delete_kwargs: Mapping[str, object],
    fetch_interaction: FetchInteraction = fetch_background_interaction,
    store: BackgroundSettlementStore | None = None,
) -> SettlementOutcome | None:
    entry: Final = _ACTIVE_POLLS.get(interaction_id)
    if entry is not None and not entry.context.resumed:
        return await _settle_before_delete(entry.context, await _fetch_before_delete(entry.context, fetch_interaction))
    settlement_store: Final = store or _STORE.store
    pending: Final = await _pending(settlement_store, interaction_id)
    if pending is None:
        return None
    api_key: Final = delete_kwargs.get("api_key")
    api_base: Final = delete_kwargs.get("api_base")
    context: Final = BackgroundInteractionPollContext(
        interaction_id=interaction_id,
        custom_llm_provider=pending.custom_llm_provider,
        logging_obj=_rebuild_logging_obj(pending.create_context),
        api_key=api_key if isinstance(api_key, str) else None,
        api_base=api_base if isinstance(api_base, str) else None,
        store=settlement_store,
    )
    try:
        response: Final = await fetch_interaction(context)
    except Exception:
        verbose_logger.debug(
            "Failing the delete of background interaction %s: this process could not fetch it with the delete's "
            "credentials, so the poll that created it keeps the bill",
            interaction_id,
        )
        raise
    return await _settle_before_delete(context, response)


async def _unclaimed(store: BackgroundSettlementStore) -> Sequence[PendingBackgroundInteraction]:
    try:
        return await store.unclaimed()
    except Exception:  # noqa: BLE001  # an unreadable store at startup leaves its rows for the next boot
        verbose_logger.exception("Could not list the unsettled background interactions")
        return ()


@dataclass(frozen=True, slots=True)
class PollSchedule:
    initial_interval_seconds: float = BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS
    max_interval_seconds: float = BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS
    timeout_seconds: float = BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS


DEFAULT_POLL_SCHEDULE: Final = PollSchedule()


def _resumed_context(
    row: PendingBackgroundInteraction, store: BackgroundSettlementStore, schedule: PollSchedule
) -> BackgroundInteractionPollContext:
    age_seconds: Final = (datetime.now(timezone.utc) - row.created_at).total_seconds()
    return BackgroundInteractionPollContext(
        interaction_id=row.interaction_id,
        custom_llm_provider=row.custom_llm_provider,
        logging_obj=_rebuild_logging_obj(row.create_context),
        initial_interval_seconds=schedule.initial_interval_seconds,
        max_interval_seconds=schedule.max_interval_seconds,
        timeout_seconds=max(schedule.timeout_seconds - age_seconds, schedule.initial_interval_seconds),
        store=store,
        resumed=True,
    )


async def resume_unsettled_background_interactions(
    store: BackgroundSettlementStore,
    fetch_interaction: FetchInteraction = fetch_background_interaction,
    schedule: PollSchedule = DEFAULT_POLL_SCHEDULE,
) -> tuple["asyncio.Task[SettlementOutcome | None]", ...]:
    """
    Pick up every settlement no process has claimed, which is what a replica
    that died mid-poll leaves behind. Each resumed poll keeps the remaining
    share of the original timeout and gets at least one fetch, so a completed
    interaction is still billed however late the resume comes.
    """
    return tuple(
        _track_poll(_resumed_context(row, store, schedule), fetch_interaction)
        for row in await _unclaimed(store)
        if row.interaction_id not in _ACTIVE_POLLS
    )

import asyncio
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm.litellm_core_utils.asyncify import asyncify, is_event_loop_running
from litellm.litellm_core_utils.hidden_params import get_hidden_params, served_from_cache, set_hidden_params
from litellm.litellm_core_utils.litellm_logging import BilledRoundsSpend, UncachedPrice
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.mcp.parked_stream_dispatch import (
    DISPATCH_RELEASE,
    HELD_ROUND_SPEND,
    IN_FLIGHT_ROUND,
    PARKED_DISPATCH,
    SERVED_FROM_CACHE,
    DeferredStreamRelease,
    parked_success,
)
from litellm.responses.mcp.round_billing import billed_for_every_round, merged_chat_usage, summed_cost_breakdown
from litellm.responses.streaming_iterator import spawn_logging_task
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

Parked: TypeAlias = tuple[object, ...]
PartialAnswer: TypeAlias = Callable[[], object]


@dataclass(frozen=True, slots=True)
class _AssembledRound:
    response: ModelResponse
    usage: Usage
    cache_hit: bool
    parked: Parked


@dataclass(frozen=True, slots=True)
class _PricedRound:
    response: ModelResponse
    usage: Usage
    price: UncachedPrice
    cache_hit: bool

    @property
    def spend(self) -> BilledRoundsSpend:
        return BilledRoundsSpend(self.usage, self.price.cost or 0.0, self.price.cost_breakdown)


@dataclass(frozen=True, slots=True)
class _HeldUsageChunk:
    chunk: ModelResponseStream
    usage: Usage


@dataclass(frozen=True, slots=True)
class _FirstRound:
    priced: _PricedRound
    parked: Parked
    logged_alone: Parked
    banked: LiteLLMLoggingObj | None = None
    usage_chunk: _HeldUsageChunk | None = None

    def client_usage_chunk(self, billed: ModelResponse | None) -> ModelResponseStream | None:
        if self.usage_chunk is None:
            return None
        usage: Final = (
            self.priced.usage.model_copy(update=MappingProxyType({"cost": self.priced.price.cost}))
            if billed is None
            else merged_chat_usage(self.priced.usage, self.usage_chunk.usage).model_copy(
                update=MappingProxyType({"cost": _cost(billed)})
            )
        )
        return self.usage_chunk.chunk.model_copy(
            update=MappingProxyType({"id": self.priced.response.id, "usage": usage})
        )


def _cost(response: ModelResponse) -> object:
    hidden_params: Final = get_hidden_params(response)
    return None if hidden_params is None else hidden_params.get("response_cost")


def _assembled_round(parked: Parked | None) -> _AssembledRound | None:
    match parked:
        case (ModelResponse() as response, cache_hit) if isinstance(usage := getattr(response, "usage", None), Usage):
            return _AssembledRound(response, usage, cache_hit is True, parked)
        case _:
            return None


def engaged_round_billing(logging_obj: object) -> "ChatStreamRoundBilling | None":
    return ChatStreamRoundBilling(logging_obj) if isinstance(logging_obj, LiteLLMLoggingObj) else None


class ChatStreamRoundBilling:
    def __init__(self, logging_obj: LiteLLMLoggingObj) -> None:
        self._logging_obj: Final = logging_obj
        self._first: _FirstRound | None = None
        self._partial_follow_up: PartialAnswer | None = None
        self._reading_loop: asyncio.AbstractEventLoop | None = None

    @contextmanager
    def armed_while_reading(self) -> Generator[None]:
        armed: Final = getattr(self._logging_obj, DISPATCH_RELEASE, None) is None
        if armed:
            setattr(self._logging_obj, DISPATCH_RELEASE, self._log_success)
        self._reading_loop = asyncio.get_running_loop()
        try:
            yield
        finally:
            if armed and self._peek_parked() is None:
                self._disarm()

    def complete_first_round(self) -> None:
        assembled: Final = _assembled_round(self._peek_parked())
        if assembled is not None:
            self._first = _FirstRound(self._priced(assembled), assembled.parked, assembled.parked)
            setattr(self._logging_obj, HELD_ROUND_SPEND, self._held_round_one_spend)

    def bank_first_round(self) -> BilledRoundsSpend | None:
        self._bank_apart_from_the_follow_up()
        return None if self._first is None else self._first.priced.spend

    def client_chunk(self, chunk: ModelResponseStream) -> ModelResponseStream | None:
        usage: Final[object] = getattr(chunk, "usage", None)
        if self._first is None or not isinstance(usage, Usage):
            return chunk
        self._first = replace(self._first, usage_chunk=_HeldUsageChunk(chunk, usage))
        return None

    def finish(self) -> ModelResponseStream | None:
        first: Final = self._first
        billed: Final = self._release_rounds()
        return None if first is None else first.client_usage_chunk(billed)

    def fail(self) -> None:
        self._bank_apart_from_the_follow_up()
        first: Final = self._first
        if first is None or first.banked is None or not self._holds_round_one_alone(first):
            self.finish()
            return
        self._first = None
        setattr(self._logging_obj, PARKED_DISPATCH, None)
        self._log_through_the_release((*first.parked, first.banked.guardrail_scoped_copy()))

    def follow_up_started(self, partial_follow_up: PartialAnswer) -> None:
        self._partial_follow_up = partial_follow_up
        setattr(self._logging_obj, IN_FLIGHT_ROUND, self.bill_in_flight_round)

    async def bill_in_flight_round(self) -> None:
        partial_follow_up: Final = self._partial_follow_up
        if partial_follow_up is None or self._first_round_held_alone() is None:
            return
        final_round: Final = _assembled_round((await asyncify(partial_follow_up)(), False))
        first: Final = self._first_round_held_alone()
        if final_round is None or first is None:
            return
        self._first = None
        setattr(
            self._logging_obj, PARKED_DISPATCH, self._billed_for_both_rounds(first.priced, self._priced(final_round))
        )

    def release_abandoned(self) -> None:
        if not self._owns_release():
            return
        if is_event_loop_running() or self._idle_reading_loop() is not None:
            self.finish()
            return
        setattr(self._logging_obj, PARKED_DISPATCH, None)
        self._first = None
        self._disarm()

    def _bank_apart_from_the_follow_up(self) -> None:
        first: Final = self._first
        if first is None or first.banked is not None or self._peek_parked() is not first.parked:
            return
        banked: Final = self._logging_obj.attempt_scoped_copy()
        if not self._owns_release():
            self._first = replace(first, banked=banked)
            return
        logged_alone: Final = (*first.parked, banked)
        self._first = replace(first, logged_alone=logged_alone, banked=banked)
        setattr(self._logging_obj, PARKED_DISPATCH, logged_alone)

    def _release_rounds(self) -> ModelResponse | None:
        parked: Final = self._take_parked()
        first: Final = self._first
        self._first = None
        if parked is None:
            self._disarm()
            return None
        if first is None or parked is first.logged_alone:
            self._release(parked)
            return None
        final_round: Final = _assembled_round(parked)
        if final_round is None:
            self._release(first.logged_alone)
            return None
        billed, cache_hit = self._billed_for_both_rounds(first.priced, self._priced(final_round))
        self._release((billed, cache_hit))
        return billed

    def _billed_for_both_rounds(self, first: _PricedRound, final: _PricedRound) -> tuple[ModelResponse, bool]:
        first_breakdown: Final = first.price.cost_breakdown
        final_breakdown: Final = final.price.cost_breakdown
        rounds: Final = billed_for_every_round(first.response, first_breakdown, final.response, final_breakdown)
        usage: Final[object] = getattr(rounds, "usage", None)
        charged_usage: Final = (
            usage.model_copy(update=MappingProxyType({"cost": _cost(rounds)})) if isinstance(usage, Usage) else usage
        )
        billed: Final = rounds.model_copy(update=MappingProxyType({"id": first.response.id, "usage": charged_usage}))
        cache_hit: Final = first.cache_hit and final.cache_hit
        set_hidden_params(billed, {**(get_hidden_params(rounds) or {}), "cache_hit": cache_hit})
        self._logging_obj.cost_breakdown = summed_cost_breakdown(
            first.response, first_breakdown, final.response, final_breakdown
        )
        return billed, cache_hit

    def _priced(self, assembled: _AssembledRound) -> _PricedRound:
        served: Final = assembled.cache_hit or served_from_cache(assembled.response)
        price: Final = SERVED_FROM_CACHE if served else self._logging_obj.uncached_price(assembled.response)
        priced: Final = assembled.response.model_copy()
        set_hidden_params(
            priced,
            {**(get_hidden_params(assembled.response) or {}), "response_cost": price.cost, "cache_hit": served},
        )
        return _PricedRound(priced, assembled.usage, price, served)

    def _peek_parked(self) -> Parked | None:
        parked: Final[Parked | None] = getattr(self._logging_obj, PARKED_DISPATCH, None)
        return parked if isinstance(parked, tuple) else None

    def _take_parked(self) -> Parked | None:
        parked: Final = self._peek_parked()
        setattr(self._logging_obj, PARKED_DISPATCH, None)
        return parked

    def _first_round_held_alone(self) -> _FirstRound | None:
        first: Final = self._first
        return first if first is not None and self._holds_round_one_alone(first) else None

    def _held_round_one_spend(self) -> BilledRoundsSpend | None:
        first: Final = self._first_round_held_alone()
        return None if first is None else first.priced.spend

    def _holds_round_one_alone(self, first: _FirstRound) -> bool:
        parked: Final = self._peek_parked()
        return parked is first.parked or parked is first.logged_alone

    def _owns_release(self) -> bool:
        release: Final[object] = getattr(self._logging_obj, DISPATCH_RELEASE, None)
        return release == self._log_success

    def _release(self, parked: Parked) -> None:
        if getattr(self._logging_obj, DISPATCH_RELEASE, None) is not None and not self._owns_release():
            setattr(self._logging_obj, PARKED_DISPATCH, parked)
            return
        self._log_through_the_release(parked)

    def _log_through_the_release(self, parked: Parked) -> None:
        armed: Final[DeferredStreamRelease | None] = getattr(self._logging_obj, DISPATCH_RELEASE, None)
        setattr(self._logging_obj, DISPATCH_RELEASE, None)
        logged: Final = (self._log_success if armed is None else armed)(*parked)
        idle_reading_loop: Final = None if is_event_loop_running() else self._idle_reading_loop()
        if idle_reading_loop is not None:
            idle_reading_loop.run_until_complete(logged)
            return
        spawn_logging_task(asyncio.get_running_loop(), logged, task_name="Chat completions MCP stream success log")

    def _idle_reading_loop(self) -> asyncio.AbstractEventLoop | None:
        loop: Final = self._reading_loop
        return loop if loop is not None and not loop.is_closed() and not loop.is_running() else None

    def _disarm(self) -> None:
        if self._owns_release():
            setattr(self._logging_obj, DISPATCH_RELEASE, None)

    async def _log_success(self, *parked: object) -> None:
        success: Final = parked_success(parked)
        if success is not None:
            await success.dispatch(self._logging_obj)

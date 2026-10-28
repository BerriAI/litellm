import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

from litellm.litellm_core_utils.asyncify import is_event_loop_running
from litellm.litellm_core_utils.hidden_params import get_hidden_params, served_from_cache, set_hidden_params
from litellm.litellm_core_utils.litellm_logging import BilledRoundsSpend, UncachedPrice
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.mcp.round_billing import billed_for_every_round, merged_chat_usage, summed_cost_breakdown
from litellm.responses.streaming_iterator import spawn_logging_task
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

Parked: TypeAlias = tuple[object, ...]

_PARKED_DISPATCH: Final = "_deferred_stream_complete_args"
_DISPATCH_RELEASE: Final = "_on_deferred_stream_complete"
_SERVED_FROM_CACHE: Final = UncachedPrice(cost=0.0, cost_breakdown=None)


class _DispatchesSuccessHandlers(Protocol):
    async def dispatch_success_handlers(
        self,
        result: object = None,
        start_time: object = None,
        end_time: object = None,
        cache_hit: object = None,
        prefer_async_handlers: bool = False,
    ) -> None: ...


def _success_dispatcher(logger: LiteLLMLoggingObj) -> _DispatchesSuccessHandlers:
    return logger


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
class _FirstRound:
    priced: _PricedRound
    parked: Parked
    logged_alone: Parked
    banked: LiteLLMLoggingObj | None = None


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

    @contextmanager
    def armed_while_reading(self) -> Generator[None]:
        armed: Final = getattr(self._logging_obj, _DISPATCH_RELEASE, None) is None
        if armed:
            setattr(self._logging_obj, _DISPATCH_RELEASE, self._log_success)
        try:
            yield
        finally:
            if armed and self._peek_parked() is None:
                self._disarm()

    def complete_first_round(self) -> None:
        assembled: Final = _assembled_round(self._peek_parked())
        if assembled is not None:
            self._first = _FirstRound(self._priced(assembled), assembled.parked, assembled.parked)

    def bank_first_round(self) -> BilledRoundsSpend | None:
        self._bank_apart_from_the_follow_up()
        return None if self._first is None else self._first.priced.spend

    def client_chunk(self, chunk: ModelResponseStream) -> ModelResponseStream:
        usage: Final[object] = getattr(chunk, "usage", None)
        if self._first is None or not isinstance(usage, Usage):
            return chunk
        return chunk.model_copy(update=MappingProxyType({"usage": merged_chat_usage(self._first.priced.usage, usage)}))

    def finish(self) -> None:
        parked: Final = self._take_parked()
        first: Final = self._first
        self._first = None
        if parked is None:
            self._disarm()
            return
        if first is None or parked is first.logged_alone:
            self._release(parked)
            return
        final_round: Final = _assembled_round(parked)
        if final_round is None:
            self._release(first.logged_alone)
            return
        self._release(self._billed_for_both_rounds(first.priced, self._priced(final_round)))

    def fail(self) -> None:
        self._bank_apart_from_the_follow_up()
        first: Final = self._first
        if first is None or first.banked is None or self._peek_parked() is not first.parked:
            self.finish()
            return
        self._first = None
        setattr(self._logging_obj, _PARKED_DISPATCH, None)
        self._log_apart((*first.parked, first.banked))

    def release_abandoned(self) -> None:
        if not self._owns_release():
            return
        if is_event_loop_running():
            self.finish()
            return
        setattr(self._logging_obj, _PARKED_DISPATCH, None)
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
        setattr(self._logging_obj, _PARKED_DISPATCH, logged_alone)

    def _billed_for_both_rounds(self, first: _PricedRound, final: _PricedRound) -> Parked:
        first_breakdown: Final = first.price.cost_breakdown
        final_breakdown: Final = final.price.cost_breakdown
        billed: Final = billed_for_every_round(first.response, first_breakdown, final.response, final_breakdown)
        cache_hit: Final = first.cache_hit and final.cache_hit
        set_hidden_params(billed, {**(get_hidden_params(billed) or {}), "cache_hit": cache_hit})
        self._logging_obj.cost_breakdown = summed_cost_breakdown(
            first.response, first_breakdown, final.response, final_breakdown
        )
        return billed, cache_hit

    def _priced(self, assembled: _AssembledRound) -> _PricedRound:
        served: Final = assembled.cache_hit or served_from_cache(assembled.response)
        price: Final = _SERVED_FROM_CACHE if served else self._logging_obj.uncached_price(assembled.response)
        priced: Final = assembled.response.model_copy()
        set_hidden_params(
            priced,
            {**(get_hidden_params(assembled.response) or {}), "response_cost": price.cost, "cache_hit": served},
        )
        return _PricedRound(priced, assembled.usage, price, served)

    def _peek_parked(self) -> Parked | None:
        parked: Final[Parked | None] = getattr(self._logging_obj, _PARKED_DISPATCH, None)
        return parked if isinstance(parked, tuple) else None

    def _take_parked(self) -> Parked | None:
        parked: Final = self._peek_parked()
        setattr(self._logging_obj, _PARKED_DISPATCH, None)
        return parked

    def _owns_release(self) -> bool:
        release: Final[object] = getattr(self._logging_obj, _DISPATCH_RELEASE, None)
        return release == self._log_success

    def _release(self, parked: Parked) -> None:
        if getattr(self._logging_obj, _DISPATCH_RELEASE, None) is not None and not self._owns_release():
            setattr(self._logging_obj, _PARKED_DISPATCH, parked)
            return
        setattr(self._logging_obj, _DISPATCH_RELEASE, None)
        self._log_apart(parked)

    def _log_apart(self, parked: Parked) -> None:
        loop: Final = asyncio.get_running_loop()
        spawn_logging_task(loop, self._log_success(*parked), task_name="Chat completions MCP stream success log")

    def _disarm(self) -> None:
        if self._owns_release():
            setattr(self._logging_obj, _DISPATCH_RELEASE, None)

    async def _log_success(self, assembled: object, cache_hit: object, banked: object = None) -> None:
        logger: Final = banked if isinstance(banked, LiteLLMLoggingObj) else self._logging_obj
        await _success_dispatcher(logger).dispatch_success_handlers(
            assembled, cache_hit=cache_hit, start_time=None, end_time=None, prefer_async_handlers=True
        )

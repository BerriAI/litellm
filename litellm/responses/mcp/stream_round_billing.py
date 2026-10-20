import asyncio
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol, TypeAlias

from litellm.litellm_core_utils.hidden_params import served_from_cache
from litellm.litellm_core_utils.litellm_logging import BilledRoundsDispatch, BilledRoundsSpend, UncachedPrice
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.mcp.round_billing import billed_for_every_round, merged_chat_usage, summed_cost_breakdown
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator, spawn_logging_task
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesAPIStreamingResponse
from litellm.types.utils import CostBreakdown, ModelResponse, Usage

Parked: TypeAlias = tuple[object, ...]
RestoresHeaders: TypeAlias = Callable[[ResponsesAPIStreamingResponse], None]

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


def _event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _priced(response: ResponsesAPIResponse, price: UncachedPrice) -> ResponsesAPIResponse:
    priced: Final = response.model_copy()
    priced.hidden_params = {**response.hidden_params, "response_cost": price.cost or 0.0}
    return priced


def _bridged_round_usage(parked: Parked | None) -> Usage | None:
    match parked:
        case (ModelResponse() as assembled, bool()):
            usage: Final[object] = getattr(assembled, "usage", None)
            return usage if isinstance(usage, Usage) else None
        case _:
            return None


def _usage_of_bridged_rounds(billed: Usage | None, round_usage: Usage | None, cost: object) -> Usage | None:
    if billed is None or round_usage is None:
        return None
    return merged_chat_usage(billed, round_usage).model_copy(update={"cost": cost})


def _logged_cache_hit(response: ResponsesAPIResponse, chat_usage: Usage | None) -> bool | None:
    if chat_usage is not None:
        return served_from_cache(response)
    return served_from_cache(response) or None


@dataclass(frozen=True, slots=True)
class _BilledRounds:
    response: ResponsesAPIResponse
    breakdown: CostBreakdown | None
    chat_usage: Usage | None
    rounds: int = 1

    @property
    def cost(self) -> float:
        cost: Final = self.response.hidden_params.get("response_cost")
        return float(cost) if isinstance(cost, (int, float)) else 0.0

    @property
    def spend(self) -> BilledRoundsSpend:
        usage: Final = (
            ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(self.response.usage)
            if self.chat_usage is None
            else self.chat_usage
        )
        return BilledRoundsSpend(usage, self.cost, self.breakdown)

    def with_round(
        self, priced: ResponsesAPIResponse, breakdown: CostBreakdown | None, chat_usage: Usage | None
    ) -> "_BilledRounds":
        merged: Final = billed_for_every_round(self.response, self.breakdown, priced, breakdown)
        total: Final = merged.hidden_params.get("response_cost")
        charged: Final = merged.model_copy(
            update={"usage": None if merged.usage is None else merged.usage.model_copy(update={"cost": total})}
        )
        charged.hidden_params = {
            **merged.hidden_params,
            "cache_hit": served_from_cache(self.response) and served_from_cache(priced),
        }
        return _BilledRounds(
            charged,
            summed_cost_breakdown(self.response, self.breakdown, priced, breakdown),
            _usage_of_bridged_rounds(self.chat_usage, chat_usage, total),
            self.rounds + 1,
        )


class StreamedRoundBilling:
    """One success row per streamed auto-execute request, billed for every round it ran.

    Each round's iterator parks its success dispatch on the shared logging object rather than sending it, either as a
    ready dispatch or, for a stream bridged through chat completions, as the assembled response and its cache hit.
    When a round completes, before its tools run, it is folded into the rounds before it and a dispatch for every
    round so far stays parked, so a client disconnect or a later round that fails still bills them. Only the latest
    dispatch is sent, through the proxy's deferred release when the proxy armed one."""

    def __init__(self, logging_obj: LiteLLMLoggingObj) -> None:
        self._logging_obj: Final = logging_obj
        self._dispatcher: Final[_DispatchesSuccessHandlers] = logging_obj
        self._billed: _BilledRounds | None = None
        self._banked: Parked | None = None
        if getattr(logging_obj, _DISPATCH_RELEASE, None) is None:
            setattr(logging_obj, _DISPATCH_RELEASE, self._run_parked)

    def complete_round(
        self,
        completed: ResponsesAPIStreamingResponse,
        response: ResponsesAPIResponse,
        client_output: Sequence[object],
        restore_headers: RestoresHeaders,
    ) -> None:
        round_parked: Final = self._take_parked()
        price: Final = self._price(response, round_parked)
        priced: Final = _priced(response, price)
        chat_usage: Final = _bridged_round_usage(round_parked)
        if self._billed is None:
            self._billed = _BilledRounds(priced, price.cost_breakdown, chat_usage)
            self._bank(round_parked)
            return
        billed: Final = self._billed.with_round(priced, price.cost_breakdown, chat_usage)
        self._close(round_parked)
        self._close(self._banked)
        self._logging_obj.model_call_details.pop("combined_usage_object", None)
        self._logging_obj.model_call_details.pop("response_cost", None)
        self._billed = billed
        self._logging_obj.cost_breakdown = billed.breakdown
        self._bank(
            None
            if round_parked is None
            else (
                self._dispatch(
                    completed,
                    billed.response.model_copy(update={"output": list(client_output)}),
                    billed,
                    restore_headers,
                ),
            )
        )

    def record_billed_rounds(self, follow_up: object) -> None:
        if self._billed is None or not isinstance(follow_up, BaseResponsesAPIStreamingIterator):
            return
        follow_up.bill_failure_after_earlier_rounds(self._billed.spend)

    def charge_final_round(
        self, completed: ResponsesAPIStreamingResponse, response: ResponsesAPIResponse, restore_headers: RestoresHeaders
    ) -> ResponsesAPIResponse:
        parked: Final = self._take_parked()
        self._banked = None
        if self._billed is None or self._billed.rounds == 1:
            if parked is not None:
                self._release(parked)
            return response
        self._close(parked)
        charged: Final = response.model_copy(update={"usage": self._billed.response.usage})
        charged.hidden_params = {**self._billed.response.hidden_params}
        if parked is not None:
            self._release((self._dispatch(completed, charged, self._billed, restore_headers),))
        return charged

    def finish(self) -> None:
        parked: Final = self._take_parked()
        self._banked = None
        if parked is not None:
            self._release(parked)
            return
        self._disarm()

    def release_abandoned(self) -> None:
        if not self._owns_release():
            return
        if _event_loop_running():
            self.finish()
            return
        self.abandon()

    def abandon(self) -> None:
        self._close(self._take_parked())
        self._close(self._banked)
        self._banked = None
        self._disarm()

    def _price(self, response: ResponsesAPIResponse, round_parked: Parked | None) -> UncachedPrice:
        if served_from_cache(response):
            return _SERVED_FROM_CACHE
        match round_parked:
            case (ModelResponse() as assembled, bool()):
                return self._logging_obj.uncached_price(assembled)
            case _:
                return self._logging_obj.uncached_price(response)

    def _bank(self, parked: Parked | None) -> None:
        self._banked = parked
        if parked is not None:
            setattr(self._logging_obj, _PARKED_DISPATCH, parked)

    def _dispatch(
        self,
        completed: ResponsesAPIStreamingResponse,
        response: ResponsesAPIResponse,
        billed: _BilledRounds,
        restore_headers: RestoresHeaders,
    ) -> BilledRoundsDispatch | Coroutine[object, object, None]:
        logged_response: Final = response.model_copy()
        logged_response.hidden_params = {**response.hidden_params}
        logged: Final = completed.model_copy(update={"response": logged_response})
        restore_headers(logged)
        dispatch: Final = self._log_success(
            logged,
            datetime.now(),  # noqa: DTZ005  # naive to match the logging pipeline start_time
            _logged_cache_hit(response, billed.chat_usage),
            billed.chat_usage,
        )
        if billed.chat_usage is None:
            return dispatch
        return BilledRoundsDispatch(dispatch, billed.spend)

    async def _log_success(
        self,
        logged: ResponsesAPIStreamingResponse,
        end_time: datetime,
        cache_hit: bool | None,
        chat_usage: Usage | None,
    ) -> None:
        if chat_usage is not None:
            self._logging_obj.model_call_details["combined_usage_object"] = chat_usage
        await self._dispatcher.dispatch_success_handlers(
            logged,
            start_time=self._logging_obj.start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            prefer_async_handlers=True,
        )

    def _take_parked(self) -> Parked | None:
        parked: Final[Parked | None] = getattr(self._logging_obj, _PARKED_DISPATCH, None)
        if not isinstance(parked, tuple):
            return None
        setattr(self._logging_obj, _PARKED_DISPATCH, None)
        return parked

    async def _run_parked(self, *parked: object) -> None:
        match parked:
            case (BilledRoundsDispatch(dispatch=dispatch),):
                await dispatch
            case (dispatch,) if asyncio.iscoroutine(dispatch):
                await dispatch
            case (assembled, cache_hit):
                await self._dispatcher.dispatch_success_handlers(
                    assembled, cache_hit=cache_hit, start_time=None, end_time=None, prefer_async_handlers=True
                )
            case _:
                return

    def _owns_release(self) -> bool:
        release: Final[object] = getattr(self._logging_obj, _DISPATCH_RELEASE, None)
        return release == self._run_parked

    def _release(self, parked: Parked) -> None:
        if getattr(self._logging_obj, _DISPATCH_RELEASE, None) is not None and not self._owns_release():
            setattr(self._logging_obj, _PARKED_DISPATCH, parked)
            return
        setattr(self._logging_obj, _DISPATCH_RELEASE, None)
        spawn_logging_task(
            asyncio.get_running_loop(), self._run_parked(*parked), task_name="Responses MCP stream success logging"
        )

    def _disarm(self) -> None:
        if self._owns_release():
            setattr(self._logging_obj, _DISPATCH_RELEASE, None)

    @staticmethod
    def _close(parked: Parked | None) -> None:
        match parked:
            case (BilledRoundsDispatch(dispatch=dispatch),):
                dispatch.close()
            case (dispatch,) if asyncio.iscoroutine(dispatch):
                dispatch.close()
            case _:
                return

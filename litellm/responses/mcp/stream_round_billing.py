import asyncio
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol, TypeAlias

from litellm.litellm_core_utils.hidden_params import served_from_cache
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.mcp.round_billing import billed_for_every_round, summed_cost_breakdown
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator, spawn_logging_task
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesAPIStreamingResponse
from litellm.types.utils import CostBreakdown

Dispatch: TypeAlias = Coroutine[object, object, object]
RestoresHeaders: TypeAlias = Callable[[ResponsesAPIStreamingResponse], None]

_PARKED_DISPATCH: Final = "_deferred_stream_complete_args"
_DISPATCH_RELEASE: Final = "_on_deferred_stream_complete"


class _DispatchesSuccessHandlers(Protocol):
    async def dispatch_success_handlers(
        self,
        result: object = None,
        start_time: object = None,
        end_time: object = None,
        cache_hit: object = None,
        prefer_async_handlers: bool = False,
    ) -> None: ...


async def _run_dispatch(dispatch: Dispatch) -> None:
    await dispatch


def _event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _round_cost(response: ResponsesAPIResponse) -> float:
    if served_from_cache(response) or response.usage is None:
        return 0.0
    return response.usage.cost or 0.0


def _priced(response: ResponsesAPIResponse) -> ResponsesAPIResponse:
    priced: Final = response.model_copy()
    priced.hidden_params = {**response.hidden_params, "response_cost": _round_cost(response)}
    return priced


@dataclass(frozen=True, slots=True)
class _BilledRounds:
    response: ResponsesAPIResponse
    breakdown: CostBreakdown | None
    rounds: int = 1

    @property
    def cost(self) -> float:
        cost: Final = self.response.hidden_params.get("response_cost")
        return float(cost) if isinstance(cost, (int, float)) else 0.0

    def with_round(self, priced: ResponsesAPIResponse, breakdown: CostBreakdown | None) -> "_BilledRounds":
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
            charged, summed_cost_breakdown(self.response, self.breakdown, priced, breakdown), self.rounds + 1
        )


class StreamedRoundBilling:
    """One success row per streamed auto-execute request, billed for every round it ran.

    Each round's iterator parks its success dispatch on the shared logging object rather than sending it. When a
    round completes, before its tools run, it is folded into the rounds before it and a dispatch for every round so
    far stays parked, so a client disconnect still bills them. Only the request's final response is dispatched,
    through the proxy's deferred release when the proxy armed one."""

    def __init__(self, logging_obj: LiteLLMLoggingObj) -> None:
        self._logging_obj: Final = logging_obj
        self._dispatcher: Final[_DispatchesSuccessHandlers] = logging_obj
        self._billed: _BilledRounds | None = None
        self._banked_dispatch: Dispatch | None = None
        if getattr(logging_obj, _DISPATCH_RELEASE, None) is None:
            setattr(logging_obj, _DISPATCH_RELEASE, _run_dispatch)

    def complete_round(
        self,
        completed: ResponsesAPIStreamingResponse,
        response: ResponsesAPIResponse,
        client_output: Sequence[object],
        restore_headers: RestoresHeaders,
    ) -> None:
        round_dispatch: Final = self._take_parked()
        priced: Final = _priced(response)
        breakdown: Final = self._logging_obj.cost_breakdown
        if self._billed is None:
            self._billed = _BilledRounds(priced, breakdown)
            self._bank(round_dispatch)
            return
        billed: Final = self._billed.with_round(priced, breakdown)
        self._close(round_dispatch)
        self._close(self._banked_dispatch)
        self._logging_obj.model_call_details.pop("combined_usage_object", None)
        self._logging_obj.model_call_details.pop("response_cost", None)
        self._billed = billed
        self._logging_obj.cost_breakdown = billed.breakdown
        self._bank(
            None
            if round_dispatch is None
            else self._dispatch(
                completed, billed.response.model_copy(update={"output": list(client_output)}), restore_headers
            )
        )

    def record_billed_rounds(self, follow_up: object) -> None:
        if self._billed is None:
            return
        billed_usage: Final = ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(
            self._billed.response.usage
        )
        self._logging_obj.record_partial_usage_for_failure(billed_usage, self._billed.cost)
        if isinstance(follow_up, BaseResponsesAPIStreamingIterator):
            follow_up.bill_failure_after_earlier_rounds(billed_usage, self._billed.cost)

    def charge_final_round(
        self, completed: ResponsesAPIStreamingResponse, response: ResponsesAPIResponse, restore_headers: RestoresHeaders
    ) -> ResponsesAPIResponse:
        parked: Final = self._take_parked()
        self._banked_dispatch = None
        if self._billed is None or self._billed.rounds == 1:
            if parked is not None:
                self._release(parked)
            return response
        self._close(parked)
        charged: Final = response.model_copy(update={"usage": self._billed.response.usage})
        charged.hidden_params = {**self._billed.response.hidden_params}
        if parked is not None:
            self._release(self._dispatch(completed, charged, restore_headers))
        return charged

    def finish(self) -> None:
        parked: Final = self._take_parked()
        self._banked_dispatch = None
        if parked is not None:
            self._release(parked)
            return
        self._disarm()

    def release_abandoned(self) -> None:
        if getattr(self._logging_obj, _DISPATCH_RELEASE, None) is not _run_dispatch:
            return
        if _event_loop_running():
            self.finish()
            return
        self.abandon()

    def abandon(self) -> None:
        self._close(self._take_parked())
        self._close(self._banked_dispatch)
        self._banked_dispatch = None
        self._disarm()

    def _bank(self, dispatch: Dispatch | None) -> None:
        self._banked_dispatch = dispatch
        if dispatch is not None:
            setattr(self._logging_obj, _PARKED_DISPATCH, (dispatch,))

    def _dispatch(
        self, completed: ResponsesAPIStreamingResponse, response: ResponsesAPIResponse, restore_headers: RestoresHeaders
    ) -> Dispatch:
        logged_response: Final = response.model_copy()
        logged_response.hidden_params = {**response.hidden_params}
        logged: Final = completed.model_copy(update={"response": logged_response})
        restore_headers(logged)
        return self._dispatcher.dispatch_success_handlers(
            logged,
            start_time=self._logging_obj.start_time,
            end_time=datetime.now(),  # noqa: DTZ005  # naive to match the logging pipeline start_time
            cache_hit=served_from_cache(response) or None,
            prefer_async_handlers=True,
        )

    def _take_parked(self) -> Dispatch | None:
        parked: Final[tuple[object, ...] | None] = getattr(self._logging_obj, _PARKED_DISPATCH, None)
        match parked:
            case (dispatch,) if asyncio.iscoroutine(dispatch):
                setattr(self._logging_obj, _PARKED_DISPATCH, None)
                return dispatch
            case _:
                return None

    def _release(self, dispatch: Dispatch) -> None:
        release: Final[object] = getattr(self._logging_obj, _DISPATCH_RELEASE, None)
        if release is not None and release is not _run_dispatch:
            setattr(self._logging_obj, _PARKED_DISPATCH, (dispatch,))
            return
        setattr(self._logging_obj, _DISPATCH_RELEASE, None)
        spawn_logging_task(asyncio.get_running_loop(), dispatch, task_name="Responses MCP stream success logging")

    def _disarm(self) -> None:
        if getattr(self._logging_obj, _DISPATCH_RELEASE, None) is _run_dispatch:
            setattr(self._logging_obj, _DISPATCH_RELEASE, None)

    @staticmethod
    def _close(dispatch: Dispatch | None) -> None:
        if dispatch is not None:
            dispatch.close()

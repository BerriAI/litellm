import asyncio
import contextvars
from asyncio import CancelledError
from collections.abc import Awaitable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

from litellm._internal_context import internal_sub_call
from litellm.litellm_core_utils.hidden_params import get_hidden_params, served_from_cache, set_hidden_params
from litellm.litellm_core_utils.litellm_logging import BilledRoundsDispatch, BilledRoundsSpend, UncachedPrice
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.messages.native.fake_stream_iterator import FakeAnthropicMessagesStreamIterator
from litellm.responses.mcp.parked_stream_dispatch import (
    DISPATCH_RELEASE,
    PARKED_DISPATCH,
    SERVED_FROM_CACHE,
    success_dispatcher,
)
from litellm.responses.mcp.round_billing import RoundCharge, merged_anthropic_usage, merged_chat_usage
from litellm.responses.streaming_iterator import spawn_logging_task
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse, AnthropicUsage
from litellm.types.utils import CostBreakdown, ModelResponse, Usage


class _AnthropicUsageTranslation(Protocol):
    def calculate_usage(self, usage_object: Mapping[str, object], reasoning_content: str | None) -> Usage: ...


def _anthropic_usage_translation() -> _AnthropicUsageTranslation:
    return AnthropicConfig()


_USAGE_TRANSLATION: Final = _anthropic_usage_translation()


def _chat_usage_of(usage: AnthropicUsage | None) -> Usage | None:
    if usage is None:
        return None
    return _USAGE_TRANSLATION.calculate_usage(usage_object=usage, reasoning_content=None)


def _merged(first: Usage | None, final: Usage | None) -> Usage | None:
    if first is None or final is None:
        return final if first is None else first
    return merged_chat_usage(first, final)


def _event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class BilledRounds:
    response: AnthropicMessagesResponse
    usage: AnthropicUsage | None
    charge: RoundCharge
    chat_usage: Usage | None
    bridged: ModelResponse | None
    rounds: int = 1

    def then(self, later: "BilledRounds") -> "BilledRounds":
        return BilledRounds(
            later.response,
            merged_anthropic_usage(self.usage, later.usage),
            self.charge.then(later.charge),
            _merged(self.chat_usage, later.chat_usage),
            later.bridged,
            self.rounds + 1,
        )

    @property
    def spend(self) -> BilledRoundsSpend:
        return BilledRoundsSpend(
            self.chat_usage if self.chat_usage is not None else Usage(),
            self.charge.cost or 0.0,
            self.charge.cost_breakdown,
        )

    @property
    def logged_cache_hit(self) -> bool | None:
        return self.charge.served_from_cache or None

    def served(self, client_response: AnthropicMessagesResponse) -> AnthropicMessagesResponse:
        if self.rounds == 1 or self.usage is None:
            return client_response
        return {**client_response, "usage": self.usage}

    def _bridged_row(self, bridged: ModelResponse, cost: float) -> ModelResponse:
        row: Final = bridged.model_copy(update={"usage": self.chat_usage})
        set_hidden_params(row, {**(get_hidden_params(bridged) or {}), "response_cost": cost})
        return row

    def record_on(self, logging_obj: LiteLLMLoggingObj) -> None:
        if self.rounds == 1:
            return
        logging_obj.record_every_round_usage(self.chat_usage, self.charge.served_from_cache)
        logging_obj.record_every_round_breakdown(self.charge.cost_breakdown)
        if self.charge.cost is None:
            return
        logging_obj.record_route_cost(self.charge.cost)
        if self.bridged is not None:
            logging_obj.record_bridged_model_response(self._bridged_row(self.bridged, self.charge.cost))


@dataclass(frozen=True, slots=True)
class _RoundStart:
    bridged: ModelResponse | None
    breakdown: CostBreakdown | None


@dataclass(frozen=True, slots=True)
class RoundBilling:
    logging_obj: LiteLLMLoggingObj | None

    @classmethod
    def of(cls, logging_obj: object) -> "RoundBilling":
        return cls(logging_obj if isinstance(logging_obj, LiteLLMLoggingObj) else None)

    def start(self) -> None:
        if self.logging_obj is not None:
            self.logging_obj.carry_failed_attempt_usage()

    async def round(self, call: Awaitable[AnthropicMessagesResponse]) -> BilledRounds:
        logging_obj: Final = self.logging_obj
        if logging_obj is None:
            unlogged: Final = await call
            usage: Final = unlogged.get("usage")
            return BilledRounds(unlogged, usage, RoundCharge(None, None, False), _chat_usage_of(usage), None)
        before: Final = _RoundStart(logging_obj.bridged_model_response, logging_obj.cost_breakdown)
        with internal_sub_call(logging_obj.litellm_call_id):
            response: Final = await call
        return self._billed(logging_obj, response, before)

    @staticmethod
    def _billed(
        logging_obj: LiteLLMLoggingObj, response: AnthropicMessagesResponse, before: _RoundStart
    ) -> BilledRounds:
        bridged: Final = logging_obj.bridged_model_response
        fresh_bridged: Final = None if bridged is before.bridged else bridged
        cached: Final = (
            served_from_cache(response)
            or logging_obj.model_call_details.get("cache_hit") is True
            or (before.bridged is not None and fresh_bridged is None)
        )
        price: Final[UncachedPrice] = SERVED_FROM_CACHE if cached else logging_obj.uncached_price(response)
        breakdown: Final = None if price.cost_breakdown is before.breakdown else price.cost_breakdown
        usage: Final = response.get("usage")
        bridged_usage: Final[object] = getattr(fresh_bridged, "usage", None)
        return BilledRounds(
            response,
            usage,
            RoundCharge(price.cost, breakdown, cached),
            bridged_usage if isinstance(bridged_usage, Usage) else _chat_usage_of(usage),
            bridged,
        )

    @contextmanager
    def billed_on_failure(self, billed: BilledRounds) -> Generator[None]:
        try:
            yield
        except (Exception, CancelledError):
            if self.logging_obj is not None and not billed.charge.served_from_cache:
                self.logging_obj.record_billed_rounds_for_failure(billed.spend)
            raise

    def finish(
        self, billed: BilledRounds, client_response: AnthropicMessagesResponse, stream: bool
    ) -> AnthropicMessagesResponse | FakeAnthropicMessagesStreamIterator:
        served: Final = billed.served(client_response)
        logging_obj: Final = self.logging_obj
        if logging_obj is not None:
            billed.record_on(logging_obj)
        if not stream:
            return served
        if logging_obj is None:
            return FakeAnthropicMessagesStreamIterator(served)
        return BilledRoundsStream(served, _ParkedRoundsLog(logging_obj, served, billed))


class _ParkedRoundsLog:
    def __init__(self, logging_obj: LiteLLMLoggingObj, logged: AnthropicMessagesResponse, billed: BilledRounds) -> None:
        self._logging_obj: Final = logging_obj
        dispatch: Final = self._log(
            logging_obj,
            logged,
            billed,
            datetime.now(),  # noqa: DTZ005  # naive to match the logging pipeline start_time
        )
        setattr(logging_obj, PARKED_DISPATCH, (BilledRoundsDispatch(dispatch, billed.spend),))
        if getattr(logging_obj, DISPATCH_RELEASE, None) is None:
            setattr(logging_obj, DISPATCH_RELEASE, self._run_parked)

    @staticmethod
    async def _log(
        logging_obj: LiteLLMLoggingObj, logged: AnthropicMessagesResponse, billed: BilledRounds, end_time: datetime
    ) -> None:
        billed.record_on(logging_obj)
        await success_dispatcher(logging_obj).dispatch_success_handlers(
            logged,
            start_time=logging_obj.start_time,
            end_time=end_time,
            cache_hit=billed.logged_cache_hit,
            prefer_async_handlers=True,
        )

    @staticmethod
    async def _run_parked(*parked: object) -> None:
        match parked:
            case (BilledRoundsDispatch(dispatch=dispatch),):
                await dispatch
            case _:
                return

    def _owns_release(self) -> bool:
        release: Final[object] = getattr(self._logging_obj, DISPATCH_RELEASE, None)
        return release == self._run_parked

    def _take_parked(self) -> BilledRoundsDispatch | None:
        parked: Final[object] = getattr(self._logging_obj, PARKED_DISPATCH, None)
        setattr(self._logging_obj, PARKED_DISPATCH, None)
        setattr(self._logging_obj, DISPATCH_RELEASE, None)
        match parked:
            case (BilledRoundsDispatch() as dispatch,):
                return dispatch
            case _:
                return None

    def finish(self) -> None:
        if not self._owns_release():
            return
        parked: Final = self._take_parked()
        if parked is None:
            return
        if _event_loop_running():
            spawn_logging_task(
                asyncio.get_running_loop(), parked.dispatch, task_name="Anthropic messages MCP stream success logging"
            )
            return
        from litellm.litellm_core_utils.thread_pool_executor import executor

        executor.submit(contextvars.copy_context().run, asyncio.run, parked.dispatch)


class BilledRoundsStream(FakeAnthropicMessagesStreamIterator):
    def __init__(self, response: AnthropicMessagesResponse, parked: _ParkedRoundsLog) -> None:
        super().__init__(response)
        self._parked: Final = parked

    async def __anext__(self) -> bytes:
        try:
            return await super().__anext__()
        except StopAsyncIteration:
            self._parked.finish()
            raise

    def __next__(self) -> bytes:
        try:
            return super().__next__()
        except StopIteration:
            self._parked.finish()
            raise

    async def aclose(self) -> None:
        self._parked.finish()

    def __del__(self) -> None:
        self._parked.finish()

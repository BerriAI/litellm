import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, TypeAlias, runtime_checkable

from pydantic import ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.realtime_errors import realtime_error_event
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.utils import CallTypes, Usage

TURN_LIMIT_CHECK_FAILED_MESSAGE: Final = "The proxy could not check this response against its token rate limits"


@runtime_checkable
class RealtimeTurnLimiter(Protocol):
    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None, call_type: str = ...
    ) -> None: ...

    async def charge_realtime_turn_usage(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None, usage: Usage
    ) -> None: ...


@runtime_checkable
class RealtimeCapacityCheck(Protocol):
    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None: ...


@runtime_checkable
class RealtimeCapacityCharge(Protocol):
    async def charge_realtime_capacity_turn(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None, usage: Usage
    ) -> None: ...


RealtimeTurnCharge: TypeAlias = RealtimeTurnLimiter | RealtimeCapacityCharge
RealtimeTurnLimiterSource: TypeAlias = Callable[[], Sequence[object]]


def registered_realtime_turn_limiters() -> Sequence[object]:
    return tuple(litellm.callbacks)


@dataclass(frozen=True, slots=True)
class TurnRefusal:
    event: str
    over_limit: bool


def _refusal(error: Exception) -> TurnRefusal:
    match error:
        case RateLimitError():
            return TurnRefusal(realtime_error_event(str(error), error_type="rate_limit_exceeded"), over_limit=True)
        case ServiceUnavailableError():
            return TurnRefusal(realtime_error_event(str(error), error_type="server_error"), over_limit=False)
        case _:
            return TurnRefusal(
                realtime_error_event(TURN_LIMIT_CHECK_FAILED_MESSAGE, error_type="server_error"), over_limit=False
            )


def _reported_usage(event: Mapping[str, object]) -> Mapping[str, object] | None:
    match event.get("type"):
        case "response.done":
            response: Final = as_str_mapping(event.get("response"))
            return None if response is None else as_str_mapping(response.get("usage"))
        case "conversation.item.input_audio_transcription.completed":
            return as_str_mapping(event.get("usage"))
        case _:
            return None


def turn_token_usage(event: Mapping[str, object]) -> Usage | None:
    usage: Final = _reported_usage(event)
    if usage is None:
        return None
    try:
        converted: Final = ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(usage)
    except ValidationError as error:
        verbose_logger.warning("Ignoring realtime usage the provider reported in an unexpected shape: %s", error)
        return None
    return converted if converted.total_tokens or converted.prompt_tokens or converted.completion_tokens else None


def realtime_response_id(event: Mapping[str, object]) -> str | None:
    response: Final = as_str_mapping(event.get("response"))
    response_id: Final = None if response is None else response.get("id")
    return response_id if isinstance(response_id, str) and response_id else None


@dataclass(frozen=True, slots=True)
class _UnchargedTurn:
    charger: RealtimeTurnCharge
    usage: Usage


@dataclass(slots=True)
class RealtimeTurnLimits:
    limiters: tuple[RealtimeTurnLimiter, ...]
    user_api_key_dict: UserAPIKeyAuth
    requested_model: str
    capacity_checks: tuple[RealtimeCapacityCheck, ...] = ()
    capacity_charges: tuple[RealtimeCapacityCharge, ...] = ()
    _charged_response_ids: frozenset[str] = field(default=frozenset(), init=False)
    _uncharged: tuple[_UnchargedTurn, ...] = field(default=(), init=False)
    _turn_refused: bool = field(default=False, init=False)

    @property
    def turn_refused(self) -> bool:
        return self._turn_refused

    async def refuse_turn(self) -> TurnRefusal | None:
        refusal: Final = await self.refusal()
        self._turn_refused = refusal is not None
        return refusal

    async def refusal(self) -> TurnRefusal | None:
        await self._settle_uncharged()
        try:
            for limiter in self.limiters:
                await limiter.enforce_token_limits_for_frame(
                    self.user_api_key_dict, self.requested_model, call_type=CallTypes.arealtime.value
                )
            for capacity_check in self.capacity_checks:
                await capacity_check.enforce_token_limits_for_frame(self.user_api_key_dict, self.requested_model)
        except RateLimitError as refused:
            return _refusal(refused)
        except Exception as error:  # noqa: BLE001  # a response the limits could not check must not reach the provider
            verbose_logger.exception("Realtime turn token limit check failed: %s", error)
            return _refusal(error)
        self._turn_refused = False
        return None

    async def charge(self, event: Mapping[str, object]) -> None:
        if event.get("type") == "response.done":
            self._turn_refused = False
        usage: Final = turn_token_usage(event)
        if usage is None:
            return
        response_id: Final = realtime_response_id(event)
        if response_id in self._charged_response_ids:
            return
        if response_id is not None:
            self._charged_response_ids = self._charged_response_ids | {response_id}
        chargers: Final = (*self.limiters, *self.capacity_charges)
        self._uncharged = (*self._uncharged, *(_UnchargedTurn(charger, usage) for charger in chargers))
        await self._settle_uncharged()

    async def _still_uncharged(self, turn: _UnchargedTurn) -> bool:
        try:
            match turn.charger:
                case RealtimeTurnLimiter() as limiter:
                    await limiter.charge_realtime_turn_usage(self.user_api_key_dict, self.requested_model, turn.usage)
                case capacity:
                    await capacity.charge_realtime_capacity_turn(
                        self.user_api_key_dict, self.requested_model, turn.usage
                    )
        except Exception as error:  # noqa: BLE001  # the response already ran, so a failed charge must not end the session
            verbose_logger.exception("Realtime turn token charge failed: %s", error)
            return True
        return False

    async def _settle_uncharged(self) -> None:
        settling: Final = self._uncharged
        if not settling:
            return
        self._uncharged = ()
        outcomes: Final = await asyncio.gather(*(self._still_uncharged(turn) for turn in settling))
        failed: Final = tuple(turn for turn, still_uncharged in zip(settling, outcomes, strict=True) if still_uncharged)
        self._uncharged = (*failed, *self._uncharged)


def _session_model_group(model_call_details: object, model: str) -> str:
    details: Final = as_str_mapping(model_call_details) or {}
    litellm_params: Final = as_str_mapping(details.get("litellm_params")) or {}
    router_metadata: Final = as_str_mapping(litellm_params.get("litellm_metadata")) or {}
    model_group: Final = router_metadata.get("model_group")
    return model_group if isinstance(model_group, str) and model_group else model


def realtime_turn_limits(
    source: RealtimeTurnLimiterSource, user_api_key_dict: object, model_call_details: object, model: str
) -> RealtimeTurnLimits | None:
    if not isinstance(user_api_key_dict, UserAPIKeyAuth):
        return None
    callbacks: Final = tuple(source())
    limiters: Final = tuple(callback for callback in callbacks if isinstance(callback, RealtimeTurnLimiter))
    capacity_checks: Final = tuple(
        callback
        for callback in callbacks
        if isinstance(callback, RealtimeCapacityCheck) and not isinstance(callback, RealtimeTurnLimiter)
    )
    if not limiters and not capacity_checks:
        return None
    capacity_charges: Final = tuple(check for check in capacity_checks if isinstance(check, RealtimeCapacityCharge))
    return RealtimeTurnLimits(
        limiters, user_api_key_dict, _session_model_group(model_call_details, model), capacity_checks, capacity_charges
    )

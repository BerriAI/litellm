from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias, runtime_checkable

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


RealtimeTurnLimiterSource: TypeAlias = Callable[[], Sequence[object]]


def registered_realtime_turn_limiters() -> Sequence[object]:
    return tuple(litellm.callbacks)


def _refusal_event(error: Exception) -> str:
    match error:
        case RateLimitError():
            return realtime_error_event(str(error), error_type="rate_limit_exceeded")
        case ServiceUnavailableError():
            return realtime_error_event(str(error), error_type="server_error")
        case _:
            return realtime_error_event(TURN_LIMIT_CHECK_FAILED_MESSAGE, error_type="server_error")


def response_done_usage(event: Mapping[str, object]) -> Usage | None:
    response: Final = as_str_mapping(event.get("response")) if event.get("type") == "response.done" else None
    usage: Final = None if response is None else as_str_mapping(response.get("usage"))
    if usage is None:
        return None
    converted: Final = ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(usage)
    return converted if converted.total_tokens or converted.prompt_tokens or converted.completion_tokens else None


@dataclass(frozen=True, slots=True)
class RealtimeTurnLimits:
    limiters: tuple[RealtimeTurnLimiter, ...]
    user_api_key_dict: UserAPIKeyAuth
    requested_model: str

    async def refusal(self) -> str | None:
        try:
            for limiter in self.limiters:
                await limiter.enforce_token_limits_for_frame(
                    self.user_api_key_dict, self.requested_model, call_type=CallTypes.arealtime.value
                )
        except RateLimitError as refused:
            return _refusal_event(refused)
        except Exception as error:  # noqa: BLE001  # a response the limits could not check must not reach the provider
            verbose_logger.exception("Realtime turn token limit check failed: %s", error)
            return _refusal_event(error)
        return None

    async def charge(self, event: Mapping[str, object]) -> None:
        usage: Final = response_done_usage(event)
        if usage is None:
            return
        try:
            for limiter in self.limiters:
                await limiter.charge_realtime_turn_usage(self.user_api_key_dict, self.requested_model, usage)
        except Exception as error:  # noqa: BLE001  # the response already ran, so a failed charge must not end the session
            verbose_logger.exception("Realtime turn token charge failed: %s", error)


def _session_model_group(model_call_details: object, model: str) -> str:
    details: Final = as_str_mapping(model_call_details) or {}
    litellm_params: Final = as_str_mapping(details.get("litellm_params")) or {}
    metadata: Final = as_str_mapping(litellm_params.get("metadata")) or {}
    model_group: Final = metadata.get("model_group")
    return model_group if isinstance(model_group, str) and model_group else model


def realtime_turn_limits(
    source: RealtimeTurnLimiterSource, user_api_key_dict: object, model_call_details: object, model: str
) -> RealtimeTurnLimits | None:
    if not isinstance(user_api_key_dict, UserAPIKeyAuth):
        return None
    limiters: Final = tuple(callback for callback in source() if isinstance(callback, RealtimeTurnLimiter))
    if not limiters:
        return None
    return RealtimeTurnLimits(limiters, user_api_key_dict, _session_model_group(model_call_details, model))

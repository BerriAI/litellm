from typing import Final

from litellm.proxy.hooks.model_capacity_native import NativeDynamicRateLimiter
from litellm.proxy.hooks.parallel_request_limiter_v3 import (  # legacy module exports
    RATE_LIMIT_UNVERIFIABLE_MESSAGE,
    BorrowingTeamModelHold,
    BorrowingTeamModelLimits,
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitDescriptorRateLimitObject,
    RateLimitedModel,
    RateLimitResponse,
    RateLimitStatus,
    RateLimitUnverifiableError,
    RequestRateLimiterStash,
    ReservationOperation,
    ReservationWindowIdentity,
    WindowGuardedIncrement,
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # backwards-compatible package export
    borrowing_team_model_limits,
    callback_kwargs_mapping,
    claim_request_stash_for_data,
    get_or_create_request_stash,
    get_request_stash,
    request_stash_for_callback,
)

PROXY_DynamicRateLimitHandlerV3: Final = NativeDynamicRateLimiter
_PROXY_DynamicRateLimitHandlerV3: Final = NativeDynamicRateLimiter

__all__ = [
    "RATE_LIMIT_UNVERIFIABLE_MESSAGE",
    "BorrowingTeamModelHold",
    "BorrowingTeamModelLimits",
    "PROXY_DynamicRateLimitHandlerV3",
    "PROXY_MaxParallelRequestsHandler_v3",
    "RateLimitDescriptor",
    "RateLimitDescriptorRateLimitObject",
    "RateLimitResponse",
    "RateLimitStatus",
    "RateLimitUnverifiableError",
    "RateLimitedModel",
    "RequestRateLimiterStash",
    "ReservationOperation",
    "ReservationWindowIdentity",
    "WindowGuardedIncrement",
    "_PROXY_DynamicRateLimitHandlerV3",
    "_PROXY_MaxParallelRequestsHandler_v3",
    "borrowing_team_model_limits",
    "callback_kwargs_mapping",
    "claim_request_stash_for_data",
    "get_or_create_request_stash",
    "get_request_stash",
    "request_stash_for_callback",
]

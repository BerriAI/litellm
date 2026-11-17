"""
Re-check per-model rate limits before the router takes a fallback.

The v3 limiter admits a request once, against the model group the caller asked for. When that
group fails at the provider and the router moves to a fallback group, nothing re-checked the
caller's per-model limits on the fallback, so a provider outage on one model removed the cap on
the other for every caller at once. `RouterFallbackRateLimitCheck` runs the limiter's
model-scoped descriptors for the fallback target and reserves capacity on them, so the router
skips a target the caller has exhausted.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.fallback_budget import (
    _user_api_key_auth_from_request,  # pyright: ignore[reportPrivateUsage]  # shared auth lookup for fallback checks
)
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import PROXY_DynamicRateLimitHandlerV3
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # the proxy's limiter type
)
from litellm.router_api import RoutingContext


@dataclass(frozen=True, slots=True)
class RouterFallbackRateLimitCheck:
    """
    `FallbackRateLimitCheck` for the proxy's router: a fallback target is attempted only when the
    caller still has per-model rate limit capacity on it, and the attempt reserves that capacity.
    Requests that carry no key (for example internal health checks) and proxies running without
    the v3 limiter are not restricted.
    """

    limiter: Callable[[], object]

    async def __call__(self, *, model: str, request_kwargs: Mapping[str, object], ctx: RoutingContext) -> bool:
        limiter: Final = self.limiter()
        if not isinstance(limiter, PROXY_MaxParallelRequestsHandler_v3):
            return True
        valid_token: Final = _user_api_key_auth_from_request(request_kwargs)
        if valid_token is None:
            return True
        try:
            dynamic_limiter: Final = next(
                (cb for cb in litellm.callbacks if isinstance(cb, PROXY_DynamicRateLimitHandlerV3)), None
            )
            if dynamic_limiter is not None and not await dynamic_limiter.async_fallback_target_allowed(
                valid_token, model, request_kwargs, commit=False
            ):
                return False
            if not await limiter.async_fallback_target_allowed(valid_token, model, request_kwargs):
                return False
            return dynamic_limiter is None or await dynamic_limiter.async_fallback_target_allowed(
                valid_token, model, request_kwargs
            )
        except Exception as e:  # noqa: BLE001  # fail closed: a limiter failure must not hand out unchecked capacity
            verbose_proxy_logger.warning("Skipping fallback to model=%s: rate limit check failed: %s", model, e)
            return False

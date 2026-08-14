"""Adaptive router strategy. See README.md for design overview."""

from litellm.router_strategy.adaptive_router.adaptive_router import AdaptiveRouter
from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook

__all__ = ["AdaptiveRouter", "AdaptiveRouterPostCallHook"]

"""
Efficiency reward for the adaptive router's (request_type, model) bandit cells.
"""

from __future__ import annotations

import math
from typing import Final

from litellm.router_strategy.adaptive_router.config import (
    DEFAULT_TARGET_LATENCY_SECONDS,
    EFFICIENCY_GAMMA,
    EFFICIENCY_THROUGHPUT_MAX,
)


def composite_efficiency_reward(
    latency_seconds: float,
    completion_tokens: int,
    gamma: float = EFFICIENCY_GAMMA,
    target_latency_seconds: float = DEFAULT_TARGET_LATENCY_SECONDS,
    throughput_max: float = EFFICIENCY_THROUGHPUT_MAX,
) -> float:
    """Blend a latency-decay score and a throughput score into a single reward in [0, 1]."""
    safe_latency: Final = max(latency_seconds, 1e-6)
    latency_score: Final = math.exp(-safe_latency / max(target_latency_seconds, 1e-6))
    throughput_score: Final = min(max(completion_tokens, 0) / safe_latency / max(throughput_max, 1e-6), 1.0)
    reward: Final = gamma * latency_score + (1.0 - gamma) * throughput_score
    return min(max(reward, 0.0), 1.0)

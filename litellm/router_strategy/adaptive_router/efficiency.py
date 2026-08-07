"""
Efficiency reward for the adaptive router's (request_type, model) bandit cells.

Ported from the "Elastic Unified Router" (EURo) reference design's composite reward
(efficiency_router.py: latency_score/throughput_score blend). Pure arithmetic on
already-observed latency and completion tokens -- no classifier, no judge LLM, no
extra request in the hot path. See PHASE1_PLAN.md for why this piece specifically
was chosen as a first, low-risk cut of that design.

    latency_score    = exp(-latency_seconds / target_latency_seconds)
    throughput_score = min(completion_tokens / latency_seconds / throughput_max, 1.0)
    reward           = gamma * latency_score + (1 - gamma) * throughput_score

    gamma=1.0 -> pure latency (fast responses favored regardless of length)
    gamma=0.0 -> pure throughput (tokens/sec favored regardless of absolute latency)

On failure (non-2xx / exception), callers should pass reward=0.0 directly rather than
computing this function -- see hooks.py's response_status handling, which reuses the
same status already threaded through quality-signal detection instead of adding a
second failure-detection path.
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
    """Blend a latency-decay score and a throughput score into a single reward in [0, 1].

    latency_seconds <= 0 is treated as a caller error (clock skew / bad instrumentation),
    not a signal to reward -- clamped to a small epsilon so the exponential and division
    below stay finite instead of raising or returning an inflated score.
    """
    safe_latency: Final = max(latency_seconds, 1e-6)
    latency_score: Final = math.exp(-safe_latency / max(target_latency_seconds, 1e-6))
    throughput_score: Final = min(max(completion_tokens, 0) / safe_latency / max(throughput_max, 1e-6), 1.0)
    reward: Final = gamma * latency_score + (1.0 - gamma) * throughput_score
    return min(max(reward, 0.0), 1.0)

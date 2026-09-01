"""
Configuration constants for the adaptive_router strategy.

All magic numbers are first-pass guesses (D3-D6 in the handoff plan).
Expect to retune after first 1000 sessions of real traffic.
"""

from typing import Final

from litellm.types.router import RequestType  # re-export for convenience  # noqa: F401

# D3 — Score weights (default; user-overridable via AdaptiveRouterConfig.weights)
DEFAULT_QUALITY_WEIGHT: Final[float] = 0.7  # UNVALIDATED — calibrated against [0] sessions
DEFAULT_COST_WEIGHT: Final[float] = 0.3  # UNVALIDATED — calibrated against [0] sessions

# D4 — Cold-start prior: (alpha + beta) total mass = COLD_START_MASS
# Mean of Beta = base_tier_weight + (strength_bonus if declared)
BASE_TIER_WEIGHT: Final[dict[int, float]] = {1: 0.3, 2: 0.5, 3: 0.7}  # UNVALIDATED
STRENGTH_BONUS: Final[float] = 0.3  # UNVALIDATED
COLD_START_MASS: Final[float] = 10.0

# D5 — Sample cap. Hard cap, no rescaling (drift handling is v1).
SAMPLE_CAP: Final[int] = 200

# D6 — Clean-trace credit: minimum turns before α += 1 can fire.
MIN_TURNS_FOR_CLEAN_CREDIT: Final[int] = 3

# D2 — Owner-cache TTL (seconds). 24h.
# A conversation's first-picked model "owns" the bandit-update slot for
# this long. Subsequent turns of the same conversation only contribute a
# bandit/state update when the same model is re-sampled.
OWNER_CACHE_TTL_SECONDS: Final[int] = 24 * 3600

# Below this many messages we skip post-call signal recording. Most signals
# (misalignment, stagnation, satisfaction-in-response-to-prior-turn) need at
# least one full prior exchange to be meaningful.
SIGNAL_GATE_MIN_MESSAGES: Final[int] = 4

# Detector thresholds (from Plano/Chen 2026 paper).
MISALIGNMENT_JACCARD_THRESHOLD: Final[float] = 0.45
STAGNATION_JACCARD_NEAR_DUP: Final[float] = 0.50
LOOP_REPEAT_THRESHOLD: Final[int] = 3
TOOL_CALL_HISTORY_MAX: Final[int] = 20

# D1 — Caller filter for min quality tier.
MIN_QUALITY_TIER_HEADER: Final[str] = "x-litellm-min-quality-tier"
MIN_QUALITY_TIER_METADATA_KEY: Final[str] = "min_quality_tier"

# Pre-routing -> post-call relay: the chosen logical model is stashed on
# request_kwargs["metadata"][ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY] by the
# pre-routing hook, then read by the post-call hook to surface as the
# ADAPTIVE_ROUTER_RESPONSE_HEADER response header.
ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY: Final[str] = "adaptive_router_chosen_model"
ADAPTIVE_ROUTER_RESPONSE_HEADER: Final[str] = "x-litellm-adaptive-router-model"

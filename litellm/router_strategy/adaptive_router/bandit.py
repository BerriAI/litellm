"""
Thompson sampling and prior initialization for the adaptive router bandit.

Each (router, request_type, model) cell is a Beta(alpha, beta) posterior.
- alpha = pseudo-successes
- beta  = pseudo-failures
- mean  = alpha / (alpha + beta)
- total samples = alpha + beta - COLD_START_MASS  (informative prior, not data)

Hot path: thompson_sample() — pure function, no I/O.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from litellm.router_strategy.adaptive_router.config import (
    BASE_TIER_WEIGHT,
    COLD_START_MASS,
    DEFAULT_COST_WEIGHT,
    DEFAULT_QUALITY_WEIGHT,
    SAMPLE_CAP,
    STRENGTH_BONUS,
)
from litellm.types.router import AdaptiveRouterPreferences, RequestType


@dataclass(frozen=True)
class BanditCell:
    """Posterior state for a single (router, request_type, model) cell."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    @property
    def total_samples(self) -> int:
        return max(0, int(self.alpha + self.beta - COLD_START_MASS))


def initial_cell(prefs: AdaptiveRouterPreferences, request_type: RequestType) -> BanditCell:
    """
    Cold-start prior for a (model, request_type) cell.

    mean = base_tier_weight[tier] + (STRENGTH_BONUS if request_type in strengths else 0)
    capped at 0.95 to avoid an over-confident prior.
    Total mass = COLD_START_MASS so that ~10 real observations can move it noticeably.
    """
    if prefs.quality_tier not in BASE_TIER_WEIGHT:
        valid = sorted(BASE_TIER_WEIGHT)
        raise ValueError(f"quality_tier={prefs.quality_tier} is not supported; valid tiers are {valid}")
    base = BASE_TIER_WEIGHT[prefs.quality_tier]
    bonus = STRENGTH_BONUS if request_type in prefs.strengths else 0.0
    mean = min(0.95, base + bonus)
    alpha = mean * COLD_START_MASS
    beta = (1.0 - mean) * COLD_START_MASS
    return BanditCell(alpha=alpha, beta=beta)


def apply_delta(cell: BanditCell, delta_alpha: float, delta_beta: float) -> BanditCell:
    """
    Apply a learning update to a cell, enforcing the sample cap.

    SAMPLE_CAP is a HARD cap on (alpha + beta). When the cap would be exceeded,
    we drop the update. (D5: hard cap, no rescaling — keep v0 simple.)
    """
    new_alpha = cell.alpha + delta_alpha
    new_beta = cell.beta + delta_beta
    if new_alpha + new_beta > SAMPLE_CAP:
        return cell
    return BanditCell(alpha=new_alpha, beta=new_beta)


def thompson_sample(cell: BanditCell, rng: Optional[random.Random] = None) -> float:
    """Draw a sample from Beta(alpha, beta). Returns a quality estimate in [0, 1]."""
    r = rng if rng is not None else random
    return r.betavariate(cell.alpha, cell.beta)


def normalized_cost(model_cost: float, all_costs: List[float]) -> float:
    """
    Map a raw $/1k-token cost into [0, 1] where 0 = most expensive, 1 = cheapest.
    Returns 0.5 when there's no spread.
    """
    if not all_costs:
        return 0.5
    lo, hi = min(all_costs), max(all_costs)
    if hi == lo:
        return 0.5
    return 1.0 - ((model_cost - lo) / (hi - lo))


def score(
    quality_sample: float,
    model_cost: float,
    all_costs: List[float],
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
) -> float:
    """
    Multi-objective score. V0 is a weighted linear sum of (quality, normalized_cost).
    Higher is better. Both inputs are in [0, 1].
    """
    cost_score = normalized_cost(model_cost, all_costs)
    return quality_weight * quality_sample + cost_weight * cost_score


def _draw_once(
    cells: Dict[str, BanditCell],
    model_costs: Dict[str, float],
    all_costs: List[float],
    quality_weight: float,
    cost_weight: float,
    rng: Optional[random.Random],
) -> str:
    """One joint Thompson draw: sample every arm, score, return the argmax."""
    best_model: Optional[str] = None
    best_score = float("-inf")
    for model, cell in cells.items():
        q = thompson_sample(cell, rng=rng)
        s = score(q, model_costs[model], all_costs, quality_weight, cost_weight)
        if s > best_score:
            best_score = s
            best_model = model
    assert best_model is not None
    return best_model


def pick_best(
    cells: Dict[str, BanditCell],
    model_costs: Dict[str, float],
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Sample once per model, score each, return the model with highest score.

    cells: {model_name: BanditCell}
    model_costs: {model_name: $/1k tokens}
    """
    if not cells:
        raise ValueError("pick_best called with no models")
    all_costs = list(model_costs.values())
    return _draw_once(cells, model_costs, all_costs, quality_weight, cost_weight, rng)


def estimate_propensity(
    cells: Dict[str, BanditCell],
    model_costs: Dict[str, float],
    chosen: str,
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    rng: Optional[random.Random] = None,
    propensity_samples: int = 0,
) -> Optional[float]:
    """
    Estimate P(`chosen` is picked) in the current posterior state.

    Returns None when propensity_samples == 0, which is the default and costs
    nothing.

    Why estimate rather than compute: a model wins when its scored Beta draw
    beats every other model's. For more than two models that has no closed
    form, so P(chosen) has to be measured while the posteriors are in hand or
    it cannot be recovered at all. A randomized routing decision logged without
    it is data no off-policy estimator can use.

    The already-made choice counts as one observation, which guarantees the
    result is >= 1/(propensity_samples + 1). A zero would be self-contradictory
    -- the model demonstrably was picked -- and would become an infinite
    importance weight downstream. The cost is an upward bias of order
    1/propensity_samples, far below the Monte Carlo noise it displaces.

    Hot path: propensity_samples * len(cells) betavariate draws, all of it
    skipped at the default.
    """
    if propensity_samples < 0:
        raise ValueError(f"propensity_samples must be >= 0, got {propensity_samples}")
    # Checked before anything else: at the default this function does no work
    # and must not be able to fail, whatever state the caller is in.
    if propensity_samples == 0:
        return None
    if not cells:
        raise ValueError("estimate_propensity called with no models")
    if len(cells) == 1:
        # Nothing was chosen between, so the probability is exactly 1 and
        # sampling would only add error.
        return 1.0

    all_costs = list(model_costs.values())
    wins = 1  # the decision that was actually made
    for _ in range(propensity_samples):
        if _draw_once(cells, model_costs, all_costs, quality_weight, cost_weight, rng) == chosen:
            wins += 1
    return wins / (propensity_samples + 1)

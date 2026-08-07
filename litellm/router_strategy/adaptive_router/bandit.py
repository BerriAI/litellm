"""
Thompson sampling and prior initialization for the adaptive router bandit.

Each (router, request_type, model) cell holds two independent Beta posteriors:
- quality (alpha, beta): fed by behavioral signal detection (signals.py) -- unchanged from v0.
- efficiency (alpha_eff, beta_eff): fed by the latency/throughput composite reward
  (efficiency.py) -- new. Kept as a separate Beta rather than folded into quality because
  the two signals answer different questions ("was this response good" vs "was this response
  fast") and an operator may want to weight them independently (AdaptiveRouterWeights.efficiency).
  Defaults to an uninformative Beta(1, 1) prior -- no cold-start bias by quality_tier, since
  quality_tier says nothing about a model's expected latency.
- mean  = alpha / (alpha + beta)
- total samples = alpha + beta - COLD_START_MASS  (informative prior, not data)

Hot path: thompson_sample() — pure function, no I/O.
"""

import random
from dataclasses import dataclass
from typing import Final

from litellm.router_strategy.adaptive_router.config import (
    BASE_TIER_WEIGHT,
    COLD_START_MASS,
    DEFAULT_COST_WEIGHT,
    DEFAULT_EFFICIENCY_WEIGHT,
    DEFAULT_QUALITY_WEIGHT,
    SAMPLE_CAP,
    STRENGTH_BONUS,
)
from litellm.types.router import AdaptiveRouterPreferences, RequestType

# Uninformative prior for the efficiency posterior -- see class docstring above.
_EFFICIENCY_PRIOR_ALPHA: Final = 1.0
_EFFICIENCY_PRIOR_BETA: Final = 1.0


@dataclass(frozen=True)
class BanditCell:
    """Posterior state for a single (router, request_type, model) cell."""

    alpha: float
    beta: float
    alpha_eff: float = _EFFICIENCY_PRIOR_ALPHA
    beta_eff: float = _EFFICIENCY_PRIOR_BETA

    @property
    def mean(self) -> float:
        total: Final = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    @property
    def efficiency_mean(self) -> float:
        total: Final = self.alpha_eff + self.beta_eff
        return self.alpha_eff / total if total > 0 else 0.5

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
        valid: Final = sorted(BASE_TIER_WEIGHT)
        raise ValueError(f"quality_tier={prefs.quality_tier} is not supported; valid tiers are {valid}")
    base: Final = BASE_TIER_WEIGHT[prefs.quality_tier]
    bonus: Final = STRENGTH_BONUS if request_type in prefs.strengths else 0.0
    mean: Final = min(0.95, base + bonus)
    alpha: Final = mean * COLD_START_MASS
    beta: Final = (1.0 - mean) * COLD_START_MASS
    return BanditCell(alpha=alpha, beta=beta)


def apply_delta(cell: BanditCell, delta_alpha: float, delta_beta: float) -> BanditCell:
    """
    Apply a learning update to the quality posterior, enforcing the sample cap.

    SAMPLE_CAP is a HARD cap on (alpha + beta). When the cap would be exceeded,
    we drop the update. (D5: hard cap, no rescaling — keep v0 simple.)
    Leaves the efficiency posterior untouched -- use apply_efficiency_delta for that.
    """
    new_alpha: Final = cell.alpha + delta_alpha
    new_beta: Final = cell.beta + delta_beta
    if new_alpha + new_beta > SAMPLE_CAP:
        return cell
    return BanditCell(alpha=new_alpha, beta=new_beta, alpha_eff=cell.alpha_eff, beta_eff=cell.beta_eff)


def apply_efficiency_delta(cell: BanditCell, delta_alpha_eff: float, delta_beta_eff: float) -> BanditCell:
    """
    Apply a learning update to the efficiency posterior, enforcing the same sample cap
    as the quality posterior (D5). Leaves the quality posterior untouched.
    """
    new_alpha_eff: Final = cell.alpha_eff + delta_alpha_eff
    new_beta_eff: Final = cell.beta_eff + delta_beta_eff
    if new_alpha_eff + new_beta_eff > SAMPLE_CAP:
        return cell
    return BanditCell(alpha=cell.alpha, beta=cell.beta, alpha_eff=new_alpha_eff, beta_eff=new_beta_eff)


def thompson_sample(cell: BanditCell, rng: random.Random | None = None) -> float:
    """Draw a sample from the quality posterior Beta(alpha, beta). Returns an estimate in [0, 1]."""
    r: Final = rng if rng is not None else random
    return r.betavariate(cell.alpha, cell.beta)


def thompson_sample_efficiency(cell: BanditCell, rng: random.Random | None = None) -> float:
    """Draw a sample from the efficiency posterior Beta(alpha_eff, beta_eff)."""
    r: Final = rng if rng is not None else random
    return r.betavariate(cell.alpha_eff, cell.beta_eff)


def normalized_cost(model_cost: float, all_costs: list[float]) -> float:
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
    all_costs: list[float],
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    efficiency_sample: float | None = None,
    efficiency_weight: float = DEFAULT_EFFICIENCY_WEIGHT,
) -> float:
    """
    Multi-objective score. V0 was a weighted linear sum of (quality, normalized_cost);
    efficiency is an additive third term so a caller that never passes efficiency_sample
    (efficiency_weight defaults to 0.0) gets byte-identical scores to before this was added.
    Higher is better. All inputs are in [0, 1].
    """
    cost_score: Final = normalized_cost(model_cost, all_costs)
    total: Final = quality_weight * quality_sample + cost_weight * cost_score
    if efficiency_sample is None or efficiency_weight == 0.0:
        return total
    return total + efficiency_weight * efficiency_sample


def pick_best(
    cells: dict[str, BanditCell],
    model_costs: dict[str, float],
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    efficiency_weight: float = DEFAULT_EFFICIENCY_WEIGHT,
    rng: random.Random | None = None,
) -> str:
    """
    Sample once per model, score each, return the model with highest score.

    cells: {model_name: BanditCell}
    model_costs: {model_name: $/1k tokens}

    Efficiency is only sampled when efficiency_weight != 0.0, so a caller that never
    passes it (the default) does one fewer betavariate draw per candidate -- same hot
    path cost as before this was added, not just the same score.
    """
    if not cells:
        raise ValueError("pick_best called with no models")
    all_costs: Final = list(model_costs.values())
    best_model: str | None = None
    best_score = float("-inf")
    for model, cell in cells.items():
        q = thompson_sample(cell, rng=rng)
        eff = thompson_sample_efficiency(cell, rng=rng) if efficiency_weight != 0.0 else None
        s = score(q, model_costs[model], all_costs, quality_weight, cost_weight, eff, efficiency_weight)
        if s > best_score:
            best_score = s
            best_model = model
    assert best_model is not None
    return best_model

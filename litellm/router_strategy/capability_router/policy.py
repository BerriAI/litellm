"""Deterministic policy for selecting the cheapest qualified model."""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict

from .config import (
    CapabilityBoundary,
    CapabilityCandidateScore,
    CapabilityClassifierVerdict,
    CapabilityRouterCandidate,
    CapabilityRouterConfig,
    CapabilitySelectionReason,
    indexed_rules,
)

BOUNDARY_THRESHOLD_STEPS: Final[Mapping[CapabilityBoundary, int]] = MappingProxyType(
    {"supported": 0, "uncertain": 1, "unmatched": 1, "unsupported": 2}
)


def effective_boundary(candidate: CapabilityRouterCandidate, score: CapabilityCandidateScore) -> CapabilityBoundary:
    """With a rule card, the matched rule's operator-declared boundary overrides the judge's opinion."""
    if not candidate.rules:
        return score.capability_boundary
    boundaries: Final[Mapping[str, CapabilityBoundary]] = MappingProxyType(
        {rule_id: rule.boundary for rule_id, rule in indexed_rules(candidate)}
    )
    return boundaries.get(score.primary_rule, "unmatched")


def calibrated_probability(
    candidate: CapabilityRouterCandidate, raw_probability: float, primary_rule: str = "none"
) -> float:
    rule_probability: Final = next(
        (
            rule.observed_success_probability
            for rule_id, rule in indexed_rules(candidate)
            if rule_id == primary_rule and rule.observed_success_probability is not None
        ),
        None,
    )
    if rule_probability is not None:
        return rule_probability
    return next(
        (bucket.probability for bucket in candidate.probability_calibration if raw_probability <= bucket.upper_bound),
        raw_probability,
    )


class CapabilityCandidateAssessment(BaseModel):
    model: str
    raw_p_solve: float
    p_solve: float
    reason: str
    primary_rule: str
    capability_boundary: CapabilityBoundary
    estimated_cost: float | None
    qualified: bool

    model_config = ConfigDict(extra="forbid")


class CapabilityRoutingDecision(BaseModel):
    selected_model: str
    reason: CapabilitySelectionReason
    candidates: tuple[CapabilityCandidateAssessment, ...] = ()

    model_config = ConfigDict(extra="forbid")


def fallback_decision(
    config: CapabilityRouterConfig,
    reason: CapabilitySelectionReason,
    candidates: tuple[CapabilityCandidateAssessment, ...] = (),
) -> CapabilityRoutingDecision:
    return CapabilityRoutingDecision(
        selected_model=config.fallback_model,
        reason=reason,
        candidates=candidates,
    )


def select_capability_model(
    config: CapabilityRouterConfig,
    verdict: CapabilityClassifierVerdict,
    estimated_costs: Mapping[str, float | None],
) -> CapabilityRoutingDecision:
    """Choose the cheapest candidate whose p_solve clears its boundary-stepped threshold."""
    configured: Final = MappingProxyType({candidate.model: candidate for candidate in config.candidates})
    scores: Final = MappingProxyType({candidate.model: candidate for candidate in verdict.candidates})
    if frozenset(scores) != frozenset(configured):
        return fallback_decision(config, "invalid_classifier_verdict")

    boundaries: Final[Mapping[str, CapabilityBoundary]] = MappingProxyType(
        {model: effective_boundary(configured[model], scores[model]) for model in configured}
    )
    assessments: Final = tuple(
        CapabilityCandidateAssessment(
            model=model,
            raw_p_solve=scores[model].p_solve,
            p_solve=calibrated_probability(configured[model], scores[model].p_solve, scores[model].primary_rule),
            reason=scores[model].reason,
            primary_rule=scores[model].primary_rule,
            capability_boundary=boundaries[model],
            estimated_cost=estimated_costs.get(model),
            qualified=calibrated_probability(configured[model], scores[model].p_solve, scores[model].primary_rule)
            > round(
                config.probability_threshold + BOUNDARY_THRESHOLD_STEPS[boundaries[model]] * config.threshold_step,
                9,
            ),
        )
        for model in configured
    )
    qualified: Final = tuple(candidate for candidate in assessments if candidate.qualified)
    if not qualified:
        return fallback_decision(config, "no_qualified_candidate", assessments)
    if any(candidate.estimated_cost is None for candidate in qualified):
        return fallback_decision(config, "missing_candidate_price", assessments)

    order: Final = MappingProxyType({model: index for index, model in enumerate(configured)})
    selected: Final = min(
        qualified,
        key=lambda candidate: (
            candidate.estimated_cost if candidate.estimated_cost is not None else math.inf,
            order[candidate.model],
        ),
    )
    return CapabilityRoutingDecision(
        selected_model=selected.model,
        reason="cheapest_qualified",
        candidates=assessments,
    )

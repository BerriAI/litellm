"""Deterministic policy for selecting the cheapest qualified model."""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict

from .config import (
    CapabilityBoundary,
    CapabilityClassifierVerdict,
    CapabilityRouterConfig,
    CapabilitySelectionReason,
)

BOUNDARY_THRESHOLD_STEPS: Final[Mapping[CapabilityBoundary, int]] = MappingProxyType(
    {"supported": 0, "uncertain": 1, "unmatched": 1, "unsupported": 2}
)


class CapabilityCandidateAssessment(BaseModel):
    model: str
    p_solve: float
    reason: str
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
    configured_models: Final = tuple(candidate.model for candidate in config.candidates)
    scores: Final = MappingProxyType({candidate.model: candidate for candidate in verdict.candidates})
    if frozenset(scores) != frozenset(configured_models):
        return fallback_decision(config, "invalid_classifier_verdict")

    assessments: Final = tuple(
        CapabilityCandidateAssessment(
            model=model,
            p_solve=scores[model].p_solve,
            reason=scores[model].reason,
            capability_boundary=scores[model].capability_boundary,
            estimated_cost=estimated_costs.get(model),
            qualified=scores[model].p_solve
            > round(
                config.probability_threshold
                + BOUNDARY_THRESHOLD_STEPS[scores[model].capability_boundary] * config.threshold_step,
                9,
            ),
        )
        for model in configured_models
    )
    qualified: Final = tuple(candidate for candidate in assessments if candidate.qualified)
    if not qualified:
        return fallback_decision(config, "no_qualified_candidate", assessments)
    if any(candidate.estimated_cost is None for candidate in qualified):
        return fallback_decision(config, "missing_candidate_price", assessments)

    order: Final = MappingProxyType({model: index for index, model in enumerate(configured_models)})
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

"""Deterministic policy for selecting the cheapest qualified model."""

import math
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from .config import CapabilityClassifierVerdict, CapabilityRouterConfig, CapabilitySelectionReason


class CapabilityCandidateAssessment(BaseModel):
    model: str
    p_solve: float
    reason: str
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
    """Choose the cheapest candidate at or above the configured probability."""
    configured_models = tuple(candidate.model for candidate in config.candidates)
    scores = {candidate.model: candidate for candidate in verdict.candidates}
    if set(scores) != set(configured_models):
        return fallback_decision(config, "invalid_classifier_verdict")

    assessments = tuple(
        CapabilityCandidateAssessment(
            model=model,
            p_solve=scores[model].p_solve,
            reason=scores[model].reason,
            estimated_cost=estimated_costs.get(model),
            qualified=scores[model].p_solve >= config.probability_threshold,
        )
        for model in configured_models
    )
    qualified = tuple(candidate for candidate in assessments if candidate.qualified)
    if not qualified:
        return fallback_decision(config, "no_qualified_candidate", assessments)
    if any(candidate.estimated_cost is None for candidate in qualified):
        return fallback_decision(config, "missing_candidate_price", assessments)

    order = {model: index for index, model in enumerate(configured_models)}
    selected = min(
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

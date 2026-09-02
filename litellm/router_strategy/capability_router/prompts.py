"""Classifier prompt and response schema for capability routing."""

from typing import Any

from .config import CapabilityRouterConfig

CAPABILITY_BOUNDARIES = ("supported", "uncertain", "unsupported", "unmatched")


def build_classifier_prompt(config: CapabilityRouterConfig) -> str:
    candidates = "\n".join(f"- {candidate.model}: {candidate.description}" for candidate in config.candidates)
    return f"""You forecast task outcomes for a model router.

For each candidate model below, forecast one binary event. SUCCESS means the candidate completes the newest user task correctly and completely in one fresh attempt, using only the tools available in the request. FAILURE is any other outcome. The two outcomes are exhaustive.

Use only evidence in the conversation and each candidate's capability description. Do not assume hidden state, unmentioned tools, or future clarifications. The descriptions are qualitative evidence, not measured success rates.

Assessment procedure, per candidate:
1. State the crux in "reason": the hardest material requirement for whole-task success.
2. Set "capability_boundary": "supported" if the description covers the crux, "unsupported" if it excludes it, "uncertain" if coverage is unclear, "unmatched" if the description does not speak to this task.
3. Estimate "p_solve" last. It is the probability of SUCCESS, not confidence in this assessment and not a route recommendation.

Interpret p_solve as a natural frequency: at 0.7, about 70 of 100 comparable fresh attempts succeed. Use the full range when justified, and reserve 0 and 1 for outcomes that are logically impossible or certain. "supported" does not mean 1 and "unsupported" does not mean 0. Do not consider price or the routing threshold; both are handled separately.

Candidates:
{candidates}

Return one entry for every candidate using the exact model names."""


def build_classifier_response_schema(config: CapabilityRouterConfig) -> dict[str, Any]:
    model_names = [candidate.model for candidate in config.candidates]
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": len(model_names),
                "maxItems": len(model_names),
                "items": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string", "enum": model_names},
                        "reason": {"type": "string", "minLength": 1},
                        "capability_boundary": {"type": "string", "enum": list(CAPABILITY_BOUNDARIES)},
                        "p_solve": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["model", "reason", "capability_boundary", "p_solve"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }

"""Classifier prompt and response schema for capability routing."""

from typing import Any

from .config import CapabilityRouterConfig


def build_classifier_prompt(config: CapabilityRouterConfig) -> str:
    candidates = "\n".join(
        f"- {candidate.model}: {candidate.description}" for candidate in config.candidates
    )
    return f"""You route tasks between language models.

For each candidate below, estimate the probability that it can complete the user's task correctly and completely. Base the estimate only on the task and the operator-provided description. Do not consider price; price is handled separately.

Candidates:
{candidates}

Return one score for every candidate using the exact model names. p_solve must be a number from 0 to 1. reason must briefly identify the capability that most affects the estimate."""


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
                        "p_solve": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": ["model", "p_solve", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }

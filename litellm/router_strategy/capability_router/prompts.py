"""Classifier prompt and response schema for capability routing."""

from typing import Final, Literal, TypedDict

from typing_extensions import ReadOnly

from .config import CapabilityRouterConfig

CAPABILITY_BOUNDARIES: Final = ("supported", "uncertain", "unsupported", "unmatched")


class _StringEnumSchema(TypedDict):
    type: ReadOnly[Literal["string"]]
    enum: ReadOnly[list[str]]  # mutable-ok: provider json_schema transforms only rewrite list-typed arrays


class _NonBlankStringSchema(TypedDict):
    type: ReadOnly[Literal["string"]]
    minLength: ReadOnly[int]


class _UnitIntervalSchema(TypedDict):
    type: ReadOnly[Literal["number"]]
    minimum: ReadOnly[int]
    maximum: ReadOnly[int]


class _CandidateProperties(TypedDict):
    model: ReadOnly[_StringEnumSchema]
    reason: ReadOnly[_NonBlankStringSchema]
    capability_boundary: ReadOnly[_StringEnumSchema]
    p_solve: ReadOnly[_UnitIntervalSchema]


class _CandidateItemSchema(TypedDict):
    type: ReadOnly[Literal["object"]]
    properties: ReadOnly[_CandidateProperties]
    required: ReadOnly[list[str]]  # mutable-ok: provider json_schema transforms only rewrite list-typed arrays
    additionalProperties: ReadOnly[bool]


class _CandidateArraySchema(TypedDict):
    type: ReadOnly[Literal["array"]]
    minItems: ReadOnly[int]
    maxItems: ReadOnly[int]
    items: ReadOnly[_CandidateItemSchema]


class _VerdictProperties(TypedDict):
    candidates: ReadOnly[_CandidateArraySchema]


class ClassifierResponseSchema(TypedDict):
    type: ReadOnly[Literal["object"]]
    properties: ReadOnly[_VerdictProperties]
    required: ReadOnly[list[str]]  # mutable-ok: provider json_schema transforms only rewrite list-typed arrays
    additionalProperties: ReadOnly[bool]


def build_classifier_prompt(config: CapabilityRouterConfig) -> str:
    candidates: Final = "\n".join(f"- {candidate.model}: {candidate.description}" for candidate in config.candidates)
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


def build_classifier_response_schema(config: CapabilityRouterConfig) -> ClassifierResponseSchema:
    model_names: Final = [candidate.model for candidate in config.candidates]  # mutable-ok: wire arrays are lists
    schema: Final[ClassifierResponseSchema] = {
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
                        "capability_boundary": {
                            "type": "string",
                            "enum": list(CAPABILITY_BOUNDARIES),  # mutable-ok: wire arrays are lists
                        },
                        "p_solve": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": [  # mutable-ok: wire arrays are lists
                        "model",
                        "reason",
                        "capability_boundary",
                        "p_solve",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],  # mutable-ok: wire arrays are lists
        "additionalProperties": False,
    }
    return schema

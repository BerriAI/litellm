"""Classifier prompt and response schema for capability routing."""

from typing import Final, Literal, TypedDict

from typing_extensions import ReadOnly

from .config import CapabilityRouterCandidate, CapabilityRouterConfig, indexed_rules

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
    primary_rule: ReadOnly[_NonBlankStringSchema]
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


def _candidate_card(candidate: CapabilityRouterCandidate) -> str:
    header: Final = f"- {candidate.model}: {candidate.description}"
    rules: Final = tuple(f"  {rule_id}: {rule.rule}" for rule_id, rule in indexed_rules(candidate))
    return "\n".join((header, *rules))


def build_classifier_prompt(config: CapabilityRouterConfig) -> str:
    candidates: Final = "\n".join(_candidate_card(candidate) for candidate in config.candidates)
    return f"""You are the routing classifier for a model gateway.

Score each candidate model on a single yes-or-no outcome. SUCCESS: given the request exactly as shown, including its available tools, the candidate delivers a correct and complete answer to the newest user message on the first try. Anything else, including a partial or plausible-but-wrong answer, counts as FAILURE.

Ground every judgment in what the conversation and the candidate descriptions actually say. Never credit a candidate with tools, context, or clarifications that are absent from the request, and read each description as the operator's stated opinion, not a measured track record.

Fill the fields for each candidate in this order:
1. "reason": name the single requirement of this task most likely to decide success or failure.
2. "primary_rule": when the candidate lists rule ids below its description, give the id of the one rule whose text best matches that requirement, or "none" when no listed rule fits. A candidate without rules always takes "none". Rule ids are arbitrary labels; weigh a rule only by its text.
3. "capability_boundary": "supported" when the candidate's card covers that requirement, "unsupported" when it rules it out, "uncertain" when coverage is unclear, "unmatched" when the card says nothing relevant to this task. When you selected a rule, base this on that rule.
4. "p_solve": fill this only after the other fields. It answers one question, how often this candidate would fully succeed here. It is not your confidence and not a routing recommendation.

Treat p_solve as a frequency over repeated independent tries; 0.6 claims six successes in ten. Spread scores across the whole 0 to 1 range as the evidence warrants, keeping the exact endpoints for certainty. A "supported" boundary still permits a low p_solve and an "unsupported" one a high p_solve. Ignore pricing and ignore whatever threshold the router applies; both belong to the router, not to you.

Candidates:
{candidates}

Return one entry for every candidate, using each model name exactly as written above."""


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
                        "primary_rule": {"type": "string", "minLength": 1},
                        "capability_boundary": {
                            "type": "string",
                            "enum": list(CAPABILITY_BOUNDARIES),  # mutable-ok: wire arrays are lists
                        },
                        "p_solve": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": [  # mutable-ok: wire arrays are lists
                        "model",
                        "reason",
                        "primary_rule",
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

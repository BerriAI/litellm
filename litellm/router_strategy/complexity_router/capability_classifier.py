# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability forecast contract and routing policy adapted from NVIDIA NeMo Switchyard."""

from collections.abc import Mapping
from sys import float_info
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, TypeAdapter, model_validator

CapabilityBoundary: TypeAlias = Literal["supported", "uncertain", "unsupported", "unmatched"]
CapabilityRule: TypeAlias = Literal[
    "SUP-1",
    "SUP-2",
    "SUP-3",
    "SUP-4",
    "SUP-5",
    "UNC-1",
    "UNC-2",
    "LIM-1",
    "LIM-2",
    "none",
]

CAPABILITY_CLASSIFIER_SYSTEM_PROMPT: Final = """You are a task-level probability forecaster for a model router. You receive the
task's opening instruction and, when present, its latest user follow-up, plus
the qualitative capability card below.

Forecast one binary event:

SUCCESS means that the efficient agent completes the whole task correctly on
one fresh run under the actual harness, tools, and budget, as judged by the
final verifier. FAILURE means any other outcome. The two outcomes are
exhaustive.

Use only evidence in the instruction and the capability card. Do not assume
hidden repository state, unmentioned tools, validators, documentation, access,
or future work habits. Do not invent empirical counts, success rates, or base
rates. The capability card is qualitative evidence, not a measured prior.

# Assessment procedure

1. State the crux: the hardest material requirement for whole-task success.
2. Select the one capability rule that best describes the crux. Use
   primary_rule=none and capability_boundary=unmatched when no rule applies.
   Rule ids are opaque labels. Do not infer a boundary from an id's spelling.
3. Privately identify the strongest instruction-visible reasons for SUCCESS
   and FAILURE, then imagine the most likely concrete failure.
4. Privately consider material unknowns. Missing information should limit
   extreme estimates, but it is not evidence that p_solve must equal 0.50.
5. Estimate p_solve last. It is the probability of whole-task SUCCESS, not
   confidence in this assessment, a route recommendation, or a cost judgment.

Interpret probabilities as natural frequencies. If p_solve is 0.70 for 100
comparable fresh runs, about 70 should succeed and 30 should fail. Use the full
range when justified. Reserve 0.00 and 1.00 for outcomes that are logically
impossible or certain under the visible contract. Supported does not mean 1.00,
and unsupported does not mean 0.00. The downstream routing threshold is not
part of this forecast.

# Efficient-agent capability card

The route verbs in this source card are inherited qualitative descriptions.
They do not ask you to output a route and do not assign a fixed probability to
any boundary.

- SUP-1 [supported]: Route to the Efficient model when the task provides a complete output contract and a deterministic local validator that covers the material requirements.
- SUP-2 [supported]: Route to the Efficient model when all required inputs are available, the target environment can be inspected, and correctness can be verified end-to-end without inaccessible external state.
- SUP-3 [supported]: Route to the Efficient model when mathematical behavior, interfaces, shapes, data types, tolerances, and performance requirements are explicit and exercised by a representative harness.
- SUP-4 [supported]: Route to the Efficient model when the required mechanism is identified, the relevant search space is bounded, and the success condition is executable. Do not infer this rule merely from the task's technical domain.
- SUP-5 [supported]: Route to the Efficient model when reconstruction or behavioral reproduction is constrained by an executable reference, parser, format specification, or checker strong enough to distinguish correct from merely plausible output.
- UNC-1 [uncertain]: Treat the route as uncertain when multiple reasonable interpretations of preprocessing, representation, indexing, naming, or output placement would produce different results and neither the instructions nor a validator resolve the choice.
- UNC-2 [uncertain]: Treat the route as uncertain when success requires finding every relevant item across heterogeneous inputs or environment state, but the task does not define the search boundary or provide a completeness check.
- LIM-1 [unsupported]: Prefer the Capable model when correctness depends primarily on extracting precise information from noisy visual, temporal, or rendered media and no machine-checkable extraction or replay mechanism is available.
- LIM-2 [unsupported]: Prefer the Capable model when success depends on reproducing undocumented reference behavior, hidden intermediate state, or an unknown configuration, and small deviations fail despite satisfying the visible specification.

# Output

Return exactly one JSON object matching the response schema supplied with the
request. Do not include markdown or commentary.

p_solve must be between 0.00 and 1.00. p_fail is exactly 1.00 - p_solve and
must not be emitted separately. Do not output recommended_route, confidence,
abstain, counts, task totals, empirical rates, or any other field."""

_BOUNDARY_STEPS: Final = MappingProxyType(
    {
        "supported": 0,
        "uncertain": 1,
        "unmatched": 1,
        "unsupported": 2,
    }
)

_RULE_BOUNDARIES: Final = MappingProxyType(
    {
        "SUP-1": "supported",
        "SUP-2": "supported",
        "SUP-3": "supported",
        "SUP-4": "supported",
        "SUP-5": "supported",
        "UNC-1": "uncertain",
        "UNC-2": "uncertain",
        "LIM-1": "unsupported",
        "LIM-2": "unsupported",
        "none": "unmatched",
    }
)


class CapabilityClassifierVerdict(BaseModel):
    """Strict structured verdict returned by the capability forecaster."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    crux: str = Field(min_length=1)
    primary_rule: CapabilityRule
    capability_boundary: CapabilityBoundary
    p_solve: StrictFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_rule_boundary_pair(self) -> "CapabilityClassifierVerdict":
        if not self.crux.strip():
            raise ValueError("crux must contain non-whitespace text")
        expected: Final = _RULE_BOUNDARIES[self.primary_rule]
        if self.capability_boundary != expected:
            raise ValueError(
                f"primary_rule {self.primary_rule!r} requires capability_boundary {expected!r}, "
                f"got {self.capability_boundary!r}"
            )
        return self

    def routing_threshold(self, base_threshold: float, threshold_step: float) -> float:
        """Required efficient-model solve probability for this boundary."""
        return base_threshold + _BOUNDARY_STEPS[self.capability_boundary] * threshold_step

    def meets_routing_threshold(self, threshold: float) -> bool:
        """Inclusive comparison with Switchyard's one-epsilon rounding guard."""
        return self.p_solve >= threshold or abs(threshold - self.p_solve) <= float_info.epsilon


_CAPABILITY_CLASSIFIER_RESPONSE_FORMAT_JSON: Final = """{
  "type": "json_schema",
  "json_schema": {
    "name": "CapabilityClassifierDecision",
    "strict": true,
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": ["crux", "primary_rule", "capability_boundary", "p_solve"],
      "properties": {
        "crux": {"type": "string", "minLength": 1},
        "primary_rule": {
          "type": "string",
          "enum": ["SUP-1", "SUP-2", "SUP-3", "SUP-4", "SUP-5", "UNC-1", "UNC-2", "LIM-1", "LIM-2", "none"]
        },
        "capability_boundary": {
          "type": "string",
          "enum": ["supported", "uncertain", "unsupported", "unmatched"]
        },
        "p_solve": {"type": "number", "minimum": 0.0, "maximum": 1.0}
      }
    }
  }
}"""

_RESPONSE_FORMAT_ADAPTER: Final = TypeAdapter(Mapping[str, object])


def capability_classifier_response_format() -> Mapping[str, object]:
    """Fresh copy of Switchyard's packaged strict JSON Schema wrapper."""
    return _RESPONSE_FORMAT_ADAPTER.validate_json(_CAPABILITY_CLASSIFIER_RESPONSE_FORMAT_JSON)


def parse_capability_classifier_verdict(content: str) -> CapabilityClassifierVerdict:
    """Parse raw JSON or the fenced JSON shape tolerated by Switchyard."""
    text: Final = content.strip()
    if not text.startswith("```"):
        return CapabilityClassifierVerdict.model_validate_json(text)
    unfenced: Final = text.removeprefix("```").removeprefix("json").lstrip("\n\r")
    return CapabilityClassifierVerdict.model_validate_json(unfenced.removesuffix("```").strip())

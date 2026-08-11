"""Calibration examples for the LLM classifier's built-in rubric.

A preset contributes worked examples and nothing else: the tier criteria, the trust-boundary
paragraph, and the closing line are shared by every preset. Stating the tier boundaries as prose
leaves them where the reader of that prose puts them, and a rubric written against consumer chat puts
"non-trivial code, multi-step technical work" at the top of the scale. That is the median request in
developer and agent traffic, so ordinary engineering reads as top-tier and the router pays for the
most expensive model on it. Examples move the boundary where more rules only restate the taxonomy,
which is why the presets differ in examples rather than in criteria.

`AGENTIC` carries engineering anchors that place routine installs, builds, multi-file edits, and
standard debugging at MEDIUM. `CHAT` omits them, for a deployment serving only conversational
traffic where those anchors describe requests it never sees.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, NamedTuple

from .config import ComplexityTier, RubricPreset


class _CalibrationExample(NamedTuple):
    request: str
    tiers: tuple[ComplexityTier, ...]
    note: str | None
    presets: frozenset[RubricPreset]


class _ExampleGroup(NamedTuple):
    heading: str
    examples: tuple[_CalibrationExample, ...]
    presets: frozenset[RubricPreset]


_EVERY_PRESET: Final = frozenset(RubricPreset)
_AGENTIC_ONLY: Final = frozenset({RubricPreset.AGENTIC})

_GENERAL_HEADING: Final = "Calibration examples:"
_ENGINEERING_HEADING: Final = (
    "Calibration on engineering tasks, which is where the boundary matters most. These are typical of agent "
    "and terminal work:"
)

_GENERAL_EXAMPLES: Final[tuple[_CalibrationExample, ...]] = (
    _CalibrationExample('"what\'s the capital of France?"', (ComplexityTier.SIMPLE,), None, _EVERY_PRESET),
    _CalibrationExample(
        'three paragraphs of context ending in "what time does the building open on Saturdays?"',
        (ComplexityTier.SIMPLE,),
        "the ask is a lookup",
        _EVERY_PRESET,
    ),
    _CalibrationExample(
        '"Think step by step and reason carefully: what is 7 times 8?"',
        (ComplexityTier.SIMPLE,),
        "the framing does not change the task",
        _EVERY_PRESET,
    ),
    _CalibrationExample(
        '"in python, how do I check if a dict has a key?"',
        (ComplexityTier.SIMPLE,),
        "technical vocabulary but one obvious answer",
        _EVERY_PRESET,
    ),
    _CalibrationExample('"write a regex for a US phone number"', (ComplexityTier.MEDIUM,), None, _EVERY_PRESET),
    _CalibrationExample('"explain REST vs gRPC and when to use each"', (ComplexityTier.MEDIUM,), None, _EVERY_PRESET),
    _CalibrationExample(
        '"implement a distributed token bucket rate limiter on Redis, correct under concurrency"',
        (ComplexityTier.COMPLEX,),
        None,
        _EVERY_PRESET,
    ),
    _CalibrationExample(
        '"why does our p99 latency triple when we double the replica count?"',
        (ComplexityTier.COMPLEX,),
        "casual and short, but the answer needs a real causal model",
        _AGENTIC_ONLY,
    ),
    _CalibrationExample(
        '"prove the halting problem is undecidable"',
        (ComplexityTier.COMPLEX, ComplexityTier.REASONING),
        "short but genuinely hard",
        _EVERY_PRESET,
    ),
    _CalibrationExample(
        '"A farmer has 17 sheep. All but 9 die. How many are left?"',
        (ComplexityTier.REASONING,),
        "the arithmetic is trivial and the trap is not",
        _AGENTIC_ONLY,
    ),
    _CalibrationExample(
        '"should we use Postgres or Mongo given these constraints? commit to an answer"',
        (ComplexityTier.REASONING,),
        None,
        _EVERY_PRESET,
    ),
    _CalibrationExample(
        'after a turn offering to work through a Raft safety argument, a bare "yes"',
        (ComplexityTier.REASONING,),
        "it inherits that work",
        _EVERY_PRESET,
    ),
    _CalibrationExample(
        'after a turn about the weather API, a bare "yes"',
        (ComplexityTier.SIMPLE,),
        "it inherits that work",
        _EVERY_PRESET,
    ),
)

_ENGINEERING_EXAMPLES: Final[tuple[_CalibrationExample, ...]] = (
    _CalibrationExample(
        '"write /app/ode_solve.py, a small RK4 initial value problem solver, with the interface the tests import"',
        (ComplexityTier.MEDIUM,),
        None,
        _AGENTIC_ONLY,
    ),
    _CalibrationExample(
        '"set up a Jupyter server with token auth on port 8888 and confirm it serves"',
        (ComplexityTier.MEDIUM,),
        None,
        _AGENTIC_ONLY,
    ),
    _CalibrationExample(
        '"update this Fortran project\'s build to use gfortran instead of the legacy toolchain"',
        (ComplexityTier.MEDIUM,),
        None,
        _AGENTIC_ONLY,
    ),
    _CalibrationExample(
        '"a secret was committed then removed by rewriting history; recover it and prove which commit introduced it"',
        (ComplexityTier.MEDIUM,),
        None,
        _AGENTIC_ONLY,
    ),
    _CalibrationExample(
        '"complete the missing forward pass in this attention-based multiple instance learning model"',
        (ComplexityTier.MEDIUM,),
        None,
        _AGENTIC_ONLY,
    ),
    _CalibrationExample(
        '"solve this 5x4 Huarong Dao sliding block puzzle in the fewest moves"',
        (ComplexityTier.COMPLEX,),
        "it needs a real search formulation",
        _AGENTIC_ONLY,
    ),
    _CalibrationExample(
        '"allocate rare-earth minerals across 1,000 variables under these constraints, optimally"',
        (ComplexityTier.COMPLEX,),
        None,
        _AGENTIC_ONLY,
    ),
    _CalibrationExample(
        '"separability_matrix computes the wrong result for nested CompoundModels; find and fix the root cause"',
        (ComplexityTier.COMPLEX,),
        "the bug is in the semantics, not the syntax",
        _AGENTIC_ONLY,
    ),
)

_GROUPS: Final[tuple[_ExampleGroup, ...]] = (
    _ExampleGroup(_GENERAL_HEADING, _GENERAL_EXAMPLES, _EVERY_PRESET),
    _ExampleGroup(_ENGINEERING_HEADING, _ENGINEERING_EXAMPLES, _AGENTIC_ONLY),
)


def _render_example(example: _CalibrationExample, labels: Mapping[ComplexityTier, str]) -> str:
    tiers: Final = " or ".join(labels[tier] for tier in example.tiers)
    note: Final = f", {example.note}" if example.note else ""
    return f"- {example.request} -> {tiers}{note}"


def calibration_examples_section(preset: RubricPreset, labels: Mapping[ComplexityTier, str]) -> str:
    """The preset's worked-example blocks, each example's tier named in the operator's own vocabulary.

    The examples name tiers on the wire the same way the criteria bullets do, so a router that renamed
    its tiers does not ship a rubric whose examples demand a label absent from the response schema.
    """
    return "\n\n".join(
        "\n".join(
            (
                group.heading,
                *(_render_example(example, labels) for example in group.examples if preset in example.presets),
            )
        )
        for group in _GROUPS
        if preset in group.presets
    )

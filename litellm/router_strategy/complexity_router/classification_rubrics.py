"""Calibration examples for the LLM classifier's built-in rubric.

A preset contributes worked examples and nothing else: the tier criteria, the trust-boundary paragraph,
and the closing line are shared. Stating the tier boundaries as prose alone leaves them where the reader
of that prose puts them, and a rubric written for consumer chat puts "non-trivial code, multi-step
technical work" at the top of the scale. That is the median request in developer and agent traffic, so
ordinary engineering reads as top-tier and the router pays for the most expensive model on it. Examples
move the boundary where more rules only restate the taxonomy.

Each preset holds its examples in full rather than sharing a common block. They are measured artifacts:
the accuracy reported for one describes that exact text, so tuning the chat examples must not silently
edit the agentic ones. `ClassificationRubric.LEGACY` has no examples and so appears nowhere here.

Tiers are written as format placeholders because the response schema's enum is built from the operator's
tier_labels; an example naming a canonical tier would tell the classifier to emit a label it is not
allowed to return.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from .config import ClassificationRubric, ComplexityTier

_CHAT_EXAMPLES: Final = """Calibration examples:
- "what's the capital of France?" -> {SIMPLE}
- three paragraphs of context ending in "what time does the building open on Saturdays?" -> {SIMPLE}, the ask is a lookup
- "Think step by step and reason carefully: what is 7 times 8?" -> {SIMPLE}, the framing does not change the task
- "in python, how do I check if a dict has a key?" -> {SIMPLE}, technical vocabulary but one obvious answer
- "write a regex for a US phone number" -> {MEDIUM}
- "explain REST vs gRPC and when to use each" -> {MEDIUM}
- "implement a distributed token bucket rate limiter on Redis, correct under concurrency" -> {COMPLEX}
- "prove the halting problem is undecidable" -> {COMPLEX} or {REASONING}, short but genuinely hard
- "should we use Postgres or Mongo given these constraints? commit to an answer" -> {REASONING}
- after a turn offering to work through a Raft safety argument, a bare "yes" -> {REASONING}, it inherits that work
- after a turn about the weather API, a bare "yes" -> {SIMPLE}, it inherits that work"""

_AGENTIC_EXAMPLES: Final = """Calibration examples:
- "what's the capital of France?" -> {SIMPLE}
- three paragraphs of context ending in "what time does the building open on Saturdays?" -> {SIMPLE}, the ask is a lookup
- "Think step by step and reason carefully: what is 7 times 8?" -> {SIMPLE}, the framing does not change the task
- "in python, how do I check if a dict has a key?" -> {SIMPLE}, technical vocabulary but one obvious answer
- "write a regex for a US phone number" -> {MEDIUM}
- "explain REST vs gRPC and when to use each" -> {MEDIUM}
- "implement a distributed token bucket rate limiter on Redis, correct under concurrency" -> {COMPLEX}
- "why does our p99 latency triple when we double the replica count?" -> {COMPLEX}, casual and short, but the answer needs a real causal model
- "prove the halting problem is undecidable" -> {COMPLEX} or {REASONING}, short but genuinely hard
- "A farmer has 17 sheep. All but 9 die. How many are left?" -> {REASONING}, the arithmetic is trivial and the trap is not
- "should we use Postgres or Mongo given these constraints? commit to an answer" -> {REASONING}
- after a turn offering to work through a Raft safety argument, a bare "yes" -> {REASONING}, it inherits that work
- after a turn about the weather API, a bare "yes" -> {SIMPLE}, it inherits that work

Calibration on engineering tasks, which is where the boundary matters most. These are typical of agent and terminal work:
- "write /app/ode_solve.py, a small RK4 initial value problem solver, with the interface the tests import" -> {MEDIUM}
- "set up a Jupyter server with token auth on port 8888 and confirm it serves" -> {MEDIUM}
- "update this Fortran project's build to use gfortran instead of the legacy toolchain" -> {MEDIUM}
- "a secret was committed then removed by rewriting history; recover it and prove which commit introduced it" -> {MEDIUM}
- "complete the missing forward pass in this attention-based multiple instance learning model" -> {MEDIUM}
- "solve this 5x4 Huarong Dao sliding block puzzle in the fewest moves" -> {COMPLEX}, it needs a real search formulation
- "allocate rare-earth minerals across 1,000 variables under these constraints, optimally" -> {COMPLEX}
- "separability_matrix computes the wrong result for nested CompoundModels; find and fix the root cause" -> {COMPLEX}, the bug is in the semantics, not the syntax"""

_CALIBRATION_EXAMPLES: Final[Mapping[ClassificationRubric, str]] = MappingProxyType(
    {
        ClassificationRubric.CHAT: _CHAT_EXAMPLES,
        ClassificationRubric.AGENTIC: _AGENTIC_EXAMPLES,
    }
)


def calibration_examples_section(
    preset: ClassificationRubric, labeled_tiers: Sequence[tuple[ComplexityTier, str]]
) -> str:
    """The preset's worked examples, each tier named in the operator's own vocabulary."""
    return _CALIBRATION_EXAMPLES[preset].format_map(
        MappingProxyType({tier.value: label for tier, label in labeled_tiers})
    )

"""Calibration examples for the LLM classifier's built-in rubric.

A preset contributes worked examples and, for BUSINESS, its own tier criteria: the trust-boundary
paragraph and the closing line are shared. Stating the tier boundaries as prose alone leaves them where the reader
of that prose puts them, and a rubric written for consumer chat puts "non-trivial code, multi-step
technical work" at the top of the scale. That is the median request in developer and agent traffic, so
ordinary engineering reads as top-tier and the router pays for the most expensive model on it. Examples
move the boundary where more rules only restate the taxonomy.

Each preset holds its examples in full rather than sharing a common block. They are measured artifacts:
the accuracy reported for one describes that exact text, so tuning the chat examples must not silently
edit the agentic ones. `ClassificationRubric.LEGACY` has no examples and so appears nowhere here.

BUSINESS carries its own tier criteria because the shared criteria are engineering-flavored ("non-trivial
code, architecture..."), which the business sweep found was the bottleneck for business traffic: swapping
the criteria moved accuracy more than any examples block did. Its criteria draw the COMPLEX/REASONING
boundary at decision-making rather than at analysis, so data-determined diagnosis does not route to the
most expensive tier. The four tier names are unchanged, so escalation, adaptive selection, session
affinity, and tier renames all still apply.

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

_BUSINESS_EXAMPLES: Final = """Calibration examples:
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
- after a turn about the weather API, a bare "yes" -> {SIMPLE}, it inherits that work

Calibration on business and sales tasks, which is where the boundary matters most. Routine drafting, rewriting, and summarizing are everyday work, not analysis:
- "what's our refund policy?" -> {SIMPLE}
- a pasted email thread ending in "when does the Q3 promo end?" -> {SIMPLE}, the ask is a lookup
- "make this one-line reply to a customer sound friendlier" -> {SIMPLE}, one obvious transformation
- "draft a cold outreach email for a VP of Engineering at a fintech" -> {MEDIUM}
- "write an email to re-engage a prospect who went dark after the trial" -> {MEDIUM}, drafting that needs judgment is still routine work
- "summarize this discovery call transcript into next steps and owners" -> {MEDIUM}, long input but routine extraction
- "summarize what changed in this contract redline for a non-lawyer" -> {MEDIUM}
- "write a five-touch outreach sequence for this persona" -> {MEDIUM}, volume of output does not raise the tier
- "build a competitive battlecard against this vendor from these source docs" -> {COMPLEX}
- "here's our cohort table, diagnose why churn spiked" -> {COMPLEX}, hard analysis, but the data determines the answer
- "draft a counter-proposal for a multi-year enterprise renewal under these constraints" -> {COMPLEX}
- analysis that follows from supplied data is {COMPLEX} even when heavy with numbers; reserve {REASONING} for committing to a decision under conflicting tradeoffs or a genuine optimization
- "do we discount to close this quarter or hold price and risk slipping? commit to a recommendation" -> {REASONING}
- "design territories assigning our reps across these named accounts, optimally" -> {REASONING}"""

_CALIBRATION_EXAMPLES: Final[Mapping[ClassificationRubric, str]] = MappingProxyType(
    {
        ClassificationRubric.CHAT: _CHAT_EXAMPLES,
        ClassificationRubric.AGENTIC: _AGENTIC_EXAMPLES,
        ClassificationRubric.BUSINESS: _BUSINESS_EXAMPLES,
    }
)

BUSINESS_TIER_CRITERIA: Final[Mapping[ComplexityTier, str]] = MappingProxyType(
    {
        ComplexityTier.SIMPLE: (
            "greetings, chitchat, or lookups of a fact, policy, price, or date with a short known answer. "
            "Never for analysis, strategy, or non-trivial work, even if the request is only one sentence."
        ),
        ComplexityTier.MEDIUM: (
            "everyday working requests: drafting, rewriting, summarizing, routine explanations, light "
            "reasoning, or minor technical content, regardless of output length."
        ),
        ComplexityTier.COMPLEX: (
            "multi-step analysis or synthesis whose answer is determined by the material at hand: diagnosing "
            "metrics from data, multi-source deliverables, non-trivial code, or specialized domain depth."
        ),
        ComplexityTier.REASONING: (
            "committing to a decision under conflicting tradeoffs, genuine optimization or proof, or anything "
            "where being right requires extended deliberation rather than applying a known procedure."
        ),
    }
)


def calibration_examples_section(
    preset: ClassificationRubric, labeled_tiers: Sequence[tuple[ComplexityTier, str]]
) -> str:
    """The preset's worked examples, each tier named in the operator's own vocabulary."""
    return _CALIBRATION_EXAMPLES[preset].format_map(
        MappingProxyType({tier.value: label for tier, label in labeled_tiers})
    )

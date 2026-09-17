"""Golden generator for the cost suite. Run:

    uv run python tests/e2e/cost_calculation/generate_expected.py

Loads the derived matrix (models x applicable cases), computes the golden for
each exact-spend cell from the rate arithmetic, and writes ``expected.json``
with sorted keys. Default behaviour adds missing cells and drops stale cells
but never overwrites an existing cell's values (a reviewed golden is
authoritative); ``--rewrite`` recomputes everything. Prints added/removed/kept
counts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cost_matrix import (  # noqa: E402  # path bootstrap before package-local imports
    EXPECTED_PATH,
    FRONTIER_MODELS,
    TIER_THRESHOLD_TOKENS,
    Case,
    CostMapEntry,
    FrontierModel,
    cases_for,
    expected_key,
)

# Wires whose response surface reports a real web-search call count; the
# chat-completions wires only expose url_citation annotations, so their billed
# count floors to one.
_EXACT_WEB_SEARCH_WIRES: Final = frozenset(
    {"openai_responses", "anthropic_messages", "gemini_generate", "vertex_generate"}
)


def billed_web_search_calls(model: FrontierModel, case: Case) -> int:
    if case.usage.web_search_calls == 0:
        return 0
    return case.usage.web_search_calls if model.wire in _EXACT_WEB_SEARCH_WIRES else 1


@dataclass(frozen=True, slots=True)
class ExpectedCost:
    """The expected bill split the way the spend row's cost_breakdown reports
    it: the gross input component (cache reads/writes folded in), the output
    component, and the tool-usage component."""

    input_cost: float
    output_cost: float
    tool_cost: float

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost + self.tool_cost


def expected_breakdown(model: FrontierModel, case: Case) -> ExpectedCost:
    """Literal arithmetic on the test-map rates over the scripted token counts.

    Input = fresh*in + read*read + 5m*create + 1h*create_1h + audio_in*audio_in;
    output = text*out + reasoning*reasoning + audio_out*audio_out; plus the
    billed web-search calls at the medium search-context rate. Above-threshold
    swaps every input/output rate to its ``_above_200k_tokens`` variant when
    total prompt tokens exceed the threshold; a service tier swaps input/output
    to the tier's variants, falling back to the base rate when a variant is
    unset -- mirroring _get_token_base_cost in litellm's cost calculator.
    """
    rates: Final[CostMapEntry] = model.override_rates if case.response_model_override else model.rates
    u: Final = case.usage
    prompt_tokens: Final = (
        u.fresh_input_tokens + u.cache_read_tokens + u.cache_write_5m_tokens
        + u.cache_write_1h_tokens + u.audio_input_tokens
    )
    tiered: Final = prompt_tokens > TIER_THRESHOLD_TOKENS
    in_rate: Final = (
        (rates.input_cost_per_token_above_200k_tokens if tiered else None)
        or (rates.input_cost_per_token_priority if case.service_tier == "priority" else None)
        or (rates.input_cost_per_token_flex if case.service_tier == "flex" else None)
        or rates.input_cost_per_token
        or 0.0
    )
    out_rate: Final = (
        (rates.output_cost_per_token_above_200k_tokens if tiered else None)
        or (rates.output_cost_per_token_priority if case.service_tier == "priority" else None)
        or (rates.output_cost_per_token_flex if case.service_tier == "flex" else None)
        or rates.output_cost_per_token
        or 0.0
    )
    # The biller charges cache writes at the input rate when the entry carries
    # no cache_creation rate (cost_calculator.py:2452), and at the 5m write
    # rate when the 1h variant is unset; cache reads bill only at their own
    # rate (zero when the entry lacks one).
    write_5m_rate: Final = rates.cache_creation_input_token_cost or in_rate
    input_cost: Final = (
        u.fresh_input_tokens * in_rate
        + u.cache_read_tokens * (rates.cache_read_input_token_cost or 0.0)
        + u.cache_write_5m_tokens * write_5m_rate
        + u.cache_write_1h_tokens * (rates.cache_creation_input_token_cost_above_1hr or write_5m_rate)
        + u.audio_input_tokens * (rates.input_cost_per_audio_token or 0.0)
    )
    output_cost: Final = (
        u.output_tokens * out_rate
        + u.reasoning_tokens * (rates.output_cost_per_reasoning_token or out_rate)
        + u.audio_output_tokens * (rates.output_cost_per_audio_token or out_rate)
    )
    search: Final = rates.search_context_cost_per_query
    tool_cost: Final = billed_web_search_calls(model, case) * (
        search.search_context_size_medium if search and search.search_context_size_medium else 0.0
    )
    return ExpectedCost(input_cost=input_cost, output_cost=output_cost, tool_cost=tool_cost)


def expected_token_columns(model: FrontierModel, case: Case) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) the spend row should carry, per the
    wire's normalization: Anthropic folds cache read/write into prompt_tokens,
    everyone else reports the totals the wire emitted."""
    u: Final = case.usage
    if model.wire in ("anthropic_messages", "bedrock_converse"):
        return (
            u.fresh_input_tokens + u.cache_read_tokens + u.cache_write_5m_tokens + u.cache_write_1h_tokens,
            u.output_tokens,
        )
    if model.wire in ("gemini_generate", "vertex_generate"):
        return (
            u.fresh_input_tokens + u.cache_read_tokens + u.audio_input_tokens,
            u.output_tokens + u.reasoning_tokens + u.audio_output_tokens,
        )
    if model.wire == "openai_responses":
        return (
            u.fresh_input_tokens + u.cache_read_tokens,
            u.output_tokens + u.reasoning_tokens,
        )
    return (
        u.fresh_input_tokens
        + u.cache_read_tokens
        + u.cache_write_5m_tokens
        + u.cache_write_1h_tokens
        + u.audio_input_tokens,
        u.output_tokens + u.reasoning_tokens + u.audio_output_tokens,
    )


def _proposed() -> dict[str, dict[str, object]]:
    return {
        expected_key(model, case): (
            lambda breakdown, tokens: {
                "spend": breakdown.total,
                "input_cost": breakdown.input_cost,
                "output_cost": breakdown.output_cost,
                "prompt_tokens": tokens[0],
                "completion_tokens": tokens[1],
            }
        )(expected_breakdown(model, case), expected_token_columns(model, case))
        for model in FRONTIER_MODELS
        for case in cases_for(model)
        if case.exact_spend
    }


def main() -> None:
    rewrite: Final = "--rewrite" in sys.argv[1:]
    proposed: Final = _proposed()
    existing: Final = (
        json.loads(EXPECTED_PATH.read_text()) if EXPECTED_PATH.exists() else {}
    )
    merged: Final = {
        key: (proposed[key] if rewrite or key not in existing else existing[key])
        for key in sorted(proposed)
    }
    added: Final = sum(1 for key in proposed if key not in existing)
    removed: Final = sum(1 for key in existing if key not in proposed)
    kept: Final = sum(1 for key in proposed if key in existing and not rewrite)
    rewritten: Final = sum(1 for key in proposed if key in existing and rewrite)
    EXPECTED_PATH.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(
        f"expected.json: {added} added, {removed} removed, {kept} kept, "
        f"{rewritten} rewritten ({len(merged)} cells)"
    )


if __name__ == "__main__":
    main()

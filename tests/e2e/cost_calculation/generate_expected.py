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
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from cost_matrix import (
    EXPECTED_PATH,
    FRONTIER_MODELS,
    TIER_THRESHOLD_TOKENS,
    Case,
    CostMapEntry,
    ExpectedCell,
    FrontierModel,
    cases_for,
    expected_key,
)
from pydantic import TypeAdapter


def _first_present(*rates: float | None) -> float | None:
    return next((rate for rate in rates if rate is not None), None)


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
    billed web-search calls at the medium search-context rate. Every billed
    token is a token the provider charged for: a component whose entry has no
    dedicated rate bills at the ordinary input or output rate, and a present
    rate (including an explicit 0.0) is authoritative. When the total prompt
    tokens exceed the threshold, input/output rates come from the
    ``_above_200k_tokens`` variants; a service tier takes its ``_priority`` or
    ``_flex`` variant when the entry carries one, and otherwise bills at the
    base rate.
    """
    rates: Final[CostMapEntry] = model.override_rates if case.response_model_override else model.rates
    u: Final = case.usage
    prompt_tokens: Final = (
        u.fresh_input_tokens + u.cache_read_tokens + u.cache_write_5m_tokens
        + u.cache_write_1h_tokens + u.audio_input_tokens
    )
    tiered: Final = prompt_tokens > TIER_THRESHOLD_TOKENS
    in_rate: Final = (
        _first_present(
            rates.input_cost_per_token_above_200k_tokens if tiered else None,
            rates.input_cost_per_token_priority if case.service_tier == "priority" else None,
            rates.input_cost_per_token_flex if case.service_tier == "flex" else None,
            rates.input_cost_per_token,
        )
        or 0.0
    )
    out_rate: Final = (
        _first_present(
            rates.output_cost_per_token_above_200k_tokens if tiered else None,
            rates.output_cost_per_token_priority if case.service_tier == "priority" else None,
            rates.output_cost_per_token_flex if case.service_tier == "flex" else None,
            rates.output_cost_per_token,
        )
        or 0.0
    )
    read_rate: Final = _first_present(rates.cache_read_input_token_cost, in_rate) or 0.0
    write_rate: Final = _first_present(rates.cache_creation_input_token_cost, in_rate) or 0.0
    write_1h_rate: Final = (
        _first_present(rates.cache_creation_input_token_cost_above_1hr, write_rate) or 0.0
    )
    audio_in_rate: Final = _first_present(rates.input_cost_per_audio_token, in_rate) or 0.0
    reasoning_rate: Final = _first_present(rates.output_cost_per_reasoning_token, out_rate) or 0.0
    audio_out_rate: Final = _first_present(rates.output_cost_per_audio_token, out_rate) or 0.0
    input_cost: Final = (
        u.fresh_input_tokens * in_rate
        + u.cache_read_tokens * read_rate
        + u.cache_write_5m_tokens * write_rate
        + u.cache_write_1h_tokens * write_1h_rate
        + u.audio_input_tokens * audio_in_rate
    )
    output_cost: Final = (
        u.output_tokens * out_rate
        + u.reasoning_tokens * reasoning_rate
        + u.audio_output_tokens * audio_out_rate
    )
    search: Final = rates.search_context_cost_per_query
    medium_rate: Final = (
        search.search_context_size_medium if search is not None else None
    )
    if u.web_search_calls and medium_rate is None:
        raise ValueError(
            f"{model.map_key}: case {case.name} bills {u.web_search_calls} web-search "
            "calls but the entry has no search_context_cost_per_query medium rate"
        )
    tool_cost: Final = u.web_search_calls * (medium_rate if medium_rate is not None else 0.0)
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


def _cell(model: FrontierModel, case: Case) -> ExpectedCell:
    breakdown: Final = expected_breakdown(model, case)
    prompt_tokens, completion_tokens = expected_token_columns(model, case)
    return ExpectedCell(
        spend=breakdown.total,
        input_cost=breakdown.input_cost,
        output_cost=breakdown.output_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _proposed() -> Mapping[str, ExpectedCell]:
    return MappingProxyType(
        {
            expected_key(model, case): _cell(model, case)
            for model in FRONTIER_MODELS
            for case in cases_for(model)
            if case.exact_spend
        }
    )


def main() -> None:
    rewrite: Final = "--rewrite" in sys.argv[1:]
    proposed: Final = _proposed()
    proposed_values: Final = {key: cell.model_dump() for key, cell in proposed.items()}
    existing: Final[Mapping[str, ExpectedCell]] = (
        TypeAdapter(dict[str, ExpectedCell]).validate_python(
            json.loads(EXPECTED_PATH.read_text())
        )
        if EXPECTED_PATH.exists()
        else {}
    )
    merged: Final = {
        key: (
            proposed_values[key]
            if rewrite or key not in existing
            else existing[key].model_dump()
        )
        for key in sorted(proposed_values)
    }
    added: Final = sum(1 for key in proposed_values if key not in existing)
    removed: Final = sum(1 for key in existing if key not in proposed_values)
    kept: Final = sum(1 for key in proposed_values if key in existing and not rewrite)
    rewritten: Final = sum(1 for key in proposed_values if key in existing and rewrite)
    EXPECTED_PATH.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(  # noqa: T201  # CLI summary is the tool output
        f"expected.json: {added} added, {removed} removed, {kept} kept, "
        f"{rewritten} rewritten ({len(merged)} cells)"
    )


if __name__ == "__main__":
    main()

from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import (
    BranchFacts,
    FeeBilling,
    ReceiptFacts,
    adjusted_cost,
    billed_token_rates,
    branch_receipt,
    part_costs,
)
from litellm.rust_bridge._native import NativeReceipt
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

Resolve = Callable[..., float]


@pytest.fixture
def resolve_cost_discount(monkeypatch: pytest.MonkeyPatch) -> Resolve:
    """The discount fraction the Rust pricing core applies under ``config``, 0.0 when no key matches."""

    def resolve(
        config: Mapping[str, float], provider: str | None, model: str | None, region: str | None = None
    ) -> float:
        monkeypatch.setattr(litellm, "cost_discount_config", dict(config))
        adjusted: Final = adjusted_cost(1.0, provider, model, region)
        assert adjusted.cost == 1.0 - adjusted.discount_percent
        return adjusted.discount_percent

    return resolve


def test_a_discount_config_replaced_twice_prices_with_the_one_installed_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cost_margin_config", {})

    def priced_with(fraction: float) -> float:
        monkeypatch.setattr(litellm, "cost_discount_config", {"anthropic": fraction})
        return adjusted_cost(10.0, "anthropic").cost

    costs: Final = tuple(priced_with(fraction) for fraction in (0.1, 0.5, 0.1))

    assert costs == pytest.approx((9.0, 5.0, 9.0))


@pytest.mark.parametrize(
    ("config", "edit", "expected"),
    [
        ("cost_discount_config", ("anthropic", 0.5), 5.0),
        ("cost_margin_config", ("global", 0.1), 11.0),
    ],
)
def test_a_cost_config_edited_in_place_applies_on_the_next_call(
    monkeypatch: pytest.MonkeyPatch, config: str, edit: tuple[str, float], expected: float
) -> None:
    monkeypatch.setattr(litellm, "cost_discount_config", {})
    monkeypatch.setattr(litellm, "cost_margin_config", {})
    unadjusted: Final = adjusted_cost(10.0, "anthropic").cost

    monkeypatch.setitem(getattr(litellm, config), *edit)
    edited: Final = adjusted_cost(10.0, "anthropic").cost

    assert unadjusted == pytest.approx(10.0)
    assert edited == pytest.approx(expected)


def test_resolve_cost_discount_exact_key_beats_glob_and_bare(resolve_cost_discount: Resolve):
    config = {"vertex_ai/claude-sonnet-4-5": 0.3, "vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.3


def test_resolve_cost_discount_glob_beats_bare_provider(resolve_cost_discount: Resolve):
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.2


def test_resolve_cost_discount_longest_literal_glob_wins(resolve_cost_discount: Resolve):
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai/claude-sonnet-*": 0.25, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.25


def test_resolve_cost_discount_strips_provider_prefix_from_model(resolve_cost_discount: Resolve):
    config = {"vertex_ai/claude-*": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "vertex_ai/claude-sonnet-4-5") == 0.2


@pytest.mark.parametrize("model", ["openai/openai/gpt-4", "openai/openai/openai/gpt-4"])
def test_resolve_cost_discount_collapses_repeated_provider_prefix(model: str, resolve_cost_discount: Resolve):
    config = {"openai": 0.05, "openai/gpt-*": 0.20}
    assert resolve_cost_discount(config, "openai", model) == 0.20


def test_resolve_cost_discount_prefixed_pattern_beats_stripped_glob(resolve_cost_discount: Resolve):
    config = {"openrouter": 0.05, "openrouter/openrouter/aurora-*": 0.30, "openrouter/aurora-*": 0.20}
    assert resolve_cost_discount(config, "openrouter", "openrouter/openrouter/aurora-alpha") == 0.30


def test_resolve_cost_discount_fully_stripped_name_still_matches(resolve_cost_discount: Resolve):
    config = {"openrouter": 0.05, "openrouter/aurora-*": 0.20}
    assert resolve_cost_discount(config, "openrouter", "openrouter/openrouter/aurora-alpha") == 0.20


def test_resolve_cost_discount_exact_prefixed_name_beats_stripped_exact(resolve_cost_discount: Resolve):
    config = {"openrouter/openrouter/auto": 0.40, "openrouter/auto": 0.10}
    assert resolve_cost_discount(config, "openrouter", "openrouter/openrouter/auto") == 0.40
    assert resolve_cost_discount(config, "openrouter", "openrouter/auto") == 0.10


def test_resolve_cost_discount_longest_literal_across_candidate_names(resolve_cost_discount: Resolve):
    config = {"openai/*": 0.10, "openai/gpt-*": 0.20}
    assert resolve_cost_discount(config, "openai", "openai/openai/gpt-4") == 0.20


def test_resolve_cost_discount_glob_skips_prefixed_candidate_names(resolve_cost_discount: Resolve):
    config = {"openai/o*": 0.30, "openai/gpt-*": 0.20}
    assert resolve_cost_discount(config, "openai", "openai/openai/gpt-4") == 0.20


def test_resolve_cost_discount_prefixless_glob_only_sees_fully_stripped_name(resolve_cost_discount: Resolve):
    config = {"openai": 0.05, "openai/o*": 0.30}
    assert resolve_cost_discount(config, "openai", "openai/openai/gpt-4") == 0.05


def test_resolve_cost_discount_slash_glob_sees_prefixed_candidate_names(resolve_cost_discount: Resolve):
    config = {"openrouter": 0.05, "openrouter/*/auto": 0.25}
    assert resolve_cost_discount(config, "openrouter", "openrouter/openrouter/auto") == 0.25


def test_resolve_cost_discount_non_matching_model_falls_back_to_bare(resolve_cost_discount: Resolve):
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "gemini-3-pro-preview") == 0.05


def test_resolve_cost_discount_non_matching_model_no_bare_returns_none(resolve_cost_discount: Resolve):
    config = {"vertex_ai/claude-*": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "gemini-3-pro-preview") == 0.0


def test_resolve_cost_discount_other_provider_untouched(resolve_cost_discount: Resolve):
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "openai", "claude-sonnet-4-5") == 0.0


def test_resolve_cost_discount_none_provider_returns_none(resolve_cost_discount: Resolve):
    config = {"vertex_ai": 0.05}
    assert resolve_cost_discount(config, None, "claude-sonnet-4-5") == 0.0


def test_resolve_cost_discount_empty_provider_returns_none(resolve_cost_discount: Resolve):
    config = {"vertex_ai": 0.05}
    assert resolve_cost_discount(config, "", "claude-sonnet-4-5") == 0.0


def test_resolve_cost_discount_none_model_only_bare_matches(resolve_cost_discount: Resolve):
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", None) == 0.05
    assert resolve_cost_discount({"vertex_ai/claude-*": 0.2}, "vertex_ai", None) == 0.0


def test_resolve_cost_discount_char_class_counts_as_wildcard(resolve_cost_discount: Resolve):
    config = {"vertex_ai/claude-sonnet-*": 0.3, "vertex_ai/claude-[abcdefghijklmnopqrstuvwxyz]*": 0.1}
    assert resolve_cost_discount(config, "vertex_ai", "claude-sonnet-4-5") == 0.3


def test_resolve_cost_discount_unclosed_bracket_matches_literally(resolve_cost_discount: Resolve):
    config = {"vertex_ai/weird[": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "weird[") == 0.2
    assert resolve_cost_discount(config, "vertex_ai", "weirdx") == 0.0


def test_resolve_cost_discount_unclosed_negated_class_counts_as_literal(resolve_cost_discount: Resolve):
    config = {"vertex_ai/gemini-?*": 0.2, "vertex_ai/gemini-[!]*": 0.3}
    assert resolve_cost_discount(config, "vertex_ai", "gemini-[!]") == 0.3


def test_resolve_cost_discount_char_class_with_leading_bracket(resolve_cost_discount: Resolve):
    config = {"vertex_ai/a[]x]*": 0.1, "vertex_ai/a]*": 0.2}
    assert resolve_cost_discount(config, "vertex_ai", "a]q") == 0.2
    assert resolve_cost_discount(config, "vertex_ai", "axq") == 0.1


def test_resolve_cost_discount_glob_crosses_slash_in_model(resolve_cost_discount: Resolve):
    config = {"bedrock/*anthropic.claude-*": 0.15, "bedrock": 0.05}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-east-1/anthropic.claude-v2:1") == 0.15


def test_resolve_cost_discount_exact_pattern_key_does_not_count_as_exact(resolve_cost_discount: Resolve):
    config = {"vertex_ai/claude-*": 0.2, "vertex_ai": 0.05}
    assert resolve_cost_discount(config, "vertex_ai", "vertex_ai") == 0.05


def test_resolve_cost_discount_region_stripped_name_matches_pattern(resolve_cost_discount: Resolve):
    config = {"bedrock": 0.05, "bedrock/claude-*": 0.2}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-gov-west-1/claude-x", "us-gov-west-1") == 0.2


def test_resolve_cost_discount_region_stripped_exact_beats_glob(resolve_cost_discount: Resolve):
    config = {"bedrock/claude-x": 0.3, "bedrock/claude-*": 0.2}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-gov-west-1/claude-x", "us-gov-west-1") == 0.3


def test_resolve_cost_discount_region_qualified_pattern_wins_longest_literal(resolve_cost_discount: Resolve):
    config = {"bedrock": 0.05, "bedrock/us-gov-west-1/*": 0.4, "bedrock/claude-*": 0.2}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-gov-west-1/claude-x", "us-gov-west-1") == 0.4


def test_resolve_cost_discount_prefixless_glob_cannot_see_region_prefix(resolve_cost_discount: Resolve):
    config = {"bedrock": 0.05, "bedrock/us*": 0.9}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-gov-west-1/claude-x", "us-gov-west-1") == 0.05


def test_resolve_cost_discount_no_region_name_keeps_qualified_name_hidden(resolve_cost_discount: Resolve):
    config = {"bedrock": 0.05, "bedrock/claude-*": 0.2}
    assert resolve_cost_discount(config, "bedrock", "bedrock/us-gov-west-1/claude-x") == 0.05


_PRICED_ROW: Final = {"litellm_provider": "openai", "input_cost_per_token": 2.0, "output_cost_per_token": 1.0}
_LOOKED_UP_ROW: Final = {
    "litellm_provider": "openai",
    "input_cost_per_token": 5.0,
    "output_cost_per_token": 1.0,
    "cache_read_input_token_cost": 0.5,
}
_ROUTER_ROW: Final = {"litellm_provider": "azure_ai", "input_cost_per_token": 0.25}


def _receipt(facts: ReceiptFacts) -> NativeReceipt:
    return branch_receipt(
        "model",
        "openai",
        Usage(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=4),
        ),
        _PRICED_ROW,
        branch=BranchFacts(rules="generic"),
        receipt=facts,
    )


@pytest.mark.parametrize(
    ("facts", "cache_read_rate"),
    [
        (ReceiptFacts(), 2.0),
        (ReceiptFacts(breakdown="row", breakdown_row=_LOOKED_UP_ROW), 0.5),
        (ReceiptFacts(breakdown="absent"), None),
    ],
)
def test_the_stored_breakdown_reads_the_row_the_caller_names(facts: ReceiptFacts, cache_read_rate: float | None) -> None:
    receipt: Final = _receipt(facts)
    rates: Final = billed_token_rates(receipt)

    assert receipt.prompt_cost == pytest.approx(10 * 2.0)
    assert (None if rates is None else rates.cache_read_input_token_cost) == cache_read_rate


@pytest.mark.parametrize(
    ("billed", "prompt_cost", "fees"),
    [("prompt", 20.0 + 2.5, {}), ("additional", 20.0, {"Azure Model Router Flat Cost": 2.5})],
)
def test_the_router_fee_bills_where_the_caller_says(
    billed: FeeBilling, prompt_cost: float, fees: Mapping[str, float]
) -> None:
    receipt: Final = _receipt(ReceiptFacts(router_fee=billed, fee_row=_ROUTER_ROW, tool_fees=1.0))

    assert receipt.prompt_cost == pytest.approx(prompt_cost)
    assert dict(part_costs(receipt)) == pytest.approx(fees)
    assert receipt.tools_cost == 1.0
    assert receipt.total == pytest.approx(prompt_cost + 2.0 + sum(fees.values()) + 1.0)

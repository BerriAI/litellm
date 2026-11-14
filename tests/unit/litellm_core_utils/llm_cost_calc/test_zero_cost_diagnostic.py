from collections.abc import Mapping
from typing import Final

import pytest

from litellm.litellm_core_utils.llm_cost_calc.zero_cost_diagnostic import (
    ZERO_COST_COUNTER_NAME,
    diagnose_zero_cost,
    is_explicitly_free_usage,
    used_pricing_keys,
    zero_cost_warning,
)
from litellm.types.llms.base import CachedTokensDetails
from litellm.types.utils import CompletionTokensDetailsWrapper, PromptTokensDetailsWrapper, Usage

PER_SECOND_ENTRY: Final = {"cost_per_second": 0.00042}
FREE_ENTRY: Final = {"input_cost_per_token": 0, "output_cost_per_token": 0, "cache_read_input_token_cost": 2e-08}
PRICED_ENTRY: Final = {"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}
TEXT_USAGE: Final = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)


@pytest.mark.parametrize(
    ("usage", "rates", "expected"),
    [
        (TEXT_USAGE, FREE_ENTRY, True),
        (TEXT_USAGE, PRICED_ENTRY, False),
        (TEXT_USAGE, {}, False),
        (TEXT_USAGE, {"input_cost_per_token": 0.0}, False),
        (TEXT_USAGE, {"input_cost_per_token": False, "output_cost_per_token": 0.0}, False),
        (Usage(prompt_tokens=10, completion_tokens=0), {"input_cost_per_token": 0.0}, True),
        (Usage(prompt_tokens=0, completion_tokens=0), FREE_ENTRY, False),
        (
            Usage(prompt_tokens=10, completion_tokens=0, prompt_tokens_details={"cached_tokens": 10}),
            FREE_ENTRY,
            False,
        ),
        (
            Usage(prompt_tokens=10, completion_tokens=0, prompt_tokens_details={"cached_tokens": 10}),
            {"input_cost_per_token": 0.0},
            True,
        ),
    ],
)
def test_free_usage_requires_explicit_zero_rates_for_every_used_token_kind(
    usage: Usage, rates: Mapping[str, object], expected: bool
) -> None:
    assert is_explicitly_free_usage(usage, rates) is expected


@pytest.mark.parametrize(
    ("input_tokens", "tier_output_rate", "expected"),
    [(10, 0.0, True), (11, 0.0, False), (10, 0.001, False), (10, None, False)],
)
def test_free_usage_uses_the_selected_tier_and_preserves_missing_output_prices(
    input_tokens: int, tier_output_rate: float | None, expected: bool
) -> None:
    rates: Final = {
        "input_cost_per_token": 0.01,
        "tiered_pricing": [
            {"range": [0, 10], "input_cost_per_token": 0.0, "output_cost_per_token": tier_output_rate},
            {"range": [10, 20], "input_cost_per_token": 0.01, "output_cost_per_token": 0.02},
        ],
    }

    assert is_explicitly_free_usage(Usage(prompt_tokens=input_tokens, completion_tokens=5), rates) is expected


@pytest.mark.parametrize("cache_rate", (None, 0.01))
def test_free_cached_usage_uses_the_selected_tier_cache_rate_or_its_input_rate(cache_rate: float | None) -> None:
    rates: Final = {
        "input_cost_per_token": 0.01,
        "cache_read_input_token_cost": 0.02,
        "tiered_pricing": [
            {"range": [0, 20], "input_cost_per_token": 0.0, "cache_read_input_token_cost": cache_rate}
        ],
    }
    usage: Final = Usage(prompt_tokens=10, completion_tokens=0, prompt_tokens_details={"cached_tokens": 10})

    assert is_explicitly_free_usage(usage, rates) is (cache_rate is None)


@pytest.mark.parametrize(
    "tiers",
    [
        ["invalid tier"],
        [],
        [{"range": [0, 20], "output_cost_per_token": 0.0}],
        [{"range": "invalid range", "input_cost_per_token": 0.0, "output_cost_per_token": 0.0}],
    ],
)
def test_invalid_or_inactive_tiers_do_not_make_paid_usage_free(tiers: object) -> None:
    rates: Final = {
        "input_cost_per_token": 0.01,
        "output_cost_per_token": 0.02,
        "tiered_pricing": tiers,
    }

    assert is_explicitly_free_usage(Usage(prompt_tokens=10, completion_tokens=5), rates) is False


def test_missing_pricing_key_names_every_rate_the_usage_needs() -> None:
    diagnostic = diagnose_zero_cost(
        usage=TEXT_USAGE, pricing_model="dep-1", pricing_entry=PER_SECOND_ENTRY, calculation_failed=False
    )

    assert diagnostic == {
        "reason": "missing_pricing_key",
        "pricing_model": "dep-1",
        "missing_pricing_keys": ("input_cost_per_token", "output_cost_per_token"),
    }


def test_only_the_absent_rate_is_reported() -> None:
    diagnostic = diagnose_zero_cost(
        usage=TEXT_USAGE, pricing_model="dep-1", pricing_entry={"input_cost_per_token": 1e-06}, calculation_failed=False
    )

    assert diagnostic is not None
    assert diagnostic["missing_pricing_keys"] == ("output_cost_per_token",)


def test_free_model_stays_silent() -> None:
    assert (
        diagnose_zero_cost(usage=TEXT_USAGE, pricing_model="dep-1", pricing_entry=FREE_ENTRY, calculation_failed=False)
        is None
    )


@pytest.mark.parametrize("calculation_failed", [False, True])
def test_request_without_usage_stays_silent(calculation_failed: bool) -> None:
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    assert (
        diagnose_zero_cost(
            usage=usage, pricing_model="dep-1", pricing_entry=PER_SECOND_ENTRY, calculation_failed=calculation_failed
        )
        is None
    )


@pytest.mark.parametrize(
    "entry",
    [
        {"litellm_provider": "openai", "mode": "chat", "supports_prompt_caching": True},
        {"tiered_pricing": [{"range": [0, 128000], "input_cost_per_token": 0, "output_cost_per_token": 0}]},
        {"tiered_pricing": "not a tier table", "litellm_provider": "openai"},
    ],
)
def test_entry_that_declares_no_rate_stays_silent(entry: Mapping[str, object]) -> None:
    assert (
        diagnose_zero_cost(usage=TEXT_USAGE, pricing_model="dep-1", pricing_entry=entry, calculation_failed=False)
        is None
    )


def test_tiered_rate_counts_as_a_declared_rate() -> None:
    entry = {"tiered_pricing": [{"range": [0, 128000], "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}]}

    diagnostic = diagnose_zero_cost(
        usage=TEXT_USAGE, pricing_model="dep-1", pricing_entry=entry, calculation_failed=False
    )

    assert diagnostic is not None
    assert diagnostic["reason"] == "missing_pricing_key"


def test_priced_entry_that_still_prices_to_zero_is_pricing_not_applied() -> None:
    diagnostic = diagnose_zero_cost(
        usage=TEXT_USAGE, pricing_model="dep-1", pricing_entry=PRICED_ENTRY, calculation_failed=False
    )

    assert diagnostic == {"reason": "pricing_not_applied", "pricing_model": "dep-1", "missing_pricing_keys": ()}


def test_calculator_failure_on_a_priced_entry_is_cost_calculation_error() -> None:
    diagnostic = diagnose_zero_cost(
        usage=TEXT_USAGE, pricing_model="dep-1", pricing_entry=PRICED_ENTRY, calculation_failed=True
    )

    assert diagnostic == {"reason": "cost_calculation_error", "pricing_model": "dep-1", "missing_pricing_keys": ()}


def test_calculator_failure_on_a_free_entry_stays_silent() -> None:
    assert (
        diagnose_zero_cost(usage=TEXT_USAGE, pricing_model="dep-1", pricing_entry=FREE_ENTRY, calculation_failed=True)
        is None
    )


def test_calculator_failure_on_an_entry_that_declares_no_rate_stays_silent() -> None:
    entry: Final = {"litellm_provider": "openai", "mode": "chat", "supports_prompt_caching": True}
    assert (
        diagnose_zero_cost(usage=TEXT_USAGE, pricing_model="dep-1", pricing_entry=entry, calculation_failed=True)
        is None
    )


def test_audio_tokens_need_the_audio_rates() -> None:
    usage = Usage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_tokens_details=PromptTokensDetailsWrapper(audio_tokens=10, text_tokens=0),
        completion_tokens_details=CompletionTokensDetailsWrapper(audio_tokens=5, text_tokens=15),
    )

    assert used_pricing_keys(usage) == (
        "input_cost_per_audio_token",
        "output_cost_per_token",
        "output_cost_per_audio_token",
    )
    diagnostic = diagnose_zero_cost(
        usage=usage, pricing_model="gemini-audio", pricing_entry=PRICED_ENTRY, calculation_failed=False
    )
    assert diagnostic is not None
    assert diagnostic["missing_pricing_keys"] == ("input_cost_per_audio_token", "output_cost_per_audio_token")


def test_cached_audio_uses_its_cache_rate_in_zero_cost_diagnostic() -> None:
    usage: Final = Usage(
        prompt_tokens=20,
        completion_tokens=0,
        total_tokens=20,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=0,
            audio_tokens=20,
            cached_tokens=20,
            cached_tokens_details=CachedTokensDetails(audio_tokens=20),
        ),
    )
    pricing_entry: Final = {
        "input_cost_per_token": 1e-6,
        "input_cost_per_audio_token": 2e-6,
        "cache_read_input_audio_token_cost": 0,
    }

    assert used_pricing_keys(usage) == ("cache_read_input_audio_token_cost",)
    assert (
        diagnose_zero_cost(
            usage=usage,
            pricing_model="gemini-audio",
            pricing_entry=pricing_entry,
            calculation_failed=False,
        )
        is None
    )


def test_warning_names_the_request_the_entry_the_missing_keys_and_the_counter() -> None:
    diagnostic = diagnose_zero_cost(
        usage=TEXT_USAGE, pricing_model="dep-1", pricing_entry=PER_SECOND_ENTRY, calculation_failed=False
    )
    assert diagnostic is not None

    message = zero_cost_warning(
        diagnostic,
        model_group="per-second-priced-chat",
        model="openai/gpt-5.4-nano",
        custom_llm_provider="openai",
        usage=TEXT_USAGE,
    )

    assert "model_group=per-second-priced-chat" in message
    assert "model=openai/gpt-5.4-nano" in message
    assert "provider=openai" in message
    assert "prompt_tokens=10 completion_tokens=20" in message
    assert "pricing entry 'dep-1' has no input_cost_per_token, output_cost_per_token" in message
    assert f'{ZERO_COST_COUNTER_NAME}{{reason="missing_pricing_key"}}' in message

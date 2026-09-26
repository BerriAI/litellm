from collections.abc import Mapping
from typing import Final

import pytest

from litellm.litellm_core_utils.llm_cost_calc.zero_cost_diagnostic import (
    ZERO_COST_COUNTER_NAME,
    diagnose_zero_cost,
    used_pricing_keys,
    zero_cost_warning,
)
from litellm.types.utils import CompletionTokensDetailsWrapper, PromptTokensDetailsWrapper, Usage

PER_SECOND_ENTRY: Final = {"input_cost_per_second": 0.00042, "output_cost_per_second": 0.00042}
FREE_ENTRY: Final = {"input_cost_per_token": 0, "output_cost_per_token": 0, "cache_read_input_token_cost": 2e-08}
PRICED_ENTRY: Final = {"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}
TEXT_USAGE: Final = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)


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

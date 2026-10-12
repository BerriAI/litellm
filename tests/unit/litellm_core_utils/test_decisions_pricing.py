from typing import Final, cast

import pytest

import litellm
from litellm import cost_calculator
from litellm.litellm_core_utils.decisions_pricing import (
    apply_decisions_pricing,
    decisions_pricing_fields,
    is_decisions_call_type,
)
from litellm.types.utils import CallTypesLiteral, PromptTokensDetailsWrapper, Usage

_ROW: Final[dict[str, object]] = {
    "input_cost_per_token": 1e-07,
    "input_cost_per_token_above_272k_tokens": 2e-07,
    "output_cost_per_token": 5e-07,
    "output_cost_per_token_decisions": 0.0,
    "cache_read_input_token_cost": 1e-08,
    "cache_read_input_token_cost_decisions": 1e-07,
    "litellm_provider": "openai",
    "mode": "chat",
}


def test_is_decisions_call_type_covers_decisions_and_systemone_only() -> None:
    assert all(
        is_decisions_call_type(call_type) for call_type in ("decisions", "adecisions", "systemone", "asystemone")
    )
    assert not any(is_decisions_call_type(call_type) for call_type in ("completion", "acompletion", "aresponses"))


def test_decisions_pricing_fields_maps_suffixed_fields_onto_plain_names() -> None:
    assert decisions_pricing_fields(_ROW) == {
        "output_cost_per_token": 0.0,
        "cache_read_input_token_cost": 1e-07,
    }
    assert decisions_pricing_fields({"input_cost_per_token": 1e-07, "mode": "evaluation"}) == {}


def test_apply_decisions_pricing_overrides_plain_fields_and_keeps_the_rest() -> None:
    priced: Final = apply_decisions_pricing(_ROW)

    assert priced == {
        "input_cost_per_token": 1e-07,
        "input_cost_per_token_above_272k_tokens": 2e-07,
        "output_cost_per_token": 0.0,
        "cache_read_input_token_cost": 1e-07,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    assert _ROW["output_cost_per_token"] == 5e-07


def test_apply_decisions_pricing_leaves_rows_without_decisions_fields_unchanged() -> None:
    plain_row: Final[dict[str, object]] = {
        "input_cost_per_token": 4e-08,
        "output_cost_per_token": 0.0,
        "mode": "evaluation",
    }

    assert apply_decisions_pricing(plain_row) == plain_row


_LUNA_USAGE: Final = Usage(
    prompt_tokens=100,
    completion_tokens=10,
    total_tokens=110,
    prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=50),
)


@pytest.fixture
def luna_with_decisions_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(litellm.model_cost, "gpt-6-luna", {**litellm.model_cost["gpt-6-luna"], **_ROW})


@pytest.mark.parametrize("call_type", ["decisions", "adecisions", "systemone", "asystemone"])
def test_cost_per_token_bills_decisions_call_types_from_decisions_fields(
    luna_with_decisions_fields: None, call_type: str
) -> None:
    prompt_cost, completion_cost = cost_calculator.cost_per_token(
        model="gpt-6-luna",
        custom_llm_provider="openai",
        usage_object=_LUNA_USAGE,
        call_type=cast(CallTypesLiteral, call_type),
    )

    assert prompt_cost == pytest.approx(50 * 1e-07 + 50 * 1e-07)
    assert completion_cost == 0.0


def test_cost_per_token_keeps_chat_on_plain_fields(luna_with_decisions_fields: None) -> None:
    prompt_cost, completion_cost = cost_calculator.cost_per_token(
        model="gpt-6-luna",
        custom_llm_provider="openai",
        usage_object=_LUNA_USAGE,
        call_type="completion",
    )

    assert prompt_cost == pytest.approx(50 * 1e-07 + 50 * 1e-08)
    assert completion_cost == pytest.approx(10 * 5e-07)


def test_cost_per_token_falls_back_to_plain_fields_without_decisions_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    plain_row: Final[dict[str, object]] = {
        "input_cost_per_token": 4e-08,
        "output_cost_per_token": 0.0,
        "litellm_provider": "perplexity",
        "mode": "evaluation",
    }
    monkeypatch.setitem(litellm.model_cost, "perplexity/pplx-decider-v1-27b", plain_row)

    prompt_cost, completion_cost = cost_calculator.cost_per_token(
        model="pplx-decider-v1-27b",
        custom_llm_provider="perplexity",
        usage_object=_LUNA_USAGE,
        call_type="adecisions",
    )

    assert prompt_cost == pytest.approx(100 * 4e-08)
    assert completion_cost == 0.0

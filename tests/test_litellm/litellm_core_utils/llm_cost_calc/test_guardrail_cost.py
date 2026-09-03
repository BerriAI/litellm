import os

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
    bedrock_guardrail_cost,
    cost_breakdown_with_guardrail,
    guardrail_information_cost,
)


@pytest.fixture
def synthetic_cost_map(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "bedrock/guardrails": {
                "guardrail_cost_per_unit": {
                    "contentPolicyUnits": 0.00015,
                    "topicPolicyUnits": 0.00015,
                    "wordPolicyUnits": 0.0,
                }
            },
            "bedrock/eu-west-1/guardrails": {"guardrail_cost_per_unit": {"contentPolicyUnits": 0.0002}},
            "bedrock/us-west-2/guardrails": {"guardrail_cost_per_unit": "malformed"},
        },
    )


def test_bedrock_guardrail_cost_prices_each_counter(synthetic_cost_map):
    cost = bedrock_guardrail_cost(
        usage_units={"contentPolicyUnits": 2, "topicPolicyUnits": 1, "wordPolicyUnits": 5},
        aws_region_name="us-east-1",
    )
    assert cost == pytest.approx(0.00045)


def test_bedrock_guardrail_cost_prefers_regional_entry(synthetic_cost_map):
    cost = bedrock_guardrail_cost(usage_units={"contentPolicyUnits": 1}, aws_region_name="eu-west-1")
    assert cost == pytest.approx(0.0002)


def test_bedrock_guardrail_cost_unknown_counter_is_free(synthetic_cost_map):
    assert bedrock_guardrail_cost(usage_units={"someFutureCounter": 3}, aws_region_name="us-east-1") == 0.0


def test_bedrock_guardrail_cost_malformed_regional_entry_falls_back(synthetic_cost_map):
    cost = bedrock_guardrail_cost(usage_units={"contentPolicyUnits": 1}, aws_region_name="us-west-2")
    assert cost == pytest.approx(0.00015)


def test_bedrock_guardrail_cost_no_pricing_entry(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", {})
    assert bedrock_guardrail_cost(usage_units={"contentPolicyUnits": 1}, aws_region_name="us-east-1") == 0.0


def test_shipped_bedrock_guardrail_prices_match_aws_pricing_page(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    assert litellm.model_cost["bedrock/guardrails"]["guardrail_cost_per_unit"] == {
        "automatedReasoningPolicyUnits": 0.00017,
        "contentPolicyImageUnits": 0.00075,
        "contentPolicyUnits": 0.00015,
        "contextualGroundingPolicyUnits": 0.0001,
        "sensitiveInformationPolicyFreeUnits": 0.0,
        "sensitiveInformationPolicyUnits": 0.0001,
        "topicPolicyUnits": 0.00015,
        "wordPolicyUnits": 0.0,
    }
    assert "bedrock/guardrails" not in litellm.bedrock_models


def test_guardrail_information_cost_sums_entries():
    entries = [
        {"guardrail_name": "a", "guardrail_cost": 0.0003},
        {"guardrail_name": "b", "guardrail_cost": None},
        {"guardrail_name": "c"},
        {"guardrail_name": "d", "guardrail_cost": 0.0001},
    ]
    assert guardrail_information_cost(entries) == pytest.approx(0.0004)


def test_guardrail_information_cost_single_entry_and_garbage():
    assert guardrail_information_cost({"guardrail_cost": 0.0001}) == pytest.approx(0.0001)
    assert guardrail_information_cost(None) == 0.0
    assert guardrail_information_cost("not-guardrail-info") == 0.0
    assert guardrail_information_cost([{"guardrail_cost": "bad"}]) == 0.0


def test_guardrail_information_cost_ignores_negative_and_non_finite():
    entries = [
        {"guardrail_name": "forged-negative", "guardrail_cost": -0.005},
        {"guardrail_name": "forged-nan", "guardrail_cost": float("nan")},
        {"guardrail_name": "forged-inf", "guardrail_cost": float("inf")},
        {"guardrail_name": "real", "guardrail_cost": 0.0003},
    ]
    assert guardrail_information_cost(entries) == pytest.approx(0.0003)
    assert guardrail_information_cost({"guardrail_cost": -1.0}) == 0.0


def test_cost_breakdown_with_guardrail_merges_and_creates():
    assert cost_breakdown_with_guardrail(None, 0.0) is None
    untouched = {"input_cost": 0.1, "total_cost": 0.4}
    assert cost_breakdown_with_guardrail(untouched, 0.0) is untouched
    merged = cost_breakdown_with_guardrail({"input_cost": 0.1, "total_cost": 0.4}, 0.0003)
    assert merged is not None
    assert merged["guardrail_cost"] == pytest.approx(0.0003)
    assert merged["total_cost"] == pytest.approx(0.4003)
    assert merged["input_cost"] == pytest.approx(0.1)
    created = cost_breakdown_with_guardrail(None, 0.0003)
    assert created == {"guardrail_cost": 0.0003, "total_cost": 0.0003}


def test_azure_prompt_shield_guardrail_cost_paid_tier_prices_text_records():
    from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
        azure_prompt_shield_guardrail_cost,
    )

    cost = azure_prompt_shield_guardrail_cost(
        usage_units={"text_records": 3, "requests": 1, "input_characters": 2100},
        cost_tier="paid",
        price_per_1000_text_records=0.38,
    )
    assert cost == pytest.approx(0.00114)


def test_azure_prompt_shield_guardrail_cost_free_tier_is_zero():
    from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
        azure_prompt_shield_guardrail_cost,
    )

    assert azure_prompt_shield_guardrail_cost({"text_records": 50}, "free", 0.38) == 0.0


def test_azure_prompt_shield_guardrail_cost_unconfigured_is_none():
    from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
        azure_prompt_shield_guardrail_cost,
    )

    assert azure_prompt_shield_guardrail_cost({"text_records": 50}, None, None) is None


def test_azure_prompt_shield_guardrail_cost_no_text_records_is_zero():
    from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
        azure_prompt_shield_guardrail_cost,
    )

    assert azure_prompt_shield_guardrail_cost({}, None, 0.38) == 0.0


def test_guardrail_information_cost_excludes_entries_marked_not_in_spend():
    entries = [
        {"guardrail_name": "azure-shield", "guardrail_cost": 0.5, "guardrail_cost_in_spend": False},
        {"guardrail_name": "bedrock", "guardrail_cost": 0.0003},
    ]
    assert guardrail_information_cost(entries) == pytest.approx(0.0003)
    assert guardrail_information_cost({"guardrail_cost": 0.5, "guardrail_cost_in_spend": False}) == 0.0
    assert guardrail_information_cost({"guardrail_cost": 0.5, "guardrail_cost_in_spend": True}) == pytest.approx(0.5)


def test_guardrail_information_cost_treats_none_in_spend_as_billed():
    """An explicit ``guardrail_cost_in_spend: None`` (the TypedDict sanctions it)
    keeps the default billed behavior AND must not fail union validation, which
    would silently zero a sibling entry's real cost."""
    assert guardrail_information_cost({"guardrail_cost": 0.5, "guardrail_cost_in_spend": None}) == pytest.approx(0.5)
    entries = [
        {"guardrail_name": "azure-shield", "guardrail_cost": 0.5, "guardrail_cost_in_spend": None},
        {"guardrail_name": "bedrock", "guardrail_cost": 0.0003},
    ]
    assert guardrail_information_cost(entries) == pytest.approx(0.5003)


def test_guardrail_information_cost_skips_malformed_entry_keeps_siblings():
    """Entries are validated one by one: a malformed entry (a custom hook stamping
    a non-boolean guardrail_cost_in_spend) prices to 0.0 by itself and must not
    zero a sibling entry's real billable cost."""
    entries = [
        {"guardrail_name": "custom", "guardrail_cost": 0.5, "guardrail_cost_in_spend": "maybe"},
        {"guardrail_name": "bedrock", "guardrail_cost": 0.0003},
    ]
    assert guardrail_information_cost(entries) == pytest.approx(0.0003)
    assert guardrail_information_cost({"guardrail_cost": 0.5, "guardrail_cost_in_spend": "maybe"}) == 0.0

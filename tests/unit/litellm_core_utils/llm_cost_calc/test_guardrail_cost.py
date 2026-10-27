import json
from typing import Final

import pytest
from pydantic import JsonValue

import litellm
from litellm.litellm_core_utils.llm_cost_calc import guardrail_cost
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
    bedrock_guardrail_cost,
    bedrock_guardrail_cost_by_unit,
    billed_guardrail_cost_by_unit,
    cost_breakdown_with_guardrail,
    guardrail_cost_total,
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


def test_bedrock_guardrail_cost_by_unit_prices_every_counter_it_was_given(synthetic_cost_map):
    """LIT-5652: the daily rollup stores one row per counter, so pricing must come
    back at that grain, keyed exactly like the usage. An explicit 0.0 in the cost
    map is free; a counter the map does not list is unknown (None), never free,
    while the scalar the spend path bills still sums only the known prices."""
    usage = {"contentPolicyUnits": 2, "topicPolicyUnits": 1, "wordPolicyUnits": 5, "someFutureCounter": 3}
    by_unit = bedrock_guardrail_cost_by_unit(usage_units=usage, aws_region_name="us-east-1")
    assert by_unit is not None
    assert by_unit.keys() == usage.keys()
    assert by_unit["contentPolicyUnits"] == pytest.approx(0.0003)
    assert by_unit["topicPolicyUnits"] == pytest.approx(0.00015)
    assert by_unit["wordPolicyUnits"] == 0.0
    assert by_unit["someFutureCounter"] is None
    assert guardrail_cost_total(by_unit) == pytest.approx(0.00045)
    assert guardrail_cost_total(by_unit) == pytest.approx(
        bedrock_guardrail_cost(usage_units=usage, aws_region_name="us-east-1")
    )


def test_bedrock_guardrail_cost_by_unit_is_none_without_pricing_so_unpriced_is_not_free(monkeypatch):
    """The scalar keeps returning 0.0 for the spend path; the per-unit view must
    say "unknown" instead so the rollup stores NULL rather than a $0 that would
    hide the exact silent-spend problem this feature exists to surface."""
    monkeypatch.setattr(litellm, "model_cost", {})
    assert bedrock_guardrail_cost_by_unit(usage_units={"contentPolicyUnits": 1}, aws_region_name="us-east-1") is None


def test_billed_guardrail_cost_by_unit_reads_the_hook_stamp():
    entry = {
        "guardrail_name": "bedrock",
        "guardrail_cost_by_unit": {"contentPolicyUnits": 0.15, "wordPolicyUnits": 0, "someFutureCounter": None},
    }
    assert billed_guardrail_cost_by_unit(entry) == {
        "contentPolicyUnits": 0.15,
        "wordPolicyUnits": 0.0,
        "someFutureCounter": None,
    }


@pytest.mark.parametrize(
    "entry",
    [
        {"guardrail_name": "no-pricing", "guardrail_usage": {"contentPolicyUnits": 1}},
        {"guardrail_cost_by_unit": {"text_records": 0.5}, "guardrail_cost_in_spend": False},
        {"guardrail_cost_by_unit": {"contentPolicyUnits": -0.5}},
        {"guardrail_cost_by_unit": {"contentPolicyUnits": float("nan")}},
        {"guardrail_cost_by_unit": {"contentPolicyUnits": float("inf")}},
        {"guardrail_cost_by_unit": {"contentPolicyUnits": "bad"}},
        {"guardrail_cost_by_unit": "not-a-map"},
        {"guardrail_cost_by_unit": {"contentPolicyUnits": 0.1}, "guardrail_cost_in_spend": "maybe"},
        "not-an-entry",
    ],
)
def test_billed_guardrail_cost_by_unit_is_none_when_unpriced_report_only_or_forged(entry):
    assert billed_guardrail_cost_by_unit(entry) is None


def test_billed_guardrail_cost_by_unit_treats_none_in_spend_as_billed():
    entry = {"guardrail_cost_by_unit": {"contentPolicyUnits": 0.15}, "guardrail_cost_in_spend": None}
    assert billed_guardrail_cost_by_unit(entry) == {"contentPolicyUnits": 0.15}


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


_TRACE_USAGE_IN: Final = {
    "topicPolicyUnits": 1,
    "contentPolicyUnits": 1,
    "wordPolicyUnits": 1,
    "sensitiveInformationPolicyUnits": 0,
    "contextualGroundingPolicyUnits": 0,
    "automatedReasoningPolicies": 0,
}
_TRACE_USAGE_OUT: Final = {
    **_TRACE_USAGE_IN,
    "topicPolicyUnits": 2,
    "contentPolicyUnits": 2,
    "sensitiveInformationPolicyUnits": 2,
}
_TRACE_PRICING: Final = {
    "bedrock/guardrails": {
        "litellm_provider": "bedrock",
        "mode": "guardrails",
        "guardrail_cost_per_unit": {
            "topicPolicyUnits": 0.00015,
            "contentPolicyUnits": 0.00015,
            "wordPolicyUnits": 0.0,
            "sensitiveInformationPolicyUnits": 0.0001,
        },
    }
}


def _converse_trace(action: str = "NONE") -> dict[str, JsonValue]:
    return {
        "guardrail": {
            "inputAssessment": {"gid": {"invocationMetrics": {"usage": _TRACE_USAGE_IN}}},
            "outputAssessments": {
                "gid": [
                    {
                        "topicPolicy": {"topics": [{"name": "t", "type": "DENY", "action": action}]},
                        "invocationMetrics": {"usage": _TRACE_USAGE_OUT},
                    }
                ]
            },
        }
    }


def _invoke_trace() -> dict[str, JsonValue]:
    return {
        "guardrail": {
            "input": {"gid": {"invocationMetrics": {"usage": _TRACE_USAGE_IN}}},
            "outputs": [{"gid": {"invocationMetrics": {"usage": _TRACE_USAGE_OUT}}}],
        }
    }


@pytest.mark.parametrize("trace", [_converse_trace(), _invoke_trace()], ids=["converse", "invoke"])
def test_native_trace_sums_every_assessment_and_prices_it_like_the_proxy_guardrail(monkeypatch, trace) -> None:
    monkeypatch.setattr(litellm, "model_cost", _TRACE_PRICING)
    entry: Final = guardrail_cost.native_bedrock_guardrail_information(
        trace, aws_region_name=None, guardrail_identifier=None
    )
    assert entry is not None
    assert entry["guardrail_usage"] == {
        "topicPolicyUnits": 3,
        "contentPolicyUnits": 3,
        "wordPolicyUnits": 2,
        "sensitiveInformationPolicyUnits": 2,
        "contextualGroundingPolicyUnits": 0,
    }
    assert entry["guardrail_cost"] == pytest.approx(3 * 0.00015 + 3 * 0.00015 + 2 * 0.0001)
    assert entry["guardrail_cost_by_unit"]["contextualGroundingPolicyUnits"] is None
    assert entry["guardrail_name"] == "gid"
    assert entry["guardrail_provider"] == "bedrock"
    assert entry["guardrail_status"] == "success"
    assert guardrail_cost.guardrail_information_cost([entry]) == pytest.approx(entry["guardrail_cost"])


def test_native_trace_marks_a_blocked_assessment_intervened_and_prefers_the_configured_identifier(monkeypatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", _TRACE_PRICING)
    entry: Final = guardrail_cost.native_bedrock_guardrail_information(
        _converse_trace("BLOCKED"), aws_region_name="us-east-1", guardrail_identifier="configured-id"
    )
    assert entry is not None
    assert entry["guardrail_status"] == "guardrail_intervened"
    assert entry["guardrail_name"] == "configured-id"


def test_native_trace_without_usage_is_not_billed(monkeypatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", _TRACE_PRICING)
    unpriced: Final = {"guardrail": {"inputAssessment": {"gid": {"topicPolicy": {"topics": []}}}}}
    assert guardrail_cost.native_bedrock_guardrail_information(unpriced, None, None) is None
    assert guardrail_cost.bedrock_guardrail_trace({"usage": {}}) is None


def test_with_guardrail_entry_records_the_same_trace_once() -> None:
    entry: Final = {"guardrail_name": "gid", "guardrail_cost": 0.001}
    metadata: dict = {"standard_logging_guardrail_information": {"guardrail_name": "hook"}}
    guardrail_cost.with_guardrail_entry(metadata, entry)  # pyright: ignore[reportArgumentType]  # partial entry
    guardrail_cost.with_guardrail_entry(metadata, entry)  # pyright: ignore[reportArgumentType]  # partial entry
    assert metadata["standard_logging_guardrail_information"] == [{"guardrail_name": "hook"}, entry]


def test_native_trace_entry_redacts_the_pii_matches_before_they_reach_spend_logs(monkeypatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", _TRACE_PRICING)
    trace: Final = _converse_trace()
    trace["guardrail"]["inputAssessment"]["gid"]["sensitiveInformationPolicy"] = {
        "piiEntities": [{"type": "US_SSN", "match": "123-45-6789", "action": "ANONYMIZED"}]
    }
    entry: Final = guardrail_cost.native_bedrock_guardrail_information(
        trace, aws_region_name=None, guardrail_identifier=None
    )
    assert entry is not None
    assert entry["guardrail_status"] == "guardrail_intervened"
    assert "123-45-6789" not in json.dumps(entry["guardrail_response"])
    assert entry["guardrail_usage"]["sensitiveInformationPolicyUnits"] == 2

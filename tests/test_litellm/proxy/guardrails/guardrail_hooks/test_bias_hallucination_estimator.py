from typing import Any

import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.bias_hallucination_estimator import (
    BiasHallucinationEstimatorGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.data_sources import (
    ContextDocumentDataSource,
    FileDataSource,
)
from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.estimator_core import (
    BiasDetector,
    HallucinationDetector,
)
from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.grounding_checker import (
    GroundingChecker,
)
from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.risk_scorer import (
    RiskScorer,
)
from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.utils import (
    clip_example,
    split_sentences,
    unique_preserve_order,
)
from litellm.types.guardrails import GuardrailEventHooks, SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator import (
    BiasAnalysis,
    BiasHallucinationEstimatorConfigModel,
    HallucinationAnalysis,
)


# ---------------------------------------------------------------------------
# BiasDetector
# ---------------------------------------------------------------------------


def test_bias_detector_detects_dogmatic_language() -> None:
    analysis = BiasDetector().detect(
        "Everyone knows this is obviously the only correct answer. It will definitely work for every user."
    )

    assert analysis.bias_detected is True
    assert analysis.score > 0
    assert "dogmatic_language" in analysis.patterns_found
    assert "overconfidence" in analysis.patterns_found


def test_bias_detector_detects_opinion_as_fact() -> None:
    analysis = BiasDetector().detect("I believe this should be mandatory for all developers.")

    assert analysis.bias_detected is True
    assert "opinion_as_fact" in analysis.patterns_found


def test_bias_detector_detects_sweeping_generalization() -> None:
    analysis = BiasDetector().detect("All engineers are skilled at math.")

    assert analysis.bias_detected is True
    assert "sweeping_generalization" in analysis.patterns_found


def test_bias_detector_score_accumulates_across_patterns() -> None:
    analysis = BiasDetector().detect(
        "Obviously all developers must be certified. I believe it is 100% guaranteed."
    )

    assert analysis.score > 0.3
    assert len(analysis.patterns_found) >= 2
    assert len(analysis.examples) >= 2


def test_bias_detector_does_not_flag_neutral_text() -> None:
    analysis = BiasDetector().detect("The timeout defaults to ten seconds and can be configured per request.")

    assert analysis.bias_detected is False
    assert analysis.score == 0
    assert analysis.patterns_found == []


def test_bias_detector_score_capped_at_one() -> None:
    heavily_biased = " ".join(
        [
            "Obviously everyone knows this will definitely always be true.",
            "I believe it must be that all users should never question it.",
            "The fact is it is 100% guaranteed and certainly cannot be wrong.",
        ]
    )
    analysis = BiasDetector().detect(heavily_biased)

    assert analysis.score <= 1.0


def test_bias_detector_reasoning_reflects_patterns() -> None:
    analysis = BiasDetector().detect("Obviously this is the only solution.")

    assert "dogmatic_language" in analysis.reasoning


def test_bias_detector_neutral_text_reasoning() -> None:
    analysis = BiasDetector().detect("Configure the retry limit in your settings file.")

    assert analysis.reasoning == "No bias indicators found."


# ---------------------------------------------------------------------------
# HallucinationDetector
# ---------------------------------------------------------------------------


def test_hallucination_detector_detects_unsourced_statistics_and_citation_gaps() -> None:
    analysis = HallucinationDetector().detect("Research shows 73% of users switch products on March 14, 2022.")

    assert analysis.hallucination_detected is True
    assert analysis.score >= 0.5
    assert "unsourced_statistics" in analysis.patterns_found
    assert "missing_citations" in analysis.patterns_found
    assert "fabricated_specificity" in analysis.patterns_found


def test_hallucination_detector_allows_sourced_statistics() -> None:
    analysis = HallucinationDetector().detect(
        "According to the 2024 usage report, 73% of requests completed successfully."
    )

    assert analysis.hallucination_detected is False
    assert analysis.score == 0
    assert analysis.unsourced_claims == []


def test_hallucination_detector_detects_vague_authority() -> None:
    analysis = HallucinationDetector().detect("It is widely known that this approach is superior.")

    assert "missing_citations" in analysis.patterns_found
    assert analysis.hallucination_detected is True


def test_hallucination_detector_detects_overly_precise_number() -> None:
    analysis = HallucinationDetector().detect("The system processed exactly 1,234 requests last month.")

    assert "fabricated_specificity" in analysis.patterns_found


def test_hallucination_detector_multiple_unsourced_claims_increase_score() -> None:
    multi_claim = (
        "Studies show 45% of users prefer option A. "
        "Research shows 62% switch after six months. "
        "Scientists found 3 out of 4 developers agree."
    )
    analysis = HallucinationDetector().detect(multi_claim)

    assert analysis.score > 0.5
    assert len(analysis.unsourced_claims) >= 2


def test_hallucination_detector_score_capped_at_one() -> None:
    very_risky = " ".join(
        [
            "Research shows 73% of users agree.",
            "Studies found 2 out of 3 experts concur.",
            "It has been proven that exactly 1,234,567 cases exist.",
            "According to experts, on January 15, 2023 the rate was 89%.",
            "Data proves that scientists found 99% accuracy.",
        ]
    )
    analysis = HallucinationDetector().detect(very_risky)

    assert analysis.score <= 1.0


def test_hallucination_detector_citation_indicator_clears_number_in_sentence() -> None:
    analysis = HallucinationDetector().detect(
        "Published in Nature journal: 85% of trials showed improvement."
    )

    assert analysis.unsourced_claims == []


# ---------------------------------------------------------------------------
# RiskScorer
# ---------------------------------------------------------------------------


def test_risk_scorer_blocks_when_hallucination_threshold_is_crossed() -> None:
    risk = RiskScorer().compute_risk(
        bias_analysis=BiasAnalysis(score=0.1),
        hallucination_analysis=HallucinationAnalysis(
            hallucination_detected=True,
            score=0.6,
            patterns_found=["unsourced_statistics"],
        ),
    )

    assert risk.overall_risk_percentage == 40
    assert risk.recommendation == "block"
    assert risk.detected_issues == ["hallucination:unsourced_statistics"]


def test_risk_scorer_flags_medium_weighted_risk() -> None:
    risk = RiskScorer(
        bias_threshold=0.9,
        hallucination_threshold=0.9,
    ).compute_risk(
        bias_analysis=BiasAnalysis(
            bias_detected=True,
            score=0.4,
            patterns_found=["dogmatic_language"],
        ),
        hallucination_analysis=HallucinationAnalysis(score=0.2),
    )

    assert risk.overall_risk_percentage == 28
    assert risk.recommendation == "flag"


def test_risk_scorer_passes_low_risk() -> None:
    risk = RiskScorer().compute_risk(
        bias_analysis=BiasAnalysis(score=0.1),
        hallucination_analysis=HallucinationAnalysis(score=0.1),
    )

    assert risk.recommendation == "pass"
    assert risk.overall_risk_percentage < 25


def test_risk_scorer_blocks_on_high_bias_score_alone() -> None:
    risk = RiskScorer(bias_threshold=0.4).compute_risk(
        bias_analysis=BiasAnalysis(bias_detected=True, score=0.5, patterns_found=["overconfidence"]),
        hallucination_analysis=HallucinationAnalysis(score=0.0),
    )

    assert risk.recommendation == "block"
    assert "bias:overconfidence" in risk.detected_issues


def test_risk_scorer_detected_issues_prefix_by_type() -> None:
    risk = RiskScorer().compute_risk(
        bias_analysis=BiasAnalysis(
            bias_detected=True, score=0.6, patterns_found=["dogmatic_language", "overconfidence"]
        ),
        hallucination_analysis=HallucinationAnalysis(
            hallucination_detected=True, score=0.6, patterns_found=["unsourced_statistics"]
        ),
    )

    assert "bias:dogmatic_language" in risk.detected_issues
    assert "bias:overconfidence" in risk.detected_issues
    assert "hallucination:unsourced_statistics" in risk.detected_issues


def test_risk_scorer_custom_weights_change_overall_percentage() -> None:
    bias_only = RiskScorer(bias_weight=1.0, hallucination_weight=0.0).compute_risk(
        bias_analysis=BiasAnalysis(score=0.5),
        hallucination_analysis=HallucinationAnalysis(score=0.0),
    )
    hallucination_only = RiskScorer(bias_weight=0.0, hallucination_weight=1.0).compute_risk(
        bias_analysis=BiasAnalysis(score=0.0),
        hallucination_analysis=HallucinationAnalysis(score=0.5),
    )

    assert bias_only.overall_risk_percentage == hallucination_only.overall_risk_percentage == 50


def test_risk_scorer_zero_weight_total_returns_zero_percentage() -> None:
    # When both weights are 0 the weighted formula returns 0% — but per-score
    # thresholds are still evaluated, so passing scores below the threshold is
    # required to also get a "pass" recommendation.
    risk = RiskScorer(bias_weight=0.0, hallucination_weight=0.0).compute_risk(
        bias_analysis=BiasAnalysis(score=0.1),
        hallucination_analysis=HallucinationAnalysis(score=0.1),
    )

    assert risk.overall_risk_percentage == 0
    assert risk.recommendation == "pass"


# ---------------------------------------------------------------------------
# BiasHallucinationEstimatorGuardrail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardrail_blocks_high_risk_response_and_logs_metadata() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(
        guardrail_name="bias-hallucination",
        event_hook=GuardrailEventHooks.post_call,
    )
    request_data: dict[str, Any] = {}

    with pytest.raises(GuardrailRaisedException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["Research shows 73% of users switch products on March 14, 2022."]},
            request_data=request_data,
            input_type="response",
        )

    assert "High bias/hallucination risk detected" in str(exc.value)
    logging_entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert logging_entries[0]["guardrail_status"] == "guardrail_intervened"
    assert logging_entries[0]["guardrail_response"]["decision"] == "blocked"


@pytest.mark.asyncio
async def test_guardrail_log_only_flags_without_blocking() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(
        guardrail_name="bias-hallucination",
        event_hook=GuardrailEventHooks.post_call,
        log_only=True,
    )
    request_data: dict[str, Any] = {}

    result = await guardrail.apply_guardrail(
        inputs={"texts": ["Research shows 73% of users switch products on March 14, 2022."]},
        request_data=request_data,
        input_type="response",
    )

    assert result["texts"][0].startswith("Research shows")
    logging_entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert logging_entries[0]["guardrail_status"] == "success"
    assert logging_entries[0]["guardrail_response"]["decision"] == "flagged"


@pytest.mark.asyncio
async def test_guardrail_skips_request_by_default() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(
        guardrail_name="bias-hallucination",
        event_hook=GuardrailEventHooks.pre_call,
    )
    request_data: dict[str, Any] = {}

    result = await guardrail.apply_guardrail(
        inputs={"texts": ["Research shows 73% of users switch products."]},
        request_data=request_data,
        input_type="request",
    )

    assert result["texts"][0].startswith("Research shows")
    assert request_data == {}


@pytest.mark.asyncio
async def test_guardrail_passes_low_risk_text() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(
        guardrail_name="bias-hallucination",
        event_hook=GuardrailEventHooks.post_call,
    )
    request_data: dict[str, Any] = {}

    result = await guardrail.apply_guardrail(
        inputs={"texts": ["The timeout defaults to ten seconds and can be changed in configuration."]},
        request_data=request_data,
        input_type="response",
    )

    assert result["texts"][0].startswith("The timeout")
    logging_entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert logging_entries[0]["guardrail_response"]["decision"] == "passed"


@pytest.mark.asyncio
async def test_guardrail_respects_custom_violation_message() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(
        guardrail_name="bias-hallucination",
        violation_message="Custom block message.",
    )

    with pytest.raises(GuardrailRaisedException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["Research shows 73% of users switch products on March 14, 2022."]},
            request_data={},
            input_type="response",
        )

    assert "Custom block message." in str(exc.value)


@pytest.mark.asyncio
async def test_guardrail_tool_calls_are_analyzed() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(
        guardrail_name="bias-hallucination",
        event_hook=GuardrailEventHooks.post_call,
    )
    request_data: dict[str, Any] = {}

    with pytest.raises(GuardrailRaisedException):
        await guardrail.apply_guardrail(
            inputs={
                "texts": [],
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "Research shows 73% switch on March 14, 2022"}',
                        }
                    }
                ],
            },
            request_data=request_data,
            input_type="response",
        )


@pytest.mark.asyncio
async def test_guardrail_empty_inputs_returns_unchanged() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(guardrail_name="bias-hallucination")
    request_data: dict[str, Any] = {}

    result = await guardrail.apply_guardrail(
        inputs={"texts": []},
        request_data=request_data,
        input_type="response",
    )

    assert result == {"texts": []}
    assert request_data == {}


@pytest.mark.asyncio
async def test_guardrail_check_request_enabled_detects_bias() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(
        guardrail_name="bias-hallucination",
        check_request=True,
        check_response=False,
        event_hook=GuardrailEventHooks.pre_call,
    )

    with pytest.raises(GuardrailRaisedException):
        await guardrail.apply_guardrail(
            inputs={"texts": ["Research shows 73% of users switch products on March 14, 2022."]},
            request_data={},
            input_type="request",
        )


def test_guardrail_estimate_returns_structured_result() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(guardrail_name="bias-hallucination")
    result = guardrail.estimate_bias_hallucination(
        "Everyone knows this is obviously the only correct answer."
    )

    assert "bias" in result
    assert "hallucination" in result
    assert "risk" in result
    assert isinstance(result["risk"], dict)
    assert "overall_risk_percentage" in result["risk"]  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------


def test_config_model_defaults_and_ui_name() -> None:
    config = BiasHallucinationEstimatorConfigModel()

    assert config.bias_threshold == 0.5
    assert config.hallucination_threshold == 0.5
    assert config.check_response is True
    assert BiasHallucinationEstimatorConfigModel.ui_friendly_name() == "LiteLLM Bias & Hallucination Estimator"


def test_config_model_rejects_out_of_range_threshold() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        BiasHallucinationEstimatorConfigModel(bias_threshold=1.5)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_guardrail_package_exports_registry_entries() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator import (
        guardrail_class_registry,
        guardrail_initializer_registry,
        initialize_guardrail,
    )

    key = SupportedGuardrailIntegrations.BIAS_HALLUCINATION_ESTIMATOR.value

    assert key == "bias_hallucination_estimator"
    assert guardrail_initializer_registry[key] == initialize_guardrail
    assert guardrail_class_registry[key] == BiasHallucinationEstimatorGuardrail


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def test_split_sentences_basic() -> None:
    result = split_sentences("Hello world. This is a test! Is it working?")

    assert result == ("Hello world.", "This is a test!", "Is it working?")


def test_split_sentences_empty_string() -> None:
    assert split_sentences("") == ()


def test_split_sentences_normalizes_whitespace() -> None:
    result = split_sentences("  Hello   world.   Next sentence.  ")

    assert result == ("Hello world.", "Next sentence.")


def test_clip_example_short_text_unchanged() -> None:
    assert clip_example("Short text") == "Short text"


def test_clip_example_truncates_long_text() -> None:
    long_text = "a" * 200
    result = clip_example(long_text)

    assert len(result) == 160
    assert result.endswith("...")


def test_unique_preserve_order_deduplicates() -> None:
    result = unique_preserve_order(["b", "a", "b", "c", "a"])

    assert result == ("b", "a", "c")


def test_unique_preserve_order_filters_empty_strings() -> None:
    result = unique_preserve_order(["a", "", "b", ""])

    assert result == ("a", "b")


# ---------------------------------------------------------------------------
# FileDataSource
# ---------------------------------------------------------------------------


def test_file_data_source_nonexistent_path_returns_empty() -> None:
    source = FileDataSource("/nonexistent/path/facts.json")

    assert source._documents == []


@pytest.mark.asyncio
async def test_file_data_source_search_returns_empty_for_empty_source() -> None:
    source = FileDataSource("/nonexistent/path/facts.json")
    results = await source.search("some query")

    assert results == []


@pytest.mark.asyncio
async def test_context_document_source_finds_matching_content() -> None:
    source = ContextDocumentDataSource(
        documents=[
            {"text": "The company was founded in 2015 by Jane Smith."},
            {"text": "The product has 3.2 million active users."},
        ]
    )

    results = await source.search("company founded 2015")

    assert len(results) > 0
    assert any("2015" in r.text for r in results)


@pytest.mark.asyncio
async def test_context_document_source_no_match_returns_empty() -> None:
    source = ContextDocumentDataSource(
        documents=[{"text": "The company was founded in 2015."}]
    )

    results = await source.search("quantum physics neutron stars")

    assert results == []


# ---------------------------------------------------------------------------
# GroundingChecker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grounding_checker_empty_claim_returns_ungrounded() -> None:
    checker = GroundingChecker(data_sources=[])
    result = await checker.check_claim_grounding("")

    assert result.is_grounded is False
    assert "empty" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_grounding_checker_no_enabled_sources_returns_ungrounded() -> None:
    checker = GroundingChecker(data_sources=[])
    result = await checker.check_claim_grounding("Company founded in 2015")

    assert result.is_grounded is False


@pytest.mark.asyncio
async def test_grounding_checker_verifies_claim_from_context_docs() -> None:
    source = ContextDocumentDataSource(
        documents=[{"text": "The company was founded in 2015 by Jane Smith."}]
    )
    checker = GroundingChecker(data_sources=[source], confidence_threshold=0.3)
    result = await checker.check_claim_grounding("company founded 2015")

    assert result.is_grounded is True
    assert result.confidence >= 0.3
    assert len(result.supporting_docs) > 0


@pytest.mark.asyncio
async def test_grounding_checker_unverifiable_claim_not_grounded() -> None:
    source = ContextDocumentDataSource(
        documents=[{"text": "The company was founded in 2015 by Jane Smith."}]
    )
    checker = GroundingChecker(data_sources=[source], confidence_threshold=0.6)
    result = await checker.check_claim_grounding("quantum entanglement photon polarization")

    assert result.is_grounded is False


@pytest.mark.asyncio
async def test_grounding_checker_verify_multiple_claims() -> None:
    source = ContextDocumentDataSource(
        documents=[{"text": "The company was founded in 2015."}]
    )
    checker = GroundingChecker(data_sources=[source], confidence_threshold=0.3)
    results = await checker.verify_multiple_claims(
        ["company founded 2015", "quantum physics neutron stars"]
    )

    assert len(results) == 2
    grounded_flags = [r.is_grounded for r in results]
    assert True in grounded_flags
    assert False in grounded_flags

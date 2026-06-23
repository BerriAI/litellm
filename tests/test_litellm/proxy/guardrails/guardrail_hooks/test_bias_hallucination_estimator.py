import json
import tempfile
import unittest.mock
from typing import Any

import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.bias_hallucination_estimator import (
    BiasHallucinationEstimatorGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.data_sources import (
    ContextDocumentDataSource,
    DataSourceResult,
    FactCheckDataSource,
    FileDataSource,
    KnowledgeGraphDataSource,
    URLDataSource,
    VectorStoreDataSource,
    _get_doc_text,
    _keyword_search,
    _parse_json_docs,
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
    analysis = BiasDetector().detect(
        "I believe this should be mandatory for all developers."
    )

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
    analysis = BiasDetector().detect(
        "The timeout defaults to ten seconds and can be configured per request."
    )

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


def test_hallucination_detector_detects_unsourced_statistics_and_citation_gaps() -> (
    None
):
    analysis = HallucinationDetector().detect(
        "Research shows 73% of users switch products on March 14, 2022."
    )

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
    analysis = HallucinationDetector().detect(
        "It is widely known that this approach is superior."
    )

    assert "missing_citations" in analysis.patterns_found
    assert analysis.hallucination_detected is True


def test_hallucination_detector_detects_overly_precise_number() -> None:
    analysis = HallucinationDetector().detect(
        "The system processed exactly 1,234 requests last month."
    )

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
        bias_analysis=BiasAnalysis(
            bias_detected=True, score=0.5, patterns_found=["overconfidence"]
        ),
        hallucination_analysis=HallucinationAnalysis(score=0.0),
    )

    assert risk.recommendation == "block"
    assert "bias:overconfidence" in risk.detected_issues


def test_risk_scorer_detected_issues_prefix_by_type() -> None:
    risk = RiskScorer().compute_risk(
        bias_analysis=BiasAnalysis(
            bias_detected=True,
            score=0.6,
            patterns_found=["dogmatic_language", "overconfidence"],
        ),
        hallucination_analysis=HallucinationAnalysis(
            hallucination_detected=True,
            score=0.6,
            patterns_found=["unsourced_statistics"],
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
    hallucination_only = RiskScorer(
        bias_weight=0.0, hallucination_weight=1.0
    ).compute_risk(
        bias_analysis=BiasAnalysis(score=0.0),
        hallucination_analysis=HallucinationAnalysis(score=0.5),
    )

    assert (
        bias_only.overall_risk_percentage
        == hallucination_only.overall_risk_percentage
        == 50
    )


def test_risk_scorer_zero_weight_total_returns_zero_percentage() -> None:
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
            inputs={
                "texts": [
                    "Research shows 73% of users switch products on March 14, 2022."
                ]
            },
            request_data=request_data,
            input_type="response",
        )

    assert "High bias/hallucination risk detected" in str(exc.value)
    logging_entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert logging_entries[0]["guardrail_status"] == "guardrail_intervened"
    assert logging_entries[0]["guardrail_response"]["decision"] == "blocked"


@pytest.mark.asyncio
async def test_guardrail_log_payload_excludes_text_snippets() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(
        guardrail_name="bias-hallucination",
        log_only=True,
    )
    request_data: dict[str, Any] = {}

    await guardrail.apply_guardrail(
        inputs={
            "texts": ["Research shows 73% of users switch products on March 14, 2022."]
        },
        request_data=request_data,
        input_type="response",
    )

    response = request_data["metadata"]["standard_logging_guardrail_information"][0][
        "guardrail_response"
    ]
    snippet_fields = {
        "examples",
        "unsourced_claims",
        "missing_citations",
        "fabricated_specificity",
    }
    for entry in response["bias"]:
        assert (
            not snippet_fields & entry.keys()
        ), f"bias log entry contains snippet fields: {entry.keys()}"
    for entry in response["hallucination"]:
        assert (
            not snippet_fields & entry.keys()
        ), f"hallucination log entry contains snippet fields: {entry.keys()}"


@pytest.mark.asyncio
async def test_guardrail_log_only_flags_without_blocking() -> None:
    guardrail = BiasHallucinationEstimatorGuardrail(
        guardrail_name="bias-hallucination",
        event_hook=GuardrailEventHooks.post_call,
        log_only=True,
    )
    request_data: dict[str, Any] = {}

    result = await guardrail.apply_guardrail(
        inputs={
            "texts": ["Research shows 73% of users switch products on March 14, 2022."]
        },
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
        inputs={
            "texts": [
                "The timeout defaults to ten seconds and can be changed in configuration."
            ]
        },
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
            inputs={
                "texts": [
                    "Research shows 73% of users switch products on March 14, 2022."
                ]
            },
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
            inputs={
                "texts": [
                    "Research shows 73% of users switch products on March 14, 2022."
                ]
            },
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
    assert config.bias_weight == 0.4
    assert config.hallucination_weight == 0.6
    assert (
        BiasHallucinationEstimatorConfigModel.ui_friendly_name()
        == "LiteLLM Bias & Hallucination Estimator"
    )


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
    result = await checker.check_claim_grounding(
        "quantum entanglement photon polarization"
    )

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


# ---------------------------------------------------------------------------
# DataSourceResult
# ---------------------------------------------------------------------------


def test_data_source_result_clamps_confidence_below_zero() -> None:
    result = DataSourceResult(text="hello", source="s", confidence=-0.5)

    assert result.confidence == 0.0


def test_data_source_result_clamps_confidence_above_one() -> None:
    result = DataSourceResult(text="hello", source="s", confidence=1.5)

    assert result.confidence == 1.0


def test_data_source_result_default_metadata_is_empty_dict() -> None:
    result = DataSourceResult(text="hello", source="s")

    assert result.metadata == {}


def test_data_source_result_repr_contains_source_and_confidence() -> None:
    result = DataSourceResult(text="hello", source="my_source", confidence=0.75)

    assert "my_source" in repr(result)
    assert "0.75" in repr(result)


# ---------------------------------------------------------------------------
# DataSource.verify_fact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_data_source_verify_fact_returns_true_when_match_found() -> None:
    source = ContextDocumentDataSource(
        documents=[{"text": "The company was founded in 2015."}]
    )
    found, text = await source.verify_fact("company founded 2015")

    assert found is True
    assert text is not None
    assert "2015" in text


@pytest.mark.asyncio
async def test_data_source_verify_fact_returns_false_when_no_match() -> None:
    source = ContextDocumentDataSource(
        documents=[{"text": "The company was founded in 2015."}]
    )
    found, text = await source.verify_fact("quantum entanglement photon")

    assert found is False
    assert text is None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_get_doc_text_returns_string_directly() -> None:
    assert _get_doc_text("plain string") == "plain string"


def test_get_doc_text_extracts_text_key_from_dict() -> None:
    assert _get_doc_text({"text": "from dict"}) == "from dict"


def test_get_doc_text_falls_back_to_str_when_no_text_key() -> None:
    result = _get_doc_text({"other": "value"})

    assert "other" in result


def test_parse_json_docs_wraps_non_list_in_list() -> None:
    result = _parse_json_docs({"text": "single doc"})

    assert isinstance(result, list)
    assert len(result) == 1


def test_parse_json_docs_returns_list_unchanged() -> None:
    docs = [{"text": "a"}, {"text": "b"}]
    result = _parse_json_docs(docs)

    assert result == docs


def test_keyword_search_returns_empty_for_empty_query() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.data_sources import (
        _build_keyword_index,
    )

    docs = [{"text": "some content"}]
    index = _build_keyword_index(docs)
    results = _keyword_search("", docs, index, "src", 5)

    assert results == []


def test_keyword_search_returns_empty_when_no_docs_match() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.data_sources import (
        _build_keyword_index,
    )

    docs = [{"text": "apples and oranges"}]
    index = _build_keyword_index(docs)
    results = _keyword_search("quantum neutron", docs, index, "src", 5)

    assert results == []


def test_keyword_search_respects_limit() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.data_sources import (
        _build_keyword_index,
    )

    docs = [{"text": f"document about python {i}"} for i in range(10)]
    index = _build_keyword_index(docs)
    results = _keyword_search("python document", docs, index, "src", 3)

    assert len(results) <= 3


def test_keyword_search_scores_by_word_overlap() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.data_sources import (
        _build_keyword_index,
    )

    docs = [
        {"text": "python programming language"},
        {"text": "python snake reptile"},
    ]
    index = _build_keyword_index(docs)
    results = _keyword_search("python programming", docs, index, "src", 5)

    assert results[0].text == "python programming language"
    assert results[0].confidence > results[1].confidence


# ---------------------------------------------------------------------------
# FileDataSource with real files
# ---------------------------------------------------------------------------


def test_file_data_source_loads_json_file() -> None:
    docs = [
        {"text": "Paris is the capital of France."},
        {"text": "Berlin is the capital of Germany."},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(docs, f)
        path = f.name

    source = FileDataSource(path)

    assert len(source._documents) == 2


@pytest.mark.asyncio
async def test_file_data_source_searches_json_content() -> None:
    docs = [
        {"text": "Paris is the capital of France."},
        {"text": "Berlin is the capital of Germany."},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(docs, f)
        path = f.name

    source = FileDataSource(path)
    results = await source.search("capital France Paris")

    assert len(results) > 0
    assert any("Paris" in r.text for r in results)


def test_file_data_source_loads_txt_file() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("First fact about astronomy.\nSecond fact about physics.\n")
        path = f.name

    source = FileDataSource(path)

    assert len(source._documents) == 2


def test_file_data_source_loads_csv_file() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("row one data\nrow two data\n")
        path = f.name

    source = FileDataSource(path)

    assert len(source._documents) == 2


def test_file_data_source_uses_stem_as_default_name() -> None:
    source = FileDataSource("/some/path/my_facts.json")

    assert source.name == "file_my_facts"


def test_file_data_source_loads_json_object_not_list() -> None:
    doc = {"text": "Single document as object."}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(doc, f)
        path = f.name

    source = FileDataSource(path)

    assert len(source._documents) == 1


# ---------------------------------------------------------------------------
# URLDataSource
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_data_source_returns_empty_without_aiohttp() -> None:
    source = URLDataSource(urls=["http://example.com/data.json"])

    with unittest.mock.patch(
        "litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.data_sources._AIOHTTP_AVAILABLE",
        False,
    ):
        results = await source.search("any query")

    assert results == []


@pytest.mark.asyncio
async def test_url_data_source_parse_content_handles_json() -> None:
    json_content = json.dumps([{"text": "fact one"}, {"text": "fact two"}])
    parsed = URLDataSource._parse_content(json_content)

    assert len(parsed) == 2
    assert parsed[0] == {"text": "fact one"}  # type: ignore[comparison-overlap]


@pytest.mark.asyncio
async def test_url_data_source_parse_content_handles_plain_text() -> None:
    parsed = URLDataSource._parse_content("This is not JSON at all.")

    assert len(parsed) == 1
    assert parsed[0] == {"text": "This is not JSON at all."}  # type: ignore[comparison-overlap]


@pytest.mark.asyncio
async def test_url_data_source_search_returns_cached_on_second_call() -> None:
    source = URLDataSource(urls=[])
    source._fetched = True
    source._documents = [{"text": "cached document about python"}]
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.data_sources import (
        _build_keyword_index,
    )

    source._index = _build_keyword_index(source._documents)

    results = await source.search("python")

    assert len(results) == 1
    assert "python" in results[0].text


@pytest.mark.asyncio
async def test_url_data_source_concurrent_fetch_only_runs_once() -> None:
    fetch_count = 0

    async def fake_fetch_all() -> list[Any]:
        nonlocal fetch_count
        fetch_count += 1
        return [{"text": "fetched document"}]

    source = URLDataSource(urls=["http://example.com"])

    with unittest.mock.patch.object(source, "_fetch_all", side_effect=fake_fetch_all):
        await source.search("document")
        await source.search("document")

    assert fetch_count == 1


# ---------------------------------------------------------------------------
# VectorStoreDataSource
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_store_data_source_returns_empty_without_client() -> None:
    source = VectorStoreDataSource(provider="unknown_provider")

    assert source.client is None
    results = await source.search("anything")

    assert results == []


@pytest.mark.asyncio
async def test_vector_store_data_source_returns_empty_without_embedding_model() -> None:
    source = VectorStoreDataSource(
        provider="pinecone",
        client=unittest.mock.MagicMock(),
        embedding_model=None,
    )

    results = await source.search("query")

    assert results == []


def test_vector_store_data_source_default_name_uses_provider() -> None:
    source = VectorStoreDataSource(provider="pinecone")

    assert source.name == "vectorstore_pinecone"


# ---------------------------------------------------------------------------
# KnowledgeGraphDataSource
# ---------------------------------------------------------------------------


def test_knowledge_graph_sparql_strips_injection_chars() -> None:
    malicious = 'normal text"; DROP TABLE items; --'
    query = KnowledgeGraphDataSource._build_sparql_query(malicious)
    label_value = query.split('label "')[1].split('"')[0]

    assert ";" not in label_value
    assert '"' not in label_value


def test_knowledge_graph_sparql_caps_at_128_chars() -> None:
    long_input = "a" * 200
    query = KnowledgeGraphDataSource._build_sparql_query(long_input)
    label_value = query.split('label "')[1].split('"')[0]

    assert len(label_value) <= 128


def test_knowledge_graph_sparql_preserves_alphanumeric_and_spaces() -> None:
    query = KnowledgeGraphDataSource._build_sparql_query("Marie Curie 1867")
    label_value = query.split('label "')[1].split('"')[0]

    assert label_value == "Marie Curie 1867"


def test_knowledge_graph_extract_text_uses_itemlabel_first() -> None:
    binding = {"itemLabel": {"value": "Paris"}, "label": {"value": "Other"}}

    assert KnowledgeGraphDataSource._extract_text(binding) == "Paris"


def test_knowledge_graph_extract_text_falls_back_to_result_key() -> None:
    binding = {"result": {"value": "Some result"}}

    assert KnowledgeGraphDataSource._extract_text(binding) == "Some result"


def test_knowledge_graph_extract_text_falls_back_to_label_key() -> None:
    binding = {"label": {"value": "A label"}}

    assert KnowledgeGraphDataSource._extract_text(binding) == "A label"


def test_knowledge_graph_extract_text_returns_empty_for_unknown_keys() -> None:
    binding = {"unknown_key": {"value": "ignored"}}

    assert KnowledgeGraphDataSource._extract_text(binding) == ""


@pytest.mark.asyncio
async def test_knowledge_graph_returns_empty_without_aiohttp() -> None:
    source = KnowledgeGraphDataSource()

    with unittest.mock.patch(
        "litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.data_sources._AIOHTTP_AVAILABLE",
        False,
    ):
        results = await source.search("Paris")

    assert results == []


# ---------------------------------------------------------------------------
# FactCheckDataSource
# ---------------------------------------------------------------------------


def test_fact_check_data_source_default_name_uses_provider() -> None:
    source = FactCheckDataSource()

    assert source.name == "factcheck_snopes"


def test_fact_check_data_source_custom_name_overrides_default() -> None:
    source = FactCheckDataSource(name="my_checker")

    assert source.name == "my_checker"


@pytest.mark.asyncio
async def test_fact_check_data_source_search_returns_empty() -> None:
    source = FactCheckDataSource(provider="snopes", api_key="dummy")
    results = await source.search("any claim")

    assert results == []


# ---------------------------------------------------------------------------
# URLDataSource._fetch_url via aiohttp mock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_data_source_fetch_url_returns_parsed_json() -> None:
    payload = json.dumps([{"text": "fact from url"}])
    mock_response = unittest.mock.AsyncMock()
    mock_response.status = 200
    mock_response.text = unittest.mock.AsyncMock(return_value=payload)
    mock_response.__aenter__ = unittest.mock.AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    mock_session = unittest.mock.AsyncMock()
    mock_session.get = unittest.mock.MagicMock(return_value=mock_response)
    mock_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_session):
        source = URLDataSource(urls=["http://example.com/data.json"])
        results = await source.search("fact url")

    assert any("fact from url" in r.text for r in results)


@pytest.mark.asyncio
async def test_url_data_source_fetch_url_handles_non_200_response() -> None:
    mock_response = unittest.mock.AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = unittest.mock.AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    mock_session = unittest.mock.AsyncMock()
    mock_session.get = unittest.mock.MagicMock(return_value=mock_response)
    mock_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_session):
        source = URLDataSource(urls=["http://example.com/data.json"])
        results = await source.search("any query")

    assert results == []


# ---------------------------------------------------------------------------
# VectorStoreDataSource.search with mocked client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_store_data_source_pinecone_returns_results() -> None:
    mock_client = unittest.mock.MagicMock()
    mock_client.query.return_value = {
        "matches": [
            {"metadata": {"text": "pinecone result one"}, "score": 0.9},
            {"metadata": {"text": "pinecone result two"}, "score": 0.7},
        ]
    }
    mock_model = unittest.mock.MagicMock()
    mock_model.encode.return_value = [0.1, 0.2, 0.3]
    source = VectorStoreDataSource(
        provider="pinecone", client=mock_client, embedding_model=mock_model
    )

    results = await source.search("any query")

    assert len(results) == 2
    assert results[0].text == "pinecone result one"
    assert results[0].confidence == 0.9


@pytest.mark.asyncio
async def test_vector_store_data_source_weaviate_returns_results() -> None:
    mock_client = unittest.mock.MagicMock()
    (
        mock_client.query.get.return_value.with_near_vector.return_value.with_limit.return_value.do.return_value
    ) = {"data": {"Get": [{"text": "weaviate result"}]}}
    mock_model = unittest.mock.MagicMock()
    mock_model.encode.return_value = [0.1, 0.2, 0.3]
    source = VectorStoreDataSource(
        provider="weaviate", client=mock_client, embedding_model=mock_model
    )

    results = await source.search("any query")

    assert len(results) == 1
    assert results[0].text == "weaviate result"


@pytest.mark.asyncio
async def test_vector_store_data_source_embedding_list_passthrough() -> None:
    mock_client = unittest.mock.MagicMock()
    mock_client.query.return_value = {"matches": []}
    mock_model = unittest.mock.MagicMock()
    mock_model.encode.return_value = [0.5, 0.6]
    source = VectorStoreDataSource(
        provider="pinecone", client=mock_client, embedding_model=mock_model
    )

    results = await source.search("query")

    assert results == []
    mock_model.encode.assert_called_once_with("query", convert_to_tensor=False)


# ---------------------------------------------------------------------------
# KnowledgeGraphDataSource.search via aiohttp mock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_knowledge_graph_returns_results_from_sparql_response() -> None:
    sparql_response = {
        "results": {
            "bindings": [
                {"itemLabel": {"value": "Paris"}},
                {"itemLabel": {"value": "Lyon"}},
            ]
        }
    }
    mock_response = unittest.mock.AsyncMock()
    mock_response.status = 200
    mock_response.json = unittest.mock.AsyncMock(return_value=sparql_response)
    mock_response.__aenter__ = unittest.mock.AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    mock_session = unittest.mock.AsyncMock()
    mock_session.get = unittest.mock.MagicMock(return_value=mock_response)
    mock_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_session):
        source = KnowledgeGraphDataSource()
        results = await source.search("French cities")

    assert len(results) == 2
    assert results[0].text == "Paris"
    assert results[0].confidence == 0.9


@pytest.mark.asyncio
async def test_knowledge_graph_skips_bindings_with_empty_text() -> None:
    sparql_response = {
        "results": {
            "bindings": [
                {"unknown_key": {"value": "ignored"}},
                {"itemLabel": {"value": "Berlin"}},
            ]
        }
    }
    mock_response = unittest.mock.AsyncMock()
    mock_response.status = 200
    mock_response.json = unittest.mock.AsyncMock(return_value=sparql_response)
    mock_response.__aenter__ = unittest.mock.AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    mock_session = unittest.mock.AsyncMock()
    mock_session.get = unittest.mock.MagicMock(return_value=mock_response)
    mock_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_session):
        source = KnowledgeGraphDataSource()
        results = await source.search("German cities")

    assert len(results) == 1
    assert results[0].text == "Berlin"


@pytest.mark.asyncio
async def test_knowledge_graph_handles_non_200_response() -> None:
    mock_response = unittest.mock.AsyncMock()
    mock_response.status = 500
    mock_response.__aenter__ = unittest.mock.AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    mock_session = unittest.mock.AsyncMock()
    mock_session.get = unittest.mock.MagicMock(return_value=mock_response)
    mock_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_session):
        source = KnowledgeGraphDataSource()
        results = await source.search("query")

    assert results == []


# ---------------------------------------------------------------------------
# initialize_guardrail (__init__.py coverage)
# ---------------------------------------------------------------------------


def test_initialize_guardrail_wires_up_guardrail_from_litellm_params() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator import (
        initialize_guardrail,
    )
    from litellm.types.guardrails import Guardrail, LitellmParams

    litellm_params = LitellmParams(
        guardrail="bias_hallucination_estimator",
        mode="post_call",
    )
    guardrail: Guardrail = {
        "guardrail_name": "test-guardrail",
        "litellm_params": litellm_params,
        "guardrail_id": "grd-001",
    }

    instance = initialize_guardrail(litellm_params=litellm_params, guardrail=guardrail)

    assert isinstance(instance, BiasHallucinationEstimatorGuardrail)
    assert instance.guardrail_id == "grd-001"
    assert instance.bias_threshold == 0.5
    assert instance.hallucination_threshold == 0.5


def test_initialize_guardrail_uses_custom_thresholds() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator import (
        initialize_guardrail,
    )
    from litellm.types.guardrails import Guardrail, LitellmParams

    litellm_params = LitellmParams(
        guardrail="bias_hallucination_estimator",
        mode="post_call",
        bias_threshold=0.3,
        hallucination_threshold=0.4,
        block_on_high_risk=False,
        log_only=True,
    )
    guardrail: Guardrail = {
        "guardrail_name": "custom-guardrail",
        "litellm_params": litellm_params,
    }

    instance = initialize_guardrail(litellm_params=litellm_params, guardrail=guardrail)

    assert instance.bias_threshold == 0.3
    assert instance.hallucination_threshold == 0.4
    assert instance.block_on_high_risk is False
    assert instance.log_only is True


# ---------------------------------------------------------------------------
# BiasHallucinationEstimatorGuardrail — uncovered static branches
# ---------------------------------------------------------------------------


def test_normalize_event_hook_with_mode_returns_mode() -> None:
    from litellm.types.guardrails import Mode

    mode = Mode(tags={}, default=None)
    result = BiasHallucinationEstimatorGuardrail._normalize_event_hook(mode)

    assert result == mode


def test_normalize_event_hook_with_list_returns_list_of_hooks() -> None:
    result = BiasHallucinationEstimatorGuardrail._normalize_event_hook(
        ["pre_call", "post_call"]
    )

    assert isinstance(result, list)
    assert GuardrailEventHooks.pre_call in result
    assert GuardrailEventHooks.post_call in result


def test_detected_issues_returns_empty_for_non_dict() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.bias_hallucination_estimator import (
        BiasHallucinationEstimatorGuardrail,
    )

    assert BiasHallucinationEstimatorGuardrail._detected_issues("not a dict") == ()


def test_detected_issues_returns_empty_when_detected_issues_not_list() -> None:
    assert (
        BiasHallucinationEstimatorGuardrail._detected_issues(
            {"detected_issues": "not a list"}
        )
        == ()
    )


def test_violation_categories_returns_empty_when_risk_scores_not_list() -> None:
    assert (
        BiasHallucinationEstimatorGuardrail._violation_categories(
            {"risk_scores": "bad"}
        )
        == []
    )


def test_tool_call_text_returns_none_for_dict_without_function() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.bias_hallucination_estimator import (
        BiasHallucinationEstimatorGuardrail,
    )

    assert BiasHallucinationEstimatorGuardrail._tool_call_text({"other": "key"}) is None


def test_tool_call_text_returns_none_for_object_without_function_attr() -> None:
    assert BiasHallucinationEstimatorGuardrail._tool_call_text(object()) is None


def test_tool_call_text_handles_protocol_object() -> None:
    class FakeFunction:
        name = "my_tool"
        arguments = '{"x": 1}'

    class FakeToolCall:
        function = FakeFunction()

    result = BiasHallucinationEstimatorGuardrail._tool_call_text(FakeToolCall())

    assert result == 'my_tool {"x": 1}'


def test_string_value_returns_empty_for_none() -> None:
    assert BiasHallucinationEstimatorGuardrail._string_value(None) == ""


def test_string_value_converts_non_string() -> None:
    assert BiasHallucinationEstimatorGuardrail._string_value(42) == "42"


def test_get_config_model_returns_config_class() -> None:
    result = BiasHallucinationEstimatorGuardrail.get_config_model()

    assert result is not None


# ---------------------------------------------------------------------------
# GroundingChecker — uncovered branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grounding_checker_no_verifiable_elements_returns_ungrounded() -> None:
    source = ContextDocumentDataSource(documents=[{"text": "some content"}])
    checker = GroundingChecker(data_sources=[source])
    result = await checker.check_claim_grounding("a an the and or")

    assert result.is_grounded is False
    assert "verifiable" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_grounding_checker_source_failure_returns_ungrounded() -> None:
    source = ContextDocumentDataSource(documents=[{"text": "company founded 2015"}])
    source.name = "failing_source"

    async def always_fails(query: str, limit: int = 5) -> list[Any]:
        raise RuntimeError("connection refused")

    source.search = always_fails  # type: ignore[method-assign]
    checker = GroundingChecker(data_sources=[source], confidence_threshold=0.3)
    result = await checker.check_claim_grounding("company founded 2015")

    assert result.is_grounded is False


@pytest.mark.asyncio
async def test_grounding_checker_partial_match_below_threshold_is_not_grounded() -> (
    None
):
    source = ContextDocumentDataSource(
        documents=[{"text": "The company was founded in 2015 by Jane Smith."}]
    )
    checker = GroundingChecker(data_sources=[source], confidence_threshold=0.99)
    # 2 of 5 query words match → base confidence 0.4; boosted still well below 0.99
    result = await checker.check_claim_grounding(
        "company founded quantum photon neutron"
    )

    assert result.is_grounded is False
    assert "confidence" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_grounding_checker_timeout_returns_empty_not_error() -> None:
    import asyncio as _asyncio

    source = ContextDocumentDataSource(documents=[{"text": "anything"}])

    async def slow_search(query: str, limit: int = 5) -> list[Any]:
        await _asyncio.sleep(10)
        return []

    source.search = slow_search  # type: ignore[method-assign]
    checker = GroundingChecker(
        data_sources=[source], confidence_threshold=0.3, timeout_per_source=0.01
    )
    result = await checker.check_claim_grounding("company founded 2015")

    assert result.is_grounded is False


# ---------------------------------------------------------------------------
# FileDataSource — exception path (corrupt file)
# ---------------------------------------------------------------------------


def test_file_data_source_corrupt_json_returns_empty() -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{ this is not valid json }")
        path = f.name

    source = FileDataSource(path)

    assert source._documents == []


# ---------------------------------------------------------------------------
# DataSource.verify_fact — base class helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_data_source_verify_fact_returns_true_when_result_found() -> None:
    source = ContextDocumentDataSource(
        documents=[{"text": "Python was created by Guido van Rossum."}]
    )
    found, text = await source.verify_fact("Python created Guido")
    assert found is True
    assert text is not None
    assert "Python" in text


@pytest.mark.asyncio
async def test_data_source_verify_fact_returns_false_when_no_result() -> None:
    source = ContextDocumentDataSource(documents=[])
    found, text = await source.verify_fact("anything")
    assert found is False
    assert text is None


# ---------------------------------------------------------------------------
# _keyword_search — empty query_words branch
# ---------------------------------------------------------------------------


def test_keyword_search_returns_empty_for_non_word_query() -> None:
    documents: list[str | dict[str, Any]] = [{"text": "hello world"}]
    index = {"hello": [0], "world": [0]}
    result = _keyword_search("!!!", documents, index, "src", 5)
    assert result == []


# ---------------------------------------------------------------------------
# URLDataSource._fetch_url — exception path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_data_source_fetch_url_handles_exception() -> None:
    source = URLDataSource(urls=["http://example.com"])

    async def boom(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("network error")

    with unittest.mock.patch("aiohttp.ClientSession.get", side_effect=boom):
        import sys

        sys.modules.setdefault("aiohttp", unittest.mock.MagicMock())
        result = await source._fetch_url("http://example.com")

    assert result == []


# ---------------------------------------------------------------------------
# VectorStoreDataSource — _initialize_client provider paths
# ---------------------------------------------------------------------------


def test_vector_store_initialize_client_pinecone_returns_index() -> None:
    mock_index = object()
    mock_pinecone = unittest.mock.MagicMock()
    mock_pinecone.Index.return_value = mock_index

    with unittest.mock.patch.dict(
        __import__("sys").modules, {"pinecone": mock_pinecone}
    ):
        client = VectorStoreDataSource._initialize_client(
            "pinecone", {"api_key": "k", "index_name": "idx"}
        )

    assert client is mock_index


def test_vector_store_initialize_client_pinecone_no_credentials_returns_none() -> None:
    mock_pinecone = unittest.mock.MagicMock()
    with unittest.mock.patch.dict(
        __import__("sys").modules, {"pinecone": mock_pinecone}
    ):
        client = VectorStoreDataSource._initialize_client("pinecone", {})

    assert client is None


def test_vector_store_initialize_client_weaviate_returns_client() -> None:
    mock_client_obj = object()
    mock_weaviate = unittest.mock.MagicMock()
    mock_weaviate.Client.return_value = mock_client_obj

    with unittest.mock.patch.dict(
        __import__("sys").modules, {"weaviate": mock_weaviate}
    ):
        client = VectorStoreDataSource._initialize_client(
            "weaviate", {"url": "http://localhost:8080"}
        )

    assert client is mock_client_obj


def test_vector_store_initialize_client_weaviate_no_url_returns_none() -> None:
    mock_weaviate = unittest.mock.MagicMock()
    with unittest.mock.patch.dict(
        __import__("sys").modules, {"weaviate": mock_weaviate}
    ):
        client = VectorStoreDataSource._initialize_client("weaviate", {})

    assert client is None


def test_vector_store_initialize_client_weaviate_import_error_returns_none() -> None:
    import sys

    saved = sys.modules.pop("weaviate", None)
    try:
        with unittest.mock.patch.dict(sys.modules, {"weaviate": None}):  # type: ignore[dict-item]
            client = VectorStoreDataSource._initialize_client(
                "weaviate", {"url": "http://localhost:8080"}
            )
        assert client is None
    finally:
        if saved is not None:
            sys.modules["weaviate"] = saved


# ---------------------------------------------------------------------------
# VectorStoreDataSource — _load_embedding_model
# ---------------------------------------------------------------------------


def test_vector_store_load_embedding_model_returns_transformer() -> None:
    mock_model = object()
    mock_st = unittest.mock.MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model

    with unittest.mock.patch.dict(
        __import__("sys").modules, {"sentence_transformers": mock_st}
    ):
        model = VectorStoreDataSource._load_embedding_model()

    assert model is mock_model


# ---------------------------------------------------------------------------
# VectorStoreDataSource.search — exception path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_store_search_exception_returns_empty() -> None:
    mock_client = unittest.mock.MagicMock()
    mock_client.query.side_effect = RuntimeError("pinecone unavailable")
    mock_model = unittest.mock.MagicMock()
    mock_model.encode.return_value = [0.1, 0.2, 0.3]

    source = VectorStoreDataSource(
        provider="pinecone", client=mock_client, embedding_model=mock_model
    )
    results = await source.search("test query")

    assert results == []


# ---------------------------------------------------------------------------
# KnowledgeGraphDataSource.search — exception path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_knowledge_graph_search_exception_returns_empty() -> None:
    import sys

    mock_aiohttp = unittest.mock.MagicMock()
    mock_session = unittest.mock.AsyncMock()
    mock_session.__aenter__ = unittest.mock.AsyncMock(
        side_effect=RuntimeError("connection error")
    )
    mock_aiohttp.ClientSession.return_value = mock_session

    with unittest.mock.patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
        source = KnowledgeGraphDataSource()
        with unittest.mock.patch(
            "litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.data_sources._AIOHTTP_AVAILABLE",
            True,
        ):
            results = await source.search("test")

    assert results == []


# ---------------------------------------------------------------------------
# GroundingChecker._boost_confidence — entity match branch (line 164)
# ---------------------------------------------------------------------------


def test_boost_confidence_entity_match_increases_score() -> None:
    from litellm.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator.grounding_checker import (
        GroundingChecker,
    )

    result = DataSourceResult(
        text="Albert Einstein published the theory of relativity.",
        source="test",
        confidence=0.5,
    )
    claim_elements = {
        "numbers": [],
        "dates": [],
        "entities": ["Albert Einstein"],
        "keywords": ["theory", "relativity"],
    }
    boosted = GroundingChecker._boost_confidence(result, claim_elements)

    assert boosted > 0.5

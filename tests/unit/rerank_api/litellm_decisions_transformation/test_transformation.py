from typing import Final

import pytest

import litellm
from litellm.rerank_api.litellm_decisions_transformation.transformation import (
    DecisionRerankRequest,
    DecisionRerankResult,
    LiteLLMDecisionsRerankConfig,
)
from litellm.types.decisions import (
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
    OpenAIPredicateAnswer,
    OpenAIRefusalAnswer,
)


def test_invalid_score_keeps_completed_responses_and_usage_for_failure_billing():
    request: Final = DecisionRerankRequest(strategy="pointwise", query="warranty", documents=("relevant", "invalid"))
    responses: Final = tuple(
        OpenAIDecisionResponse(
            model="decision-unit-test",
            answers=(OpenAIPredicateAnswer(name="relevance", probability=score),),
            usage=OpenAIDecisionUsage(input_tokens=10, output_tokens=2, total_tokens=12),
        )
        for score in (0.9, 1.4)
    )
    outcomes: Final = tuple(
        DecisionRerankResult(index, request.document_text(index), response=response)
        for index, response in enumerate(responses)
    )
    result, error = LiteLLMDecisionsRerankConfig.transform_decisions_response(
        outcomes, request, "decision-unit-test", "typesafe"
    )
    assert isinstance(error, litellm.InternalServerError)
    assert result.results == [{"index": 0, "relevance_score": 0.9}]
    assert result.meta["tokens"] == {"input_tokens": 20, "output_tokens": 4}
    assert result.decision_responses[0] is responses[0]
    assert result.decision_responses[1] is responses[1]


@pytest.mark.parametrize("strategy,limit", [("pointwise", 1000), ("multiple_choice", 255)])
def test_document_limit_is_enforced_at_request_boundary(strategy, limit):
    params: Final = {"query": "warranty", "documents": ("passage",) * limit}
    request: Final = LiteLLMDecisionsRerankConfig.validate_request(params, "decision-unit-test", "typesafe", strategy)
    assert len(request.documents) == limit
    with pytest.raises(litellm.BadRequestError, match=str(limit)):
        LiteLLMDecisionsRerankConfig.validate_request(
            {**params, "documents": ("passage",) * (limit + 1)}, "decision-unit-test", "typesafe", strategy
        )


@pytest.mark.parametrize("per_request,global_default", [(True, False), (None, True), (False, True)])
@pytest.mark.parametrize(
    "unsupported",
    [
        {"rank_fields": ["text"]},
        {"max_chunks_per_doc": 1},
        {"max_tokens_per_doc": 10},
        {"instruction": "Rank only identical text"},
    ],
)
def test_drop_params_preserves_supported_request_fields(per_request, global_default, unsupported, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", global_default)
    request: Final = LiteLLMDecisionsRerankConfig.validate_request(
        {
            "query": "warranty",
            "documents": ("passage",),
            "top_n": 1,
            "return_documents": True,
            "drop_params": per_request,
            "extra_headers": {"x-request-id": "supported-header"},
            "safety_identifier": "supported-identifier",
            **unsupported,
        },
        "decision-unit-test",
        "typesafe",
        "pointwise",
    )
    assert request.query == "warranty"
    assert request.documents == ("passage",)
    assert request.top_n == 1
    assert request.return_documents is True
    assert request.extra_headers == {"x-request-id": "supported-header"}
    assert request.safety_identifier == "supported-identifier"
    assert request.timeout == 600.0


def test_unsupported_parameters_still_fail_when_dropping_is_disabled(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    with pytest.raises(litellm.UnsupportedParamsError, match="rank_fields"):
        LiteLLMDecisionsRerankConfig.validate_request(
            {"query": "warranty", "documents": ("passage",), "rank_fields": ["text"], "drop_params": False},
            "decision-unit-test",
            "typesafe",
            "pointwise",
        )


@pytest.mark.parametrize("strategy", ["pointwise", "multiple_choice"])
def test_refusal_retains_billable_usage_and_returns_content_policy_error(strategy):
    request: Final = DecisionRerankRequest(strategy=strategy, query="warranty", documents=("first", "second"))
    response: Final = OpenAIDecisionResponse(
        model="decision-unit-test",
        answers=(OpenAIRefusalAnswer(name="relevance"),),
        usage=OpenAIDecisionUsage(input_tokens=10, output_tokens=2, total_tokens=12),
    )
    result, error = LiteLLMDecisionsRerankConfig.transform_decisions_response(
        (DecisionRerankResult(0, request.document_text(0), response=response),),
        request,
        "decision-unit-test",
        "typesafe",
    )
    assert isinstance(error, litellm.ContentPolicyViolationError)
    assert result.results == []
    assert result.meta["tokens"] == {"input_tokens": 10, "output_tokens": 2}
    assert result.decision_responses == (response,)

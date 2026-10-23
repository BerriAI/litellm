from typing import Final

import litellm
from litellm.rerank_api.litellm_decisions_transformation.transformation import (
    DecisionRerankRequest,
    DecisionRerankResult,
    LiteLLMDecisionsRerankConfig,
)
from litellm.types.decisions import OpenAIDecisionResponse, OpenAIDecisionUsage, OpenAIPredicateAnswer


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

"""Convert rerank requests and Decisions responses independently of provider transport."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, model_validator

import litellm
from litellm.types.decisions import (
    OpenAIChoiceAnswer,
    OpenAIChoiceOption,
    OpenAIChoiceQuestion,
    OpenAIDecisionRequestBody,
    OpenAIDecisionResponse,
    OpenAIPredicateAnswer,
    OpenAIPredicateQuestion,
    OpenAIRefusalAnswer,
)
from litellm.types.rerank import DecisionRerankConfig, DecisionRerankResponse, RerankResponseResult

_RELEVANCE: Final = (
    "Is the passage relevant to the query, meaning it contains information that answers, supports or contradicts it?"
)
_UNSUPPORTED: Final = frozenset(
    {
        "rank_fields",
        "max_chunks_per_doc",
        "max_tokens_per_doc",
        "instruction",
        "truncate_prompt_tokens",
        "max_tokens_per_query",
        "truncation_side",
    }
)


class _TextDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    text: str


class DecisionRerankRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    rerank_mode: Literal["decisions"]
    decision_rerank: DecisionRerankConfig = DecisionRerankConfig()
    query: str
    documents: Annotated[tuple[str | _TextDocument, ...], Field(min_length=1)]
    top_n: Annotated[StrictInt | None, Field(ge=1)] = None
    return_documents: bool | None = None
    timeout: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 600.0
    extra_headers: Mapping[str, str] | None = None
    safety_identifier: str | None = None

    @model_validator(mode="after")
    def validate_choice_request(self) -> "DecisionRerankRequest":
        if self.decision_rerank.strategy == "multiple_choice":
            if not 2 <= len(self.documents) <= 255:
                raise ValueError("multiple_choice requires between 2 and 255 documents")
            if self.decision_rerank.refusal_policy != "error":
                raise ValueError("multiple_choice requires refusal_policy='error'")
        return self

    def document_text(self, index: int) -> str:
        document: Final = self.documents[index]
        return document if isinstance(document, str) else document.text


@dataclass(frozen=True, slots=True)
class DecisionRerankResult:
    index: int
    text: str
    response: OpenAIDecisionResponse | None = None
    error: Exception | None = None


def _invalid_response(index: int, reason: str, model: str, provider: str) -> litellm.InternalServerError:
    return litellm.InternalServerError(
        message=f"Invalid decision rerank response for document {index}: {reason}",
        model=model,
        llm_provider=provider,
    )


def _score(
    outcome: DecisionRerankResult, refusal_policy: Literal["error", "zero"], model: str, provider: str
) -> float | Exception:
    if outcome.error is not None:
        return outcome.error
    response: Final = outcome.response
    if response is None or len(response.answers) != 1:
        return _invalid_response(outcome.index, "expected one relevance answer", model, provider)
    answer: Final = response.answers[0]
    if isinstance(answer, OpenAIRefusalAnswer):
        if refusal_policy == "zero":
            return 0.0
        return litellm.ContentPolicyViolationError(
            message=f"Decision model refused to score document {outcome.index}",
            model=model,
            llm_provider=provider,
        )
    if not isinstance(answer, OpenAIPredicateAnswer) or answer.name != "relevance":
        return _invalid_response(outcome.index, "expected the relevance predicate", model, provider)
    if not math.isfinite(answer.probability) or not 0 <= answer.probability <= 1:
        return _invalid_response(outcome.index, "probability must be finite and between zero and one", model, provider)
    return answer.probability


def _result(index: int, text: str, score: float, return_documents: bool) -> RerankResponseResult:
    if return_documents:
        return RerankResponseResult(
            index=index,
            relevance_score=score,
            document={"text": text},
        )
    return RerankResponseResult(index=index, relevance_score=score)


def _is_refusal(response: OpenAIDecisionResponse) -> bool:
    return any(isinstance(answer, OpenAIRefusalAnswer) for answer in response.answers)


def _choice_scores(
    outcome: DecisionRerankResult, request: DecisionRerankRequest, model: str, provider: str
) -> tuple[float, ...] | Exception:
    if outcome.error is not None:
        return outcome.error
    response: Final = outcome.response
    if response is None or len(response.answers) != 1:
        return _invalid_response(outcome.index, "expected one choice answer", model, provider)
    answer: Final = response.answers[0]
    if isinstance(answer, OpenAIRefusalAnswer):
        return litellm.ContentPolicyViolationError(
            message="Decision model refused to rank the candidate set",
            model=model,
            llm_provider=provider,
        )
    if not isinstance(answer, OpenAIChoiceAnswer) or answer.name != "relevance":
        return _invalid_response(outcome.index, "expected the relevance choice", model, provider)
    labels: Final = tuple(f"p{index}" for index in range(len(request.documents)))
    probabilities: Final = {item.value: item.probability for item in answer.probabilities}
    if len(answer.probabilities) != len(labels) or set(probabilities) != set(labels):
        return _invalid_response(outcome.index, "expected each document label exactly once", model, provider)
    scores: Final = tuple(probabilities[label] for label in labels)
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        return _invalid_response(
            outcome.index, "probabilities must be finite and between zero and one", model, provider
        )
    # Two-decimal provider rounding can move the total by half a cent per option.
    if (
        sum(scores) <= 0
        or sum(max(0.0, score - 0.005) for score in scores) > 1.0 + 1e-9
        or sum(min(1.0, score + 0.005) for score in scores) < 1.0 - 1e-9
    ):
        return _invalid_response(
            outcome.index,
            "choice probabilities must sum to one within rounding precision",
            model,
            provider,
        )
    return scores


def _scores(
    outcomes: Sequence[DecisionRerankResult], request: DecisionRerankRequest, model: str, provider: str
) -> tuple[float | Exception, ...]:
    if request.decision_rerank.strategy == "multiple_choice":
        scores: Final = _choice_scores(outcomes[0], request, model, provider) if outcomes else ()
        return (scores,) if isinstance(scores, Exception) else scores
    return tuple(_score(outcome, request.decision_rerank.refusal_policy, model, provider) for outcome in outcomes)


def _ranked_results(
    outcomes: Sequence[DecisionRerankResult], request: DecisionRerankRequest, scores: Sequence[float | Exception]
) -> tuple[RerankResponseResult, ...]:
    if request.decision_rerank.strategy == "multiple_choice":
        return tuple(
            _result(index, request.document_text(index), score, bool(request.return_documents))
            for index, score in enumerate(scores)
            if isinstance(score, float)
        )
    return tuple(
        _result(outcome.index, outcome.text, score, bool(request.return_documents))
        for outcome, score in zip(outcomes, scores, strict=True)
        if isinstance(score, float)
    )


def _response(
    outcomes: Sequence[DecisionRerankResult], request: DecisionRerankRequest, scores: Sequence[float | Exception]
) -> DecisionRerankResponse:
    responses: Final = tuple(outcome.response for outcome in outcomes if outcome.response is not None)
    results: Final = _ranked_results(outcomes, request, scores)
    ordered: Final = sorted(results, key=lambda result: (-result["relevance_score"], result["index"]))
    return DecisionRerankResponse(
        results=ordered[: request.top_n],
        meta={
            "tokens": {
                "input_tokens": sum(response.usage.input_tokens for response in responses),
                "output_tokens": sum(response.usage.output_tokens for response in responses),
            },
            "refused_document_indices": [
                outcome.index for outcome in outcomes if outcome.response is not None and _is_refusal(outcome.response)
            ],
        },
        decision_responses=responses,
    )


class LiteLLMDecisionsRerankConfig:
    @staticmethod
    def validate_request(params: Mapping[str, object], model: str, provider: str) -> DecisionRerankRequest:
        unsupported: Final = tuple(key for key in _UNSUPPORTED if params.get(key) is not None)
        if unsupported:
            raise litellm.UnsupportedParamsError(
                message=f"Decision reranking does not support: {', '.join(sorted(unsupported))}",
                model=model,
                llm_provider=provider,
            )
        try:
            return DecisionRerankRequest.model_validate(params)
        except ValidationError as error:
            raise litellm.BadRequestError(
                message="Invalid decision rerank request: "
                + "; ".join(detail["msg"] for detail in error.errors(include_input=False, include_url=False)),
                model=model,
                llm_provider=provider,
            ) from error

    @staticmethod
    def transform_rerank_request(request: DecisionRerankRequest, index: int) -> OpenAIDecisionRequestBody:
        if request.decision_rerank.strategy == "multiple_choice":
            return OpenAIDecisionRequestBody(
                input=json.dumps(
                    {
                        "query": request.query,
                        "passages": {
                            f"p{index}": request.document_text(index) for index in range(len(request.documents))
                        },
                    },
                    ensure_ascii=False,
                ),
                questions=(
                    OpenAIChoiceQuestion(
                        type="choice",
                        name="relevance",
                        instructions="Which passage is the most relevant to the query?",
                        choices=tuple(OpenAIChoiceOption(value=f"p{index}") for index in range(len(request.documents))),
                    ),
                ),
            )
        return OpenAIDecisionRequestBody(
            input=json.dumps({"query": request.query, "passage": request.document_text(index)}, ensure_ascii=False),
            questions=(OpenAIPredicateQuestion(type="predicate", name="relevance", instructions=_RELEVANCE),),
        )

    @staticmethod
    def transform_decisions_response(
        outcomes: Sequence[DecisionRerankResult], request: DecisionRerankRequest, model: str, provider: str
    ) -> tuple[DecisionRerankResponse, Exception | None]:
        """Retain completed responses for billing even when another result fails validation."""
        scores: Final = _scores(outcomes, request, model, provider)
        return _response(outcomes, request, scores), next(
            (score for score in scores if isinstance(score, Exception)), None
        )

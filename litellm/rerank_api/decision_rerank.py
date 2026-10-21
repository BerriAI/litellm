import asyncio
import json
import math
import time
from collections.abc import Awaitable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from functools import partial
from typing import Annotated, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    model_validator,
)

import litellm
from litellm.decisions.call import (
    DecisionsCall,
    OpenAIDecisionRequestFields,
    asend,
    prepare_call,
    send,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.decisions import (
    OpenAIChoiceAnswer,
    OpenAIDecisionResponse,
    OpenAIPredicateAnswer,
    OpenAIRefusalAnswer,
)
from litellm.types.rerank import (
    DecisionRerankConfig,
    DecisionRerankResponse,
    RerankResponseResult,
)
from litellm.types.utils import Usage

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
_CHILD_OMIT: Final = frozenset({"litellm_logging_obj", "rerank_mode", "decision_rerank", "litellm_call_id"})


class _TextDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    text: str


class _Request(BaseModel):
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
    def validate_choice_request(self) -> "_Request":
        if self.decision_rerank.strategy == "multiple_choice":
            if not 2 <= len(self.documents) <= 255:
                raise ValueError("multiple_choice requires between 2 and 255 documents")
            if self.decision_rerank.refusal_policy != "error":
                raise ValueError("multiple_choice requires refusal_policy='error'")
        return self


@dataclass(frozen=True, slots=True)
class _DocumentCall:
    index: int
    text: str
    call: DecisionsCall


@dataclass(frozen=True, slots=True)
class _Outcome:
    document: _DocumentCall
    response: OpenAIDecisionResponse | None = None
    error: Exception | None = None


def _timeout(model: str, provider: str) -> litellm.Timeout:
    return litellm.Timeout(
        message="Decision reranking exceeded the request timeout",
        model=model,
        llm_provider=provider,
    )


def _call_with_deadline(document: _DocumentCall, deadline: float) -> DecisionsCall | litellm.Timeout:
    remaining: Final = deadline - time.monotonic()
    if remaining <= 0:
        return _timeout(document.call.requested_model, document.call.custom_llm_provider)
    return replace(document.call, timeout=remaining)


def _send(document: _DocumentCall, deadline: float) -> _Outcome:
    call: Final = _call_with_deadline(document, deadline)
    if isinstance(call, litellm.Timeout):
        return _Outcome(document, error=call)
    try:
        response: Final = send(call)
        if isinstance(response, OpenAIDecisionResponse):
            return _Outcome(document, response=response)
        return _Outcome(document, error=_invalid_response(document, "expected a Decisions response"))
    except Exception as error:  # noqa: BLE001  # preserve mapped provider failures alongside completed billable calls
        return _Outcome(document, error=error)


async def _asend(document: _DocumentCall, deadline: float, semaphore: asyncio.Semaphore) -> _Outcome:
    async with semaphore:
        call: Final = _call_with_deadline(document, deadline)
        if isinstance(call, litellm.Timeout):
            return _Outcome(document, error=call)
        try:
            response: Final = await asend(call)
            if isinstance(response, OpenAIDecisionResponse):
                return _Outcome(document, response=response)
            return _Outcome(
                document,
                error=_invalid_response(document, "expected a Decisions response"),
            )
        except Exception as error:  # noqa: BLE001  # preserve mapped provider failures alongside completed billable calls
            return _Outcome(document, error=error)


def _invalid_response(document: _DocumentCall, reason: str) -> litellm.InternalServerError:
    return litellm.InternalServerError(
        message=f"Invalid decision rerank response for document {document.index}: {reason}",
        model=document.call.requested_model,
        llm_provider=document.call.custom_llm_provider,
    )


def _score(outcome: _Outcome, refusal_policy: Literal["error", "zero"]) -> float | Exception:
    if outcome.error is not None:
        return outcome.error
    response: Final = outcome.response
    if response is None or len(response.answers) != 1:
        return _invalid_response(outcome.document, "expected one relevance answer")
    answer: Final = response.answers[0]
    if isinstance(answer, OpenAIRefusalAnswer):
        if refusal_policy == "zero":
            return 0.0
        return litellm.ContentPolicyViolationError(
            message=f"Decision model refused to score document {outcome.document.index}",
            model=outcome.document.call.requested_model,
            llm_provider=outcome.document.call.custom_llm_provider,
        )
    if not isinstance(answer, OpenAIPredicateAnswer) or answer.name != "relevance":
        return _invalid_response(outcome.document, "expected the relevance predicate")
    if not math.isfinite(answer.probability) or not 0 <= answer.probability <= 1:
        return _invalid_response(outcome.document, "probability must be finite and between zero and one")
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


def _choice_scores(outcome: _Outcome, request: _Request) -> tuple[float, ...] | Exception:
    if outcome.error is not None:
        return outcome.error
    response: Final = outcome.response
    if response is None or len(response.answers) != 1:
        return _invalid_response(outcome.document, "expected one choice answer")
    answer: Final = response.answers[0]
    if isinstance(answer, OpenAIRefusalAnswer):
        return litellm.ContentPolicyViolationError(
            message="Decision model refused to rank the candidate set",
            model=outcome.document.call.requested_model,
            llm_provider=outcome.document.call.custom_llm_provider,
        )
    if not isinstance(answer, OpenAIChoiceAnswer) or answer.name != "relevance":
        return _invalid_response(outcome.document, "expected the relevance choice")
    labels: Final = tuple(f"p{index}" for index in range(len(request.documents)))
    probabilities: Final = {item.value: item.probability for item in answer.probabilities}
    if len(answer.probabilities) != len(labels) or set(probabilities) != set(labels):
        return _invalid_response(outcome.document, "expected each document label exactly once")
    scores: Final = tuple(probabilities[label] for label in labels)
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        return _invalid_response(outcome.document, "probabilities must be finite and between zero and one")
    # Two-decimal provider rounding can move the total by half a cent per option.
    if (
        sum(scores) <= 0
        or sum(max(0.0, score - 0.005) for score in scores) > 1.0 + 1e-9
        or sum(min(1.0, score + 0.005) for score in scores) < 1.0 - 1e-9
    ):
        return _invalid_response(
            outcome.document,
            "choice probabilities must sum to one within rounding precision",
        )
    return scores


def _scores(outcomes: Sequence[_Outcome], request: _Request) -> tuple[float | Exception, ...]:
    if request.decision_rerank.strategy == "multiple_choice":
        scores: Final = _choice_scores(outcomes[0], request) if outcomes else ()
        return (scores,) if isinstance(scores, Exception) else scores
    return tuple(_score(outcome, request.decision_rerank.refusal_policy) for outcome in outcomes)


def _ranked_results(
    outcomes: Sequence[_Outcome], request: _Request, scores: Sequence[float | Exception]
) -> tuple[RerankResponseResult, ...]:
    if request.decision_rerank.strategy == "multiple_choice":
        return tuple(
            _result(index, _text(request.documents[index]), score, bool(request.return_documents))
            for index, score in enumerate(scores)
            if isinstance(score, float)
        )
    return tuple(
        _result(outcome.document.index, outcome.document.text, score, bool(request.return_documents))
        for outcome, score in zip(outcomes, scores, strict=True)
        if isinstance(score, float)
    )


def _response(
    outcomes: Sequence[_Outcome], request: _Request, scores: Sequence[float | Exception]
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
                outcome.document.index
                for outcome in outcomes
                if outcome.response is not None and _is_refusal(outcome.response)
            ],
        },
        decision_responses=responses,
    )


def _record_partial(response: DecisionRerankResponse, logging_obj: Logging) -> None:
    responses: Final = response.decision_responses
    if not responses:
        return
    logging_obj.record_partial_usage_for_failure(
        Usage(
            prompt_tokens=sum(item.usage.input_tokens for item in responses),
            completion_tokens=sum(item.usage.output_tokens for item in responses),
        ),
        logging_obj.response_cost_calculator(result=response) or 0.0,
    )


def _finish(
    outcomes: Sequence[_Outcome],
    request: _Request,
    logging_obj: Logging,
    timed_out: bool,
    model: str,
    provider: str,
) -> DecisionRerankResponse:
    scores: Final = _scores(outcomes, request)
    response: Final = _response(outcomes, request, scores)
    failure: Final = next((score for score in scores if isinstance(score, Exception)), None) or (
        _timeout(model, provider) if timed_out else None
    )
    if failure is not None:
        _record_partial(response, logging_obj)
        raise failure
    logging_obj.model_call_details["combined_usage_object"] = Usage(  # rebind-ok: log aggregate usage
        prompt_tokens=sum(item.usage.input_tokens for item in response.decision_responses),
        completion_tokens=sum(item.usage.output_tokens for item in response.decision_responses),
    )
    return response


def _rerank(
    documents: tuple[_DocumentCall, ...],
    request: _Request,
    logging_obj: Logging,
    model: str,
    provider: str,
) -> DecisionRerankResponse:
    deadline: Final = time.monotonic() + request.timeout
    executor: Final = ThreadPoolExecutor(max_workers=request.decision_rerank.max_concurrency)
    futures: Final = tuple(executor.submit(_send, document, deadline) for document in documents)
    try:
        _, pending = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
        outcomes: Final = tuple(future.result() for future in futures if future.done() and not future.cancelled())
        return _finish(outcomes, request, logging_obj, bool(pending), model, provider)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


async def _arerank(
    documents: tuple[_DocumentCall, ...],
    request: _Request,
    logging_obj: Logging,
    model: str,
    provider: str,
) -> DecisionRerankResponse:
    deadline: Final = time.monotonic() + request.timeout
    semaphore: Final = asyncio.Semaphore(request.decision_rerank.max_concurrency)
    tasks: Final = tuple(asyncio.create_task(_asend(document, deadline, semaphore)) for document in documents)
    try:
        _, pending = await asyncio.wait(tasks, timeout=max(0.0, deadline - time.monotonic()))
        outcomes: Final = tuple(
            _Outcome(
                document,
                error=litellm.APIConnectionError(
                    message="Decision rerank provider request was cancelled",
                    model=model,
                    llm_provider=provider,
                ),
            )
            if task.cancelled()
            else task.result()
            for document, task in zip(documents, tasks, strict=True)
            if task.done()
        )
        return _finish(outcomes, request, logging_obj, bool(pending), model, provider)
    except asyncio.CancelledError:
        outcomes_on_cancel: Final = tuple(task.result() for task in tasks if task.done() and not task.cancelled())
        _record_partial(_response(outcomes_on_cancel, request, _scores(outcomes_on_cancel, request)), logging_obj)
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _text(document: str | _TextDocument) -> str:
    return document if isinstance(document, str) else document.text


def _prepare_document(
    index: int,
    document: str | _TextDocument,
    request: _Request,
    model: str,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    kwargs: Mapping[str, object],
) -> _DocumentCall:
    text: Final = _text(document)
    return _DocumentCall(
        index=index,
        text=text,
        call=prepare_call(
            model=model,
            fields=_choice_fields(request)
            if request.decision_rerank.strategy == "multiple_choice"
            else OpenAIDecisionRequestFields(
                input=json.dumps({"query": request.query, "passage": text}, ensure_ascii=False),
                questions=(
                    {
                        "type": "predicate",
                        "name": "relevance",
                        "instructions": _RELEVANCE,
                    },
                ),
            ),
            safety_identifier=request.safety_identifier,
            api_key=api_key,
            api_base=api_base,
            timeout=request.timeout,
            custom_llm_provider=provider,
            extra_headers=request.extra_headers,
            kwargs={key: value for key, value in kwargs.items() if key not in _CHILD_OMIT},
        ),
    )


def _choice_fields(request: _Request) -> OpenAIDecisionRequestFields:
    return OpenAIDecisionRequestFields(
        input=json.dumps(
            {
                "query": request.query,
                "passages": {f"p{index}": _text(item) for index, item in enumerate(request.documents)},
            },
            ensure_ascii=False,
        ),
        questions=(
            {
                "type": "choice",
                "name": "relevance",
                "instructions": "Which passage is the most relevant to the query?",
                "choices": tuple({"value": f"p{index}"} for index in range(len(request.documents))),
            },
        ),
    )


def rerank_with_decisions(
    *,
    model: str,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    logging_obj: Logging,
    is_async: bool,
    params: Mapping[str, object],
    kwargs: Mapping[str, object],
) -> DecisionRerankResponse | Awaitable[DecisionRerankResponse]:
    unsupported: Final = tuple(key for key in _UNSUPPORTED if params.get(key) is not None)
    if unsupported:
        raise litellm.UnsupportedParamsError(
            message=f"Decision reranking does not support: {', '.join(sorted(unsupported))}",
            model=model,
            llm_provider=provider,
        )
    try:
        request: Final = _Request.model_validate(params)
    except ValidationError as error:
        raise litellm.BadRequestError(
            message="Invalid decision rerank request: "
            + "; ".join(detail["msg"] for detail in error.errors(include_input=False, include_url=False)),
            model=model,
            llm_provider=provider,
        ) from error
    logging_obj.update_from_kwargs(  # pyright: ignore[reportUnknownMemberType]  # Logging accepts legacy provider kwargs
        kwargs=dict(kwargs),
        model=model,
        custom_llm_provider=provider,
        optional_params=request.model_dump(
            exclude={"extra_headers", "safety_identifier", "timeout"}, exclude_none=True
        ),
        litellm_params={key: value for key, value in kwargs.items() if key != "litellm_logging_obj"},
    )
    prepare: Final = partial(
        _prepare_document,
        request=request,
        model=model,
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        kwargs=kwargs,
    )
    documents: Final = (
        (prepare(0, request.documents[0]),)
        if request.decision_rerank.strategy == "multiple_choice"
        else tuple(prepare(index, document) for index, document in enumerate(request.documents))
    )
    logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # Logging accepts the logical rerank request
        input=request.query,
        api_key=documents[0].call.api_key,
        additional_args={
            "complete_input_dict": {
                "model": model,
                **request.model_dump(
                    mode="json", exclude={"extra_headers", "safety_identifier", "timeout"}, exclude_none=True
                ),
            },
            "api_base": documents[0].call.api_base,
            "headers": dict(documents[0].call.headers),
        },
    )
    if is_async:
        return _arerank(documents, request, logging_obj, model, provider)
    return _rerank(documents, request, logging_obj, model, provider)

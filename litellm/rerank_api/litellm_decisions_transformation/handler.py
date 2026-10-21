"""Execute rerank requests through LiteLLM's existing Decisions transport."""

import asyncio
import time
from collections.abc import Awaitable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from functools import partial
from typing import Final

import litellm
from litellm.decisions.call import DecisionsCall, OpenAIDecisionRequestFields, asend, prepare_call, send
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rerank_api.litellm_decisions_transformation.transformation import (
    DecisionRerankRequest,
    DecisionRerankResult,
    LiteLLMDecisionsRerankConfig,
)
from litellm.types.decisions import OpenAIDecisionResponse
from litellm.types.rerank import DecisionRerankResponse
from litellm.types.utils import Usage

_CHILD_OMIT: Final = frozenset({"litellm_logging_obj", "rerank_mode", "decision_rerank", "litellm_call_id"})


@dataclass(frozen=True, slots=True)
class _DocumentCall:
    index: int
    text: str
    call: DecisionsCall


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


def _send(document: _DocumentCall, deadline: float) -> DecisionRerankResult:
    call: Final = _call_with_deadline(document, deadline)
    if isinstance(call, litellm.Timeout):
        return DecisionRerankResult(document.index, document.text, error=call)
    try:
        response: Final = send(call)
        if isinstance(response, OpenAIDecisionResponse):
            return DecisionRerankResult(document.index, document.text, response=response)
        return DecisionRerankResult(
            document.index, document.text, error=_invalid_response(document, "expected a Decisions response")
        )
    except Exception as error:  # noqa: BLE001  # preserve mapped provider failures alongside completed billable calls
        return DecisionRerankResult(document.index, document.text, error=error)


async def _asend(document: _DocumentCall, deadline: float, semaphore: asyncio.Semaphore) -> DecisionRerankResult:
    async with semaphore:
        call: Final = _call_with_deadline(document, deadline)
        if isinstance(call, litellm.Timeout):
            return DecisionRerankResult(document.index, document.text, error=call)
        try:
            response: Final = await asend(call)
            if isinstance(response, OpenAIDecisionResponse):
                return DecisionRerankResult(document.index, document.text, response=response)
            return DecisionRerankResult(
                document.index,
                document.text,
                error=_invalid_response(document, "expected a Decisions response"),
            )
        except Exception as error:  # noqa: BLE001  # preserve mapped provider failures alongside completed billable calls
            return DecisionRerankResult(document.index, document.text, error=error)


def _invalid_response(document: _DocumentCall, reason: str) -> litellm.InternalServerError:
    return litellm.InternalServerError(
        message=f"Invalid decision rerank response for document {document.index}: {reason}",
        model=document.call.requested_model,
        llm_provider=document.call.custom_llm_provider,
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
    outcomes: Sequence[DecisionRerankResult],
    request: DecisionRerankRequest,
    logging_obj: Logging,
    timed_out: bool,
    model: str,
    provider: str,
) -> DecisionRerankResponse:
    response, validation_failure = LiteLLMDecisionsRerankConfig.transform_decisions_response(
        outcomes, request, model, provider
    )
    failure: Final = validation_failure or (_timeout(model, provider) if timed_out else None)
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
    request: DecisionRerankRequest,
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
    request: DecisionRerankRequest,
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
            DecisionRerankResult(
                document.index,
                document.text,
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
        response_on_cancel, _ = LiteLLMDecisionsRerankConfig.transform_decisions_response(
            outcomes_on_cancel, request, model, provider
        )
        _record_partial(response_on_cancel, logging_obj)
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _prepare_document(
    index: int,
    request: DecisionRerankRequest,
    model: str,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    kwargs: Mapping[str, object],
) -> _DocumentCall:
    text: Final = request.document_text(index)
    transformed: Final = LiteLLMDecisionsRerankConfig.transform_rerank_request(request, index)
    return _DocumentCall(
        index=index,
        text=text,
        call=prepare_call(
            model=model,
            fields=OpenAIDecisionRequestFields(input=transformed.input, questions=transformed.questions),
            safety_identifier=request.safety_identifier,
            api_key=api_key,
            api_base=api_base,
            timeout=request.timeout,
            custom_llm_provider=provider,
            extra_headers=request.extra_headers,
            kwargs={key: value for key, value in kwargs.items() if key not in _CHILD_OMIT},
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
    request: Final = LiteLLMDecisionsRerankConfig.validate_request(params, model, provider)
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
        (prepare(0),)
        if request.decision_rerank.strategy == "multiple_choice"
        else tuple(prepare(index) for index in range(len(request.documents)))
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

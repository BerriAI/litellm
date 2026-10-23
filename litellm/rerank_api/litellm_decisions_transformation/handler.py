"""Execute rerank requests through LiteLLM's existing Decisions transport."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial
from queue import Empty, SimpleQueue
from threading import Event
from typing import Final, Literal

import litellm
from litellm.decisions.call import (
    DecisionsCall,
    OpenAIDecisionRequestFields,
    asend,
    prepare_call,
    send,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rerank_api.litellm_decisions_transformation.transformation import (
    DecisionRerankRequest,
    DecisionRerankResult,
    LiteLLMDecisionsRerankConfig,
    invalid_response,
    result_error,
)
from litellm.types.decisions import OpenAIDecisionResponse
from litellm.types.rerank import DecisionRerankResponse
from litellm.types.utils import Usage

_CHILD_OMIT: Final = frozenset({"litellm_logging_obj", "litellm_call_id"})
_MAX_CONCURRENCY: Final = 8


class DecisionRerankLoggingCapture:
    def __init__(self, call_id: str) -> None:
        self.call_id: Final = call_id
        self.logging_obj: Logging | None = None

    @contextmanager
    def capture(self) -> Generator[None]:
        token: Final = _LOGGING_CAPTURE.set(self)
        try:
            yield
        finally:
            _LOGGING_CAPTURE.reset(token)

    def record(self, logging_obj: Logging) -> None:
        if logging_obj.litellm_call_id == self.call_id:
            self.logging_obj = logging_obj


_LOGGING_CAPTURE: Final[ContextVar[DecisionRerankLoggingCapture | None]] = ContextVar(
    "decision_rerank_logging_capture", default=None
)


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


def _send(
    document: _DocumentCall,
    deadline: float,
) -> DecisionRerankResult:
    call: Final = _call_with_deadline(document, deadline)
    if isinstance(call, litellm.Timeout):
        return DecisionRerankResult(document.index, document.text, error=call)
    try:
        response: Final = send(call)
        if isinstance(response, OpenAIDecisionResponse):
            return DecisionRerankResult(document.index, document.text, response=response)
        return DecisionRerankResult(
            document.index,
            document.text,
            error=invalid_response(
                document.index, "expected a Decisions response", call.requested_model, call.custom_llm_provider
            ),
        )
    except Exception as error:  # noqa: BLE001  # preserve mapped provider failures alongside completed billable calls
        return DecisionRerankResult(document.index, document.text, error=error)


async def _asend(
    document: _DocumentCall,
    deadline: float,
    cancellation_requested: asyncio.Event,
) -> DecisionRerankResult:
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
            error=invalid_response(
                document.index, "expected a Decisions response", call.requested_model, call.custom_llm_provider
            ),
        )
    except asyncio.CancelledError:
        if cancellation_requested.is_set():
            raise
        return DecisionRerankResult(
            document.index,
            document.text,
            error=litellm.APIConnectionError(
                message="Decision rerank provider request was cancelled",
                model=document.call.requested_model,
                llm_provider=document.call.custom_llm_provider,
            ),
        )
    except Exception as error:  # noqa: BLE001  # preserve mapped provider failures alongside completed billable calls
        return DecisionRerankResult(document.index, document.text, error=error)


def _aggregate_usage(response: DecisionRerankResponse) -> Usage:
    prompt_tokens: Final = sum(item.usage.input_tokens for item in response.decision_responses)
    completion_tokens: Final = sum(item.usage.output_tokens for item in response.decision_responses)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _record_partial(response: DecisionRerankResponse, logging_obj: Logging) -> None:
    if not response.decision_responses:
        return
    logging_obj.record_partial_usage_for_failure(
        _aggregate_usage(response),
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
    usage: Final = _aggregate_usage(response)
    logging_obj.model_call_details["combined_usage_object"] = usage  # rebind-ok: log aggregate usage
    return response


def _rerank(
    prepare: Callable[[int], _DocumentCall],
    request: DecisionRerankRequest,
    logging_obj: Logging,
    model: str,
    provider: str,
    deadline: float,
) -> DecisionRerankResponse:
    count: Final = request.request_count
    workers: Final = min(count, _MAX_CONCURRENCY)
    completed: Final[SimpleQueue[DecisionRerankResult]] = SimpleQueue()
    stopped: Final = Event()

    def record(outcome: DecisionRerankResult) -> None:
        if result_error(outcome, request, model, provider) is not None:
            stopped.set()
        completed.put(outcome)

    def run_one(index: int) -> None:
        try:
            document: Final = prepare(index)
            if stopped.is_set():
                return
            record(_send(document, deadline))
        except Exception as error:  # noqa: BLE001  # preparation failures retain completed billable calls
            record(DecisionRerankResult(index, request.document_text(index), error=error))

    def run(indices: range) -> None:
        for index in indices:
            if stopped.is_set() or time.monotonic() >= deadline:
                return
            run_one(index)

    executor: Final = ThreadPoolExecutor(max_workers=workers)
    futures: Final = tuple(executor.submit(run, range(worker, count, workers)) for worker in range(workers))
    try:
        waited: Final = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
        timed_out: Final = bool(waited.not_done) or time.monotonic() >= deadline
    finally:
        stopped.set()
        executor.shutdown(wait=True, cancel_futures=True)
    outcomes: Final[list[DecisionRerankResult]] = []  # mutable-ok: retain completed results for partial billing
    while True:
        try:
            outcomes.append(completed.get_nowait())
        except Empty:
            break
    return _finish(outcomes, request, logging_obj, timed_out or len(outcomes) < count, model, provider)


async def _arerank(
    prepare: Callable[[int], _DocumentCall],
    request: DecisionRerankRequest,
    logging_obj: Logging,
    model: str,
    provider: str,
    deadline: float,
) -> DecisionRerankResponse:
    count: Final = request.request_count
    workers: Final = min(count, _MAX_CONCURRENCY)
    outcomes: Final[list[DecisionRerankResult]] = []  # mutable-ok: retain completed results for partial billing
    cancellation_requested: Final = asyncio.Event()
    stopped: Final = asyncio.Event()

    def record(outcome: DecisionRerankResult) -> None:
        if result_error(outcome, request, model, provider) is not None:
            stopped.set()
        outcomes.append(outcome)

    async def run(indices: range) -> None:
        for index in indices:
            if stopped.is_set() or time.monotonic() >= deadline:
                return
            try:
                record(
                    await _asend(
                        prepare(index),
                        deadline,
                        cancellation_requested,
                    )
                )
            except Exception as error:  # noqa: BLE001  # preparation failures retain completed billable calls
                record(DecisionRerankResult(index, request.document_text(index), error=error))

    tasks: Final = tuple(asyncio.create_task(run(range(worker, count, workers))) for worker in range(workers))
    try:
        try:
            _, pending = await asyncio.wait(tasks, timeout=max(0.0, deadline - time.monotonic()))
            timed_out: Final = bool(pending) or time.monotonic() >= deadline
        finally:
            stopped.set()
            cancellation_requested.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        response_on_cancel, _ = LiteLLMDecisionsRerankConfig.transform_decisions_response(
            outcomes, request, model, provider
        )
        _record_partial(response_on_cancel, logging_obj)
        raise
    return _finish(outcomes, request, logging_obj, timed_out or len(outcomes) < count, model, provider)


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
    strategy: Literal["pointwise", "multiple_choice"],
    params: Mapping[str, object],
    kwargs: Mapping[str, object],
) -> DecisionRerankResponse | Awaitable[DecisionRerankResponse]:
    logging_capture: Final = _LOGGING_CAPTURE.get()
    if logging_capture is not None:
        logging_capture.record(logging_obj)
    request: Final = LiteLLMDecisionsRerankConfig.validate_request(params, model, provider, strategy)
    deadline: Final = time.monotonic() + request.timeout
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
    first: Final = prepare(0)

    def prepare_next(index: int) -> _DocumentCall:
        return first if index == 0 else prepare(index)

    logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # Logging accepts the logical rerank request
        input=request.query,
        api_key=first.call.api_key,
        additional_args={
            "complete_input_dict": {
                "model": model,
                **request.model_dump(
                    mode="json",
                    exclude={"extra_headers", "safety_identifier", "timeout"},
                    exclude_none=True,
                ),
            },
            "api_base": first.call.api_base,
            "headers": dict(first.call.headers),
        },
    )
    if is_async:
        return _arerank(
            prepare_next,
            request,
            logging_obj,
            model,
            provider,
            deadline,
        )
    return _rerank(prepare_next, request, logging_obj, model, provider, deadline)

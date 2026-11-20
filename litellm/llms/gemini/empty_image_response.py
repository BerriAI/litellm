from collections.abc import Mapping
from typing import Final, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.exceptions import (
    ContentPolicyViolationError,
    UnprocessableEntityError,
    error_response_without_credentials,
)
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.utils import ImageResponse, Usage

_REASONS_WITHOUT_A_POLICY_VERDICT: Final = frozenset({"NO_IMAGE", "IMAGE_OTHER"})
_ERROR_DOCS_URL: Final = "https://docs.litellm.ai/docs"


class FailedImageBilling(Protocol):
    def response_cost_calculator(self, result: ImageResponse) -> float | None: ...

    def record_partial_usage_for_failure(self, usage: Usage, response_cost: float) -> None: ...


class _Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    finish_reason: str | None = Field(default=None, alias="finishReason")
    finish_message: str | None = Field(default=None, alias="finishMessage")


class _PromptFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    block_reason: str | None = Field(default=None, alias="blockReason")
    block_reason_message: str | None = Field(default=None, alias="blockReasonMessage")


class _GenerateContentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidates: tuple[_Candidate, ...] | None = None
    prompt_feedback: _PromptFeedback | None = Field(default=None, alias="promptFeedback")


def _parsed(response: Mapping[str, object]) -> _GenerateContentResult:
    try:
        return _GenerateContentResult.model_validate(response)
    except ValidationError:
        return _GenerateContentResult()


def _is_refusal(candidate: _Candidate) -> bool:
    reason: Final = candidate.finish_reason
    return (
        reason is not None
        and reason not in _REASONS_WITHOUT_A_POLICY_VERDICT
        and map_finish_reason(reason) == "content_filter"
    )


def _bill_answer(billing: FailedImageBilling, answer: ImageResponse) -> None:
    if answer.usage is None:
        return
    cost: Final = billing.response_cost_calculator(result=answer) or 0.0
    if cost <= 0:
        return
    billing.record_partial_usage_for_failure(
        ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(answer.usage.model_dump()), cost
    )


def _with_detail(summary: str, detail: str | None) -> str:
    return f"{summary}: {detail}" if detail else summary


def _answered_as(status_code: int, raw_response: httpx.Response) -> httpx.Response:
    return error_response_without_credentials(status_code, raw_response, _ERROR_DOCS_URL)


def empty_image_response_error(
    response: Mapping[str, object],
    model: str,
    llm_provider: str,
    raw_response: httpx.Response,
    answer: ImageResponse,
    billing: FailedImageBilling,
) -> ContentPolicyViolationError | UnprocessableEntityError:
    result: Final = _parsed(response)
    feedback: Final = result.prompt_feedback
    if feedback is not None and feedback.block_reason:
        return ContentPolicyViolationError(
            message=_with_detail(
                f"Gemini blocked the image prompt, blockReason={feedback.block_reason}", feedback.block_reason_message
            ),
            model=model,
            llm_provider=llm_provider,
            response=_answered_as(400, raw_response),
            provider_specific_fields={
                "block_reason": feedback.block_reason,
                "block_reason_message": feedback.block_reason_message,
            },
        )
    candidates: Final = result.candidates or ()
    refusal: Final = next((candidate for candidate in candidates if _is_refusal(candidate)), None)
    if refusal is not None:
        return ContentPolicyViolationError(
            message=_with_detail(
                f"Gemini refused to return an image, finishReason={refusal.finish_reason}", refusal.finish_message
            ),
            model=model,
            llm_provider=llm_provider,
            response=_answered_as(400, raw_response),
            provider_specific_fields={
                "finish_reason": refusal.finish_reason,
                "finish_message": refusal.finish_message,
            },
        )
    finish_reasons: Final = ",".join(candidate.finish_reason or "none" for candidate in candidates) or "none"
    _bill_answer(billing, answer)
    return UnprocessableEntityError(
        message=f"Gemini returned no image, finishReason={finish_reasons}",
        model=model,
        llm_provider=llm_provider,
        response=_answered_as(422, raw_response),
    )

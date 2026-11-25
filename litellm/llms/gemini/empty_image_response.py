from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.exceptions import (
    ContentPolicyViolationError,
    UnprocessableEntityError,
    error_response_without_credentials,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping, map_finish_reason
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.utils import CallTypes, ImageResponse, Usage

_REASONS_WITHOUT_A_POLICY_VERDICT: Final = frozenset({"NO_IMAGE", "IMAGE_OTHER", "OTHER", "LANGUAGE"})
_REFUSALS_GOOGLE_DOES_NOT_CHARGE: Final = frozenset({"IMAGE_RECITATION"})
_ERROR_DOCS_URL: Final = "https://docs.litellm.ai/docs"


class UnservedImageError(UnprocessableEntityError):
    pass


class ImageRefusalError(ContentPolicyViolationError):
    pass


IMAGE_ANSWER_ERRORS: Final = (ImageRefusalError, UnservedImageError)
IMAGE_ANSWERS_BILLED_ON_FAILURE: Final = "image_answers_billed_on_failure"
IMAGE_ANSWER_TOKENS_COUNTED_ON_FAILURE: Final = "image_answer_tokens_counted_on_failure"
SETTLED_BY_THE_ASYNC_FAILURE_HOOK: Final = frozenset({CallTypes.aimage_generation.value, CallTypes.aimage_edit.value})


def image_answer_cost_on_failure(call_details: Mapping[str, object]) -> float:
    exception: Final = call_details.get("exception")
    response_cost: Final = call_details.get("response_cost")
    if not isinstance(exception, IMAGE_ANSWER_ERRORS) or not isinstance(response_cost, (int, float)):
        return 0.0
    return max(float(response_cost), 0.0)


@dataclass(frozen=True, slots=True)
class BilledDeploymentTokens:
    deployment_id: str
    deployment_name: str
    tokens: int


def billed_deployment_tokens(kwargs: Mapping[str, object]) -> BilledDeploymentTokens | None:
    if not isinstance(kwargs.get("exception"), IMAGE_ANSWER_ERRORS):
        return None
    logged: Final = as_str_mapping(kwargs.get("standard_logging_object")) or {}
    params: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
    routed: Final = ((as_str_mapping(params.get(c)) or {}).get("deployment") for c in ("litellm_metadata", "metadata"))
    deployment_name: Final = next((name for name in routed if isinstance(name, str)), None)
    deployment_id: Final = logged.get("model_id")
    tokens: Final = logged.get("total_tokens")
    if deployment_name is None or not isinstance(deployment_id, (str, int)) or not isinstance(tokens, int):
        return None
    return BilledDeploymentTokens(str(deployment_id), deployment_name, tokens)


def success_cost_without_failed_image_answers(call_details: Mapping[str, object], response_cost: float) -> float:
    billed_on_failure: Final = call_details.get(IMAGE_ANSWERS_BILLED_ON_FAILURE)
    if not isinstance(billed_on_failure, (int, float)):
        return response_cost
    return max(response_cost - billed_on_failure, 0.0)


def image_answer_tokens_counted_on_failure(call_details: Mapping[str, object]) -> int:
    billed: Final = billed_deployment_tokens(call_details)
    return 0 if billed is None else billed.tokens


def success_tokens_without_failed_image_answers(call_details: Mapping[str, object], total_tokens: int) -> int:
    counted_on_failure: Final = call_details.get(IMAGE_ANSWER_TOKENS_COUNTED_ON_FAILURE)
    if not isinstance(counted_on_failure, int):
        return total_tokens
    return max(total_tokens - counted_on_failure, 0)


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
    billing.record_partial_usage_for_failure(
        ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(answer.usage.model_dump()), cost
    )


def _bill_refusal(reason: str, billing: FailedImageBilling, answer: ImageResponse) -> None:
    if reason in _REFUSALS_GOOGLE_DOES_NOT_CHARGE:
        return
    _bill_answer(billing, answer)


def _answered_as(status_code: int, raw_response: httpx.Response) -> httpx.Response:
    return error_response_without_credentials(status_code, raw_response, _ERROR_DOCS_URL)


def _answered_without_model_output(raw_response: httpx.Response, result: _GenerateContentResult) -> httpx.Response:
    return httpx.Response(
        status_code=422,
        content=result.model_dump_json(by_alias=True, exclude_none=True),
        request=_answered_as(422, raw_response).request,
    )


def empty_image_response_error(
    response: Mapping[str, object],
    model: str,
    llm_provider: str,
    raw_response: httpx.Response,
    answer: ImageResponse,
    billing: FailedImageBilling,
) -> ImageRefusalError | UnservedImageError:
    result: Final = _parsed(response)
    feedback: Final = result.prompt_feedback
    if feedback is not None and feedback.block_reason:
        _bill_refusal(feedback.block_reason, billing, answer)
        return ImageRefusalError(
            message=f"Gemini blocked the image prompt, blockReason={feedback.block_reason}",
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
    if refusal is not None and refusal.finish_reason is not None:
        _bill_refusal(refusal.finish_reason, billing, answer)
        return ImageRefusalError(
            message=f"Gemini refused to return an image, finishReason={refusal.finish_reason}",
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
    return UnservedImageError(
        message=f"Gemini returned no image, finishReason={finish_reasons}",
        model=model,
        llm_provider=llm_provider,
        response=_answered_without_model_output(raw_response, result),
    )

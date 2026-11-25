from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.constants import IMAGE_CALL_TYPES
from litellm.exceptions import (
    ContentPolicyViolationError,
    UnprocessableEntityError,
    error_response_without_credentials,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping, map_finish_reason
from litellm.types.utils import ImageResponse, Usage

_REASONS_WITHOUT_A_POLICY_VERDICT: Final = frozenset({"NO_IMAGE", "IMAGE_OTHER", "OTHER", "LANGUAGE"})
_REFUSALS_GOOGLE_DOES_NOT_CHARGE: Final = frozenset({"IMAGE_RECITATION"})
_ERROR_DOCS_URL: Final = "https://docs.litellm.ai/docs"
_ATTEMPT_USAGES: Final = TypeAdapter(tuple[tuple[Usage, float], ...])


class UnservedImageError(UnprocessableEntityError):
    def __init__(
        self, message: str, model: str, llm_provider: str, response: httpx.Response, billed_answer: ImageResponse
    ) -> None:
        super().__init__(message=message, model=model, llm_provider=llm_provider, response=response)
        self.billed_answer: Final = billed_answer
        self._billed_instead: Final = partial(UnservedImageError, message, model, llm_provider, response)

    def billing(self, answer: ImageResponse) -> "UnservedImageError":
        return self._billed_instead(billed_answer=answer)


class ImageRefusalError(ContentPolicyViolationError):
    def __init__(
        self,
        message: str,
        model: str,
        llm_provider: str,
        response: httpx.Response,
        provider_specific_fields: Mapping[str, str | None],
        billed_answer: ImageResponse | None,
    ) -> None:
        super().__init__(
            message=message,
            model=model,
            llm_provider=llm_provider,
            response=response,
            provider_specific_fields=dict(provider_specific_fields),
        )
        self.billed_answer: Final = billed_answer
        self._billed_instead: Final = partial(
            ImageRefusalError, message, model, llm_provider, response, provider_specific_fields
        )

    def billing(self, answer: ImageResponse) -> "ImageRefusalError":
        return self._billed_instead(billed_answer=answer)


def is_answer_without_an_image(exception: object) -> bool:
    return isinstance(exception, (ImageRefusalError, UnservedImageError))


@dataclass(frozen=True, slots=True)
class BilledDeploymentTokens:
    deployment_id: str
    deployment_name: str
    tokens: int


def billed_deployment_tokens(kwargs: Mapping[str, object]) -> BilledDeploymentTokens | None:
    if not is_answer_without_an_image(kwargs.get("exception")):
        return None
    logged: Final = as_str_mapping(kwargs.get("standard_logging_object")) or {}
    params: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
    routed: Final = ((as_str_mapping(params.get(c)) or {}).get("deployment") for c in ("litellm_metadata", "metadata"))
    deployment_name: Final = next((name for name in routed if isinstance(name, str)), None)
    deployment_id: Final = logged.get("model_id")
    tokens: Final = logged.get("total_tokens")
    if not isinstance(tokens, int) or tokens <= 0:
        return None
    if deployment_name is None or not isinstance(deployment_id, (str, int)):
        return None
    return BilledDeploymentTokens(str(deployment_id), deployment_name, tokens)


def tokens_of_image_answers_billed_on_failure(kwargs: object) -> int:
    fields: Final = as_str_mapping(kwargs) or {}
    if fields.get("call_type") not in IMAGE_CALL_TYPES:
        return 0
    carried: Final = _ATTEMPT_USAGES.validate_python(fields.get("failed_attempts_billed_on_success", ()))
    return sum(usage.total_tokens for usage, _ in carried)


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


def _answered_as(status_code: int, raw_response: httpx.Response) -> httpx.Response:
    return error_response_without_credentials(status_code, raw_response, _ERROR_DOCS_URL)


def _answered_without_model_output(raw_response: httpx.Response, result: _GenerateContentResult) -> httpx.Response:
    return httpx.Response(
        status_code=422,
        content=result.model_dump_json(by_alias=True, exclude_none=True),
        request=_answered_as(422, raw_response).request,
    )


def _refusal_error(
    reason: str,
    message: str,
    fields: Mapping[str, str | None],
    model: str,
    llm_provider: str,
    raw_response: httpx.Response,
    answer: ImageResponse,
) -> ImageRefusalError:
    return ImageRefusalError(
        message=message,
        model=model,
        llm_provider=llm_provider,
        response=_answered_as(400, raw_response),
        provider_specific_fields=fields,
        billed_answer=None if reason in _REFUSALS_GOOGLE_DOES_NOT_CHARGE else answer,
    )


def empty_image_response_error(
    response: Mapping[str, object],
    model: str,
    llm_provider: str,
    raw_response: httpx.Response,
    answer: ImageResponse,
) -> ImageRefusalError | UnservedImageError:
    result: Final = _parsed(response)
    feedback: Final = result.prompt_feedback
    if feedback is not None and feedback.block_reason:
        return _refusal_error(
            feedback.block_reason,
            f"Gemini blocked the image prompt, blockReason={feedback.block_reason}",
            {"block_reason": feedback.block_reason, "block_reason_message": feedback.block_reason_message},
            model,
            llm_provider,
            raw_response,
            answer,
        )
    candidates: Final = result.candidates or ()
    refusal: Final = next((candidate for candidate in candidates if _is_refusal(candidate)), None)
    if refusal is not None and refusal.finish_reason is not None:
        return _refusal_error(
            refusal.finish_reason,
            f"Gemini refused to return an image, finishReason={refusal.finish_reason}",
            {"finish_reason": refusal.finish_reason, "finish_message": refusal.finish_message},
            model,
            llm_provider,
            raw_response,
            answer,
        )
    finish_reasons: Final = ",".join(candidate.finish_reason or "none" for candidate in candidates) or "none"
    return UnservedImageError(
        message=f"Gemini returned no image, finishReason={finish_reasons}",
        model=model,
        llm_provider=llm_provider,
        response=_answered_without_model_output(raw_response, result),
        billed_answer=answer,
    )

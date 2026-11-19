"""The litellm ``Usage`` a finished call's response reports, whatever shape the provider sent it in."""

from collections.abc import Mapping
from typing import Final, cast

from pydantic import BaseModel
from typing_extensions import TypedDict, assert_never

import litellm
from litellm import verbose_logger
from litellm.litellm_core_utils.usage.usage_object_transformation import (
    InteractionsUsageObjectTransformation,
    TranscriptionUsageObjectTransformation,
)
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.decisions import (
    DecisionsResponse,
    DecisionsUsage,
    OpenAIDecisionCacheWriteTokensDetails,
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
)
from litellm.types.extraction import ExtractionResponse, ExtractionUsage
from litellm.types.llms.openai import (
    HttpxBinaryResponseContent,
    ResponseAPIUsage,
)
from litellm.types.utils import (
    CacheCreationTokenDetails,
    PromptTokensDetailsWrapper,
    TranscriptionUsageDurationObject,
    TranscriptionUsageTokensObject,
    Usage,
)


def get_usage_object(
    completion_response: object,
) -> Usage | None:
    usage_obj: Final = cast(
        Usage | ResponseAPIUsage | dict | BaseModel,
        (
            completion_response.get("usage")
            if isinstance(completion_response, dict)
            else getattr(completion_response, "get", lambda x: None)("usage")
        ),
    )

    if isinstance(completion_response, (DecisionsResponse, OpenAIDecisionResponse)):
        return None if completion_response.usage is None else decisions_usage(completion_response.usage)
    if isinstance(completion_response, ExtractionResponse):
        return extraction_usage(completion_response.usage)
    if isinstance(completion_response, HttpxBinaryResponseContent):
        return completion_response.usage
    if usage_obj is None:
        return None
    if isinstance(usage_obj, Usage):
        return usage_obj
    elif isinstance(usage_obj, dict) and litellm.AnthropicConfig.is_anthropic_usage_object(usage_obj):
        return litellm.AnthropicConfig().calculate_usage(usage_object=usage_obj, reasoning_content=None)
    elif (
        usage_obj is not None
        and (isinstance(usage_obj, dict) or isinstance(usage_obj, ResponseAPIUsage))
        and ResponseAPILoggingUtils.is_response_api_usage(usage_obj)
    ):
        return ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(usage_obj)
    elif TranscriptionUsageObjectTransformation.is_transcription_usage_object(usage_obj):
        return TranscriptionUsageObjectTransformation.transform_transcription_usage_object(
            cast(
                TranscriptionUsageDurationObject | TranscriptionUsageTokensObject,
                usage_obj,
            )
        )
    elif isinstance(usage_obj, dict) and InteractionsUsageObjectTransformation.is_interactions_usage_object(usage_obj):
        return InteractionsUsageObjectTransformation.transform_interactions_usage_object(usage_obj)
    elif isinstance(usage_obj, dict):
        return Usage(**usage_obj)
    elif isinstance(usage_obj, BaseModel):
        return usage_from_model(usage_obj)
    else:
        verbose_logger.debug("Unknown usage object type: %s, usage_obj: %s", type(usage_obj), usage_obj)
        return None


class _DumpedUsageFields(TypedDict, total=False, extra_items=object):
    pass


def _dumped_usage_fields(usage_model: BaseModel) -> Mapping[str, object]:
    return usage_model.model_dump()


def usage_from_model(usage_model: BaseModel) -> Usage:
    return Usage(**_DumpedUsageFields(**_dumped_usage_fields(usage_model)))


def _decisions_cache_write_lifetimes(
    details: OpenAIDecisionCacheWriteTokensDetails | None,
) -> CacheCreationTokenDetails | None:
    if details is None:
        return None
    return CacheCreationTokenDetails(
        ephemeral_30m_input_tokens=details.ttl_30m_tokens, ephemeral_12h_input_tokens=details.ttl_12h_tokens
    )


def _decisions_prompt_tokens_details(usage: DecisionsUsage | OpenAIDecisionUsage) -> PromptTokensDetailsWrapper:
    match usage:
        case DecisionsUsage():
            return PromptTokensDetailsWrapper(
                cached_tokens=usage.cached_tokens, cache_write_tokens=usage.cache_write_tokens
            )
        case OpenAIDecisionUsage():
            return PromptTokensDetailsWrapper(
                cached_tokens=usage.input_tokens_details.cached_tokens,
                cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
                cache_creation_token_details=_decisions_cache_write_lifetimes(
                    usage.input_tokens_details.cache_write_tokens_details
                ),
            )
        case _:
            assert_never(usage)


def decisions_usage(usage: DecisionsUsage | OpenAIDecisionUsage) -> Usage:
    return Usage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
        prompt_tokens_details=_decisions_prompt_tokens_details(usage),
    )


def extraction_usage(usage: ExtractionUsage) -> Usage:
    return Usage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
    )


def is_known_usage_object(usage_obj: object) -> bool:
    """Returns True if the usage obj is a known Usage type"""
    return (
        isinstance(usage_obj, litellm.Usage)
        or isinstance(usage_obj, ResponseAPIUsage)
        or isinstance(usage_obj, DecisionsUsage)
        or isinstance(usage_obj, OpenAIDecisionUsage)
        or isinstance(usage_obj, ExtractionUsage)
        or TranscriptionUsageObjectTransformation.is_transcription_usage_object(usage_obj)
    )

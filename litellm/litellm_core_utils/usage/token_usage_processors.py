"""Usage summed over the events of a realtime or Responses WebSocket session."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, cast

from pydantic import BaseModel, TypeAdapter

from litellm.litellm_core_utils.audio_utils.utils import normalized_audio_duration_seconds
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.llms.base import CachedTokensDetails, LiteLLMBaseModel
from litellm.types.llms.openai import (
    OpenAIRealtimeStreamList,
    OpenAIRealtimeStreamResponseBaseObject,
)
from litellm.types.utils import (
    LiteLLMRealtimeStreamLoggingObject,
    PromptTokensDetailsWrapper,
    Usage,
)

_TRANSLATION_CLOSED_EVENT_TYPE: Final = "session.closed"


def translation_session_seconds(results: OpenAIRealtimeStreamList) -> tuple[float, float]:
    usage_events: Final = tuple(
        usage
        for result in results
        if (event := as_str_mapping(result)) is not None
        and event.get("type") == _TRANSLATION_CLOSED_EVENT_TYPE
        and (usage := as_str_mapping(event.get("usage"))) is not None
    )
    input_seconds: Final = sum(
        seconds
        for usage in usage_events
        if (seconds := normalized_audio_duration_seconds(usage.get("input_seconds"))) is not None
    )
    output_seconds: Final = sum(
        output_duration
        for usage in usage_events
        if (output_duration := normalized_audio_duration_seconds(usage.get("output_seconds"))) is not None
    )
    return float(input_seconds), float(output_seconds)


_ATTRIBUTE_VALUE: Final = TypeAdapter(object)


def _attribute_value(obj: object, name: str) -> object:
    return _ATTRIBUTE_VALUE.validate_python(getattr(obj, name))


def _summable_prompt_token_fields(prompt_tokens_details: BaseModel) -> list[str]:
    field_names: Final = list(type(prompt_tokens_details).model_fields)
    if getattr(prompt_tokens_details, "cache_write_tokens", None) is None:
        return field_names
    return [attr for attr in field_names if attr != "cache_creation_tokens"]


def _combine_cached_tokens_details(
    current: CachedTokensDetails | None, new: CachedTokensDetails
) -> CachedTokensDetails:
    def _sum_optional(current_value: int | None, new_value: int | None) -> int | None:
        if current_value is None and new_value is None:
            return None
        return (current_value or 0) + (new_value or 0)

    return CachedTokensDetails(
        text_tokens=_sum_optional(current.text_tokens if current is not None else None, new.text_tokens),
        audio_tokens=_sum_optional(current.audio_tokens if current is not None else None, new.audio_tokens),
        image_tokens=_sum_optional(current.image_tokens if current is not None else None, new.image_tokens),
    )


def _combine_prompt_tokens_details(
    current: PromptTokensDetailsWrapper | None, new: PromptTokensDetailsWrapper
) -> PromptTokensDetailsWrapper:
    base: Final = current if current is not None else PromptTokensDetailsWrapper()
    base_values: Final = MappingProxyType(
        {attr: getattr(base, attr) for attr in type(base).model_fields if hasattr(base, attr)}
    )
    summed: Final = MappingProxyType(
        {
            attr: (getattr(base, attr, 0) or 0) + (getattr(new, attr) or 0)
            for attr in _summable_prompt_token_fields(new)
            if hasattr(new, attr) and isinstance(getattr(new, attr) or 0, (int, float))
        }
    )
    new_cached_tokens_details: Final = getattr(new, "cached_tokens_details", None)
    cached_tokens_details: Final = (
        _combine_cached_tokens_details(getattr(base, "cached_tokens_details", None), new_cached_tokens_details)
        if isinstance(new_cached_tokens_details, CachedTokensDetails)
        else getattr(base, "cached_tokens_details", None)
    )
    return PromptTokensDetailsWrapper(
        **MappingProxyType({**base_values, **summed, "cached_tokens_details": cached_tokens_details})
    )


class BaseTokenUsageProcessor:
    @staticmethod
    def combine_usage_objects(usage_objects: list[Usage]) -> Usage:
        """
        Combine multiple Usage objects into a single Usage object, checking model keys for nested values.
        """
        from litellm.types.utils import (
            CompletionTokensDetailsWrapper,
            Usage,
        )

        combined: Final = Usage()

        # Sum basic token counts
        for usage in usage_objects:
            # Handle direct attributes by checking what exists in the model
            for attr in dir(usage):
                if not attr.startswith("_") and not callable(_attribute_value(usage, attr)):
                    current_val = getattr(combined, attr, 0)
                    new_val = getattr(usage, attr, 0)
                    if (
                        new_val is not None
                        and isinstance(new_val, (int, float))
                        and isinstance(current_val, (int, float))
                    ):
                        setattr(combined, attr, current_val + new_val)
            if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                combined.prompt_tokens_details = _combine_prompt_tokens_details(
                    getattr(combined, "prompt_tokens_details", None), usage.prompt_tokens_details
                )

            # Handle nested completion_tokens_details
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                if not hasattr(combined, "completion_tokens_details") or not combined.completion_tokens_details:
                    combined.completion_tokens_details = CompletionTokensDetailsWrapper()

                # Check what keys exist in the model's completion_tokens_details
                # Access model_fields on the class, not the instance, to avoid Pydantic 2.11+ deprecation warnings
                for attr in type(usage.completion_tokens_details).model_fields:
                    if not attr.startswith("_") and not callable(
                        _attribute_value(usage.completion_tokens_details, attr)
                    ):
                        current_val = getattr(combined.completion_tokens_details, attr, 0) or 0
                        new_val = getattr(usage.completion_tokens_details, attr, 0) or 0
                        if isinstance(new_val, (int, float)):
                            setattr(
                                combined.completion_tokens_details,
                                attr,
                                current_val + new_val,
                            )

        return combined


class RealtimeAPITokenUsageProcessor(BaseTokenUsageProcessor):
    @staticmethod
    def collect_usage_from_realtime_stream_results(
        results: OpenAIRealtimeStreamList,
    ) -> list[Usage]:
        """
        Collect usage from realtime stream results
        """
        response_done_events: Final[list[OpenAIRealtimeStreamResponseBaseObject]] = cast(
            list[OpenAIRealtimeStreamResponseBaseObject],
            [result for result in results if result["type"] == "response.done"],
        )
        usage_objects: Final[list[Usage]] = []
        for result in response_done_events:
            usage_object = ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(
                result["response"].get("usage", {})
            )
            usage_objects.append(usage_object)
        return usage_objects

    @staticmethod
    def collect_and_combine_usage_from_realtime_stream_results(
        results: OpenAIRealtimeStreamList,
    ) -> Usage:
        """
        Collect and combine usage from realtime stream results
        """
        collected_usage_objects = RealtimeAPITokenUsageProcessor.collect_usage_from_realtime_stream_results(results)
        combined_usage_object: Final = RealtimeAPITokenUsageProcessor.combine_usage_objects(collected_usage_objects)
        return combined_usage_object

    @staticmethod
    def create_logging_realtime_object(
        usage: Usage, results: OpenAIRealtimeStreamList
    ) -> LiteLLMRealtimeStreamLoggingObject:
        translated_input_seconds, _ = translation_session_seconds(results)
        return LiteLLMRealtimeStreamLoggingObject(
            usage=usage.model_copy(update={"seconds": translated_input_seconds})
            if translated_input_seconds > 0
            else usage,
            results=results,
        )


_RESPONSES_WS_BILLABLE_EVENT_TYPES: Final = frozenset({"response.completed", "response.incomplete", "response.failed"})


class _ResponsesWsEventResponse(LiteLLMBaseModel):
    id: str | None = None
    usage: Mapping[str, object] | None = None
    service_tier: str | None = None


class _ResponsesWsEvent(LiteLLMBaseModel):
    type: str = ""
    response: _ResponsesWsEventResponse | None = None


def _billable_responses_ws_events(
    results: Sequence[Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], _ResponsesWsEventResponse], ...]:
    return tuple(
        (result, event.response)
        for result in results
        if (event := _ResponsesWsEvent.model_validate(result)).type in _RESPONSES_WS_BILLABLE_EVENT_TYPES
        and event.response is not None
        and event.response.usage is not None
    )


class ResponsesWebSocketTokenUsageProcessor(BaseTokenUsageProcessor):
    @staticmethod
    def collect_usage_from_responses_ws_results(
        results: Sequence[Mapping[str, object]],
    ) -> tuple[Usage, ...]:
        return tuple(
            ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(response.usage)
            for _, response in _billable_responses_ws_events(results)
            if response.usage is not None
        )

    @staticmethod
    def partition_results_by_service_tier(
        results: Sequence[Mapping[str, object]],
    ) -> Mapping[str | None, tuple[Mapping[str, object], ...]]:
        billable: Final = _billable_responses_ws_events(results)
        tiers: Final = dict.fromkeys(response.service_tier for _, response in billable)
        return MappingProxyType(
            {tier: tuple(result for result, response in billable if response.service_tier == tier) for tier in tiers}
        )

    @staticmethod
    def collect_and_combine_usage_from_responses_ws_results(
        results: Sequence[Mapping[str, object]],
    ) -> Usage:
        collected_usage_objects: Final = ResponsesWebSocketTokenUsageProcessor.collect_usage_from_responses_ws_results(
            results
        )
        return ResponsesWebSocketTokenUsageProcessor.combine_usage_objects(list(collected_usage_objects))

    @staticmethod
    def response_id_from_responses_ws_results(results: Sequence[Mapping[str, object]]) -> str | None:
        response_ids: Final = tuple(
            response.id for _, response in _billable_responses_ws_events(results) if response.id
        )
        return response_ids[-1] if response_ids else None

    @staticmethod
    def create_logging_object(
        results: Sequence[Mapping[str, object]], litellm_model_name: str | None = None
    ) -> LiteLLMRealtimeStreamLoggingObject:
        tier_partition: Final = ResponsesWebSocketTokenUsageProcessor.partition_results_by_service_tier(results)
        logging_object: Final = LiteLLMRealtimeStreamLoggingObject(
            id=ResponsesWebSocketTokenUsageProcessor.response_id_from_responses_ws_results(results),
            usage=ResponsesWebSocketTokenUsageProcessor.collect_and_combine_usage_from_responses_ws_results(results),
            results=cast(OpenAIRealtimeStreamList, list(results)),  # cast-ok: raw WS frames, unvalidated field
            service_tier=next(iter(tier_partition)) if len(tier_partition) == 1 else None,
        )
        if litellm_model_name is not None:
            logging_object.hidden_params = {"litellm_model_name": litellm_model_name}
        return logging_object

from typing import Any, Mapping, Optional, Sequence, Union

from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    TranscriptionUsageDurationObject,
    TranscriptionUsageTokensObject,
    Usage,
)


class TranscriptionUsageObjectTransformation:
    @staticmethod
    def is_transcription_usage_object(
        usage_object: Any,
    ) -> bool:
        return isinstance(usage_object, TranscriptionUsageDurationObject) or isinstance(
            usage_object, TranscriptionUsageTokensObject
        )

    @staticmethod
    def transform_transcription_usage_object(
        usage_object: Union[TranscriptionUsageDurationObject, TranscriptionUsageTokensObject],
    ) -> Optional[Usage]:
        if isinstance(usage_object, TranscriptionUsageDurationObject):
            return None
        elif isinstance(usage_object, TranscriptionUsageTokensObject):
            return Usage(
                prompt_tokens=usage_object.input_tokens,
                completion_tokens=usage_object.output_tokens,
                total_tokens=usage_object.total_tokens,
                prompt_tokens_details=PromptTokensDetailsWrapper(
                    text_tokens=usage_object.input_token_details.text_tokens,
                    audio_tokens=usage_object.input_token_details.audio_tokens,
                ),
            )
        return None


_INTERACTIONS_MODALITY_FIELDS: Mapping[str, str] = {
    "text": "text_tokens",
    "audio": "audio_tokens",
    "image": "image_tokens",
    "video": "video_tokens",
    "document": "text_tokens",
}


def _modality_field(entry: Mapping[str, Any]) -> str | None:
    return _INTERACTIONS_MODALITY_FIELDS.get(str(entry.get("modality", "")).lower())


def _token_count(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _modality_token_sums(entries: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    fields = {field for entry in entries if (field := _modality_field(entry)) is not None}
    return {
        field: sum(_token_count(entry.get("tokens")) for entry in entries if _modality_field(entry) == field)
        for field in fields
    }


def _google_search_query_count(usage_object: Mapping[str, Any]) -> int:
    return sum(
        _token_count(entry.get("count"))
        for entry in tuple(usage_object.get("grounding_tool_count") or ())
        if isinstance(entry, Mapping) and entry.get("type") == "google_search"
    )


def _subtract_cached_from_input(
    input_sums: Mapping[str, int],
    cached_sums: Mapping[str, int],
    total_cached_tokens: int,
) -> Mapping[str, int]:
    if cached_sums:
        return {field: max(0, tokens - cached_sums.get(field, 0)) for field, tokens in input_sums.items()}
    if total_cached_tokens and "text_tokens" in input_sums:
        return {
            **input_sums,
            "text_tokens": max(0, input_sums["text_tokens"] - total_cached_tokens),
        }
    return input_sums


class InteractionsUsageObjectTransformation:
    """
    Maps the Google Interactions API usage block (total_input_tokens,
    output_tokens_by_modality, ...) into LiteLLM's chat-format ``Usage`` so the
    generic cost calculator and spend tracking can bill it.
    """

    @staticmethod
    def is_interactions_usage_object(usage_object: Any) -> bool:
        if not isinstance(usage_object, dict):
            return False
        if "prompt_tokens" in usage_object or "input_tokens" in usage_object:
            return False
        return "total_input_tokens" in usage_object or "total_output_tokens" in usage_object

    @staticmethod
    def transform_interactions_usage_object(usage_object: Mapping[str, Any]) -> Usage:
        input_entries = tuple(usage_object.get("input_tokens_by_modality") or ()) + tuple(
            usage_object.get("tool_use_tokens_by_modality") or ()
        )
        cached_sums = _modality_token_sums(tuple(usage_object.get("cached_tokens_by_modality") or ()))
        output_sums = _modality_token_sums(tuple(usage_object.get("output_tokens_by_modality") or ()))

        total_cached_tokens = _token_count(usage_object.get("total_cached_tokens"))
        input_sums = _subtract_cached_from_input(
            input_sums=_modality_token_sums(input_entries),
            cached_sums=cached_sums,
            total_cached_tokens=total_cached_tokens,
        )

        reasoning_tokens = _token_count(usage_object.get("total_reasoning_tokens")) or _token_count(
            usage_object.get("total_thought_tokens")
        )
        prompt_tokens = _token_count(usage_object.get("total_input_tokens")) + _token_count(
            usage_object.get("total_tool_use_tokens")
        )
        completion_tokens = _token_count(usage_object.get("total_output_tokens")) + reasoning_tokens
        total_tokens = _token_count(usage_object.get("total_tokens")) or (prompt_tokens + completion_tokens)

        web_search_requests = _google_search_query_count(usage_object)
        prompt_tokens_details = (
            PromptTokensDetailsWrapper(
                cached_tokens=total_cached_tokens or None,
                web_search_requests=web_search_requests or None,
                **input_sums,
            )
            if input_sums or total_cached_tokens or web_search_requests
            else None
        )
        completion_tokens_details = (
            CompletionTokensDetailsWrapper(
                reasoning_tokens=reasoning_tokens or None,
                **output_sums,
            )
            if output_sums or reasoning_tokens
            else None
        )

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=prompt_tokens_details,
            completion_tokens_details=completion_tokens_details,
            cache_read_input_tokens=total_cached_tokens or None,
        )

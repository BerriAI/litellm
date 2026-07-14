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


def _coerce_optional_token_count(value: object) -> int:
    return value if isinstance(value, int) else 0


def _tokens_by_modality(modality_entries: object) -> Mapping[str, int]:
    if not isinstance(modality_entries, Sequence):
        return {}
    return {
        str(entry.get("modality") or "").lower(): _coerce_optional_token_count(entry.get("tokens"))
        for entry in modality_entries
        if isinstance(entry, Mapping)
    }


class InteractionsUsageObjectTransformation:
    @staticmethod
    def is_interactions_usage_dict(usage_object: Mapping[str, Any]) -> bool:
        return (
            "total_input_tokens" in usage_object
            or "total_output_tokens" in usage_object
            or "input_tokens_by_modality" in usage_object
            or "output_tokens_by_modality" in usage_object
        )

    @staticmethod
    def transform_interactions_usage_to_chat_usage(
        usage_object: "Mapping[str, Any] | None",
    ) -> Usage:
        usage = usage_object or {}
        input_tokens_by_modality = _tokens_by_modality(usage.get("input_tokens_by_modality"))
        output_tokens_by_modality = _tokens_by_modality(usage.get("output_tokens_by_modality"))
        prompt_tokens = _coerce_optional_token_count(usage.get("total_input_tokens"))
        reasoning_tokens = _coerce_optional_token_count(
            usage.get("total_thought_tokens") or usage.get("total_reasoning_tokens")
        )
        completion_tokens = _coerce_optional_token_count(usage.get("total_output_tokens")) + reasoning_tokens
        cached_tokens = _coerce_optional_token_count(usage.get("total_cached_tokens"))
        total_tokens = _coerce_optional_token_count(usage.get("total_tokens")) or (prompt_tokens + completion_tokens)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=cached_tokens,
                text_tokens=input_tokens_by_modality.get("text"),
                image_tokens=input_tokens_by_modality.get("image"),
                audio_tokens=input_tokens_by_modality.get("audio"),
                video_tokens=input_tokens_by_modality.get("video"),
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=reasoning_tokens,
                text_tokens=output_tokens_by_modality.get("text"),
                image_tokens=output_tokens_by_modality.get("image"),
                audio_tokens=output_tokens_by_modality.get("audio"),
                video_tokens=output_tokens_by_modality.get("video"),
            ),
        )

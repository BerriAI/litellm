from typing import Any, Optional, Union

from litellm.types.utils import (
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
            input_token_details = usage_object.input_token_details
            prompt_tokens_details = (
                PromptTokensDetailsWrapper(
                    text_tokens=input_token_details.text_tokens,
                    audio_tokens=input_token_details.audio_tokens,
                )
                if input_token_details is not None
                else None
            )
            return Usage(
                prompt_tokens=usage_object.input_tokens,
                completion_tokens=usage_object.output_tokens,
                total_tokens=usage_object.total_tokens,
                prompt_tokens_details=prompt_tokens_details,
            )
        return None

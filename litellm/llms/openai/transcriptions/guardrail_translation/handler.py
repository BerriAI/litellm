"""
OpenAI Audio Transcription Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's audio transcription endpoint.
The handler processes the output transcribed text (input is audio, so no text to guardrail).
"""

from typing import TYPE_CHECKING, Any

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.utils import TranscriptionResponse


class OpenAIAudioTranscriptionHandler(BaseTranslation):
    """
    Handler for processing OpenAI audio transcription responses with guardrails.

    This class provides methods to:
    1. Process output transcription text (post-call hook)

    Note: Input processing is not applicable since the input is an audio file,
    not text. Only the transcribed text output is processed.
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
    ) -> Any:
        """
        Process input - not applicable for audio transcription.

        The input is an audio file, not text, so there's nothing to apply
        guardrails to. This method returns the data unchanged.

        Args:
            data: Request data dictionary containing audio file
            guardrail_to_apply: The guardrail instance (unused)

        Returns:
            Unmodified data (audio files don't need text guardrails)
        """
        verbose_proxy_logger.debug(
            "OpenAI Audio Transcription: Input processing not applicable "
            "(input is audio file, not text)"
        )
        return data

    async def process_output_response(
        self,
        response: "TranscriptionResponse",
        guardrail_to_apply: "CustomGuardrail",
    ) -> Any:
        """
        Process output transcription by applying guardrails to transcribed text.

        Args:
            response: Transcription response object containing transcribed text
            guardrail_to_apply: The guardrail instance to apply

        Returns:
            Modified response with guardrails applied to transcribed text
        """
        if not hasattr(response, "text") or response.text is None:
            verbose_proxy_logger.debug(
                "OpenAI Audio Transcription: No text in response to process"
            )
            return response

        if isinstance(response.text, str):
            original_text = response.text
            guardrailed_text = await guardrail_to_apply.apply_guardrail(
                text=original_text
            )
            response.text = guardrailed_text

            verbose_proxy_logger.debug(
                "OpenAI Audio Transcription: Applied guardrail to transcribed text. "
                "Original length: %d, New length: %d",
                len(original_text),
                len(guardrailed_text),
            )
        else:
            verbose_proxy_logger.debug(
                "OpenAI Audio Transcription: Unexpected text type: %s. Expected string.",
                type(response.text),
            )

        return response

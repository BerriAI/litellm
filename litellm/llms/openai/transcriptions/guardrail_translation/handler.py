"""
OpenAI Audio Transcription Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's audio transcription endpoint.
The handler processes the output transcribed text (input is audio, so no text to guardrail).
"""

from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import GenericGuardrailAPIInputs

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
        litellm_logging_obj: Optional[Any] = None,
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
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> Any:
        """
        Process output transcription by applying guardrails to transcribed text.

        Args:
            response: Transcription response object containing transcribed text
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

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
            # Create a request_data dict with response info and user API key metadata
            request_data: dict = {"response": response}

            # Add user API key metadata with prefixed keys
            user_metadata = self.transform_user_api_key_dict_to_metadata(
                user_api_key_dict
            )
            if user_metadata:
                request_data["litellm_metadata"] = user_metadata

            inputs = GenericGuardrailAPIInputs(texts=[original_text])
            # Include model information from the response if available
            if hasattr(response, "model") and response.model:
                inputs["model"] = response.model
            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
                logging_obj=litellm_logging_obj,
            )
            guardrailed_texts = guardrailed_inputs.get("texts", [])
            response.text = guardrailed_texts[0] if guardrailed_texts else original_text

            verbose_proxy_logger.debug(
                "OpenAI Audio Transcription: Applied guardrail to transcribed text. "
                "Original length: %d, New length: %d",
                len(original_text),
                len(response.text),
            )
        else:
            verbose_proxy_logger.debug(
                "OpenAI Audio Transcription: Unexpected text type: %s. Expected string.",
                type(response.text),
            )

        return response

"""
OpenAI Image Generation Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's image generation endpoint.
The handler processes the 'prompt' parameter for guardrails.
"""

from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.utils import ImageResponse


class OpenAIImageGenerationHandler(BaseTranslation):
    """
    Handler for processing OpenAI image generation requests with guardrails.

    This class provides methods to:
    1. Process input prompt (pre-call hook)
    2. Process output response (post-call hook) - typically not needed for images

    The handler specifically processes the 'prompt' parameter which contains
    the text description for image generation.
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
    ) -> Any:
        """
        Process input prompt by applying guardrails to text content.

        Args:
            data: Request data dictionary containing 'prompt' parameter
            guardrail_to_apply: The guardrail instance to apply

        Returns:
            Modified data with guardrails applied to prompt
        """
        prompt = data.get("prompt")
        if prompt is None:
            verbose_proxy_logger.debug(
                "OpenAI Image Generation: No prompt found in request data"
            )
            return data

        # Apply guardrail to the prompt
        if isinstance(prompt, str):
            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs={"texts": [prompt]},
                request_data=data,
                input_type="request",
                logging_obj=litellm_logging_obj,
            )
            guardrailed_texts = guardrailed_inputs.get("texts", [])
            data["prompt"] = guardrailed_texts[0] if guardrailed_texts else prompt

            verbose_proxy_logger.debug(
                "OpenAI Image Generation: Applied guardrail to prompt. "
                "Original length: %d, New length: %d",
                len(prompt),
                len(data["prompt"]),
            )
        else:
            verbose_proxy_logger.debug(
                "OpenAI Image Generation: Unexpected prompt type: %s. Expected string.",
                type(prompt),
            )

        return data

    async def process_output_response(
        self,
        response: "ImageResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> Any:
        """
        Process output response - typically not needed for image generation.

        Image responses don't contain text to apply guardrails to, so this
        method returns the response unchanged. This is provided for completeness
        and can be overridden if needed for custom image metadata processing.

        Args:
            response: Image generation response object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object (unused)
            user_api_key_dict: User API key metadata (unused)

        Returns:
            Unmodified response (images don't need text guardrails)
        """
        verbose_proxy_logger.debug(
            "OpenAI Image Generation: Output processing not needed for image responses"
        )
        return response

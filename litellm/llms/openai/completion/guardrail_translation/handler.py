"""
OpenAI Text Completion Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's text completion endpoint.
The handler processes the 'prompt' parameter for guardrails.
"""

from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.utils import TextCompletionResponse


class OpenAITextCompletionHandler(BaseTranslation):
    """
    Handler for processing OpenAI text completion requests with guardrails.

    This class provides methods to:
    1. Process input prompt (pre-call hook)
    2. Process output response (post-call hook)

    The handler specifically processes the 'prompt' parameter which can be:
    - A single string
    - A list of strings (for batch completions)
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
                "OpenAI Text Completion: No prompt found in request data"
            )
            return data

        if isinstance(prompt, str):
            # Single string prompt
            inputs = GenericGuardrailAPIInputs(texts=[prompt])
            # Include model information if available
            model = data.get("model")
            if model:
                inputs["model"] = model
            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=data,
                input_type="request",
                logging_obj=litellm_logging_obj,
            )
            guardrailed_texts = guardrailed_inputs.get("texts", [])
            data["prompt"] = guardrailed_texts[0] if guardrailed_texts else prompt

            verbose_proxy_logger.debug(
                "OpenAI Text Completion: Applied guardrail to string prompt. "
                "Original length: %d, New length: %d",
                len(prompt),
                len(data["prompt"]),
            )

        elif isinstance(prompt, list):
            # List of string prompts (batch completion)
            texts_to_check = []
            text_indices = []  # Track which prompts are strings

            for idx, p in enumerate(prompt):
                if isinstance(p, str):
                    texts_to_check.append(p)
                    text_indices.append(idx)

            if texts_to_check:
                inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
                # Include model information if available
                model = data.get("model")
                if model:
                    inputs["model"] = model
                guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                    inputs=inputs,
                    request_data=data,
                    input_type="request",
                    logging_obj=litellm_logging_obj,
                )
                guardrailed_texts = guardrailed_inputs.get("texts", [])

                # Replace guardrailed texts back
                for guardrail_idx, prompt_idx in enumerate(text_indices):
                    if guardrail_idx < len(guardrailed_texts):
                        data["prompt"][prompt_idx] = guardrailed_texts[guardrail_idx]
                        verbose_proxy_logger.debug(
                            "OpenAI Text Completion: Applied guardrail to prompt[%d]. "
                            "Original length: %d, New length: %d",
                            prompt_idx,
                            len(texts_to_check[guardrail_idx]),
                            len(guardrailed_texts[guardrail_idx]),
                        )

        else:
            verbose_proxy_logger.warning(
                "OpenAI Text Completion: Unexpected prompt type: %s. Expected string or list.",
                type(prompt),
            )

        return data

    async def process_output_response(
        self,
        response: "TextCompletionResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> Any:
        """
        Process output response by applying guardrails to completion text.

        Args:
            response: Text completion response object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified response with guardrails applied to completion text
        """
        if not hasattr(response, "choices") or not response.choices:
            verbose_proxy_logger.debug(
                "OpenAI Text Completion: No choices in response to process"
            )
            return response

        # Collect all texts to check
        texts_to_check = []
        choice_indices = []

        for idx, choice in enumerate(response.choices):
            if hasattr(choice, "text") and isinstance(choice.text, str):
                texts_to_check.append(choice.text)
                choice_indices.append(idx)

        # Apply guardrails in batch
        if texts_to_check:
            # Create a request_data dict with response info and user API key metadata
            request_data: dict = {"response": response}

            # Add user API key metadata with prefixed keys
            user_metadata = self.transform_user_api_key_dict_to_metadata(
                user_api_key_dict
            )
            if user_metadata:
                request_data["litellm_metadata"] = user_metadata

            inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
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

            # Apply guardrailed texts back to choices
            for guardrail_idx, choice_idx in enumerate(choice_indices):
                if guardrail_idx < len(guardrailed_texts):
                    original_text = response.choices[choice_idx].text
                    response.choices[choice_idx].text = guardrailed_texts[guardrail_idx]

                    verbose_proxy_logger.debug(
                        "OpenAI Text Completion: Applied guardrail to choice[%d] text. "
                        "Original length: %d, New length: %d",
                        choice_idx,
                        len(original_text),
                        len(guardrailed_texts[guardrail_idx]),
                    )

        return response

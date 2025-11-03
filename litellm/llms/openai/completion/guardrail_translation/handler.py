"""
OpenAI Text Completion Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's text completion endpoint.
The handler processes the 'prompt' parameter for guardrails.
"""

from typing import TYPE_CHECKING, Any

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation

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
            guardrailed_prompt = await guardrail_to_apply.apply_guardrail(text=prompt)
            data["prompt"] = guardrailed_prompt

            verbose_proxy_logger.debug(
                "OpenAI Text Completion: Applied guardrail to string prompt. "
                "Original length: %d, New length: %d",
                len(prompt),
                len(guardrailed_prompt),
            )

        elif isinstance(prompt, list):
            # List of string prompts (batch completion)
            guardrailed_prompts = []
            for idx, p in enumerate(prompt):
                if isinstance(p, str):
                    guardrailed_p = await guardrail_to_apply.apply_guardrail(text=p)
                    guardrailed_prompts.append(guardrailed_p)
                    verbose_proxy_logger.debug(
                        "OpenAI Text Completion: Applied guardrail to prompt[%d]. "
                        "Original length: %d, New length: %d",
                        idx,
                        len(p),
                        len(guardrailed_p),
                    )
                else:
                    # For non-string items (e.g., token lists), keep unchanged
                    guardrailed_prompts.append(p)
                    verbose_proxy_logger.debug(
                        "OpenAI Text Completion: Skipping guardrail for prompt[%d] "
                        "(not a string, type: %s)",
                        idx,
                        type(p),
                    )

            data["prompt"] = guardrailed_prompts

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
    ) -> Any:
        """
        Process output response by applying guardrails to completion text.

        Args:
            response: Text completion response object
            guardrail_to_apply: The guardrail instance to apply

        Returns:
            Modified response with guardrails applied to completion text
        """
        if not hasattr(response, "choices") or not response.choices:
            verbose_proxy_logger.debug(
                "OpenAI Text Completion: No choices in response to process"
            )
            return response

        # Apply guardrails to each choice's text
        for idx, choice in enumerate(response.choices):
            if hasattr(choice, "text") and isinstance(choice.text, str):
                original_text = choice.text
                guardrailed_text = await guardrail_to_apply.apply_guardrail(
                    text=original_text
                )
                choice.text = guardrailed_text

                verbose_proxy_logger.debug(
                    "OpenAI Text Completion: Applied guardrail to choice[%d] text. "
                    "Original length: %d, New length: %d",
                    idx,
                    len(original_text),
                    len(guardrailed_text),
                )

        return response

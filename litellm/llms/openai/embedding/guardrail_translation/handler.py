"""
OpenAI Embedding Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's embedding endpoint.
The handler processes the 'input' parameter for guardrails.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    DISABLE_MAX_GUARDRAIL_INPUT_CHECK,
    MAX_GUARDRAIL_INPUT_LENGTH,
)
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.utils import EmbeddingResponse


class OpenAIEmbeddingHandler(BaseTranslation):
    """
    Handler for processing OpenAI embedding requests with guardrails.

    This class provides methods to:
    1. Process input text (pre-call hook)
    2. Process output response (post-call hook) - typically not needed for embeddings

    The handler specifically processes the 'input' parameter which can be:
    - A single string
    - A list of strings
    - A list of token arrays (integers)
    """

    # Maximum input length for guardrail processing (safety limit)

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
    ) -> Any:
        """
        Process input text by applying guardrails to text content.

        Args:
            data: Request data dictionary containing 'input' parameter
            guardrail_to_apply: The guardrail instance to apply

        Returns:
            Modified data with guardrails applied to input
        """
        input_data = data.get("input")
        if input_data is None:
            verbose_proxy_logger.debug(
                "OpenAI Embedding: No input found in request data"
            )
            return data

        # Apply guardrail to the input
        if isinstance(input_data, str):
            # Single string input
            if (
                len(input_data) > MAX_GUARDRAIL_INPUT_LENGTH
                and not DISABLE_MAX_GUARDRAIL_INPUT_CHECK
            ):
                verbose_proxy_logger.debug(
                    "OpenAI Embedding: Skipping guardrail for input length %d (exceeds limit of %d)",
                    len(input_data),
                    MAX_GUARDRAIL_INPUT_LENGTH,
                )
            else:
                guardrailed_input = await guardrail_to_apply.apply_guardrail(
                    text=input_data
                )
                data["input"] = guardrailed_input

                verbose_proxy_logger.debug(
                    "OpenAI Embedding: Applied guardrail to single string input. "
                    "Original length: %d, New length: %d",
                    len(input_data),
                    len(guardrailed_input),
                )
        elif isinstance(input_data, list):
            # List of strings or token arrays
            if all(isinstance(item, str) for item in input_data):
                # List of strings
                skipped_count = 0
                tasks = []
                task_indices = []  # Track which indices have tasks

                for idx, text in enumerate(input_data):
                    if (
                        len(text) > MAX_GUARDRAIL_INPUT_LENGTH
                        and not DISABLE_MAX_GUARDRAIL_INPUT_CHECK
                    ):
                        verbose_proxy_logger.debug(
                            "OpenAI Embedding: Skipping guardrail for input length %d (exceeds limit of %d)",
                            len(text),
                            MAX_GUARDRAIL_INPUT_LENGTH,
                        )
                        skipped_count += 1
                    else:
                        tasks.append(guardrail_to_apply.apply_guardrail(text=text))
                        task_indices.append(idx)

                # Gather guardrailed responses
                responses = await asyncio.gather(*tasks)

                # Reconstruct the list in correct order
                guardrailed_inputs = []
                response_idx = 0
                for idx, text in enumerate(input_data):
                    if idx in task_indices:
                        # Use guardrailed response
                        guardrailed_inputs.append(responses[response_idx])
                        response_idx += 1
                    else:
                        # Keep original text (skipped)
                        guardrailed_inputs.append(text)

                data["input"] = guardrailed_inputs

                if skipped_count > 0:
                    verbose_proxy_logger.debug(
                        "OpenAI Embedding: Applied guardrail to %d string inputs, skipped %d due to length",
                        len(input_data) - skipped_count,
                        skipped_count,
                    )
                else:
                    verbose_proxy_logger.debug(
                        "OpenAI Embedding: Applied guardrail to %d string inputs",
                        len(input_data),
                    )
            elif all(isinstance(item, list) for item in input_data):
                # List of token arrays - don't apply guardrails to tokens
                verbose_proxy_logger.debug(
                    "OpenAI Embedding: Input is token arrays. Skipping guardrail application."
                )
            else:
                verbose_proxy_logger.debug(
                    "OpenAI Embedding: Mixed or unexpected input list type. Skipping guardrail."
                )
        else:
            verbose_proxy_logger.debug(
                "OpenAI Embedding: Unexpected input type: %s. Expected string or list.",
                type(input_data),
            )

        return data

    async def process_output_response(
        self,
        response: "EmbeddingResponse",
        guardrail_to_apply: "CustomGuardrail",
    ) -> Any:
        """
        Process output response - typically not needed for embeddings.

        Embedding responses contain vector embeddings, not text, so this
        method returns the response unchanged. This is provided for completeness
        and can be overridden if needed for custom embedding metadata processing.

        Args:
            response: Embedding response object
            guardrail_to_apply: The guardrail instance to apply

        Returns:
            Unmodified response (embeddings don't contain text to apply guardrails to)
        """
        verbose_proxy_logger.debug(
            "OpenAI Embedding: Output processing not needed for embedding responses"
        )
        return response

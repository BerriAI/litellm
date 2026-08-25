"""
OpenAI Embeddings Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's embeddings endpoint.
The handler processes the 'input' parameter for guardrails.
"""

from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.utils import EmbeddingResponse


class OpenAIEmbeddingsHandler(BaseTranslation):
    """
    Handler for processing OpenAI embeddings requests with guardrails.

    This class provides methods to:
    1. Process input text (pre-call hook)
    2. Process output response (post-call hook) - embeddings don't typically need output guardrails

    The handler specifically processes the 'input' parameter which can be:
    - A single string
    - A list of strings (for batch embeddings)
    - A list of integers (token IDs - not processed by guardrails)
    - A list of lists of integers (batch token IDs - not processed by guardrails)
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Any | None = None,
    ) -> Any:
        """
        Process input text by applying guardrails to text content.

        Args:
            data: Request data dictionary containing 'input' parameter
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object

        Returns:
            Modified data with guardrails applied to input
        """
        input_data: Final = data.get("input")
        if input_data is None:
            verbose_proxy_logger.debug("OpenAI Embeddings: No input found in request data")
            return data

        if isinstance(input_data, str):
            data = await self._process_string_input(data, input_data, guardrail_to_apply, litellm_logging_obj)
        elif isinstance(input_data, list):
            data = await self._process_list_input(data, input_data, guardrail_to_apply, litellm_logging_obj)
        else:
            verbose_proxy_logger.warning(
                "OpenAI Embeddings: Unexpected input type: %s. Expected string or list.",
                type(input_data),
            )

        return data

    async def _process_string_input(
        self,
        data: dict,
        input_data: str,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Any | None,
    ) -> dict:
        """Process a single string input through the guardrail."""
        inputs: Final = GenericGuardrailAPIInputs(texts=[input_data])
        if model := data.get("model"):
            inputs["model"] = model

        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )

        if guardrailed_texts := guardrailed_inputs.get("texts"):
            data["input"] = guardrailed_texts[0]
            verbose_proxy_logger.debug(
                "OpenAI Embeddings: Applied guardrail to string input. Original length: %d, New length: %d",
                len(input_data),
                len(data["input"]),
            )

        return data

    async def _process_list_input(
        self,
        data: dict,
        input_data: list[str | int | list[int]],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Any | None,
    ) -> dict:
        """Process a list input through the guardrail (if it contains strings)."""
        if len(input_data) == 0:
            return data

        first_item: Final = input_data[0]

        # Skip non-text inputs (token IDs)
        if isinstance(first_item, (int, list)):
            verbose_proxy_logger.debug("OpenAI Embeddings: Input is token IDs, skipping guardrail processing")
            return data

        if not isinstance(first_item, str):
            verbose_proxy_logger.warning(
                "OpenAI Embeddings: Unexpected input list item type: %s",
                type(first_item),
            )
            return data

        # List of strings - apply guardrail
        inputs: Final = GenericGuardrailAPIInputs(texts=input_data)
        if model := data.get("model"):
            inputs["model"] = model

        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )

        if guardrailed_texts := guardrailed_inputs.get("texts"):
            data["input"] = guardrailed_texts
            verbose_proxy_logger.debug(
                "OpenAI Embeddings: Applied guardrail to %d inputs",
                len(guardrailed_texts),
            )

        return data

    async def process_output_response(
        self,
        response: "EmbeddingResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Any | None = None,
        user_api_key_dict: Any | None = None,
        request_data: dict | None = None,
    ) -> Any:
        """
        Process output response - embeddings responses contain vectors, not text.

        For embeddings, the output is numerical vectors, so there's typically
        no text content to apply guardrails to. This method is a no-op but
        is included for interface consistency.

        Args:
            response: Embedding response object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata

        Returns:
            Unmodified response (embeddings don't have text output to guard)
        """
        verbose_proxy_logger.debug(
            "OpenAI Embeddings: Output response processing skipped - embeddings contain vectors, not text"
        )
        return response

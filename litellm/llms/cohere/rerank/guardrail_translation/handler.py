"""
Cohere Rerank Handler for Unified Guardrails

This module provides guardrail translation support for the rerank endpoint.
The handler processes only the 'query' parameter for guardrails.
"""

from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.rerank import RerankResponse


class CohereRerankHandler(BaseTranslation):
    """
    Handler for processing rerank requests with guardrails.

    This class provides methods to:
    1. Process input query (pre-call hook)
    2. Process output response (post-call hook) - not applicable for rerank

    The handler specifically processes:
    - The 'query' parameter (string)

    Note: Documents are not processed by guardrails as they are the corpus
    being searched, not user input.
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
    ) -> Any:
        """
        Process input query by applying guardrails.

        Args:
            data: Request data dictionary containing 'query'
            guardrail_to_apply: The guardrail instance to apply

        Returns:
            Modified data with guardrails applied to query only
        """
        # Process query only
        query = data.get("query")
        if query is not None and isinstance(query, str):
            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs={"texts": [query]},
                request_data=data,
                input_type="request",
                logging_obj=litellm_logging_obj,
            )
            guardrailed_texts = guardrailed_inputs.get("texts", [])
            data["query"] = guardrailed_texts[0] if guardrailed_texts else query

            verbose_proxy_logger.debug(
                "Rerank: Applied guardrail to query. "
                "Original length: %d, New length: %d",
                len(query),
                len(data["query"]),
            )
        else:
            verbose_proxy_logger.debug(
                "Rerank: No query to process or query is not a string"
            )

        return data

    async def process_output_response(
        self,
        response: "RerankResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> Any:
        """
        Process output response - not applicable for rerank.

        Rerank responses contain relevance scores and indices, not text,
        so there's nothing to apply guardrails to. This method returns
        the response unchanged.

        Args:
            response: Rerank response object with rankings
            guardrail_to_apply: The guardrail instance (unused)
            litellm_logging_obj: Optional logging object (unused)
            user_api_key_dict: User API key metadata (unused)

        Returns:
            Unmodified response (rankings don't need text guardrails)
        """
        verbose_proxy_logger.debug(
            "Rerank: Output processing not applicable "
            "(output contains relevance scores, not text)"
        )
        return response

"""
Cohere Rerank Handler for Unified Guardrails

This module provides guardrail translation support for the rerank endpoint.
The handler processes only the 'query' parameter for guardrails.
"""

from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import GenericGuardrailAPIInputs

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
    - The 'instruction' parameter (string), when present

    Note: Documents are not processed by guardrails as they are the corpus
    being searched, not user input.
    """

    # User-controlled free-text fields that reach the model and must be
    # scanned. 'instruction' is folded into the prompt by instruction-aware
    # rerankers (e.g. hosted vLLM / Qwen3-Reranker), so it is as sensitive as
    # 'query'; omitting it would let a caller smuggle content past guardrails.
    _SCANNED_FIELDS = ("query", "instruction")

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
    ) -> Any:
        """
        Process input text fields ('query' and 'instruction') by applying
        guardrails and writing the sanitized values back.

        Args:
            data: Request data dictionary containing 'query' and optionally
                'instruction'
            guardrail_to_apply: The guardrail instance to apply

        Returns:
            Modified data with guardrails applied to query/instruction only
        """
        # Collect every scannable text field in a stable order so the
        # guardrailed results can be written back to the right key by index.
        fields_to_scan = [(key, data[key]) for key in self._SCANNED_FIELDS if isinstance(data.get(key), str)]
        if not fields_to_scan:
            verbose_proxy_logger.debug("Rerank: No query/instruction to process or not strings")
            return data

        inputs = GenericGuardrailAPIInputs(texts=[value for _, value in fields_to_scan])
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

        for idx, (key, original) in enumerate(fields_to_scan):
            # Defensive: only write back when the guardrail returned a value for
            # this index; otherwise keep the original (never forward unscanned).
            if idx < len(guardrailed_texts):
                data[key] = guardrailed_texts[idx]
                verbose_proxy_logger.debug(
                    "Rerank: Applied guardrail to %s. Original length: %d, New length: %d",
                    key,
                    len(original),
                    len(data[key]),
                )

        return data

    async def process_output_response(
        self,
        response: "RerankResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[dict] = None,
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
            "Rerank: Output processing not applicable (output contains relevance scores, not text)"
        )
        return response

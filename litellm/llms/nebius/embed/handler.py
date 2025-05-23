"""
Calling logic for Nebius embeddings
"""

from typing import Optional

from litellm.llms.openai_like.embedding.handler import OpenAILikeEmbeddingHandler
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.utils import ModelResponse


def nebius_embeddings(
    model: str,
    input: list,
    api_key: str,
    api_base: str,
    logging_obj: Logging = None,
    model_response: Optional[ModelResponse] = None,
    optional_params: Optional[dict] = None,
    litellm_params: Optional[dict] = None,
    logger_fn=None,
) -> ModelResponse:
    """
    Handle embeddings for Nebius models
    """
    handler = OpenAILikeEmbeddingHandler()
    return handler.embedding(
        model=model,
        input=input,
        api_key=api_key,
        api_base=api_base,
        logging_obj=logging_obj,
        model_response=model_response,
        optional_params=optional_params or {},
        timeout=litellm_params.get("timeout", 600) if litellm_params else 600,
        client=litellm_params.get("client") if litellm_params else None,
        aembedding=litellm_params.get("aembedding") if litellm_params else None,
        custom_endpoint=True,
        headers=litellm_params.get("extra_headers") if litellm_params else None,
    ) 
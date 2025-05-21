"""
Calling logic for Nebius embeddings
"""

from typing import Optional

from litellm.llms.openai_like.embedding.handler import openai_like_embeddings
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.llms.nebius import NebiusAIEndpoint
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
    return openai_like_embeddings(
        model=model,
        input=input,
        api_key=api_key,
        api_base=api_base,
        logging_obj=logging_obj,
        model_response=model_response,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        endpoint_type=NebiusAIEndpoint.EMBEDDINGS.value.strip("/"),
    ) 
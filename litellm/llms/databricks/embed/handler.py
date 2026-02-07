"""
Calling logic for Databricks embeddings
"""

import os
from typing import Optional

from litellm.utils import EmbeddingResponse

from ...openai_like.embedding.handler import OpenAILikeEmbeddingHandler
from ..common_utils import DatabricksBase


class DatabricksEmbeddingHandler(OpenAILikeEmbeddingHandler, DatabricksBase):
    def embedding(
        self,
        model: str,
        input: list,
        timeout: float,
        logging_obj,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        model_response: Optional[EmbeddingResponse] = None,
        client=None,
        aembedding=None,
        custom_endpoint: Optional[bool] = None,
        headers: Optional[dict] = None,
    ) -> EmbeddingResponse:
        # Check for custom user agent in optional_params or environment
        # This allows partners building on LiteLLM to set their own telemetry
        # Use pop() to remove these keys so they don't get sent to the API
        custom_user_agent = (
            optional_params.pop("user_agent", None)
            or optional_params.pop("databricks_user_agent", None)
            or os.getenv("LITELLM_USER_AGENT")
            or os.getenv("DATABRICKS_USER_AGENT")
        )

        api_base, headers = self.databricks_validate_environment(
            api_base=api_base,
            api_key=api_key,
            endpoint_type="embeddings",
            custom_endpoint=custom_endpoint,
            headers=headers,
            custom_user_agent=custom_user_agent,
        )
        return super().embedding(
            model=model,
            input=input,
            timeout=timeout,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            optional_params=optional_params,
            model_response=model_response,
            client=client,
            aembedding=aembedding,
            custom_endpoint=True,
            headers=headers,
        )

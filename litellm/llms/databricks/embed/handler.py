"""
Calling logic for Databricks embeddings
"""

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
        api_base, headers = self.databricks_validate_environment(
            api_base=api_base,
            api_key=api_key,
            endpoint_type="embeddings",
            custom_endpoint=custom_endpoint,
            headers=headers,
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

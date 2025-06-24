from typing import Dict, Optional, Tuple

import litellm
from litellm.llms.base_llm.vector_store.transformation import (
    BaseVectorStoreTransformation,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils.secrets import get_secret_str


class OpenAIVectorStoreTransformation(BaseVectorStoreTransformation):
    ASSISTANTS_HEADER_KEY = "OpenAI-Beta"
    ASSISTANTS_HEADER_VALUE = "assistants=v2"

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )

        #########################################################
        # Ensure OpenAI Assistants header is includes
        #########################################################
        if self.ASSISTANTS_HEADER_KEY not in headers:
            headers.update(
                {
                    self.ASSISTANTS_HEADER_KEY: self.ASSISTANTS_HEADER_VALUE,
            }
        )

        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the Base endpoint for OpenAI Vector Stores API
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("OPENAI_BASE_URL")
            or get_secret_str("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        return f"{api_base}/vector_stores"
    

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str,
        api_base: str,
    ) -> Tuple[str, Dict]:
        url = f"{api_base}/{vector_store_id}/search"
        request_body = {
            "query": query,
            "filters": [],
            "max_num_results": 10,
            "ranking_options": {
                "relevance": {
                    "alpha": 0.5,
                    "beta": 0.5,
                }
            },
            "rewrite_query": False,
        }
        return url, request_body
    

    def transform_search_vector_store_response(self):
        pass
    

    
    
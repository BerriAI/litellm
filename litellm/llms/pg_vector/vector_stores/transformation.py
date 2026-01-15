from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from litellm.llms.openai.vector_stores.transformation import OpenAIVectorStoreConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import VectorStoreSearchOptionalRequestParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

class PGVectorStoreConfig(OpenAIVectorStoreConfig):
    """
    PG Vector Store configuration that inherits from OpenAI since it's OpenAI-compatible.

    LiteLLM Provides an OpenAI Compatible Server to connect to PG Vector.

    https://github.com/BerriAI/litellm-pgvector

    You just need to connect litellm proxy to this deployed server.
    
    Requires:
    - api_base: The base URL for the PG vector service
    - api_key: API key for authentication with the PG vector service
    """

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """
        Validate environment and set headers for PG vector service authentication
        """
        litellm_params = litellm_params or GenericLiteLLMParams()
        
        # Get API key from various sources
        api_key = (
            litellm_params.api_key
            or get_secret_str("PG_VECTOR_API_KEY")
        )
        
        if not api_key:
            raise ValueError("PG Vector API key is required. Set PG_VECTOR_API_KEY environment variable or pass api_key in litellm_params.")
        
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for PG vector service endpoints
        """
        # Get API base from various sources
        api_base = (
            api_base
            or get_secret_str("PG_VECTOR_API_BASE")
        )
        
        if not api_base:
            raise ValueError("PG Vector API base URL is required. Set PG_VECTOR_API_BASE environment variable or pass api_base in litellm_params.")

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        return f"{api_base}/v1/vector_stores" 
    

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> Tuple[str, Dict]:
        url  = f"{api_base}/{vector_store_id}/search"
        _, request_body = super().transform_search_vector_store_request(
            vector_store_id=vector_store_id,
            query=query,
            vector_store_search_optional_params=vector_store_search_optional_params,
            api_base=api_base,
            litellm_logging_obj=litellm_logging_obj,
            litellm_params=litellm_params,
        )
        return url, request_body
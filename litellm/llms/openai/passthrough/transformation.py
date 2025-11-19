from typing import TYPE_CHECKING, List, Optional, Tuple

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from httpx import URL


class OpenAIPassthroughConfig(BasePassthroughConfig):
    """
    OpenAI passthrough configuration for router models.
    
    This enables OpenAI passthrough endpoints to work with router models
    defined in config.yaml, providing load balancing and fallback support.
    """

    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        """
        Check if the request is a streaming request.
        
        For OpenAI, streaming is determined by the 'stream' parameter in the request body.
        """
        return request_data.get("stream", False)

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        """
        Get the complete URL for the OpenAI request.
        
        Args:
            api_base: The base URL for OpenAI API (e.g., "https://api.openai.com")
            api_key: The OpenAI API key
            model: The model name
            endpoint: The endpoint path (e.g., "/v1/chat/completions")
            request_query_params: Query parameters for the request
            litellm_params: Additional LiteLLM parameters
            
        Returns:
            Tuple of (complete_url, base_target_url)
        """
        # Use provided api_base or default to OpenAI's API
        base_target_url = self.get_api_base(api_base) or "https://api.openai.com"
        
        # Ensure endpoint starts with /
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
            
        # Ensure /v1 is in the path for OpenAI API
        if "/v1/" not in endpoint and not endpoint.startswith("/v1/"):
            # Insert /v1 at the beginning if not present
            endpoint = "/v1" + endpoint
        
        complete_url = self.format_url(endpoint, base_target_url, request_query_params or {})
        
        return complete_url, base_target_url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate the environment and add necessary headers for OpenAI.
        
        For passthrough with router models, the API key is managed by the router,
        so we just return the headers as-is.
        """
        # The router will handle API key authentication
        return headers

    @staticmethod
    def get_api_base(
        api_base: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get the OpenAI API base URL.
        
        Returns the provided api_base or falls back to OPENAI_API_BASE environment variable.
        """
        return api_base or get_secret_str("OPENAI_API_BASE")

    @staticmethod
    def get_api_key(
        api_key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get the OpenAI API key.
        
        Returns the provided api_key or falls back to OPENAI_API_KEY environment variable.
        """
        return api_key or get_secret_str("OPENAI_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        """
        Get the base model name.
        
        For OpenAI, the model name is used as-is without transformation.
        """
        return model

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        """
        Get list of available models.
        
        For passthrough, we rely on the router configuration rather than
        querying the OpenAI API directly.
        """
        return []


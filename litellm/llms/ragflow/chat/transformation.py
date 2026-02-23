"""
RAGFlow provider configuration for OpenAI-compatible API.

RAGFlow provides OpenAI-compatible APIs with unique path structures:
- Chat endpoint: /api/v1/chats_openai/{chat_id}/chat/completions
- Agent endpoint: /api/v1/agents_openai/{agent_id}/chat/completions

Model name format:
- Chat: ragflow/chat/{chat_id}/{model_name}
- Agent: ragflow/agent/{agent_id}/{model_name}
"""

from typing import List, Optional, Tuple

import litellm
from litellm.llms.openai.openai import OpenAIConfig
from litellm.secret_managers.main import get_secret, get_secret_str
from litellm.types.llms.openai import AllMessageValues


class RAGFlowConfig(OpenAIConfig):
    """
    Configuration for RAGFlow OpenAI-compatible API.
    
    Handles both chat and agent endpoints by parsing the model name format:
    - ragflow/chat/{chat_id}/{model_name} for chat endpoints
    - ragflow/agent/{agent_id}/{model_name} for agent endpoints
    """

    def _parse_ragflow_model(self, model: str) -> Tuple[str, str, str]:
        """
        Parse RAGFlow model name format: ragflow/{endpoint_type}/{id}/{model_name}
        
        Args:
            model: Model name in format ragflow/chat/{chat_id}/{model} or ragflow/agent/{agent_id}/{model}
            
        Returns:
            Tuple of (endpoint_type, id, model_name)
            
        Raises:
            ValueError: If model format is invalid
        """
        parts = model.split("/")
        if len(parts) < 4:
            raise ValueError(
                f"Invalid RAGFlow model format: {model}. "
                f"Expected format: ragflow/chat/{{chat_id}}/{{model}} or ragflow/agent/{{agent_id}}/{{model}}"
            )
        
        if parts[0] != "ragflow":
            raise ValueError(
                f"Invalid RAGFlow model format: {model}. Must start with 'ragflow/'"
            )
        
        endpoint_type = parts[1]
        if endpoint_type not in ["chat", "agent"]:
            raise ValueError(
                f"Invalid RAGFlow endpoint type: {endpoint_type}. Must be 'chat' or 'agent'"
            )
        
        entity_id = parts[2]
        model_name = "/".join(parts[3:])  # Handle model names that might contain slashes
        
        return endpoint_type, entity_id, model_name

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for the RAGFlow API call.
        
        Constructs URL based on endpoint type:
        - Chat: /api/v1/chats_openai/{chat_id}/chat/completions
        - Agent: /api/v1/agents_openai/{agent_id}/chat/completions
        
        Args:
            api_base: Base API URL (e.g., http://ragflow-server:port or http://ragflow-server:port/v1)
            api_key: API key (not used in URL construction)
            model: Model name in format ragflow/{endpoint_type}/{id}/{model}
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters (may contain api_base)
            stream: Whether streaming is enabled
            
        Returns:
            Complete URL for the API call
        """
        # Get api_base from multiple sources: input param, litellm_params, environment, or global litellm setting
        if litellm_params and hasattr(litellm_params, 'api_base') and litellm_params.api_base:
            api_base = api_base or litellm_params.api_base
        
        api_base = (
            api_base
            or litellm.api_base
            or get_secret("RAGFLOW_API_BASE")
            or get_secret_str("RAGFLOW_API_BASE")
        )
        
        if api_base is None:
            raise ValueError("api_base is required for RAGFlow provider. Set it via api_base parameter, RAGFLOW_API_BASE environment variable, or litellm.api_base")
        
        # Parse model name to extract endpoint type and ID
        endpoint_type, entity_id, _ = self._parse_ragflow_model(model)
        
        # Remove trailing slash from api_base if present
        api_base = api_base.rstrip("/")
        
        # Strip /v1 or /api/v1 from api_base if present, since we'll add the full path
        # Check /api/v1 first because /api/v1 ends with /v1
        if api_base.endswith("/api/v1"):
            api_base = api_base[:-7]  # Remove /api/v1
        elif api_base.endswith("/v1"):
            api_base = api_base[:-3]  # Remove /v1
        
        # Construct the RAGFlow-specific path
        if endpoint_type == "chat":
            path = f"/api/v1/chats_openai/{entity_id}/chat/completions"
        else:  # agent
            path = f"/api/v1/agents_openai/{entity_id}/chat/completions"
        
        # Ensure path starts with /
        if not path.startswith("/"):
            path = "/" + path
        
        return f"{api_base}{path}"

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Get OpenAI-compatible provider information for RAGFlow.
        
        Args:
            model: Model name (will be parsed to extract actual model name)
            api_base: Base API URL (from input params)
            api_key: API key (from input params)
            custom_llm_provider: Custom LLM provider name
            
        Returns:
            Tuple of (api_base, api_key, custom_llm_provider)
        """
        # Parse model to extract the actual model name
        # The model name will be stored in litellm_params for use in requests
        _, _, actual_model = self._parse_ragflow_model(model)
        
        # Get api_base from multiple sources: input param, environment, or global litellm setting
        dynamic_api_base = (
            api_base
            or litellm.api_base
            or get_secret("RAGFLOW_API_BASE")
            or get_secret_str("RAGFLOW_API_BASE")
        )
        
        # Get api_key from multiple sources: input param, environment, or global litellm setting
        dynamic_api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("RAGFLOW_API_KEY")
        )
        
        return dynamic_api_base, dynamic_api_key, custom_llm_provider

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
        Validate environment and set up headers for RAGFlow API.
        
        Args:
            headers: Request headers
            model: Model name
            messages: Chat messages
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters (may contain api_key)
            api_key: API key (from input params)
            api_base: Base API URL
            
        Returns:
            Updated headers dictionary
        """
        # Use api_key from litellm_params if available, otherwise fall back to other sources
        if litellm_params and hasattr(litellm_params, 'api_key') and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        
        # Get api_key from multiple sources: input param, litellm_params, environment, or global litellm setting
        api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("RAGFLOW_API_KEY")
        )
        
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Ensure Content-Type is set to application/json
        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        
        # Parse model to extract actual model name and store it
        # The actual model name should be used in the request body
        try:
            _, _, actual_model = self._parse_ragflow_model(model)
            # Store the actual model name in litellm_params for use in transform_request
            litellm_params["_ragflow_actual_model"] = actual_model
        except ValueError:
            # If parsing fails, use the original model name
            pass
        
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform request for RAGFlow API.
        
        Uses the actual model name extracted from the RAGFlow model format.
        
        Args:
            model: Model name in RAGFlow format
            messages: Chat messages
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters (may contain _ragflow_actual_model)
            headers: Request headers
            
        Returns:
            Transformed request dictionary
        """
        # Get the actual model name from litellm_params if available
        actual_model = litellm_params.get("_ragflow_actual_model")
        if actual_model is None:
            # Fallback: try to parse the model name
            try:
                _, _, actual_model = self._parse_ragflow_model(model)
            except ValueError:
                # If parsing fails, use the original model name
                actual_model = model
        
        # Use parent's transform_request with the actual model name
        return super().transform_request(
            actual_model, messages, optional_params, litellm_params, headers
        )


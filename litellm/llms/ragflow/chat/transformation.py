"""
RAGFlow provider configuration for OpenAI-compatible API.

RAGFlow provides OpenAI-compatible APIs with unique path structures:
- Chat endpoint: /api/v1/chats_openai/{chat_id}/chat/completions
- Agent endpoint: /api/v1/agents_openai/{agent_id}/chat/completions

Model name format:
- Chat: ragflow/chat/{chat_id}/{model_name}
- Agent: ragflow/agent/{agent_id}/{model_name}

Dynamic ID support:
You can pass chat_id or agent_id dynamically via extra_body:
- extra_body={"ragflow_chat_id": "1234"} for chat endpoints
- extra_body={"ragflow_agent_id": "5678"} for agent endpoints

This allows using model names like "ragflow-chat-gpt4" or "ragflow/chat/gpt-4o-mini"
without hardcoding the chat_id/agent_id in the model name.
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
    
    Supports dynamic chat_id and agent_id via extra_body:
    - extra_body={"ragflow_chat_id": "1234"} for chat endpoints
    - extra_body={"ragflow_agent_id": "5678"} for agent endpoints
    
    This allows using generic model names in the proxy config and passing
    the chat_id/agent_id dynamically per request.
    """

    def _parse_ragflow_model(self, model: str, allow_missing_id: bool = False) -> Tuple[str, Optional[str], str]:
        """
        Parse RAGFlow model name format: ragflow/{endpoint_type}/{id}/{model_name}
        
        Args:
            model: Model name in format ragflow/chat/{chat_id}/{model} or ragflow/agent/{agent_id}/{model}
                  Also supports ragflow/chat/{model} or ragflow/agent/{model} when allow_missing_id=True
            allow_missing_id: If True, allows models without ID (for dynamic ID support)
            
        Returns:
            Tuple of (endpoint_type, id, model_name) where id can be None if allow_missing_id=True
            
        Raises:
            ValueError: If model format is invalid
        """
        parts = model.split("/")
        
        # Handle model names that start with "ragflow-{type}-" (e.g., "ragflow-chat-gpt4")
        if not model.startswith("ragflow/") and model.startswith("ragflow-"):
            # Convert "ragflow-chat-gpt4" to "ragflow/chat/gpt4"
            model = model.replace("ragflow-", "ragflow/", 1).replace("-", "/", 1)
            parts = model.split("/")
        
        if len(parts) < 2:
            raise ValueError(
                f"Invalid RAGFlow model format: {model}. "
                f"Expected format: ragflow/chat/{{chat_id}}/{{model}} or ragflow/agent/{{agent_id}}/{{model}}"
            )
        
        if parts[0] != "ragflow":
            raise ValueError(
                f"Invalid RAGFlow model format: {model}. Must start with 'ragflow/' or 'ragflow-'"
            )
        
        endpoint_type = parts[1]
        if endpoint_type not in ["chat", "agent"]:
            raise ValueError(
                f"Invalid RAGFlow endpoint type: {endpoint_type}. Must be 'chat' or 'agent'"
            )
        
        # If we have 3+ parts, check if part[2] looks like an ID or model name
        if len(parts) >= 3:
            # If allow_missing_id and we only have 3 parts, treat part[2] as model name
            if allow_missing_id and len(parts) == 3:
                entity_id = None
                model_name = parts[2]
            else:
                # Assume part[2] is the entity_id
                entity_id = parts[2]
                model_name = "/".join(parts[3:]) if len(parts) > 3 else "gpt-4o-mini"  # Default model if not specified
        else:
            # Only 2 parts: ragflow/{type}
            if allow_missing_id:
                entity_id = None
                model_name = "gpt-4o-mini"  # Default model
            else:
                raise ValueError(
                    f"Invalid RAGFlow model format: {model}. "
                    f"Expected format: ragflow/chat/{{chat_id}}/{{model}} or ragflow/agent/{{agent_id}}/{{model}}"
                )
        
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
        
        Supports dynamic chat_id and agent_id via extra_body:
        - extra_body={"ragflow_chat_id": "1234"} for chat endpoints
        - extra_body={"ragflow_agent_id": "5678"} for agent endpoints
        
        Args:
            api_base: Base API URL (e.g., http://ragflow-server:port or http://ragflow-server:port/v1)
            api_key: API key (not used in URL construction)
            model: Model name in format ragflow/{endpoint_type}/{id}/{model} or ragflow/{endpoint_type}/{model}
            optional_params: Optional parameters (may contain extra_body with ragflow_chat_id or ragflow_agent_id)
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
        
        # Parse model name to extract endpoint type and ID (allow missing ID for dynamic support)
        endpoint_type, entity_id, _ = self._parse_ragflow_model(model, allow_missing_id=True)
        
        # Check for dynamic chat_id or agent_id in extra_body
        extra_body = optional_params.get("extra_body", {})
        if isinstance(extra_body, dict):
            if endpoint_type == "chat":
                # Check for ragflow_chat_id in extra_body
                dynamic_chat_id = extra_body.get("ragflow_chat_id")
                if dynamic_chat_id:
                    entity_id = str(dynamic_chat_id)
            elif endpoint_type == "agent":
                # Check for ragflow_agent_id in extra_body
                dynamic_agent_id = extra_body.get("ragflow_agent_id")
                if dynamic_agent_id:
                    entity_id = str(dynamic_agent_id)
        
        # If entity_id is still None, raise an error
        if entity_id is None:
            if endpoint_type == "chat":
                raise ValueError(
                    f"chat_id is required for RAGFlow chat endpoint. "
                    f"Either include it in the model name (ragflow/chat/{{chat_id}}/{{model}}) "
                    f"or pass it via extra_body={{'ragflow_chat_id': 'your_chat_id'}}"
                )
            else:
                raise ValueError(
                    f"agent_id is required for RAGFlow agent endpoint. "
                    f"Either include it in the model name (ragflow/agent/{{agent_id}}/{{model}}) "
                    f"or pass it via extra_body={{'ragflow_agent_id': 'your_agent_id'}}"
                )
        
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
        _, _, actual_model = self._parse_ragflow_model(model, allow_missing_id=True)
        
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
            _, _, actual_model = self._parse_ragflow_model(model, allow_missing_id=True)
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
                _, _, actual_model = self._parse_ragflow_model(model, allow_missing_id=True)
            except ValueError:
                # If parsing fails, use the original model name
                actual_model = model
        
        # Use parent's transform_request with the actual model name
        return super().transform_request(
            actual_model, messages, optional_params, litellm_params, headers
        )


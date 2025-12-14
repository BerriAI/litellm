"""
Transformation for Azure Foundry Agent Service API.

Azure Foundry Agent Service provides an Assistants-like API for running agents.
This follows the OpenAI Assistants pattern: create thread -> add messages -> create/poll run.

Model format: azure_ai/agents/<agent_id>

API Base format: https://<AIFoundryResourceName>.services.ai.azure.com/api/projects/<ProjectName>

Authentication: Uses Azure AD Bearer tokens (not API keys)
  Get token via: az account get-access-token --resource 'https://ai.azure.com'

The API uses these endpoints:
- POST /threads - Create a thread
- POST /threads/{thread_id}/messages - Add message to thread
- POST /threads/{thread_id}/runs - Create a run
- GET /threads/{thread_id}/runs/{run_id} - Poll run status
- GET /threads/{thread_id}/messages - List messages in thread

See: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any
    HTTPHandler = Any
    AsyncHTTPHandler = Any


class AzureAIAgentsError(BaseLLMException):
    """Exception class for Azure AI Agent Service API errors."""

    pass


class AzureAIAgentsConfig(BaseConfig):
    """
    Configuration for Azure AI Agent Service API.

    Azure AI Agent Service is a fully managed service for building AI agents
    that can understand natural language and perform tasks.
    
    Model format: azure_ai/agents/<agent_id>
    
    The flow is:
    1. Create a thread
    2. Add user messages to the thread
    3. Create and poll a run
    4. Retrieve the assistant's response messages
    """

    # Default API version for Azure Foundry Agent Service
    # GA version: 2025-05-01, Preview: 2025-05-15-preview
    # See: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart
    DEFAULT_API_VERSION = "2025-05-01"
    
    # Polling configuration
    MAX_POLL_ATTEMPTS = 60
    POLL_INTERVAL_SECONDS = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def is_azure_ai_agents_route(model: str) -> bool:
        """
        Check if the model is an Azure AI Agents route.
        
        Model format: azure_ai/agents/<agent_id>
        """
        return "agents/" in model

    @staticmethod
    def get_agent_id_from_model(model: str) -> str:
        """
        Extract agent ID from the model string.
        
        Model format: azure_ai/agents/<agent_id> -> <agent_id>
        or: agents/<agent_id> -> <agent_id>
        """
        if "agents/" in model:
            # Split on "agents/" and take the part after it
            parts = model.split("agents/", 1)
            if len(parts) == 2:
                return parts[1]
        return model

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get Azure AI Agent Service API base and key from params or environment.

        Returns:
            Tuple of (api_base, api_key)
        """
        from litellm.secret_managers.main import get_secret_str

        api_base = api_base or get_secret_str("AZURE_AI_API_BASE")
        api_key = api_key or get_secret_str("AZURE_AI_API_KEY")

        return api_base, api_key

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Azure Agents supports minimal OpenAI params since it's an agent runtime.
        """
        return ["stream"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Azure Agents params.
        """
        return optional_params

    def _get_api_version(self, optional_params: dict) -> str:
        """Get API version from optional params or use default."""
        return optional_params.get("api_version", self.DEFAULT_API_VERSION)

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
        Get the base URL for Azure AI Agent Service.
        
        The actual endpoint will vary based on the operation:
        - /openai/threads for creating threads
        - /openai/threads/{thread_id}/messages for adding messages
        - /openai/threads/{thread_id}/runs for creating runs
        
        This returns the base URL that will be modified for each operation.
        """
        if api_base is None:
            raise ValueError(
                "api_base is required for Azure AI Agents. Set it via AZURE_AI_API_BASE env var or api_base parameter."
            )

        # Remove trailing slash if present
        api_base = api_base.rstrip("/")

        # Return base URL - actual endpoints will be constructed during request
        return api_base

    def _get_agent_id(self, model: str, optional_params: dict) -> str:
        """
        Get the agent ID from model or optional_params.

        model format: "azure_ai/agents/<agent_id>" or "agents/<agent_id>" or just "<agent_id>"
        """
        agent_id = optional_params.get("agent_id") or optional_params.get("assistant_id")
        if agent_id:
            return agent_id

        # Extract from model name using the static method
        return self.get_agent_id_from_model(model)

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request for Azure Agents.
        
        This stores the necessary data for the multi-step agent flow.
        The actual API calls happen in the custom handler.
        """
        agent_id = self._get_agent_id(model, optional_params)

        # Convert messages to a format we can use
        converted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle content that might be a list
            if isinstance(content, list):
                content = convert_content_list_to_str(msg)

            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content)

            converted_messages.append({"role": role, "content": content})

        payload: Dict[str, Any] = {
            "agent_id": agent_id,
            "messages": converted_messages,
            "api_version": self._get_api_version(optional_params),
        }

        # Pass through thread_id if provided (for continuing conversations)
        if "thread_id" in optional_params:
            payload["thread_id"] = optional_params["thread_id"]

        # Pass through any additional instructions
        if "instructions" in optional_params:
            payload["instructions"] = optional_params["instructions"]

        verbose_logger.debug(f"Azure AI Agents request payload: {payload}")
        return payload

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
        Validate and set up environment for Azure Foundry Agents requests.
        
        Azure Foundry Agents uses Bearer token authentication with Azure AD tokens.
        Get token via: az account get-access-token --resource 'https://ai.azure.com'
        
        See: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart
        """
        headers["Content-Type"] = "application/json"

        # Azure Foundry Agents uses Bearer token authentication
        # The api_key here is expected to be an Azure AD token
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return AzureAIAgentsError(status_code=status_code, message=error_message)

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        Azure Agents uses polling, so we fake stream by returning the final response.
        """
        return True

    @property
    def has_custom_stream_wrapper(self) -> bool:
        """Azure Agents doesn't have native streaming - uses fake stream."""
        return False

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        """
        Azure Agents does not use a stream param in request body.
        """
        return False

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the Azure Agents response to LiteLLM ModelResponse format.
        """
        # This is not used since we have a custom handler
        return model_response

    @staticmethod
    def completion(
        model: str,
        messages: List,
        api_base: str,
        api_key: Optional[str],
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        timeout: Union[float, int, Any],
        acompletion: bool,
        stream: Optional[bool] = False,
        headers: Optional[dict] = None,
    ) -> Any:
        """
        Dispatch method for Azure Foundry Agents completion.
        
        Routes to sync or async completion based on acompletion flag.
        Supports native streaming via SSE when stream=True and acompletion=True.
        
        Authentication: Uses Azure AD Bearer tokens.
        - Pass api_key directly as an Azure AD token
        - Or set up Azure AD credentials via environment variables for automatic token retrieval:
          - AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET (Service Principal)
        
        See: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart
        """
        from litellm.llms.azure.common_utils import get_azure_ad_token
        from litellm.llms.azure_ai.agents.handler import azure_ai_agents_handler
        from litellm.types.router import GenericLiteLLMParams

        # If no api_key is provided, try to get Azure AD token
        if api_key is None:
            # Try to get Azure AD token using the existing Azure auth mechanisms
            # This uses the scope for Azure AI (ai.azure.com) instead of cognitive services
            # Create a GenericLiteLLMParams with the scope override for Azure Foundry Agents
            azure_auth_params = dict(litellm_params) if litellm_params else {}
            azure_auth_params["azure_scope"] = "https://ai.azure.com/.default"
            api_key = get_azure_ad_token(GenericLiteLLMParams(**azure_auth_params))
            
        if api_key is None:
            raise ValueError(
                "api_key (Azure AD token) is required for Azure Foundry Agents. "
                "Either pass api_key directly, or set AZURE_TENANT_ID, AZURE_CLIENT_ID, "
                "and AZURE_CLIENT_SECRET environment variables for Service Principal auth. "
                "Manual token: az account get-access-token --resource 'https://ai.azure.com'"
            )
        if acompletion:
            if stream:
                # Native async streaming via SSE - return the async generator directly
                return azure_ai_agents_handler.acompletion_stream(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    timeout=timeout,
                    headers=headers,
                )
            else:
                return azure_ai_agents_handler.acompletion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    api_key=api_key,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    timeout=timeout,
                    headers=headers,
                )
        else:
            # Sync completion - streaming not supported for sync
            return azure_ai_agents_handler.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=api_key,
                model_response=model_response,
                logging_obj=logging_obj,
                optional_params=optional_params,
                litellm_params=litellm_params,
                timeout=timeout,
                headers=headers,
            )

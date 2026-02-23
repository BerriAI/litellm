"""
Azure Anthropic transformation config - extends AnthropicConfig with Azure authentication
"""
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    pass


class AzureAnthropicConfig(AnthropicConfig):
    """
    Azure Anthropic configuration that extends AnthropicConfig.
    The only difference is authentication - Azure uses api-key header or Azure AD token
    instead of x-api-key header.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "azure_ai"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: Union[dict, GenericLiteLLMParams],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        """
        Validate environment and set up Azure authentication headers.
        Azure supports:
        1. API key via 'api-key' header
        2. Azure AD token via 'Authorization: Bearer <token>' header
        """
        # Convert dict to GenericLiteLLMParams if needed
        if isinstance(litellm_params, dict):
            # Ensure api_key is included if provided
            if api_key and "api_key" not in litellm_params:
                litellm_params = {**litellm_params, "api_key": api_key}
            litellm_params_obj = GenericLiteLLMParams(**litellm_params)
        else:
            litellm_params_obj = litellm_params or GenericLiteLLMParams()
            # Set api_key if provided and not already set
            if api_key and not litellm_params_obj.api_key:
                litellm_params_obj.api_key = api_key
        
        # Use Azure authentication logic
        headers = BaseAzureLLM._base_validate_azure_environment(
            headers=headers, litellm_params=litellm_params_obj
        )

        # Get tools and other anthropic-specific setup
        tools = optional_params.get("tools")
        prompt_caching_set = self.is_cache_control_set(messages=messages)
        computer_tool_used = self.is_computer_tool_used(tools=tools)
        mcp_server_used = self.is_mcp_server_used(
            mcp_servers=optional_params.get("mcp_servers")
        )
        pdf_used = self.is_pdf_used(messages=messages)
        file_id_used = self.is_file_id_used(messages=messages)
        user_anthropic_beta_headers = self._get_user_anthropic_beta_headers(
            anthropic_beta_header=headers.get("anthropic-beta")
        )

        # Get anthropic headers (but we'll replace x-api-key with Azure auth)
        anthropic_headers = self.get_anthropic_headers(
            computer_tool_used=computer_tool_used,
            prompt_caching_set=prompt_caching_set,
            pdf_used=pdf_used,
            api_key=api_key or "",  # Azure auth is already in headers
            file_id_used=file_id_used,
            is_vertex_request=optional_params.get("is_vertex_request", False),
            user_anthropic_beta_headers=user_anthropic_beta_headers,
            mcp_server_used=mcp_server_used,
        )
        # Merge headers - Azure auth (api-key or Authorization) takes precedence
        headers = {**anthropic_headers, **headers}

        # Ensure anthropic-version header is set
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"


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
        Transform request using parent AnthropicConfig, then remove unsupported params.
        Azure Anthropic doesn't support extra_body, max_retries, or stream_options parameters.
        """
        # Call parent transform_request
        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Remove unsupported parameters for Azure AI Anthropic
        data.pop("extra_body", None)
        data.pop("max_retries", None)
        data.pop("stream_options", None)

        return data


"""
Azure Anthropic messages transformation config - extends AnthropicMessagesConfig with Azure authentication
"""
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from httpx import Response


class AzureAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    Azure Anthropic messages configuration that extends AnthropicMessagesConfig.
    The only difference is authentication - Azure uses x-api-key header (not api-key)
    and Azure endpoint format.
    """

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        """
        Validate environment and set up Azure authentication headers for /v1/messages endpoint.
        """
        # Convert dict to GenericLiteLLMParams if needed
        if isinstance(litellm_params, dict):
            if api_key and "api_key" not in litellm_params:
                litellm_params = {**litellm_params, "api_key": api_key}
            litellm_params_obj = GenericLiteLLMParams(**litellm_params)
        else:
            litellm_params_obj = litellm_params or GenericLiteLLMParams()
            if api_key and not litellm_params_obj.api_key:
                litellm_params_obj.api_key = api_key

        # Use Azure authentication logic
        headers = BaseAzureLLM._base_validate_azure_environment(
            headers=headers, litellm_params=litellm_params_obj
        )

        # Azure Anthropic uses x-api-key header (not api-key)
        # Convert api-key to x-api-key if present
        if "api-key" in headers and "x-api-key" not in headers:
            headers["x-api-key"] = headers.pop("api-key")

        # Set anthropic-version header
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"

        # Set content-type header
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        headers = self._update_headers_with_anthropic_beta(
            headers=headers,
            optional_params=optional_params,
        )

        return headers, api_base

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
        Get the complete URL for Azure Anthropic /v1/messages endpoint.
        Azure Foundry endpoint format: https://<resource-name>.services.ai.azure.com/anthropic/v1/messages
        """
        from litellm.secret_managers.main import get_secret_str

        api_base = api_base or get_secret_str("AZURE_API_BASE")
        if api_base is None:
            raise ValueError(
                "Missing Azure API Base - Please set `api_base` or `AZURE_API_BASE` environment variable. "
                "Expected format: https://<resource-name>.services.ai.azure.com/anthropic"
            )

        # Ensure the URL ends with /v1/messages
        api_base = api_base.rstrip("/")
        if api_base.endswith("/v1/messages"):
            # Already correct
            pass
        elif api_base.endswith("/anthropic/v1/messages"):
            # Already correct
            pass
        else:
            # Check if /anthropic is already in the path
            if "/anthropic" in api_base:
                # /anthropic exists, ensure we end with /anthropic/v1/messages
                # Extract the base URL up to and including /anthropic
                parts = api_base.split("/anthropic", 1)
                api_base = parts[0] + "/anthropic"
            else:
                # /anthropic not in path, add it
                api_base = api_base + "/anthropic"
            # Add /v1/messages
            api_base = api_base + "/v1/messages"

        return api_base

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: "Response",
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        """
        Transform the response from Azure AI Anthropic.

        Azure AI Anthropic returns usage without total_tokens, but the logging
        utilities expect it for ResponseAPIUsage. This adds total_tokens if missing.
        """

        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise AnthropicError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        # Add total_tokens to usage if missing (Azure AI Anthropic doesn't include it)
        if "usage" in raw_response_json and isinstance(raw_response_json["usage"], dict):
            usage = raw_response_json["usage"]
            if "total_tokens" not in usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                usage["total_tokens"] = input_tokens + output_tokens

        return AnthropicMessagesResponse(**raw_response_json)


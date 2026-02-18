"""
Azure AI Anthropic CountTokens API handler.

Uses httpx for HTTP requests with Azure authentication.
"""

from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.azure_ai.anthropic.count_tokens.transformation import (
    AzureAIAnthropicCountTokensConfig,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client


class AzureAIAnthropicCountTokensHandler(AzureAIAnthropicCountTokensConfig):
    """
    Handler for Azure AI Anthropic CountTokens API requests.

    Uses httpx for HTTP requests with Azure authentication.
    """

    async def handle_count_tokens_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_key: str,
        api_base: str,
        litellm_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Dict[str, Any]:
        """
        Handle a CountTokens request using httpx with Azure authentication.

        Args:
            model: The model identifier (e.g., "claude-3-5-sonnet")
            messages: The messages to count tokens for
            api_key: The Azure AI API key
            api_base: The Azure AI API base URL
            litellm_params: Optional LiteLLM parameters
            timeout: Optional timeout for the request (defaults to litellm.request_timeout)

        Returns:
            Dictionary containing token count response

        Raises:
            AnthropicError: If the API request fails
        """
        try:
            # Validate the request
            self.validate_request(model, messages)

            verbose_logger.debug(
                f"Processing Azure AI Anthropic CountTokens request for model: {model}"
            )

            # Transform request to Anthropic format
            request_body = self.transform_request_to_count_tokens(
                model=model,
                messages=messages,
            )

            verbose_logger.debug(f"Transformed request: {request_body}")

            # Get endpoint URL
            endpoint_url = self.get_count_tokens_endpoint(api_base)

            verbose_logger.debug(f"Making request to: {endpoint_url}")

            # Get required headers with Azure authentication
            headers = self.get_required_headers(
                api_key=api_key,
                litellm_params=litellm_params,
            )

            # Use LiteLLM's async httpx client
            async_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.AZURE_AI
            )

            # Use provided timeout or fall back to litellm.request_timeout
            request_timeout = timeout if timeout is not None else litellm.request_timeout

            response = await async_client.post(
                endpoint_url,
                headers=headers,
                json=request_body,
                timeout=request_timeout,
            )

            verbose_logger.debug(f"Response status: {response.status_code}")

            if response.status_code != 200:
                error_text = response.text
                verbose_logger.error(f"Azure AI Anthropic API error: {error_text}")
                raise AnthropicError(
                    status_code=response.status_code,
                    message=error_text,
                )

            azure_response = response.json()

            verbose_logger.debug(f"Azure AI Anthropic response: {azure_response}")

            # Return Anthropic-compatible response directly - no transformation needed
            return azure_response

        except AnthropicError:
            # Re-raise Anthropic exceptions as-is
            raise
        except httpx.HTTPStatusError as e:
            # HTTP errors - preserve the actual status code
            verbose_logger.error(f"HTTP error in CountTokens handler: {str(e)}")
            raise AnthropicError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except Exception as e:
            verbose_logger.error(f"Error in CountTokens handler: {str(e)}")
            raise AnthropicError(
                status_code=500,
                message=f"CountTokens processing error: {str(e)}",
            )

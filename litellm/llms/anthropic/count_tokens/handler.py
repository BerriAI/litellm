"""
Anthropic CountTokens API handler.

Uses httpx for HTTP requests instead of the Anthropic SDK.
"""

from typing import Any, Final

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client


class AnthropicCountTokensHandler(AnthropicCountTokensConfig):
    """
    Handler for Anthropic CountTokens API requests.

    Uses httpx for HTTP requests, following the same pattern as BedrockCountTokensHandler.
    """

    async def handle_count_tokens_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_key: str,
        api_base: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        tools: list[dict[str, Any]] | None = None,
        system: Any | None = None,
    ) -> dict[str, Any]:
        """
        Handle a CountTokens request using httpx.

        Args:
            model: The model identifier (e.g., "claude-3-5-sonnet-20241022")
            messages: The messages to count tokens for
            api_key: The Anthropic API key
            api_base: Optional custom API base URL
            timeout: Optional timeout for the request (defaults to litellm.request_timeout)

        Returns:
            Dictionary containing token count response

        Raises:
            AnthropicError: If the API request fails
        """
        try:
            # Validate the request
            self.validate_request(model, messages)

            verbose_logger.debug("Processing Anthropic CountTokens request for model: %s", model)

            # Transform request to Anthropic format
            request_body: Final = self.transform_request_to_count_tokens(
                model=model,
                messages=messages,
                tools=tools,
                system=system,
            )

            verbose_logger.debug("Transformed request: %s", request_body)

            # Get endpoint URL
            endpoint_url: Final = api_base or self.get_anthropic_count_tokens_endpoint()

            verbose_logger.debug("Making request to: %s", endpoint_url)

            # Get required headers
            headers: Final = self.get_required_headers(api_key)

            # Use LiteLLM's async httpx client
            async_client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)

            # Use provided timeout or fall back to litellm.request_timeout
            request_timeout: Final = timeout if timeout is not None else litellm.request_timeout

            response: Final = await async_client.post(
                endpoint_url,
                headers=headers,
                json=request_body,
                timeout=request_timeout,
            )

            verbose_logger.debug("Response status: %s", response.status_code)

            if response.status_code != 200:
                error_text: Final = response.text
                verbose_logger.error("Anthropic API error: %s", error_text)
                raise AnthropicError(
                    status_code=response.status_code,
                    message=error_text,
                )

            anthropic_response: Final = response.json()

            verbose_logger.debug("Anthropic response: %s", anthropic_response)

            # Return Anthropic response directly - no transformation needed
            return anthropic_response

        except AnthropicError:
            # Re-raise Anthropic exceptions as-is
            raise
        except httpx.HTTPStatusError as e:
            # HTTP errors - preserve the actual status code
            verbose_logger.error("HTTP error in CountTokens handler: %s", e)
            raise AnthropicError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except Exception as e:
            verbose_logger.error("Error in CountTokens handler: %s", e)
            raise AnthropicError(
                status_code=500,
                message=f"CountTokens processing error: {e}",
            )

import os
import httpx
from typing import List, Dict, Any, Optional, Union
import litellm
from litellm.llms.anthropic.count_tokens.transformation import AnthropicCountTokensConfig
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.anthropic.common_utils import AnthropicError


class AnthropicCountTokensHandler(AnthropicCountTokensConfig):
    """
    Handler for Anthropic CountTokens API requests.
    """

    async def handle_count_tokens_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_key: str,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Handle a CountTokens request using httpx.
        """
        try:
            # Validate the request
            self.validate_request(model, messages)

            verbose_logger.debug(
                f"Processing Anthropic CountTokens request for model: {model}"
            )

            # Transform request to Anthropic format (api_base stays out of the payload body)
            request_body = self.transform_request_to_count_tokens(
                model=model,
                messages=messages,
                tools=tools,
                system=system,
            )

            verbose_logger.debug(f"Transformed request: {request_body}")

            # Get endpoint URL - prioritizing custom api_base
            endpoint_url = api_base or self.get_anthropic_count_tokens_endpoint()

            verbose_logger.info(f"Making request to: {endpoint_url}")

            # Get required headers
            headers = self.get_required_headers(api_key)

            # Use LiteLLM's async httpx client
            async_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.ANTHROPIC
            )

            request_timeout = (
                timeout if timeout is not None else litellm.request_timeout
            )

            response = await async_client.post(
                endpoint_url,
                headers=headers,
                json=request_body,
                timeout=request_timeout,
            )

            verbose_logger.debug(f"Response status: {response.status_code}")

            if response.status_code != 200:
                error_text = response.text
                verbose_logger.error(f"Anthropic API error: {error_text}")
                raise AnthropicError(
                    status_code=response.status_code,
                    message=f"CountTokens processing error: {error_text}"
                )

            return response.json()

        except AnthropicError:
            raise
        except Exception as e:
            verbose_logger.error(f"Unexpected error in CountTokens handler: {str(e)}")
            raise AnthropicError(
                status_code=500,
                message=f"CountTokens processing error: {str(e)}"
            )

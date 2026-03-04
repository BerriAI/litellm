"""
OpenAI Responses API token counting handler.

Uses httpx for HTTP requests to OpenAI's /v1/responses/input_tokens endpoint.
"""

from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
)


class OpenAICountTokensHandler(OpenAICountTokensConfig):
    """
    Handler for OpenAI Responses API token counting requests.
    """

    async def handle_count_tokens_request(
        self,
        model: str,
        input: Union[str, List[Any]],
        api_key: str,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle a token counting request to OpenAI's Responses API.

        Returns:
            Dictionary containing {"input_tokens": <number>}

        Raises:
            OpenAIError: If the API request fails
        """
        try:
            self.validate_request(model, input)

            verbose_logger.debug(
                f"Processing OpenAI CountTokens request for model: {model}"
            )

            request_body = self.transform_request_to_count_tokens(
                model=model,
                input=input,
                tools=tools,
                instructions=instructions,
            )

            endpoint_url = self.get_openai_count_tokens_endpoint(api_base)

            verbose_logger.debug(f"Making request to: {endpoint_url}")

            headers = self.get_required_headers(api_key)

            async_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.OPENAI
            )

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
                verbose_logger.error(f"OpenAI API error: {error_text}")
                raise OpenAIError(
                    status_code=response.status_code,
                    message=error_text,
                )

            openai_response = response.json()
            verbose_logger.debug(f"OpenAI response: {openai_response}")
            return openai_response

        except OpenAIError:
            raise
        except httpx.HTTPStatusError as e:
            verbose_logger.error(f"HTTP error in CountTokens handler: {str(e)}")
            raise OpenAIError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except Exception as e:
            verbose_logger.error(f"Error in CountTokens handler: {str(e)}")
            raise OpenAIError(
                status_code=500,
                message=f"CountTokens processing error: {str(e)}",
            )

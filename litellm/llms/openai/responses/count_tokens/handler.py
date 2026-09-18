"""
OpenAI Responses API token counting handler.

Uses httpx for HTTP requests to OpenAI's /v1/responses/input_tokens endpoint.
"""

import json
from typing import Any, Final

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
        input: str | list[Any],
        api_key: str,
        api_base: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
    ) -> dict[str, Any]:
        """
        Handle a token counting request to OpenAI's Responses API.

        Returns:
            Dictionary containing {"input_tokens": <number>}

        Raises:
            OpenAIError: If the API request fails
        """
        try:
            self.validate_request(model, input)

            verbose_logger.debug("Processing OpenAI CountTokens request for model: %s", model)

            request_body: Final = self.transform_request_to_count_tokens(
                model=model,
                input=input,
                tools=tools,
                instructions=instructions,
            )

            endpoint_url: Final = self.get_openai_count_tokens_endpoint(api_base)

            verbose_logger.debug("Making request to: %s", endpoint_url)

            headers: Final = self.get_required_headers(api_key)

            async_client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.OPENAI)

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
                verbose_logger.error("OpenAI API error: %s", error_text)
                raise OpenAIError(
                    status_code=response.status_code,
                    message=error_text,
                )

            openai_response: Final = response.json()
            verbose_logger.debug("OpenAI response: %s", openai_response)
            return openai_response

        except OpenAIError:
            raise
        except httpx.HTTPStatusError as e:
            verbose_logger.error("HTTP error in CountTokens handler: %s", e)
            raise OpenAIError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except (httpx.RequestError, json.JSONDecodeError, ValueError) as e:
            verbose_logger.error("Error in CountTokens handler: %s", e)
            raise OpenAIError(
                status_code=500,
                message=f"CountTokens processing error: {e}",
            )

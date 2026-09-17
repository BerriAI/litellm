"""
OpenAI Responses API token counting handler.

Uses httpx for HTTP requests to OpenAI's /v1/responses/input_tokens endpoint.
"""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Final

import httpx
from openai import APIError, AsyncOpenAI
from pydantic import Field, StrictStr, TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

_NATIVE_COUNT_FIELDS: Final = frozenset(
    {"input", "instructions", "tools", "tool_choice", "parallel_tool_calls", "reasoning", "text"}
)
_NATIVE_HEADERS: Final = TypeAdapter(dict[StrictStr, StrictStr])
_NATIVE_OBJECT: Final = TypeAdapter(dict[str, object])
_NATIVE_COUNT: Final[TypeAdapter[int]] = TypeAdapter(Annotated[int, Field(strict=True, ge=0)])


class OpenAICountTokensHandler(OpenAICountTokensConfig):
    """
    Handler for OpenAI Responses API token counting requests.
    """

    async def count_native_tokens(
        self,
        model: str,
        payload: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> int:
        handler: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.OPENAI)
        try:
            body: Final = _NATIVE_OBJECT.validate_python(
                MappingProxyType(
                    {
                        "model": model,
                        **MappingProxyType(
                            {
                                key: value
                                for key, value in payload.items()
                                if key in _NATIVE_COUNT_FIELDS and value is not None
                            }
                        ),
                    }
                )
            )
            config: Final = OpenAIResponsesAPIConfig()
            openai_base: Final = config.get_complete_url(
                api_base=api_base, litellm_params=_NATIVE_OBJECT.validate_python(MappingProxyType({}))
            ).removesuffix("/responses")
            openai_key: Final = api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")
            environment_headers: Final = _NATIVE_HEADERS.validate_python(
                config.validate_environment(
                    headers=_NATIVE_HEADERS.validate_python(headers),
                    model=model,
                    litellm_params=GenericLiteLLMParams(api_key=openai_key, api_base=openai_base),
                )
            )
            client: Final = AsyncOpenAI(
                api_key=openai_key or "",
                base_url=openai_base,
                http_client=handler.client,
                max_retries=0,
            )
            sdk_header_names: Final = MappingProxyType({name.lower(): name for name in client.default_headers})
            openai_headers: Final = MappingProxyType(
                {
                    sdk_header_names.get(name.lower(), name): value
                    for source in (environment_headers, headers)
                    for name, value in source.items()
                }
            )
            result: Final = await client.responses.input_tokens.count(
                model=model, extra_body=body, extra_headers=openai_headers, timeout=timeout
            )
            return _NATIVE_COUNT.validate_python(result.input_tokens)
        except (APIError, httpx.HTTPError, ValueError):
            raise ValueError("Provider-native token counting failed") from None

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

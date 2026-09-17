"""
Anthropic CountTokens API handler.

Uses httpx for HTTP requests instead of the Anthropic SDK.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Final

import httpx
from pydantic import Field, StrictStr, TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
from litellm.litellm_core_utils.prompt_templates.compaction import compaction_headers
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import AnthropicMessagesConfig
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

_NATIVE_COUNT_FIELDS: Final = frozenset({"messages", "system", "tools", "tool_choice", "thinking"})
_NATIVE_HEADERS: Final = TypeAdapter(dict[StrictStr, StrictStr])
_NATIVE_OBJECT: Final = TypeAdapter(dict[str, object])
_NATIVE_MESSAGES: Final = TypeAdapter(list[dict[str, object]])
_NATIVE_COUNT: Final[TypeAdapter[int]] = TypeAdapter(Annotated[int, Field(strict=True, ge=0)])


class AnthropicCountTokensHandler(AnthropicCountTokensConfig):
    """
    Handler for Anthropic CountTokens API requests.

    Uses httpx for HTTP requests, following the same pattern as BedrockCountTokensHandler.
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
        handler: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)
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
            config: Final = AnthropicMessagesConfig()
            messages_url: Final = httpx.URL(
                config.get_complete_url(
                    api_base=api_base or litellm.api_base,
                    api_key=api_key,
                    model=model,
                    optional_params=_NATIVE_OBJECT.validate_python(MappingProxyType({})),
                    litellm_params=_NATIVE_OBJECT.validate_python(MappingProxyType({})),
                )
            )
            compact_headers: Final = compaction_headers(headers)
            beta_headers: Final = _NATIVE_HEADERS.validate_python(
                MappingProxyType(
                    {
                        **compact_headers,
                        "anthropic-beta": ",".join(
                            dict.fromkeys(
                                (*compact_headers["anthropic-beta"].split(","), ANTHROPIC_TOKEN_COUNTING_BETA_VERSION)
                            )
                        ),
                    }
                )
            )
            anthropic_headers: Final = _NATIVE_HEADERS.validate_python(
                config.validate_anthropic_messages_environment(
                    headers=beta_headers,
                    model=model,
                    messages=_NATIVE_MESSAGES.validate_python(payload.get("messages")),
                    optional_params=body,
                    litellm_params=_NATIVE_OBJECT.validate_python(MappingProxyType({})),
                    api_key=api_key or litellm.anthropic_key or litellm.api_key,
                    api_base=str(messages_url),
                )[0]
            )
            response: Final = await handler.client.post(
                str(messages_url.copy_with(path=f"{messages_url.path.rstrip('/')}/count_tokens")),
                headers=anthropic_headers,
                json=body,
                timeout=timeout,
            )
            response.raise_for_status()
            return _NATIVE_COUNT.validate_python(_NATIVE_OBJECT.validate_json(response.content).get("input_tokens"))
        except (httpx.HTTPError, ValueError):
            raise ValueError("Provider-native token counting failed") from None

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

from collections.abc import Mapping, Sequence
from typing import Final

import httpx
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # factory stub leaves params untyped
)
from litellm.llms.mistral.common_utils import MistralError
from litellm.llms.mistral.count_tokens.transformation import (
    MistralCountTokensConfig,
)

_DECODED_JSON: Final = TypeAdapter(object)


class MistralCountTokensHandler(MistralCountTokensConfig):
    async def handle_count_tokens_request(
        self,
        model: str,
        input: str | Sequence[object],
        api_key: str,
        api_base: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        tools: Sequence[Mapping[str, object]] | None = None,
        instructions: str | None = None,
        text: Mapping[str, object] | None = None,
        tool_choice: JsonValue = None,
    ) -> object:
        try:
            self.validate_request(model, input)
            verbose_logger.debug("Processing Mistral CountTokens request for model: %s", model)
            request_body: Final = self.transform_request_to_count_tokens(
                model=model,
                input=input,
                tools=tools,
                instructions=instructions,
                text=text,
                tool_choice=tool_choice,
            )
        except ValueError as e:
            verbose_logger.error("Error in Mistral CountTokens handler: %s", e)
            raise MistralError(
                status_code=500, message=f"CountTokens processing error: {e}", headers=httpx.Headers()
            ) from e
        return await self.post_count_request(request_body, api_key, api_base, timeout)

    async def post_count_request(
        self,
        request_body: Mapping[str, object],
        api_key: str,
        api_base: str | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> object:
        try:
            endpoint_url: Final = self.get_mistral_count_tokens_endpoint(api_base)
            verbose_logger.debug("Making Mistral count request to: %s", endpoint_url)
            headers: Final = self.get_required_headers(api_key)
            async_client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.MISTRAL)
            request_timeout: Final = timeout if timeout is not None else litellm.request_timeout

            response: Final = await async_client.post(  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.post stub leaves params/headers untyped
                endpoint_url,
                headers=headers,
                json=dict(request_body),
                timeout=request_timeout,
            )
            verbose_logger.debug("Mistral count response status: %s", response.status_code)

            if response.status_code != 200:
                error_text: Final = response.text
                verbose_logger.error("Mistral API error: %s", error_text)
                raise MistralError(
                    status_code=response.status_code,
                    message=error_text,
                    headers=response.headers,
                )

            mistral_response: Final = _DECODED_JSON.validate_python(response.json())
            verbose_logger.debug("Mistral count response: %s", mistral_response)
        except litellm.Timeout as e:
            raise MistralError(status_code=e.status_code, message=str(e), headers=httpx.Headers()) from e
        except (httpx.RequestError, litellm.APIConnectionError, OSError) as e:
            verbose_logger.error("Connection error in Mistral CountTokens handler: %s", e)
            raise MistralError(
                status_code=500, message=f"CountTokens connection error: {e}", headers=httpx.Headers()
            ) from e
        except MistralError:
            raise
        except httpx.HTTPStatusError as e:
            verbose_logger.error("HTTP error in Mistral CountTokens handler: %s", e)
            raise MistralError(
                status_code=e.response.status_code,
                message=e.response.text,
                headers=e.response.headers,
            ) from e
        except ValueError as e:
            verbose_logger.error("Error in Mistral CountTokens handler: %s", e)
            raise MistralError(
                status_code=500,
                message=f"CountTokens processing error: {e}",
                headers=httpx.Headers(),
            ) from e
        else:
            return mistral_response

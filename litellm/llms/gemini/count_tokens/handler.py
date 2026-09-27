from collections.abc import Sequence
from typing import Any, Final

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.types.llms.gemini import GeminiCountTokensRequest
from litellm.types.llms.vertex_ai import ContentType, SystemInstructions, Tools
from litellm.types.utils import LlmProviders

# acount_tokens binds these itself, so forwarding a deployment's copy would raise a duplicate-keyword TypeError
ACOUNT_TOKENS_DEPLOYMENT_RESERVED_KEYS: Final = frozenset({"self", "system_instruction", "tools", "client"})


def build_count_tokens_request(
    model: str,
    contents: Sequence[ContentType],
    system_instruction: SystemInstructions | None,
    tools: Sequence[Tools] | None,
) -> GeminiCountTokensRequest:
    model_name: Final = f"models/{model}"
    if tools is None:
        if system_instruction is None:
            bare: Final[GeminiCountTokensRequest] = {"contents": contents}
            return bare
        with_system: Final[GeminiCountTokensRequest] = {
            "generateContentRequest": {
                "model": model_name,
                "contents": contents,
                "systemInstruction": system_instruction,
            }
        }
        return with_system
    if system_instruction is None:
        with_tools: Final[GeminiCountTokensRequest] = {
            "generateContentRequest": {"model": model_name, "contents": contents, "tools": tools}
        }
        return with_tools
    with_both: Final[GeminiCountTokensRequest] = {
        "generateContentRequest": {
            "model": model_name,
            "contents": contents,
            "systemInstruction": system_instruction,
            "tools": tools,
        }
    }
    return with_both


class GoogleAIStudioTokenCounter:
    def _clean_contents_for_gemini_api(self, contents: Any) -> Any:
        """
        Clean up contents to remove unsupported fields for the Gemini API.

        The Google Gemini API doesn't recognize the 'id' field in function responses,
        so we need to remove it to prevent 400 Bad Request errors.

        Args:
            contents: The contents to clean up

        Returns:
            Cleaned contents with unsupported fields removed
        """
        import copy

        # Handle None or empty contents
        if not contents:
            return contents

        cleaned_contents: Final = copy.deepcopy(contents)

        for content in cleaned_contents:
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            for part in parts:
                function_response = part.get("functionResponse") if isinstance(part, dict) else None
                if isinstance(function_response, dict):
                    part["functionResponse"] = {
                        key: value for key, value in function_response.items() if key != "id" and value is not None
                    }

        return cleaned_contents

    def _construct_url(self, model: str, api_base: str | None = None) -> str:
        """
        Construct the URL for the Google Gen AI Studio countTokens endpoint.
        """
        base_url: Final = api_base or "https://generativelanguage.googleapis.com"
        return f"{base_url}/v1beta/models/{model}:countTokens"

    async def validate_environment(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        headers: dict[str, object] | None = None,
        model: str = "",
        litellm_params: dict[str, object] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """
        Returns a Tuple of headers and url for the Google Gen AI Studio countTokens endpoint.
        """
        from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig

        headers = GoogleGenAIConfig().validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            litellm_params=litellm_params,
        )

        url: Final = self._construct_url(model=model, api_base=api_base)
        return headers, url

    async def acount_tokens(
        self,
        contents: Any,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        system_instruction: SystemInstructions | None = None,
        tools: Sequence[Tools] | None = None,
        client: httpx.AsyncClient | AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        """
        Count tokens using Google Gen AI Studio countTokens endpoint.

        Args:
            contents: The content to count tokens for (Google Gen AI format)
                    Example: [{"parts": [{"text": "Hello world"}]}]
            model: The model name (e.g. "gemini-1.5-flash")
            api_key: Optional Google API key (will fall back to environment)
            api_base: Optional API base URL (defaults to Google Gen AI Studio)
            timeout: Optional timeout for the request
            system_instruction: Optional system instruction to count with the contents
            tools: Optional Gemini tools to count with the contents
            client: Optional HTTP client to send the request with
            **kwargs: Additional parameters

        Returns:
            Dict containing token count information from Google Gen AI Studio API.
            Example response:
            {
                "totalTokens": 31,
                "totalBillableCharacters": 96,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 31
                    }
                ]
            }

        Raises:
            ValueError: If API key is missing
            litellm.APIError: If the API call fails
            litellm.APIConnectionError: If the connection fails
            Exception: For any other unexpected errors
        """
        headers, url = await self.validate_environment(
            api_key=api_key,
            api_base=api_base,
            headers={},  # mutable-ok: validate_environment merges into this dict
            model=model,
            litellm_params=kwargs,
        )
        request_body: Final = build_count_tokens_request(
            model=model,
            contents=self._clean_contents_for_gemini_api(contents),
            system_instruction=system_instruction,
            tools=tools,
        )
        async_httpx_client: Final = client or get_async_httpx_client(llm_provider=LlmProviders.GEMINI)

        try:
            response: Final = await async_httpx_client.post(
                url=url,
                headers=headers,
                json=request_body,  # pyright: ignore[reportArgumentType]  # post() takes a bare dict; a TypedDict is one at runtime
            )
            response.raise_for_status()

            # Parse response
            result: Final = response.json()
            return result

        except httpx.HTTPStatusError as e:
            error_msg = f"Google Gen AI Studio API error: {e.response.status_code} - {e.response.text}"
            raise litellm.APIError(
                message=error_msg,
                llm_provider="gemini",
                model=model,
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            error_msg = f"Request to Google Gen AI Studio failed: {e}"
            raise litellm.APIConnectionError(message=error_msg, llm_provider="gemini", model=model) from e
        except Exception as e:
            error_msg = f"Unexpected error during token counting: {e}"
            raise Exception(error_msg) from e

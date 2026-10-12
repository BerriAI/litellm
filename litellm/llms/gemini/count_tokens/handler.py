from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

import httpx
from pydantic import ConfigDict, JsonValue, RootModel, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.types.llms.openai import AllMessageValues, ChatCompletionSystemMessage
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.types.google_genai.main import GenerateContentContentListUnionDict
else:
    GenerateContentContentListUnionDict = Any


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _GeminiCountTokensResponse(TypedDict):
    totalTokens: NotRequired[ReadOnly[int | float]]
    tokenizer_used: NotRequired[ReadOnly[str]]


class _GeminiCountTokensResponseRoot(RootModel[_GeminiCountTokensResponse]):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


def _prepare_gemini_count_tool(tool: Mapping[str, object], model: str) -> dict[str, JsonValue]:
    if "input_schema" not in tool:
        return TypeAdapter(dict[str, JsonValue]).validate_python(tool)
    from litellm.llms.anthropic.pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter
    from litellm.types.llms.anthropic import AllAnthropicToolsValues

    parsed: Final = TypeAdapter(AllAnthropicToolsValues).validate_python(tool)
    translated, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_tools_to_openai([parsed], model=model)
    return TypeAdapter(dict[str, JsonValue]).validate_python(translated[0])


def prepare_gemini_count_tokens(
    model: str,
    messages: Sequence[Mapping[str, object]] | None,
    contents: Sequence[Mapping[str, object]] | None,
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> dict[str, JsonValue]:
    from litellm.llms.vertex_ai.gemini.transformation import (
        gemini_convert_messages_with_history,
        transform_system_message,
    )
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    parsed_messages: Final = TypeAdapter(list[AllMessageValues]).validate_python(list(messages or ()))
    leading_system: Final = (
        [TypeAdapter(ChatCompletionSystemMessage).validate_python({"role": "system", "content": system})]
        if system is not None
        else []
    )
    system_instruction, conversation = transform_system_message(
        supports_system_message=True, messages=[*leading_system, *parsed_messages]
    )
    resolved_contents: Final = (
        contents if contents is not None else gemini_convert_messages_with_history(messages=conversation, model=model)
    )
    translated_tools: Final = [_prepare_gemini_count_tool(tool, model) for tool in tools or ()]
    mapped_params: Final = VertexGeminiConfig().map_openai_params(
        non_default_params={"tools": translated_tools}, optional_params={}, model=model, drop_params=False
    )
    return TypeAdapter(dict[str, JsonValue]).validate_python(
        {
            "contents": resolved_contents,
            **({"system_instruction": system_instruction} if system_instruction is not None else {}),
            **({"tools": mapped_params["tools"]} if "tools" in mapped_params else {}),
        }
    )


def build_gemini_count_tokens_body(
    url: str,
    contents: object,
    system_instruction: Mapping[str, object] | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    if system_instruction is None and tools is None:
        return {"contents": contents}
    resource: Final = urlsplit(url).path.split("/", 2)[-1].removesuffix(":countTokens")
    generation_request: Final = {
        "model": resource,
        "contents": contents,
        **({"systemInstruction": system_instruction} if system_instruction is not None else {}),
        **({"tools": tools} if tools is not None else {}),
    }
    return generation_request if resource.startswith("projects/") else {"generateContentRequest": generation_request}


class GoogleAIStudioTokenCounter:
    def _clean_contents_for_gemini_api(self, contents: Any) -> object:
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

        from google.genai.types import FunctionResponse

        # Handle None or empty contents
        if not contents:
            return contents

        cleaned_contents: Final = copy.deepcopy(contents)

        for content in cleaned_contents:
            parts = content["parts"]
            for part in parts:
                if "functionResponse" in part:
                    function_response_data = part["functionResponse"]
                    function_response_part = FunctionResponse(**function_response_data)
                    function_response_part.id = None
                    part["functionResponse"] = function_response_part.model_dump(exclude_none=True)

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
        contents: Sequence[Mapping[str, object]] | None,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        system_instruction: Mapping[str, object] | None = None,
        tools: Sequence[Mapping[str, object]] | None = None,
        client: AsyncHTTPHandler | None = None,
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

        # Prepare headers
        headers, url = await self.validate_environment(
            api_key=api_key,
            api_base=api_base,
            headers={},
            model=model,
            litellm_params=kwargs,
        )

        # Prepare request body - clean up contents to remove unsupported fields
        cleaned_contents: Final = self._clean_contents_for_gemini_api(contents)
        request_body: Final = build_gemini_count_tokens_body(
            url=url, contents=cleaned_contents, system_instruction=system_instruction, tools=tools
        )

        async_httpx_client: Final = client or get_async_httpx_client(
            llm_provider=LlmProviders.GEMINI,
        )

        try:
            response: Final = await async_httpx_client.post(url=url, headers=headers, json=request_body)

            # Check for HTTP errors
            response.raise_for_status()

            # Parse response
            result: Final = _GeminiCountTokensResponseRoot.model_validate(response.json()).root
            return dict(result)

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

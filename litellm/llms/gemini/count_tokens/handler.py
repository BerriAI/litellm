from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import httpx
from pydantic import ConfigDict, RootModel, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm.litellm_core_utils.core_helpers import has_attributes, iter_items, json_contains
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
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


@runtime_checkable
class _SupportsGetItem(Protocol):
    def __getitem__(self, key: str, /) -> object: ...


@runtime_checkable
class _SupportsSetItem(Protocol):
    def __setitem__(self, key: str, value: object, /) -> None: ...


class _KeywordArguments(Protocol):
    def keys(self) -> Iterable[str]: ...

    def __getitem__(self, key: str, /) -> object: ...


def _subscript(value: object, key: str) -> object:
    if isinstance(value, _SupportsGetItem):
        return value[key]
    raise TypeError(f"'{type(value).__name__}' object is not subscriptable")


def _item_assignment_target(value: object) -> _SupportsSetItem:
    if isinstance(value, _SupportsSetItem):
        return value
    raise TypeError(f"'{type(value).__name__}' object does not support item assignment")


def _function_response_kwargs(value: object) -> _KeywordArguments:
    if has_attributes(value, _KeywordArguments, "keys"):
        return value
    raise TypeError(
        f"google.genai.types.FunctionResponse() argument after ** must be a mapping, not {type(value).__name__}"
    )


class GoogleAIStudioTokenCounter:
    def _clean_contents_for_gemini_api(self, contents: Sequence[Mapping[str, object]] | None) -> object:
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

        from google.genai.types import FunctionResponse

        cleaned_contents: Final = copy.deepcopy(contents)

        for content in cleaned_contents:
            parts = content["parts"]
            for part in iter_items(parts):
                if json_contains(part, "functionResponse"):
                    function_response_data = _subscript(part, "functionResponse")
                    function_response_part = FunctionResponse(**_function_response_kwargs(function_response_data))
                    function_response_part.id = None
                    _item_assignment_target(part)["functionResponse"] = function_response_part.model_dump(
                        exclude_none=True
                    )

        return cleaned_contents

    def _construct_url(self, model: str, api_base: str | None = None) -> str:
        """
        Construct the URL for the Google Gen AI Studio countTokens endpoint.
        """
        base_url: Final = api_base or "https://generativelanguage.googleapis.com"
        return join_api_base(base_url, f"v1beta/models/{model}:countTokens")

    def __init__(self, client: AsyncHTTPHandler | None = None) -> None:
        super().__init__()
        self._client: Final = client

    def _build_count_tokens_request_body(
        self,
        model: str,
        contents: object,
        system_instruction: Mapping[str, object] | None,
        tools: Sequence[Mapping[str, object]] | None,
    ) -> Mapping[str, object]:
        if system_instruction is None and not tools:
            return {"contents": contents}

        generate_content_request: Final = {
            "model": model if model.startswith("models/") else f"models/{model}",
            "contents": contents,
            **({"systemInstruction": dict(system_instruction)} if system_instruction is not None else {}),
            **({"tools": [dict(tool) for tool in tools]} if tools else {}),
        }
        return {"generateContentRequest": generate_content_request}

    async def validate_environment(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        headers: dict[str, object] | None = None,
        model: str = "",
        litellm_params: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], str]:
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
        request_body: Final = self._build_count_tokens_request_body(
            model=model,
            contents=cleaned_contents,
            system_instruction=system_instruction,
            tools=tools,
        )

        async_httpx_client: Final = (
            self._client if self._client is not None else get_async_httpx_client(llm_provider=LlmProviders.GEMINI)
        )

        try:
            response: Final = await async_httpx_client.post(url=url, headers=headers, json=dict(request_body))

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

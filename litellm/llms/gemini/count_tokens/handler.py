from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.types.google_genai.main import GenerateContentContentListUnionDict
else:
    GenerateContentContentListUnionDict = Any


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

        from google.genai.types import FunctionResponse

        # Handle None or empty contents
        if not contents:
            return contents

        cleaned_contents = copy.deepcopy(contents)

        for content in cleaned_contents:
            parts = content["parts"]
            for part in parts:
                if "functionResponse" in part:
                    function_response_data = part["functionResponse"]
                    function_response_part = FunctionResponse(**function_response_data)
                    function_response_part.id = None
                    part["functionResponse"] = function_response_part.model_dump(
                        exclude_none=True
                    )

        return cleaned_contents

    def _construct_url(self, model: str, api_base: Optional[str] = None) -> str:
        """
        Construct the URL for the Google Gen AI Studio countTokens endpoint.
        """
        base_url = api_base or "https://generativelanguage.googleapis.com"
        return f"{base_url}/v1beta/models/{model}:countTokens"

    async def validate_environment(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        model: str = "",
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
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

        url = self._construct_url(model=model, api_base=api_base)
        return headers, url

    async def acount_tokens(
        self,
        contents: Any,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
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
        cleaned_contents = self._clean_contents_for_gemini_api(contents)
        request_body = {"contents": cleaned_contents}

        async_httpx_client = get_async_httpx_client(
            llm_provider=LlmProviders.GEMINI,
        )

        try:
            response = await async_httpx_client.post(
                url=url, headers=headers, json=request_body
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Parse response
            result = response.json()
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
            error_msg = f"Request to Google Gen AI Studio failed: {str(e)}"
            raise litellm.APIConnectionError(
                message=error_msg, llm_provider="gemini", model=model
            ) from e
        except Exception as e:
            error_msg = f"Unexpected error during token counting: {str(e)}"
            raise Exception(error_msg) from e

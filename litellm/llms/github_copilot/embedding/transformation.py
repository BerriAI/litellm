"""
GitHub Copilot Embedding API Configuration.

This module provides the configuration for GitHub Copilot's Embedding API.

Implementation based on analysis of the copilot-api project by caozhiyuan:
https://github.com/caozhiyuan/copilot-api
"""
from typing import TYPE_CHECKING, Any, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import convert_to_model_response_object

from ..authenticator import Authenticator
from ..common_utils import (
    GetAccessTokenError,
    GetAPIKeyError,
    GITHUB_COPILOT_API_BASE,
    get_copilot_default_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class GithubCopilotEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration for GitHub Copilot's Embedding API.

    Reference: https://api.githubcopilot.com/embeddings
    """

    def __init__(self) -> None:
        super().__init__()

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment and set up headers for GitHub Copilot API.
        """
        try:
            if not api_key:
                raise AuthenticationError(
                    model=model,
                    llm_provider="github_copilot",
                    message="GitHub Copilot API key is required. Please authenticate via OAuth Device Flow.",
                )

            # api_key is the GitHub access token (ghu_xxx). Exchange it for
            # a short-lived Copilot inference token (cached at module level).
            authenticator = Authenticator(access_token=api_key)
            inference_token = authenticator.get_api_key()
            default_headers = get_copilot_default_headers(inference_token)

            # Merge with existing headers (user's extra_headers take priority)
            merged_headers = {**default_headers, **headers}

            verbose_logger.debug(
                f"GitHub Copilot Embedding API: Successfully configured headers for model {model}"
            )

            return merged_headers

        except (GetAPIKeyError, GetAccessTokenError) as e:
            raise AuthenticationError(
                model=model,
                llm_provider="github_copilot",
                message=str(e),
            )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for GitHub Copilot Embedding API endpoint.
        """
        # Use provided api_base or fall back to credential-resolved base or default
        if api_key:
            derived_api_base = Authenticator(access_token=api_key).get_api_base()
            api_base = api_base or derived_api_base or GITHUB_COPILOT_API_BASE
        else:
            api_base = api_base or GITHUB_COPILOT_API_BASE

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # Return the embeddings endpoint
        return f"{api_base}/embeddings"

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform embedding request to GitHub Copilot format.
        """

        # Ensure input is a list
        if isinstance(input, str):
            input = [input]

        # Strip 'github_copilot/' prefix if present
        if model.startswith("github_copilot/"):
            model = model.replace("github_copilot/", "", 1)

        return {
            "model": model,
            "input": input,
            **optional_params,
        }

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        """
        Transform embedding response from GitHub Copilot format.
        """
        logging_obj.post_call(original_response=raw_response.text)

        # GitHub Copilot returns standard OpenAI-compatible embedding response
        response_json = raw_response.json()

        return convert_to_model_response_object(
            response_object=response_json,
            model_response_object=model_response,
            response_type="embedding",
        )

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "timeout",
            "dimensions",
            "encoding_format",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param in self.get_supported_openai_params(model):
                optional_params[param] = value
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Any
    ) -> Any:
        from litellm.llms.openai.openai import OpenAIConfig

        return OpenAIConfig().get_error_class(
            error_message=error_message, status_code=status_code, headers=headers
        )

from typing import Optional, cast

import httpx

import litellm
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import _add_path_to_api_base


class AzureImageEditConfig(OpenAIImageEditConfig):
    @staticmethod
    def azure_deployment_image_edit_form_data(data: dict, request_url: str) -> dict:
        """
        Azure OpenAI ``.../openai/deployments/{deployment}/images/edits`` routes by
        deployment in the URL; including ``model`` in multipart fields can break
        the same way as image generations (LiteLLM #26316).

        Non-deployment edit URLs keep ``model`` when present.
        """
        if "images/edits" in request_url and "/openai/deployments/" in request_url:
            return {k: v for k, v in data.items() if k != "model"}
        return data

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate Azure environment and set up authentication headers.

        Delegates to ``BaseAzureLLM._base_validate_azure_environment`` so the
        Azure image-edit route uses the same auth resolution as every other
        Azure provider (videos, vector_stores, responses, containers, ...):

        - prefers the Azure-style ``api-key`` header when an API key is available
        - falls back to ``Authorization: Bearer <azure_ad_token>`` only when AAD
          auth is configured

        The previous implementation unconditionally set
        ``Authorization: Bearer <api_key>``, which is correct for OpenAI direct
        but not for Azure OpenAI / API Management gateways that expect the
        ``api-key`` header. Subscription-key-based deployments (e.g., behind
        Azure APIM) responded with ``401 "Access denied due to missing
        subscription key"``.

        API-key precedence (matches ``AzureVideosConfig``):

        - ``litellm_params["api_key"]`` is the source of truth.
        - The positional ``api_key`` kwarg only fills in when
          ``litellm_params["api_key"]`` is empty.
        - This is a deliberate change from the old ``or`` chain (where the
          positional ``api_key`` argument won) so behavior matches every other
          Azure ``validate_environment`` implementation. In production the only
          caller (``llm_http_handler.image_edit``) sources both values from
          the same ``litellm_params.api_key``, so the precedence only matters
          for direct callers of this method.
        """
        params = GenericLiteLLMParams(**(litellm_params or {}))
        if api_key is not None and params.api_key is None:
            params.api_key = api_key
        return BaseAzureLLM._base_validate_azure_environment(
            headers=headers, litellm_params=params
        )

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Constructs a complete URL for the API request.

        Args:
        - api_base: Base URL, e.g.,
            "https://litellm8397336933.openai.azure.com"
            OR
            "https://litellm8397336933.openai.azure.com/openai/deployments/<deployment_name>/images/edits?api-version=2024-05-01-preview"
        - model: Model name (deployment name).
        - litellm_params: Additional query parameters, including "api_version".

        Returns:
        - A complete URL string, e.g.,
        "https://litellm8397336933.openai.azure.com/openai/deployments/<deployment_name>/images/edits?api-version=2024-05-01-preview"
        """
        api_base = api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")
        if api_base is None:
            raise ValueError(
                f"api_base is required for Azure AI Studio. Please set the api_base parameter. Passed `api_base={api_base}`"
            )
        original_url = httpx.URL(api_base)

        # Extract api_version or use default
        api_version = cast(Optional[str], litellm_params.get("api_version"))

        # Create a new dictionary with existing params
        query_params = dict(original_url.params)

        # Add api_version if needed
        if "api-version" not in query_params and api_version:
            query_params["api-version"] = api_version

        # Add the path to the base URL using the model as deployment name
        if "/openai/deployments/" not in api_base:
            new_url = _add_path_to_api_base(
                api_base=api_base,
                ending_path=f"/openai/deployments/{model}/images/edits",
            )
        else:
            new_url = api_base

        # Use the new query_params dictionary
        final_url = httpx.URL(new_url).copy_with(params=query_params)

        return str(final_url)

    def finalize_image_edit_request_data(
        self, data: dict, resolved_request_url: str
    ) -> dict:
        return self.azure_deployment_image_edit_form_data(data, resolved_request_url)

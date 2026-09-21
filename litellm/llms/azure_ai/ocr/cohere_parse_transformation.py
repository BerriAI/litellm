"""Cohere Parse served from Azure AI Foundry (`/providers/cohere/v2/parse`)."""

from collections.abc import Mapping
from typing import Final

import httpx

from litellm.litellm_core_utils.prompt_templates.image_handling import (
    async_convert_url_to_base64,
    convert_url_to_base64,
)
from litellm.llms.azure_ai.common_utils import get_azure_ai_auth_headers
from litellm.llms.cohere.ocr.transformation import COHERE_PARSE_PATH, CohereParseConfig
from litellm.secret_managers.main import get_secret_str

AZURE_AI_API_KEY_ENV_VAR: Final = "AZURE_AI_API_KEY"
AZURE_AI_API_BASE_ENV_VAR: Final = "AZURE_AI_API_BASE"
AZURE_AI_COHERE_PROVIDER_PATH: Final = "/providers/cohere"
AZURE_AI_MODELS_PATH_SUFFIX: Final = "/models"


class AzureAICohereParseConfig(CohereParseConfig):
    """Same request and response shape as Cohere Parse, behind Azure AI auth and URL layout.

    Foundry cannot fetch external URLs, so remote images are inlined as base64 data URIs.
    """

    def get_api_key_env_var(self) -> str | None:
        return AZURE_AI_API_KEY_ENV_VAR

    def _llm_provider(self) -> str:
        return "azure_ai"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        **kwargs: object,  # kwargs-ok: BaseOCRConfig.validate_environment signature
    ) -> dict[str, str]:  # mutable-ok: BaseOCRConfig signature
        resolved_base: Final = api_base or get_secret_str(AZURE_AI_API_BASE_ENV_VAR)
        if resolved_base is None:
            raise ValueError(
                f"Missing Azure AI API Base - Set {AZURE_AI_API_BASE_ENV_VAR} environment variable "
                "or pass api_base parameter"
            )
        resolved_key: Final = api_key or get_secret_str(AZURE_AI_API_KEY_ENV_VAR)
        return {  # mutable-ok: BaseOCRConfig signature
            **get_azure_ai_auth_headers(api_key=resolved_key, litellm_params=litellm_params),
            "Content-Type": "application/json",
            **headers,
        }

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object] | None = None,
        **kwargs: object,  # kwargs-ok: BaseOCRConfig.get_complete_url signature
    ) -> str:
        resolved_base: Final = api_base or get_secret_str(AZURE_AI_API_BASE_ENV_VAR)
        if resolved_base is None:
            raise ValueError(
                f"Missing Azure AI API Base - Set {AZURE_AI_API_BASE_ENV_VAR} environment variable "
                "or pass api_base parameter"
            )
        url: Final = httpx.URL(resolved_base)
        if not url.is_absolute_url:
            raise ValueError(
                "Azure AI API Base must be an absolute URL including scheme (e.g. "
                f"'https://<resource>.services.ai.azure.com'). Got api_base={resolved_base!r}."
            )
        path: Final = url.path.rstrip("/")
        if path.endswith(COHERE_PARSE_PATH):
            return str(url.copy_with(path=path))
        if path.endswith(f"{AZURE_AI_COHERE_PROVIDER_PATH}/v2"):
            return str(url.copy_with(path=f"{path}/parse"))
        return str(
            url.copy_with(
                path=f"{path.removesuffix(AZURE_AI_MODELS_PATH_SUFFIX)}{AZURE_AI_COHERE_PROVIDER_PATH}{COHERE_PARSE_PATH}"
            )
        )

    def _resolve_image_url_sync(self, image_url: str) -> str:
        return convert_url_to_base64(image_url)

    async def _resolve_image_url_async(self, image_url: str) -> str:
        return await async_convert_url_to_base64(image_url)

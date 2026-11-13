import asyncio
from collections.abc import Mapping
from typing import Final

from pydantic import ValidationError

import litellm
from litellm.llms.azure.common_utils import (
    BaseAzureLLM,
    cached_azure_ad_token_refresh_provider,
)
from litellm.llms.openai.decisions.transformation import OpenAIDecisionsConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams


class AzureDecisionsConfig(OpenAIDecisionsConfig):
    """Azure OpenAI Decisions, with the Azure deployment name in the request body."""

    api_key_required = False

    def resolve_api_base(self, api_base: str | None) -> str | None:
        return BaseAzureLLM.resolve_azure_api_base(api_base)

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return api_key

    def resolve_credential(
        self, api_key: str | None, api_base: str, litellm_params: Mapping[str, object]
    ) -> str | None:
        return api_key

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, str]:
        try:
            params: Final = GenericLiteLLMParams.model_validate({**litellm_params, "api_key": api_key})
        except ValidationError as error:
            raise litellm.BadRequestError(
                message="Invalid Azure Decisions credential parameters",
                model=model,
                llm_provider="azure",
            ) from error
        clean_headers: Final = super().validate_environment(
            {name: value for name, value in headers.items() if name.lower() != "api-key"},
            model,
            None,
            litellm_params,
        )
        azure_headers: Final = BaseAzureLLM.base_validate_azure_environment(
            headers=dict(clean_headers), litellm_params=params
        )
        if "api-key" in azure_headers:
            return azure_headers
        if "Authorization" in azure_headers:
            return azure_headers
        scope: Final = (
            params.azure_scope or get_secret_str("AZURE_SCOPE") or "https://cognitiveservices.azure.com/.default"
        )
        azure_headers["Authorization"] = f"Bearer {cached_azure_ad_token_refresh_provider(scope)()}"
        return azure_headers

    async def avalidate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, str]:
        return await asyncio.to_thread(self.validate_environment, headers, model, api_key, litellm_params)

    def get_complete_url(self, api_base: str, model: str) -> str:
        return BaseAzureLLM.get_base_azure_url(api_base, None, route="/openai/v1/decisions")

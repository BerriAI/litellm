import asyncio
from collections.abc import Mapping
from typing import Final

from pydantic import ValidationError

import litellm
from litellm.llms.azure.common_utils import (
    BaseAzureLLM,
    cached_azure_ad_token_refresh_provider,
    get_azure_ad_token,
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
        return (
            api_key
            or litellm.api_key
            or litellm.azure_key
            or get_secret_str("AZURE_OPENAI_API_KEY")
            or get_secret_str("AZURE_API_KEY")
        )

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
        clean_headers: Final = super().validate_environment(
            {name: value for name, value in headers.items() if name.lower() != "api-key"},
            model,
            None,
            litellm_params,
        )
        if api_key:
            return {**clean_headers, "api-key": api_key}
        try:
            params: Final = GenericLiteLLMParams.model_validate(litellm_params)
        except ValidationError as error:
            raise litellm.BadRequestError(
                message="Invalid Azure Decisions credential parameters",
                model=model,
                llm_provider="azure",
            ) from error
        token: Final = self._resolve_entra_token(params)
        return {**clean_headers, "Authorization": f"Bearer {token}"}

    @staticmethod
    def _resolve_entra_token(params: GenericLiteLLMParams) -> str:
        token: Final = get_azure_ad_token(params)
        if token:
            return token
        scope: Final = (
            params.azure_scope or get_secret_str("AZURE_SCOPE") or "https://cognitiveservices.azure.com/.default"
        )
        return cached_azure_ad_token_refresh_provider(scope)()

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
